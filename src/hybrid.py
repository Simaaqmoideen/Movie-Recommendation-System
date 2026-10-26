"""Weighted content and collaborative recommendations for new-user profiles."""

import re

import pandas as pd

from src.collab import ItemItemCF
from src.recommender import ContentRecommender, _normalize_title


def _catalog_key(title: str) -> str:
    """Normalize MovieLens titles, years, and inverted leading articles."""
    cleaned = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    key = _normalize_title(cleaned)
    match = re.fullmatch(r"(.+),\s*(the|a|an)", key)
    if match:
        key = f"{match.group(2)} {match.group(1)}"
    return key


class HybridRecommender:
    """Blend a content profile with community item-item predictions.

    If no collaborative model or no title ID match is available, the result
    gracefully falls back to content relevance for that catalog item.
    """

    def __init__(
        self,
        content: ContentRecommender,
        collaborative: ItemItemCF | None = None,
        movie_catalog: pd.DataFrame | None = None,
        collaborative_weight: float = 0.5,
    ) -> None:
        if not 0 <= collaborative_weight <= 1:
            raise ValueError("collaborative_weight must be between 0 and 1")
        self.content = content
        self.collaborative = collaborative
        self.collaborative_weight = collaborative_weight
        self.catalog_ids: dict[str, int] = {}
        if movie_catalog is not None:
            required = {"movieId", "title"}
            if required.difference(movie_catalog.columns):
                raise ValueError("movie_catalog must contain movieId and title")
            self.catalog_ids = {
                _catalog_key(row.title): int(row.movieId)
                for row in movie_catalog.itertuples(index=False)
            }

    def recommend_for_ratings(
        self, ratings: dict[str, float], n: int = 5
    ) -> pd.DataFrame:
        """Return ranked unseen titles from three to five new-user ratings."""
        if not 3 <= len(ratings) <= 5:
            raise ValueError("rate between 3 and 5 movies to build a profile")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        title_keys = self.content.movies["title"].map(_normalize_title)
        title_to_position = dict(zip(title_keys, self.content.movies.index, strict=True))
        normalized_ratings: dict[int, float] = {}
        for title, rating in ratings.items():
            key = _normalize_title(title)
            if key not in title_to_position:
                raise KeyError(f"Unknown title: {title!r}")
            if not isinstance(rating, (int, float)) or not 0.5 <= rating <= 5:
                raise ValueError("profile ratings must be between 0.5 and 5.0")
            normalized_ratings[title_to_position[key]] = float(rating)
        if len(normalized_ratings) != len(ratings):
            raise ValueError("rate each movie only once")

        rated_positions = list(normalized_ratings)
        rated_values = pd.Series(normalized_ratings)
        centered = rated_values - rated_values.mean()
        similarities = self.content.sim[:, rated_positions]
        content_scores = (similarities @ centered.to_numpy()).ravel()
        normalizer = max(float(abs(centered).sum()), 1.0)
        content_scores = (content_scores / normalizer + 1) / 2

        cf_scores: dict[int, float] = {}
        if self.collaborative is not None:
            profile_ids = {
                movie_id: rating
                for title, rating in ratings.items()
                if (movie_id := self.catalog_ids.get(_catalog_key(title))) is not None
            }
            profile_predictions = self.collaborative.recommend_from_profile(profile_ids, n=len(self.content.movies))
            cf_scores = {
                int(row.movieId): (float(row.predicted_rating) - 0.5) / 4.5
                for row in profile_predictions.itertuples(index=False)
            }

        rated_set = set(rated_positions)
        results = []
        for position, movie in self.content.movies.iterrows():
            if position in rated_set:
                continue
            content_score = float(content_scores[position])
            movie_id = self.catalog_ids.get(_catalog_key(movie["title"]))
            cf_score = cf_scores.get(movie_id) if movie_id is not None else None
            if cf_score is None:
                score = content_score
            else:
                weight = self.collaborative_weight
                score = (1 - weight) * content_score + weight * cf_score
            results.append(
                {
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "score": score,
                    "content_score": content_score,
                    "collaborative_score": cf_score,
                }
            )
        result = pd.DataFrame(results)
        return result.sort_values(
            ["score", "title"], ascending=[False, True], kind="mergesort"
        ).head(n).reset_index(drop=True)
