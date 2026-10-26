"""Content-based movie recommendations using TF-IDF and cosine similarity."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parents[1] / "data" / "sample_movies.csv"
REQUIRED_COLUMNS = {"title", "genres"}


def _normalize_title(title: str) -> str:
    """Normalize user-entered titles for case-insensitive lookup."""
    return " ".join(title.casefold().split())


class ContentRecommender:
    """Recommend titles with similar genre and optional overview text."""

    def __init__(self, movies: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(movies.columns)
        if missing:
            raise ValueError(f"movies must contain {sorted(REQUIRED_COLUMNS)}")
        if movies.empty:
            raise ValueError("movies must contain at least one title")

        self.movies = movies.copy().reset_index(drop=True)
        self.movies["title"] = self.movies["title"].astype("string").str.strip()
        if self.movies["title"].isna().any() or self.movies["title"].eq("").any():
            raise ValueError("movie titles must be non-empty")
        self.movies["genres"] = self.movies["genres"].fillna("").astype(str)
        if "overview" in self.movies:
            overview = self.movies["overview"].fillna("").astype(str)
        else:
            overview = pd.Series("", index=self.movies.index, dtype="string")

        keys = self.movies["title"].map(_normalize_title)
        if keys.duplicated().any():
            raise ValueError("movie titles must be unique ignoring case and whitespace")
        self._idx = dict(zip(keys, self.movies.index, strict=True))

        text = self.movies["genres"].str.replace("|", " ", regex=False) + " " + overview
        if not text.str.strip().any():
            raise ValueError("movies must include at least one genre or overview term")
        vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = vectorizer.fit_transform(text)
        self.sim = cosine_similarity(self.matrix)

    def recommend_similar(self, title: str, n: int = 5) -> pd.DataFrame:
        """Return the top-N similar movies, excluding the selected title."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")

        key = _normalize_title(title)
        if key not in self._idx:
            raise KeyError(f"Unknown title: {title!r}")
        index = self._idx[key]
        title_key = self.movies["title"].map(_normalize_title)
        order = sorted(
            (candidate for candidate in range(len(self.movies)) if candidate != index),
            key=lambda candidate: (
                -float(self.sim[index, candidate]),
                title_key.iloc[candidate],
            ),
        )[:n]
        result = self.movies.iloc[order][["title", "genres"]].copy()
        result["score"] = [round(float(self.sim[index, item]), 3) for item in order]
        return result.reset_index(drop=True)


def load_sample() -> ContentRecommender:
    """Load the small included movie sample for an offline demo."""
    return ContentRecommender(pd.read_csv(DATA))


if __name__ == "__main__":
    recommender = load_sample()
    for movie_title in ["Toy Story", "The Matrix", "Titanic"]:
        print(f"\nBecause you liked {movie_title}:")
        print(recommender.recommend_similar(movie_title, 3).to_string(index=False))
