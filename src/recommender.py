"""Content-based recommender (TF-IDF over genres + overview)."""
import pathlib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "sample_movies.csv"


class ContentRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True)
        text = (
            self.movies["genres"].str.replace("|", " ", regex=False) + " "
            + self.movies.get("overview", "").fillna("")
        )
        self.matrix = TfidfVectorizer(stop_words="english").fit_transform(text)
        self.sim = cosine_similarity(self.matrix)
        self._idx = {t.lower(): i for i, t in enumerate(self.movies["title"])}

    def recommend_similar(self, title: str, n: int = 5) -> pd.DataFrame:
        key = title.strip().lower()
        if key not in self._idx:
            raise KeyError(f"Unknown title: {title!r}")
        i = self._idx[key]
        order = self.sim[i].argsort()[::-1]
        order = [j for j in order if j != i][:n]
        out = self.movies.iloc[order][["title", "genres"]].copy()
        out["score"] = self.sim[i][order].round(3)
        return out.reset_index(drop=True)


def load_sample() -> ContentRecommender:
    return ContentRecommender(pd.read_csv(DATA))


if __name__ == "__main__":
    rec = load_sample()
    for t in ["Toy Story", "The Matrix", "Titanic"]:
        print(f"\nBecause you liked {t}:")
        print(rec.recommend_similar(t, 3).to_string(index=False))
