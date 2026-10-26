"""Item-item collaborative filtering + evaluation skeleton.

Expects MovieLens ratings.csv (userId, movieId, rating, timestamp).
Run after:  python src/download_data.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def time_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    """Per-user split: each user's latest `test_frac` ratings go to test."""
    ratings = ratings.sort_values("timestamp")
    rank = ratings.groupby("userId").cumcount(ascending=False) + 1
    size = ratings.groupby("userId")["rating"].transform("size")
    is_test = rank <= np.ceil(size * test_frac)
    return ratings[~is_test], ratings[is_test]


class ItemItemCF:
    def fit(self, train: pd.DataFrame):
        self.mat = train.pivot_table(index="userId", columns="movieId", values="rating")
        self.global_mean = train["rating"].mean()
        filled = self.mat.sub(self.mat.mean(axis=1), axis=0).fillna(0)  # centre by user
        self.sim = pd.DataFrame(
            cosine_similarity(filled.T), index=filled.columns, columns=filled.columns
        )
        self.user_mean = self.mat.mean(axis=1)
        return self

    def predict(self, user: int, movie: int, k: int = 20) -> float:
        if user not in self.mat.index or movie not in self.sim.index:
            return self.global_mean
        rated = self.mat.loc[user].dropna()
        sims = self.sim.loc[movie, rated.index].nlargest(k)
        sims = sims[sims > 0]
        if sims.empty:
            return self.user_mean[user]
        return float(np.dot(sims, rated[sims.index]) / sims.sum())


def rmse(model: ItemItemCF, test: pd.DataFrame, limit: int = 2000) -> float:
    test = test.head(limit)
    preds = [model.predict(u, m) for u, m in zip(test["userId"], test["movieId"])]
    return float(np.sqrt(np.mean((test["rating"].values - np.array(preds)) ** 2)))


if __name__ == "__main__":
    r = pd.read_csv("data/ml-latest-small/ratings.csv")
    train, test = time_split(r)
    base = float(np.sqrt(np.mean((test["rating"] - train["rating"].mean()) ** 2)))
    print("Baseline RMSE (global mean):", round(base, 3))
    print("Item-item CF RMSE:", round(rmse(ItemItemCF().fit(train), test), 3))
