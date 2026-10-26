"""Temporal evaluation and item-item collaborative filtering for MovieLens."""

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

RATINGS_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_RELEVANCE_THRESHOLD = 4.0
RATING_MIN = 0.5
RATING_MAX = 5.0
RECOMMENDATION_SHRINKAGE = 2.0
SIMILARITY_SHRINKAGE = 10.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "ratings.csv"
METRICS_CHART_PATH = PROJECT_ROOT / "reports" / "model-metrics.png"


def _validate_ratings(ratings: pd.DataFrame, name: str) -> None:
    """Validate columns and non-empty input for rating tables."""
    missing = RATINGS_COLUMNS.difference(ratings.columns)
    if missing:
        raise ValueError(f"{name} must contain {sorted(RATINGS_COLUMNS)}")
    if ratings.empty:
        raise ValueError(f"{name} must contain at least one row")


def time_split(
    ratings: pd.DataFrame, test_frac: float = DEFAULT_TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's latest timestamp block, preserving earlier training data.

    Users with only one distinct timestamp remain in training because no valid
    chronological holdout exists for them. Timestamp ties are kept together.
    """
    _validate_ratings(ratings, "ratings")
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")

    ordered = ratings.sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    )
    train_parts = []
    test_parts = []
    for _, user_ratings in ordered.groupby("userId", sort=True):
        if len(user_ratings) < 2 or user_ratings["timestamp"].nunique() < 2:
            train_parts.append(user_ratings)
            continue
        test_size = min(ceil(len(user_ratings) * test_frac), len(user_ratings) - 1)
        cutoff = user_ratings.iloc[-test_size]["timestamp"]
        earlier = user_ratings[user_ratings["timestamp"] < cutoff]
        latest = user_ratings[user_ratings["timestamp"] >= cutoff]
        if earlier.empty:
            train_parts.append(user_ratings)
        else:
            train_parts.append(earlier)
            test_parts.append(latest)

    train = pd.concat(train_parts).sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    )
    test = (
        pd.concat(test_parts).sort_values(
            ["userId", "timestamp", "movieId"], kind="mergesort"
        )
        if test_parts
        else ordered.iloc[0:0]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def global_mean_rmse(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Score the held-out ratings with a single mean fitted on training data."""
    _validate_ratings(train, "train")
    _validate_ratings(test, "test")
    prediction = float(train["rating"].mean())
    return float(np.sqrt(np.mean((test["rating"].to_numpy() - prediction) ** 2)))


def _seen_movies(train: pd.DataFrame) -> dict[int, set[int]]:
    """Map each training user to their already-rated movie IDs."""
    return train.groupby("userId")["movieId"].agg(lambda values: set(values)).to_dict()


def _popularity_order(train: pd.DataFrame) -> list[int]:
    """Rank the training catalog by rating count, with movie ID as tie-break."""
    popularity = train.groupby("movieId").size().rename("count").reset_index()
    popularity = popularity.sort_values(
        ["count", "movieId"], ascending=[False, True], kind="mergesort"
    )
    return [int(movie_id) for movie_id in popularity["movieId"]]


def popular_recommendations(
    train: pd.DataFrame, user_id: int, k: int = 10
) -> list[int]:
    """Recommend the most-rated training movies the user has not rated."""
    _validate_ratings(train, "train")
    if k < 1:
        raise ValueError("k must be positive")
    seen = _seen_movies(train).get(user_id, set())
    return [
        movie_id
        for movie_id in _popularity_order(train)
        if movie_id not in seen
    ][:k]


def _relevant_test_items(
    train: pd.DataFrame,
    test: pd.DataFrame,
    relevance_threshold: float,
) -> dict[int, set[int]]:
    """Return held-out relevant items that exist in the train catalog."""
    catalog = set(train["movieId"])
    seen = _seen_movies(train)
    relevant = {}
    positive = test[test["rating"] >= relevance_threshold]
    for user_id, group in positive.groupby("userId"):
        relevant_items = set(group["movieId"]) & catalog
        relevant_items -= seen.get(user_id, set())
        if relevant_items:
            relevant[int(user_id)] = relevant_items
    return relevant


def precision_at_k(
    recommendations: list[int], relevant_items: set[int], k: int = 10
) -> float:
    """Compute fixed-k precision for a single user's ranked recommendations."""
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(recommendations[:k]) & relevant_items) / k


def evaluate_popularity_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict[str, float | int]:
    """Evaluate a training-only most-popular ranking on relevant held-out items."""
    _validate_ratings(train, "train")
    _validate_ratings(test, "test")
    if k < 1:
        raise ValueError("k must be positive")
    relevant = _relevant_test_items(train, test, relevance_threshold)
    if not relevant:
        raise ValueError("No held-out relevant items exist in the training catalog")
    seen = _seen_movies(train)
    popularity = _popularity_order(train)
    scores = [
        precision_at_k(
            [movie_id for movie_id in popularity if movie_id not in seen.get(user_id, set())][:k],
            items,
            k,
        )
        for user_id, items in relevant.items()
    ]
    return {
        "precision_at_k": float(np.mean(scores)),
        "eligible_users": len(scores),
        "k": k,
        "relevance_threshold": relevance_threshold,
    }


class ItemItemCF:
    """Sparse item-item collaborative filter using user-centered rating cosine."""

    def fit(self, train: pd.DataFrame) -> "ItemItemCF":
        """Fit item similarities and user means on training ratings only."""
        _validate_ratings(train, "train")
        if train.duplicated(["userId", "movieId"]).any():
            raise ValueError("train must contain at most one rating per user and movie")
        self.global_mean = float(train["rating"].mean())
        self.users = np.sort(train["userId"].unique())
        self.movies = np.sort(train["movieId"].unique())
        self._user_index = {user_id: index for index, user_id in enumerate(self.users)}
        self._movie_index = {movie_id: index for index, movie_id in enumerate(self.movies)}
        self.user_mean = train.groupby("userId")["rating"].mean().to_dict()

        rows = train["userId"].map(self._user_index).to_numpy()
        columns = train["movieId"].map(self._movie_index).to_numpy()
        deviations = (
            train["rating"] - train["userId"].map(self.user_mean)
        ).to_numpy(dtype=float)
        user_item = csr_matrix(
            (deviations, (rows, columns)),
            shape=(len(self.users), len(self.movies)),
        )
        binary_user_item = csr_matrix(
            (np.ones(len(train), dtype=float), (rows, columns)),
            shape=(len(self.users), len(self.movies)),
        )
        co_ratings = (binary_user_item.T @ binary_user_item).tocsr()
        co_ratings.data /= co_ratings.data + SIMILARITY_SHRINKAGE
        self.similarity = cosine_similarity(
            user_item.T, dense_output=False
        ).tocsr().multiply(co_ratings).tocsr()
        self._similarity_columns = self.similarity.tocsc()

        self._user_ratings = {}
        for user_id, group in train.groupby("userId", sort=False):
            positions = group["movieId"].map(self._movie_index).to_numpy(dtype=int)
            values = group["rating"].to_numpy(dtype=float)
            self._user_ratings[user_id] = (positions, values)
        return self

    def predict_many(self, user_id: int, movie_ids: list[int]) -> np.ndarray:
        """Predict multiple ratings, falling back to train means for cold entities."""
        predictions = np.full(len(movie_ids), self.global_mean, dtype=float)
        if user_id not in self._user_ratings:
            return predictions

        user_mean = float(self.user_mean[user_id])
        rated_positions, rated_values = self._user_ratings[user_id]
        predictions[:] = user_mean
        candidate_positions = np.array(
            [self._movie_index.get(movie_id, -1) for movie_id in movie_ids], dtype=int
        )
        known = candidate_positions >= 0
        if not known.any():
            return predictions

        weights = self.similarity[candidate_positions[known]][:, rated_positions].tocsr()
        weights.data = np.maximum(weights.data, 0)
        weights.eliminate_zeros()
        weight_totals = np.asarray(weights.sum(axis=1)).ravel()
        residuals = rated_values - user_mean
        adjustments = np.divide(
            np.asarray(weights @ residuals).ravel(),
            weight_totals,
            out=np.zeros_like(weight_totals),
            where=weight_totals > 0,
        )
        predictions[known] = np.clip(user_mean + adjustments, RATING_MIN, RATING_MAX)
        return predictions

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict one rating using positive similarities to the user's history."""
        return float(self.predict_many(user_id, [movie_id])[0])

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Rank unseen training-catalog movies by predicted user rating."""
        if n < 1:
            raise ValueError("n must be positive")
        if user_id not in self._user_ratings:
            raise KeyError(f"Unknown user: {user_id}")
        rated_positions, rated_values = self._user_ratings[user_id]
        is_unseen = np.ones(len(self.movies), dtype=bool)
        is_unseen[rated_positions] = False
        candidate_positions = np.flatnonzero(is_unseen)
        if not len(candidate_positions):
            return pd.DataFrame(columns=["movieId", "predicted_rating"])

        weights = self._similarity_columns[:, rated_positions][candidate_positions].tocsr()
        weights.data = np.maximum(weights.data, 0)
        weights.eliminate_zeros()
        totals = np.asarray(weights.sum(axis=1)).ravel()
        mean = float(self.user_mean[user_id])
        residuals = rated_values - mean
        adjustments = (
            np.asarray(weights @ residuals).ravel()
            / (totals + RECOMMENDATION_SHRINKAGE)
        )
        scores = np.clip(mean + adjustments, RATING_MIN, RATING_MAX)
        movie_ids = self.movies[candidate_positions]
        ordering = np.lexsort((movie_ids, -scores))[:n]
        return pd.DataFrame(
            {
                "movieId": movie_ids[ordering].astype(int),
                "predicted_rating": scores[ordering],
            }
        ).reset_index(drop=True)


def rmse(model: ItemItemCF, test: pd.DataFrame) -> float:
    """Evaluate all held-out ratings using per-user vectorized predictions."""
    _validate_ratings(test, "test")
    ordered = test.reset_index(drop=True)
    predictions = np.empty(len(ordered), dtype=float)
    for user_id, positions in ordered.groupby("userId", sort=False).indices.items():
        movies = ordered.iloc[positions]["movieId"].tolist()
        predictions[positions] = model.predict_many(user_id, movies)
    return float(np.sqrt(np.mean((ordered["rating"].to_numpy() - predictions) ** 2)))


def evaluate_collaborative_precision(
    model: ItemItemCF,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict[str, float | int]:
    """Evaluate item-item top-K recommendations on held-out relevant titles."""
    relevant = _relevant_test_items(train, test, relevance_threshold)
    if not relevant:
        raise ValueError("No held-out relevant items exist in the training catalog")
    scores = []
    for user_id, items in relevant.items():
        recommendations = model.recommend_for_user(user_id, n=k)["movieId"].tolist()
        scores.append(precision_at_k(recommendations, items, k))
    return {
        "precision_at_k": float(np.mean(scores)),
        "eligible_users": len(scores),
        "k": k,
        "relevance_threshold": relevance_threshold,
    }


def evaluate_model(
    model: ItemItemCF,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> dict[str, float | int]:
    """Compare item-item filtering with global-mean and popularity baselines."""
    popularity = evaluate_popularity_baseline(train, test, k, relevance_threshold)
    collaborative = evaluate_collaborative_precision(
        model, train, test, k, relevance_threshold
    )
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "global_mean_rmse": global_mean_rmse(train, test),
        "item_item_rmse": rmse(model, test),
        "popularity_precision_at_k": popularity["precision_at_k"],
        "item_item_precision_at_k": collaborative["precision_at_k"],
        "precision_eligible_users": collaborative["eligible_users"],
        "k": k,
        "relevance_threshold": relevance_threshold,
    }


def save_metric_chart(
    results: dict[str, float | int],
    output_path: str | Path = METRICS_CHART_PATH,
) -> Path:
    """Save separate RMSE and Precision@K baseline comparison charts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    rmse_values = [results["global_mean_rmse"], results["item_item_rmse"]]
    precision_values = [
        results["popularity_precision_at_k"],
        results["item_item_precision_at_k"],
    ]
    axes[0].bar(["Global mean", "Item-item CF"], rmse_values, color=["#9ca3af", "#6958d6"])
    axes[0].set(title="Held-out RMSE (lower is better)", ylabel="RMSE")
    axes[1].bar(["Most popular", "Item-item CF"], precision_values, color=["#9ca3af", "#e87954"])
    axes[1].set(
        title=f"Held-out Precision@{results['k']} (higher is better)",
        ylabel="Precision",
        ylim=(0, max(precision_values) * 1.25),
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=10)
        for bar in axis.patches:
            value = bar.get_height()
            axis.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> None:
    """Run temporal MovieLens evaluation and print held-out baseline comparisons."""
    if not RATINGS_PATH.is_file():
        raise FileNotFoundError("Run python src/download_data.py before this evaluation.")
    ratings = pd.read_csv(RATINGS_PATH)
    train, test = time_split(ratings)
    model = ItemItemCF().fit(train)
    results = evaluate_model(model, train, test)
    print(pd.Series(results).to_string())
    print("\nMetrics chart:", save_metric_chart(results))


if __name__ == "__main__":
    main()
