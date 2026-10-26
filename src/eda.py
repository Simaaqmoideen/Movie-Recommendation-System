"""Data loading, quality checks, and plots for the MovieLens small dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-latest-small"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
RATING_MIN = 0.5
RATING_MAX = 5.0


def load_movielens(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movie metadata, validating the expected MovieLens columns."""
    directory = Path(data_dir)
    ratings_path = directory / "ratings.csv"
    movies_path = directory / "movies.csv"
    if not ratings_path.is_file() or not movies_path.is_file():
        raise FileNotFoundError(
            f"MovieLens files are missing from {directory}. Run python src/download_data.py."
        )

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    required_ratings = {"userId", "movieId", "rating", "timestamp"}
    required_movies = {"movieId", "title", "genres"}
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain {sorted(required_ratings)}")
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"movies.csv must contain {sorted(required_movies)}")
    return ratings, movies


def summarize_ratings(
    ratings: pd.DataFrame, movie_count: int | None = None
) -> dict[str, int | float]:
    """Summarize coverage, missing values, duplicates, and rating distribution."""
    required = {"userId", "movieId", "rating", "timestamp"}
    if not required.issubset(ratings.columns):
        raise ValueError(f"ratings must contain {sorted(required)}")
    if ratings.empty:
        raise ValueError("ratings must contain at least one row")

    users = int(ratings["userId"].nunique())
    rated_movies = int(ratings["movieId"].nunique())
    catalog_size = movie_count if movie_count is not None else rated_movies
    if catalog_size < rated_movies or catalog_size <= 0:
        raise ValueError("movie_count must cover all rated movies and be positive")
    return {
        "ratings": int(len(ratings)),
        "users": users,
        "rated_movies": rated_movies,
        "catalog_movies": int(catalog_size),
        "sparsity": float(1 - len(ratings) / (users * catalog_size)),
        "mean_rating": float(ratings["rating"].mean()),
        "rating_std": float(ratings["rating"].std()),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "ratings_outside_expected_range": int(
            (~ratings["rating"].between(RATING_MIN, RATING_MAX)).sum()
        ),
        "missing_values": int(ratings.isna().sum().sum()),
        "duplicate_user_movie_pairs": int(ratings.duplicated(["userId", "movieId"]).sum()),
    }


def top_movies(
    ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Return the most-rated movies with title and mean rating."""
    if n < 1:
        raise ValueError("n must be at least 1")
    counts = ratings.groupby("movieId").agg(
        rating_count=("rating", "size"), mean_rating=("rating", "mean")
    )
    result = movies[["movieId", "title", "genres"]].merge(
        counts, on="movieId", how="inner", validate="one_to_one"
    )
    return (
        result.sort_values(
            ["rating_count", "title"], ascending=[False, True], kind="mergesort"
        )
        .head(n)
        .reset_index(drop=True)
    )


def genre_popularity(movies: pd.DataFrame) -> pd.DataFrame:
    """Count catalog titles by genre; multi-genre movies count in each genre."""
    if not {"movieId", "genres"}.issubset(movies.columns):
        raise ValueError("movies must contain movieId and genres columns")
    genre_rows = movies[["movieId", "genres"]].copy()
    genre_rows["genre"] = genre_rows["genres"].fillna("").str.split("|")
    genre_rows = genre_rows.explode("genre")
    genre_rows["genre"] = genre_rows["genre"].str.strip()
    genre_rows = genre_rows[genre_rows["genre"].ne("")]
    return (
        genre_rows.groupby("genre")["movieId"]
        .nunique()
        .rename("movie_count")
        .reset_index()
        .sort_values(["movie_count", "genre"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def save_eda_charts(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> list[Path]:
    """Save rating-distribution and popularity charts for the EDA notebook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    charts = []

    counts = ratings["rating"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(counts.index.astype(str), counts.values, color="#6958d6")
    axis.set(title="MovieLens rating distribution", xlabel="Rating", ylabel="Number of ratings")
    figure.tight_layout()
    rating_chart = directory / "rating-distribution.png"
    figure.savefig(rating_chart, dpi=150)
    plt.close(figure)
    charts.append(rating_chart)

    popular = top_movies(ratings, movies, n=10).sort_values("rating_count")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(popular["title"], popular["rating_count"], color="#e87954")
    axis.set(title="Most-rated movies", xlabel="Number of ratings", ylabel="")
    figure.tight_layout()
    popularity_chart = directory / "most-rated-movies.png"
    figure.savefig(popularity_chart, dpi=150)
    plt.close(figure)
    charts.append(popularity_chart)

    genres = genre_popularity(movies).head(10).sort_values("movie_count")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(genres["genre"], genres["movie_count"], color="#4d9a91")
    axis.set(title="Most common movie genres", xlabel="Number of catalog titles", ylabel="")
    figure.tight_layout()
    genre_chart = directory / "genre-distribution.png"
    figure.savefig(genre_chart, dpi=150)
    plt.close(figure)
    charts.append(genre_chart)
    return charts


def main() -> None:
    """Print data-quality summary and save EDA charts."""
    ratings, movies = load_movielens()
    summary = summarize_ratings(ratings, movie_count=len(movies))
    print(pd.Series(summary).to_string())
    print("\nTop-rated-by-count movies:")
    print(top_movies(ratings, movies).to_string(index=False))
    print("\nMost common genres:")
    print(genre_popularity(movies).head(10).to_string(index=False))
    print("\nCharts:")
    for path in save_eda_charts(ratings, movies):
        print(path)


if __name__ == "__main__":
    main()
