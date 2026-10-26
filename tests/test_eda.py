import pandas as pd
import pytest

from src.eda import genre_popularity, summarize_ratings, top_movies


def test_summarize_ratings_reports_coverage_and_quality():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 3.0, 4.0, 2.0],
            "timestamp": [1, 2, 3, 4],
        }
    )

    summary = summarize_ratings(ratings, movie_count=5)

    assert summary["ratings"] == 4
    assert summary["users"] == 2
    assert summary["rated_movies"] == 3
    assert summary["catalog_movies"] == 5
    assert summary["sparsity"] == pytest.approx(0.6)
    assert summary["missing_values"] == 0
    assert summary["duplicate_user_movie_pairs"] == 0


def test_top_movies_uses_rating_count_and_stable_title_ties():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 1, 2],
            "movieId": [10, 10, 20, 20],
            "rating": [5.0, 4.0, 3.0, 4.0],
            "timestamp": [1, 2, 3, 4],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20],
            "title": ["Zebra", "Apple"],
            "genres": ["Drama", "Comedy"],
        }
    )

    result = top_movies(ratings, movies, n=2)

    assert result["title"].tolist() == ["Apple", "Zebra"]
    assert result["rating_count"].tolist() == [2, 2]


def test_summarize_rejects_empty_or_invalid_data():
    with pytest.raises(ValueError, match="at least one row"):
        summarize_ratings(pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"]))


def test_top_movies_requires_positive_n():
    with pytest.raises(ValueError, match="at least 1"):
        top_movies(pd.DataFrame(), pd.DataFrame(), n=0)


def test_genre_popularity_counts_multi_genre_titles_once_per_genre():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genres": ["Drama|Comedy", "Drama", "Comedy"],
        }
    )

    result = genre_popularity(movies)

    assert result.set_index("genre")["movie_count"].to_dict() == {"Comedy": 2, "Drama": 2}
