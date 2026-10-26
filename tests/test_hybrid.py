import pandas as pd
import pytest

from src.collab import ItemItemCF
from src.hybrid import HybridRecommender, _catalog_key
from src.recommender import ContentRecommender, load_sample


def test_catalog_key_matches_inverted_articles_and_years():
    assert _catalog_key("Matrix, The (1999)") == "the matrix"


def test_profile_uses_content_and_optional_community_scores():
    train = pd.DataFrame(
        {
            "userId": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "movieId": [1, 2, 3, 4] * 3,
            "rating": [5, 4, 1, 5, 4, 5, 1, 5, 1, 2, 5, 1],
            "timestamp": [1, 2, 3, 4] * 3,
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Toy Story (1995)", "Matrix, The (1999)", "Titanic (1997)", "Dune (2021)"],
        }
    )
    content = ContentRecommender(
        pd.DataFrame(
            {
                "title": ["Toy Story", "The Matrix", "Titanic", "Dune"],
                "genres": ["Animation|Family", "Sci-Fi|Action", "Drama|Romance", "Sci-Fi|Adventure"],
            }
        )
    )
    hybrid = HybridRecommender(content, ItemItemCF().fit(train), movies)

    result = hybrid.recommend_for_ratings(
        {"Toy Story": 5, "The Matrix": 4, "Titanic": 1}, n=1
    )

    assert result.iloc[0]["title"] == "Dune"
    assert result.iloc[0]["collaborative_score"] is not None
    assert 0 <= result.iloc[0]["score"] <= 1


def test_profile_validates_cold_start_rating_count_and_titles():
    hybrid = HybridRecommender(load_sample())
    with pytest.raises(ValueError, match="between 3 and 5"):
        hybrid.recommend_for_ratings({"Toy Story": 5, "Titanic": 4})
    with pytest.raises(KeyError, match="Unknown title"):
        hybrid.recommend_for_ratings(
            {"Toy Story": 5, "Titanic": 4, "Missing": 3}
        )


def test_content_only_fallback_keeps_recommendations_available():
    hybrid = HybridRecommender(load_sample())
    result = hybrid.recommend_for_ratings(
        {"Toy Story": 5, "The Matrix": 4, "Titanic": 1}, n=3
    )

    assert len(result) == 3
    assert result["collaborative_score"].isna().all()
