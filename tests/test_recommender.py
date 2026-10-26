import pandas as pd
import pytest

from src.recommender import ContentRecommender, load_sample


def test_excludes_itself_and_returns_n():
    rec = load_sample()
    out = rec.recommend_similar("Toy Story", 4)
    assert len(out) == 4
    assert "Toy Story" not in out["title"].tolist()


def test_animation_neighbours_for_toy_story():
    out = load_sample().recommend_similar("Toy Story", 3)
    assert out["genres"].str.contains("Animation").any()


def test_unknown_title():
    with pytest.raises(KeyError, match="Unknown title"):
        load_sample().recommend_similar("Not A Movie")


def test_lookup_ignores_case_and_extra_whitespace():
    rec = load_sample()
    assert rec.recommend_similar("  tOy   sToRy ", 2).equals(
        rec.recommend_similar("Toy Story", 2)
    )


def test_missing_overview_column_uses_genres_only():
    movies = pd.DataFrame(
        {
            "title": ["Space Story", "Galaxy Quest", "Garden Party"],
            "genres": ["Sci-Fi|Adventure", "Adventure|Comedy", "Romance|Drama"],
        }
    )

    result = ContentRecommender(movies).recommend_similar("Space Story", 1)

    assert result.iloc[0]["title"] == "Galaxy Quest"


def test_rejects_invalid_n_and_blank_titles():
    rec = load_sample()
    with pytest.raises(ValueError, match="positive integer"):
        rec.recommend_similar("Toy Story", 0)
    with pytest.raises(ValueError, match="non-empty string"):
        rec.recommend_similar("   ")


def test_similarity_diagonal_is_one():
    rec = load_sample()
    assert rec.sim.diagonal().tolist() == pytest.approx([1.0] * len(rec.movies))


def test_duplicate_titles_are_rejected():
    movies = pd.DataFrame(
        {
            "title": ["Dune", " dune "],
            "genres": ["Sci-Fi", "Adventure"],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        ContentRecommender(movies)
