import pytest
from src.recommender import load_sample


def test_excludes_itself_and_returns_n():
    rec = load_sample()
    out = rec.recommend_similar("Toy Story", 4)
    assert len(out) == 4
    assert "Toy Story" not in out["title"].tolist()


def test_animation_neighbours_for_toy_story():
    out = load_sample().recommend_similar("Toy Story", 3)
    assert out["genres"].str.contains("Animation").any()


def test_unknown_title():
    with pytest.raises(KeyError):
        load_sample().recommend_similar("Not A Movie")
