import requests

from src.tmdb import fetch_movie_metadata


def test_tmdb_returns_poster_and_overview(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"poster_path": "/poster.jpg", "overview": "A story."}]}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())

    result = fetch_movie_metadata("Toy Story", "test-key")

    assert result == {
        "poster_url": "https://image.tmdb.org/t/p/w342/poster.jpg",
        "overview": "A story.",
    }


def test_tmdb_failure_is_optional(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "get", fail)

    assert fetch_movie_metadata("Toy Story", "test-key") is None
    assert fetch_movie_metadata("Toy Story", "") is None
