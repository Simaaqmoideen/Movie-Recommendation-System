"""Optional TMDB poster and overview lookup; failures never block recommendations."""

from typing import Any

import requests

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"


def fetch_movie_metadata(title: str, api_key: str) -> dict[str, str] | None:
    """Fetch a title's poster and overview, returning None on API errors."""
    if not title.strip() or not api_key.strip():
        return None
    try:
        response = requests.get(
            SEARCH_URL,
            params={"api_key": api_key, "query": title, "include_adult": "false"},
            timeout=5,
        )
        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results", [])
    except (requests.RequestException, ValueError, TypeError):
        return None
    if not results:
        return None
    match = results[0]
    metadata = {"overview": str(match.get("overview") or "")}
    poster_path = match.get("poster_path")
    if poster_path:
        metadata["poster_url"] = f"https://image.tmdb.org/t/p/w342{poster_path}"
    return metadata
