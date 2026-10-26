"""Streamlit UI for content and hybrid movie recommendations."""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from src.collab import ItemItemCF
from src.hybrid import HybridRecommender
from src.recommender import ContentRecommender, load_sample
from src.tmdb import fetch_movie_metadata

load_dotenv()
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RATINGS_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "ratings.csv"
MOVIES_PATH = PROJECT_ROOT / "data" / "ml-latest-small" / "movies.csv"


@st.cache_resource
def load_content() -> ContentRecommender:
    """Cache the small content catalog so the first page loads quickly."""
    return load_sample()


@st.cache_resource
def load_collaborative() -> tuple[ItemItemCF | None, pd.DataFrame | None]:
    """Load the optional MovieLens community model only when a profile is used."""
    if not RATINGS_PATH.is_file() or not MOVIES_PATH.is_file():
        return None, None
    ratings = pd.read_csv(RATINGS_PATH)
    movie_catalog = pd.read_csv(MOVIES_PATH)
    return ItemItemCF().fit(ratings), movie_catalog


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_tmdb_metadata(title: str, api_key: str) -> dict[str, str] | None:
    """Cache optional title lookups for one day."""
    return fetch_movie_metadata(title, api_key)


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("Find your next movie")
st.caption(
    "Choose a movie you enjoy and get recommendations based on genre and story similarity."
)

content = load_content()
movie_options = sorted(content.movies["title"].tolist(), key=str.casefold)
with st.sidebar:
    st.header("Your preferences")
    mode = st.radio("Recommendation mode", ["One movie", "Rate 3–5 movies"])
    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5,
        key="recommendation_count",
    )
    if mode == "One movie":
        favorite_movie = st.selectbox(
            "A movie you like",
            movie_options,
            index=movie_options.index("Toy Story") if "Toy Story" in movie_options else 0,
            key="favorite_movie",
        )
        profile_movies = []
        profile_ratings = {}
    else:
        profile_movies = st.multiselect(
            "Choose 3–5 movies you have seen",
            movie_options,
            max_selections=5,
            key="profile_movies",
        )
        profile_ratings = {
            title: st.slider(
                f"Your rating for {title}",
                min_value=0.5,
                max_value=5.0,
                value=4.0,
                step=0.5,
                key=f"rating_{title}",
            )
            for title in profile_movies
        }
    st.divider()
    st.caption("The sample catalog works offline. MovieLens adds community signals.")

if mode == "One movie":
    recommendations = content.recommend_similar(favorite_movie, recommendation_count)
    heading = f"Because you liked {favorite_movie}"
    description = "Ranked by content similarity. Scores are relative, not probabilities."
elif len(profile_ratings) >= 3:
    with st.spinner("Building your personalized recommendations..."):
        collaborative, movie_catalog = load_collaborative()
        recommender = HybridRecommender(content, collaborative, movie_catalog)
        recommendations = recommender.recommend_for_ratings(
            profile_ratings, recommendation_count
        )
    heading = "Picked for your ratings"
    description = (
        "Your profile blends genre/story similarity with MovieLens community ratings "
        "when MovieLens small is downloaded."
    )
else:
    recommendations = None
    heading = "Build your movie profile"
    description = "Choose and rate at least three movies in the sidebar to get personalized suggestions."

st.subheader(heading)
st.caption(description)

if recommendations is None:
    st.info(description)
elif recommendations.empty:
    st.info("No similar titles were found in this catalog. Try another movie.")
else:
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {row.title}**")
            if hasattr(row, "content_score"):
                detail = f"Hybrid score {row.score:.3f} · Content {row.content_score:.3f}"
                if row.collaborative_score is not None:
                    detail += f" · Community {row.collaborative_score:.3f}"
            else:
                detail = f"Similarity {row.score:.3f}"
            tmdb_key = os.getenv("TMDB_API_KEY", "")
            metadata = cached_tmdb_metadata(row.title, tmdb_key) if tmdb_key else None
            poster_url = metadata.get("poster_url") if metadata else None
            overview = metadata.get("overview", "") if metadata else ""
            if poster_url:
                st.image(poster_url, width=120)
            st.caption(f"{row.genres} · {detail}")
            if overview:
                st.caption(overview)

st.divider()
st.caption(
    "Offline collaborative filtering evaluation: run `python -m src.collab` after "
    "downloading MovieLens small with `python src/download_data.py`."
)
if os.getenv("TMDB_API_KEY"):
    st.caption("Movie metadata provided by TMDB; this app is not endorsed or certified by TMDB.")
