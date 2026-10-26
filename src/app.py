"""Streamlit UI for content-based movie recommendations."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import streamlit as st

from src.recommender import load_sample

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("Find your next movie")
st.caption(
    "Choose a movie you enjoy and get recommendations based on genre and story similarity."
)

recommender = load_sample()
movie_options = sorted(recommender.movies["title"].tolist(), key=str.casefold)
with st.sidebar:
    st.header("Your preferences")
    favorite_movie = st.selectbox(
        "A movie you like",
        movie_options,
        index=movie_options.index("Toy Story") if "Toy Story" in movie_options else 0,
        key="favorite_movie",
    )
    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=10,
        value=5,
        key="recommendation_count",
    )
    st.divider()
    st.caption("This demo runs from the included sample catalog; no API key is required.")

recommendations = recommender.recommend_similar(favorite_movie, recommendation_count)
st.subheader(f"Because you liked {favorite_movie}")
st.caption("Ranked by content similarity. Scores are relative similarities, not probabilities.")

if recommendations.empty:
    st.info("No similar titles were found in this catalog. Try another movie.")
else:
    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {row.title}**")
            st.caption(f"{row.genres} · Similarity {row.score:.3f}")

st.divider()
st.caption(
    "Offline collaborative filtering evaluation: run `python -m src.collab` after "
    "downloading MovieLens small with `python src/download_data.py`."
)
