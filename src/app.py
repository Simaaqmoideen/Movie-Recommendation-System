"""Streamlit UI.  Run:  streamlit run src/app.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # allow `python src/x.py`
import streamlit as st
from src.recommender import load_sample

st.title("Movie Recommender")
rec = st.cache_resource(load_sample)()
title = st.selectbox("Pick a movie you like", sorted(rec.movies["title"]))
n = st.slider("How many recommendations?", 1, 10, 5)
st.dataframe(rec.recommend_similar(title, n), hide_index=True)
