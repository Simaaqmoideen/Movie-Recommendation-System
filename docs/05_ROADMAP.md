# Roadmap: Movie Recommendation System

Suggested pace: one phase every 2-4 days (about 2-3 weeks in total).

## Phase 1: Setup and data
Download MovieLens small. Explore: rating distribution, sparsity, popular genres. Notebook `notebooks/01_eda.ipynb`.

- [x] Task completed
- [x] Notes written in the project log
- [x] Changes committed

## Phase 2: Content-based
Build TF-IDF on genres and overview, cosine similarity, `recommend_similar(title, n)`.

- [x] Task completed
- [x] Notes written in the project log
- [x] Changes committed

## Phase 3: Collaborative
Train/test split by time or per-user. Item-item cosine or SVD. Report RMSE and Precision@10.

- [x] Task completed
- [x] Notes written in the project log
- [x] Changes committed

## Phase 4: Hybrid and cold start
Blend scores (weighted). Handle new users with a short rating onboarding.

- [x] Task completed
- [x] Notes written in the project log
- [x] Changes committed

## Phase 5: UI and ship
Streamlit app, caching, README, demo.

- [x] Task completed
- [x] Notes written in the project log
- [x] Changes committed

## Project log
| Date | What I did | Blockers | Next |
|---|---|---|---|
| 2026-10-26 | Downloaded the allowed MovieLens small data, added a reproducible EDA notebook, quality summary, charts, and a Windows TLS-aware download path. | Python TLS trust failed in this environment; verified the official HTTPS download using Windows certificate validation. | Phase 2: content-based recommender. |
| 2026-10-26 | Hardened TF-IDF content recommendations, including genre-only catalogs, normalized title lookup, stable ranking, and input validation; added regression tests. | None. | Phase 3: held-out collaborative filtering and metric baselines. |
| 2026-10-26 | Added a sparse item-item CF model, per-user latest-timestamp holdout, training-only global/popularity baselines, RMSE and Precision@10 evaluation, and comparison chart. 26 tests pass. | None. | Phase 5 Must-have UI first; then revisit Phase 4 Should-have items. |
| 2026-10-26 | Shipped the sample-catalog Streamlit UI, expanded README with setup, architecture diagram, EDA/evaluation plots and measured results; browser-checked the app and manually spot-checked five titles. 26 tests passed in 10.34s. | Streamlit's AppTest runner hung in this Windows environment, so UI behavior was verified in the live browser; core logic remains covered by automated tests. | Phase 4 Should-have hybrid ranking and new-user onboarding. |
| 2026-10-26 | Added 3–5 title rating onboarding, weighted content/community profile ranking with genre-only fallback, lazy cached MovieLens fit and optional cached TMDB poster/overview lookup. Browser-checked both UI modes; `python -m pytest -q`: 34 passed in 11.59s. | TMDB lookup remains optional and requires the user to supply a key in local `.env`. | Finish final report and clean-clone checks. |
