# Project Brief: Movie Recommendation System

## One-line summary
Suggests movies based on what a user likes, the way streaming platforms do.

## Problem statement
With thousands of titles, users cannot browse everything. A recommender narrows the list to a handful of titles a user is likely to enjoy.

## Learning goals
How platforms like Netflix recommend content: content-based filtering, collaborative filtering, similarity metrics, the cold-start problem and ranking evaluation.

## Target users
- Viewers looking for something to watch
- Learners studying recommender systems

## In scope
- Content-based recommender: 'movies similar to X' using TF-IDF over genres and overview
- Collaborative filtering on user ratings (item-item or matrix factorization)
- Web UI: pick a movie or rate a few, get top-N recommendations
- Offline evaluation (Precision@K / RMSE) on a held-out split

## Out of scope (for v1)
- Production-scale infrastructure, authentication and payments
- Anything under "Could have" in `02_REQUIREMENTS.md` unless time remains

## Success criteria
- RMSE below the global-mean baseline on held-out ratings
- Precision@10 above a 'most popular' baseline
- Recommendation latency under 1 s with cached matrices
- Sanity check: 'Toy Story' returns animation / family titles

## Owner / timeline
- Owner: _your name_
- Start date: ____  Target finish: ____ (see `05_ROADMAP.md`)
