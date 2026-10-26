# Testing and Evaluation: Movie Recommendation System

## Success metrics
- RMSE below the global-mean baseline on held-out ratings
- Precision@10 above a 'most popular' baseline
- Recommendation latency under 1 s with cached matrices
- Sanity check: 'Toy Story' returns animation / family titles

## Test plan
- [x] Unit: similarity of a movie with itself is 1 and it is excluded from its own recommendations
- [x] Unit: unknown title raises a clear error
- [x] Evaluation script reports RMSE and Precision@K versus baselines
- [x] Manual: spot-check recommendations for 5 well-known movies

## How to run tests
```bash
python -m pytest -q
```

## Results log
| Date | Experiment / test | Result | Conclusion |
|---|---|---|---|
| 2026-10-26 | Phase 1 EDA and `python -m pytest -q` | 10 passed in 9.77s. MovieLens small: 100,836 ratings, 610 users, 9,724 rated movies, 9,742 catalog titles, 98.3032% sparsity, mean rating 3.5016, no missing values, no duplicate user/movie pairs, and no ratings outside 0.5–5.0. Drama (4,361 catalog titles) and Comedy (3,756) are the two largest genres by title count. Three EDA charts saved under `reports/`. | Dataset quality checks and the baseline content-recommender tests pass. Recommendation quality metrics are not evaluated in this phase. |
| 2026-10-26 | Phase 2 content-based recommender and `python -m pytest -q` | 15 passed in 11.08s. Console demo ran for Toy Story, The Matrix, and Titanic; movie titles were excluded from their own lists, and Toy Story returned animation/family results. | TF-IDF over genres and optional overview works on the sample and MovieLens-style catalogs without an overview field. This is a content similarity sanity check, not a relevance benchmark. |
| 2026-10-26 | Phase 3 item-item CF, per-user temporal holdout, and `python -m src.collab` | `python -m pytest -q`: 26 passed in 11.06s. All 80,116 training rows and 20,720 held-out rows were evaluated. Global-mean RMSE 1.068659; item-item RMSE 0.893428. Most-popular Precision@10 0.060135; item-item Precision@10 0.067061 across 592 eligible users with at least one held-out rating ≥4.0. Chart saved to `reports/model-metrics.png`. | Item-item CF beat both baselines on this MovieLens temporal holdout. These are offline metrics, not a live recommendation or satisfaction study. |
| 2026-10-26 | Phase 5 sample-catalog Streamlit UI, manual spot checks, and `python -m pytest -q` | 26 passed in 10.34s. Opened the app in a browser and verified its picker, recommendation count control, and five recommendations. Manually checked Toy Story, The Matrix, Titanic, The Lion King, and Forrest Gump; all returned three ranked suggestions. | The sample app works without downloading MovieLens or configuring credentials. |
| 2026-10-26 | Phase 4 cold-start hybrid and TMDB optional lookup; `python -m pytest -q` | 34 passed in 11.59s. Browser-checked both recommendation modes, selected five profile titles, and confirmed the app rendered ranked suggestions. A separate real MovieLens profile rating Toy Story 5, The Matrix 4, and Titanic 1 ranked Finding Nemo (0.605), The Lion King (0.584), Monsters Inc (0.537), Interstellar (0.516), and Avengers Endgame (0.511). Mocked TMDB success and timeout paths pass. | Community predictions were available for titles with MovieLens ID matches; unmatched titles used their content score. The profile output is illustrative, not a held-out metric. |
