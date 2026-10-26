# Testing and Evaluation: Movie Recommendation System

## Success metrics
- RMSE below the global-mean baseline on held-out ratings
- Precision@10 above a 'most popular' baseline
- Recommendation latency under 1 s with cached matrices
- Sanity check: 'Toy Story' returns animation / family titles

## Test plan
- [x] Unit: similarity of a movie with itself is 1 and it is excluded from its own recommendations
- [x] Unit: unknown title raises a clear error
- [ ] Evaluation script reports RMSE and Precision@K versus baselines
- [ ] Manual: spot-check recommendations for 5 well-known movies

## How to run tests
```bash
python -m pytest -q
```

## Results log
| Date | Experiment / test | Result | Conclusion |
|---|---|---|---|
| 2026-10-26 | Phase 1 EDA and `python -m pytest -q` | 10 passed in 9.77s. MovieLens small: 100,836 ratings, 610 users, 9,724 rated movies, 9,742 catalog titles, 98.3032% sparsity, mean rating 3.5016, no missing values, no duplicate user/movie pairs, and no ratings outside 0.5–5.0. Drama (4,361 catalog titles) and Comedy (3,756) are the two largest genres by title count. Three EDA charts saved under `reports/`. | Dataset quality checks and the baseline content-recommender tests pass. Recommendation quality metrics are not evaluated in this phase. |
| 2026-10-26 | Phase 2 content-based recommender and `python -m pytest -q` | 15 passed in 11.08s. Console demo ran for Toy Story, The Matrix, and Titanic; movie titles were excluded from their own lists, and Toy Story returned animation/family results. | TF-IDF over genres and optional overview works on the sample and MovieLens-style catalogs without an overview field. This is a content similarity sanity check, not a relevance benchmark. |
