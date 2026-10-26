# Final Report: Movie Recommendation System

## 1. Title, author, date

**Movie Recommendation System**
Author: Simaaq Moideen
Date: 2026-10-26

## 2. Abstract

This project recommends movies from a user's interests using content-based and collaborative filtering. The content model uses TF-IDF over genres and optional overview text, while the collaborative model learns sparse item-item similarities from MovieLens ratings. New users can rate three to five sample-catalog titles and receive a weighted hybrid ranking when MovieLens community signals are available. On a chronological per-user MovieLens holdout, item-item filtering improved RMSE over a training-mean baseline and Precision@10 over a training-only popularity baseline. The app starts from its included sample catalog and keeps the core recommendation flow local and usable without credentials.

## 3. Introduction

Movie catalogs can be difficult to browse, and useful recommendations must work both for returning users with rating histories and for new users with little or no history. This project demonstrates content-based recommendations, collaborative filtering, evaluation against simple baselines, and a cold-start flow in a small Streamlit app.

## 4. Data

The offline evaluation and EDA use MovieLens small from GroupLens. The downloaded files contained 100,836 ratings from 610 anonymous users across 9,742 catalog titles; 9,724 titles had at least one rating. The ratings file had no missing values, duplicate user/movie pairs, or out-of-range scores. Overall rating mean was 3.5016 and user-item sparsity was 98.3032%.

The evaluation split held out each user's latest timestamp block at or beyond the 80th percentile, keeping timestamp ties together. This produced 80,116 training ratings and 20,720 test ratings. Users without two distinct timestamps remained in training. The raw dataset is downloaded by `src/download_data.py` and excluded from Git. The app uses the included 15-title catalog by default.

## 5. Method

- **Content-based:** TF-IDF features from genres and optional overview text; cosine similarity ranks titles.
- **Collaborative:** sparse user-centered item-item cosine similarity, with shrinkage for weak co-rating evidence.
- **Offline evaluation:** per-user time holdout; global-mean RMSE and training-only popularity Precision@10 baselines. Fitting and popularity counts use training rows only.
- **Cold start and hybrid:** users rate 3–5 sample titles. Content relevance is blended with MovieLens profile predictions for matched title IDs. Unmatched titles and missing MovieLens data fall back to content ranking.
- **UI and optional metadata:** Streamlit caches content features and lazily loads the community model. Optional TMDB poster and overview requests require a local `.env` key and are cached in memory for a day.

## 6. Results

### Held-out collaborative evaluation

| Metric | Baseline | Item-item model | Better result |
|---|---:|---:|---|
| RMSE (lower is better) | 1.068659 global mean | 0.893428 | Item-item model |
| Precision@10 (higher is better) | 0.060135 popularity | 0.067061 | Item-item model |

Precision@10 is macro-averaged over 592 users with at least one held-out rating of 4.0 or higher. These are offline MovieLens results, not a live satisfaction study.

![Held-out RMSE and Precision@10 comparison](../reports/model-metrics.png)

### EDA

![MovieLens rating distribution](../reports/rating-distribution.png)

![Most rated titles](../reports/most-rated-movies.png)

![Movie genre distribution](../reports/genre-distribution.png)

### Cold-start example

A real profile rating Toy Story 5, The Matrix 4, and Titanic 1 produced the following top five suggestions from the included sample catalog and downloaded MovieLens community model:

| Rank | Title | Hybrid score | Content score | Community score |
|---:|---|---:|---:|---:|
| 1 | Finding Nemo | 0.605178 | 0.534839 | 0.675517 |
| 2 | The Lion King | 0.583508 | 0.491311 | 0.675705 |
| 3 | Monsters Inc | 0.536750 | 0.536750 | — |
| 4 | Interstellar | 0.515848 | 0.515848 | — |
| 5 | Avengers Endgame | 0.511029 | 0.511029 | — |

Titles without a mapped MovieLens ID used their content score. This example illustrates ranking behavior and is not an evaluation metric.

### Tests and manual checks

- `python -m pytest -q`: **34 passed in 11.64s** on the final working tree.
- The app's single-title picker, recommendation count, five-title onboarding, rating sliders, and ranked profile results were checked in a browser.
- The content recommender was manually checked on Toy Story, The Matrix, Titanic, The Lion King, and Forrest Gump; each returned three suggestions.
- Hybrid scoring on downloaded MovieLens data and mocked TMDB success/timeout paths are covered by commands and tests recorded in `docs/06_TESTING_EVALUATION.md`.

## 7. Discussion

The collaborative model improved both selected offline metrics against the simple baselines on this temporal split. The small sample-catalog content recommender passed title and genre sanity checks, and the cold-start hybrid example used community scores where movie IDs matched. The app avoids blocking its first page on the larger collaborative model by loading that model only after a profile is requested.

## 8. Limitations and ethics

MovieLens is sparse, historical, and not representative of every audience. The sample app catalog is small, and recommendation scores are relative signals rather than probabilities or endorsements. Some sample titles do not map to MovieLens IDs and therefore receive content-only ranking. New-user ratings remain in local session state and are not persisted. When TMDB is configured, displayed movie titles are sent to TMDB to retrieve optional metadata. The temporal evaluation does not establish real-world satisfaction or causal impact.

## 9. Future work

- Evaluate hybrid ranking with a separate time-based validation period before selecting blend weights.
- Add user studies and latency measurements on a clean deployment environment.
- Expand the catalog with a permitted, documented metadata source and assess catalog coverage.

## 10. References

- [MovieLens datasets, GroupLens Research](https://grouplens.org/datasets/movielens/)
- [TMDB API documentation](https://developer.themoviedb.org/docs)
- pandas, NumPy, SciPy, scikit-learn, Streamlit, pytest, and Matplotlib documentation.

## 11. Appendix: setup

See the root [`README.md`](../README.md) for Windows, macOS, and Linux setup, data download, app launch, EDA, evaluation, and test commands.
