# Architecture: Movie Recommendation System

## Overview
```
Included sample catalog ---> Cached TF-IDF content features -----------------+
                                                                              |
MovieLens ratings ---> Sparse item-item similarity ---> New-user profile ---> Hybrid rank
        |                                                                     |
        +--> Per-user time split ---> held-out RMSE / Precision@10             +--> Streamlit UI
                                                                              |
Optional TMDB API (only with .env key) --------------------------------------+
```

## Tech stack
- Python 3.10+
- pandas, numpy
- scikit-learn (TF-IDF, cosine similarity)
- scikit-surprise or implicit (optional, for collaborative filtering)
- Streamlit (UI)
- pytest

## Design decisions (fill in as you go)
| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Content features | Genres only, genres + overview | TF-IDF on genres and optional overview | The included sample has overviews; MovieLens does not, so the model uses genres when overview is absent. |
| Evaluation split | Random split, latest user interactions | Per-user chronological 80/20 holdout; equal timestamps stay together | Prevents future ratings from entering model training. Users without two distinct timestamps remain in training. |
| Collaborative model | Dense item-item cosine, SVD | Sparse user-centered item-item cosine with co-rating and recommendation shrinkage | Keeps MovieLens small manageable and dampens weak co-rating pairs. |
| Ranking baselines | No baseline, popularity | Training-only global-mean RMSE and most-rated Precision@10 | Reports model quality against simple reproducible references. |
| New-user experience | Ask for one favorite, require account | Ask for 3–5 sample-catalog ratings; blend content profile with MovieLens item-item predictions where title IDs match | Works without an account or download and falls back to content rankings if community data is absent. |
| Similarity caching | Recompute per rerun, in-memory resource cache | Cache the vectorized content catalog with Streamlit `cache_resource` | Avoids repeating feature preparation across UI reruns. |
| Optional metadata | Bundle posters, use TMDB API | Fetch poster and overview on demand only when `TMDB_API_KEY` is present | Keeps the core demo offline and credentials out of source control. |

## Module responsibilities
See the files in `src/`. Keep logic (testable functions) separate from UI and scripts.
