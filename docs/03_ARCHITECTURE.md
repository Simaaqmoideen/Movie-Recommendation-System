# Architecture: Movie Recommendation System

## Overview
```
MovieLens CSVs  ->  data/ (raw)  ->  src/data.py (clean, merge)
                                        |
              +-------------------------+-------------------------+
              v                                                   v
   Content-based (src/recommender.py)               Collaborative (src/collab.py)
   TF-IDF(genres + overview) -> cosine sim          user-item matrix -> item-item / SVD
              \                                                   /
               +----------------> Hybrid ranker <----------------+
                                        |
                                        v
                          Streamlit app (src/app.py)
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

## Module responsibilities
See the files in `src/`. Keep logic (testable functions) separate from UI and scripts.
