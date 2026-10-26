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
| | | | |

## Module responsibilities
See the files in `src/`. Keep logic (testable functions) separate from UI and scripts.
