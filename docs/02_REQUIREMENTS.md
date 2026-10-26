# Requirements: Movie Recommendation System

## Functional requirements (MoSCoW)

### Must have
- [x] Content-based recommender: 'movies similar to X' using TF-IDF over genres and overview
- [ ] Collaborative filtering on user ratings (item-item or matrix factorization)
- [ ] Web UI: pick a movie or rate a few, get top-N recommendations
- [ ] Offline evaluation (Precision@K / RMSE) on a held-out split

### Should have
- [ ] Hybrid score combining both approaches
- [ ] Cold-start handling for new users (ask for 3-5 ratings)
- [ ] Poster and metadata display via TMDB API
- [ ] Caching of the similarity matrix

### Could have (stretch)
- [ ] Explanations ('because you liked ...')
- [ ] Diversity re-ranking
- [ ] Implicit feedback (watch time)
- [ ] Deploy as an API with FastAPI

## Non-functional requirements
- **Reproducible:** a fresh clone runs end to end with the commands in README.md.
- **Secure:** secrets only in `.env`; no keys, tokens or personal data in Git.
- **Readable:** type hints on public functions, docstrings on modules, PEP 8 formatting.
- **Tested:** automated tests for core logic; they pass before every merge.
- **Documented:** README and docs updated whenever behaviour changes.

## Assumptions
- Python 3.10 or newer; internet access to install packages and fetch data.
- Free-tier / free-dataset resources are used wherever possible.

## Constraints
- Solo or small-team student project; target is a working demo, not production.
