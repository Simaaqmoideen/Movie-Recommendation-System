# Data: Movie Recommendation System

## Sources
- MovieLens (GroupLens): https://grouplens.org/datasets/movielens/ . Start with `ml-latest-small` (about 100k ratings, 9k movies). Run `python src/download_data.py`.
- TMDB API (optional, free key) for posters and overviews: https://www.themoviedb.org/documentation/api
- A 15-movie sample file is included at `data/sample_movies.csv` so the demo works with no download.

## Data handling rules
- Raw data is never edited by hand. Cleaning is done by code and written to a new file.
- Large files are not committed (see `.gitignore`); document how to download them instead.
- Respect each dataset's licence and terms of service; cite the source in the final report.
- Do not store personal or sensitive information unless it is anonymised.

## Data dictionary

### `ratings.csv`
| Column | Type | Meaning | Missing values in downloaded file? |
|---|---|---|---|
| `userId` | integer | Anonymous MovieLens user ID | No |
| `movieId` | integer | MovieLens title ID | No |
| `rating` | float | Half-star rating from 0.5 to 5.0 | No |
| `timestamp` | integer | Unix time when rating was recorded | No |

### `movies.csv`
| Column | Type | Meaning | Missing values in downloaded file? |
|---|---|---|---|
| `movieId` | integer | MovieLens title ID | No |
| `title` | string | Movie title (may include year) | No |
| `genres` | string | Pipe-separated genres | No |

The optional `overview` field used by the content recommender is present in the included sample but not in MovieLens `movies.csv`; content recommendations fall back to genre text when it is absent.

## Data quality checklist
- [x] Row count and column types checked: 100,836 ratings, 610 users, 9,724 rated titles, 9,742 catalog rows.
- [x] Missing values counted: none in the downloaded ratings file.
- [x] Duplicate `(userId, movieId)` pairs counted: none in the downloaded ratings file.
- [x] Rating bounds reviewed: observed values are 0.5–5.0, matching the dataset scale.
- [x] Train / test split strategy written down: each user's latest 20% of interactions, leaving at least one earlier interaction in training. Model fitting uses training data only.
