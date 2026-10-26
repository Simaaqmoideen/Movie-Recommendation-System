# Data folder

Only datasets listed in `docs/04_DATA.md` are used.

- MovieLens small (GroupLens): download with `python src/download_data.py`. The extracted files live in `data/ml-latest-small/` and are ignored by Git.
- `sample_movies.csv`: a small included catalog so the content-based demonstration runs without downloading MovieLens.

If the Python downloader cannot establish TLS in a restricted environment, download the official archive from <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip> and extract it into this folder. The expected paths are `data/ml-latest-small/ratings.csv` and `data/ml-latest-small/movies.csv`. On Windows, PowerShell can download and extract it with `Invoke-WebRequest -Uri https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -OutFile ml-latest-small.zip` and `Expand-Archive -LiteralPath ml-latest-small.zip -DestinationPath data`.
