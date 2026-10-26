"""Download and safely extract the official MovieLens small archive."""

import pathlib
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "ml-latest-small"


def _download_archive(destination: pathlib.Path) -> None:
    """Download with normal TLS verification, using Windows' trust store on Windows."""
    if sys.platform == "win32":
        escaped_path = str(destination).replace("'", "''")
        command = f"Invoke-WebRequest -Uri '{URL}' -OutFile '{escaped_path}'"
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            check=True,
            timeout=180,
        )
        return

    context = ssl.create_default_context()
    with urllib.request.urlopen(URL, context=context, timeout=60) as response:
        destination.write_bytes(response.read())


def _safe_extract(archive_path: pathlib.Path, destination: pathlib.Path) -> None:
    """Extract only archive entries that remain inside the data directory."""
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe path in MovieLens archive: {member.filename}")
        archive.extractall(destination)


def download_movielens() -> pathlib.Path:
    """Fetch MovieLens small into data/ and return its extracted directory."""
    if DATASET_DIR.exists():
        raise FileExistsError(
            f"Dataset already exists at {DATASET_DIR}; remove it before downloading again."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="movielens-") as temporary_dir:
        archive_path = pathlib.Path(temporary_dir) / "ml-latest-small.zip"
        _download_archive(archive_path)
        _safe_extract(archive_path, DATA_DIR)
    if not (DATASET_DIR / "ratings.csv").is_file():
        raise FileNotFoundError("The downloaded archive did not contain ratings.csv.")
    return DATASET_DIR


if __name__ == "__main__":
    print("MovieLens data ready in", download_movielens())
