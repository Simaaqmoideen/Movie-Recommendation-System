import zipfile

import pytest

from src.download_data import _safe_extract


def test_safe_extract_unpacks_into_the_requested_data_directory(tmp_path):
    archive_path = tmp_path / "dataset.zip"
    destination = tmp_path / "data"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n")

    _safe_extract(archive_path, destination)

    assert (destination / "ml-latest-small" / "ratings.csv").is_file()


def test_safe_extract_rejects_paths_outside_data_directory(tmp_path):
    archive_path = tmp_path / "malicious.zip"
    destination = tmp_path / "data"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "not allowed")

    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract(archive_path, destination)

    assert not (tmp_path / "outside.txt").exists()
