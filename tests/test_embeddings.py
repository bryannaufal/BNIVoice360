"""Embedding cache reuse and failed-download recovery without network access."""

import io

import pytest

from bnivoice360.config import Settings
from bnivoice360.data import embeddings


def test_downloads_once_and_reuses_cache(tmp_path, monkeypatch):
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return io.BytesIO(b"PAR1example dataPAR1")

    monkeypatch.setattr(embeddings, "urlopen", download)
    settings = Settings(embedding_directory=tmp_path, download_embeddings=True)
    paths = embeddings.ensure_embeddings(settings)
    assert {path.name for path in paths} == {
        "topic_id_embedding_mapping.parquet",
        "data_id_embedding_mapping.parquet",
    }
    assert all(embeddings.valid_parquet(path) for path in paths)
    assert embeddings.ensure_embeddings(settings) == paths
    assert len(calls) == 2


def test_bad_download_is_not_cached_and_can_retry(tmp_path, monkeypatch):
    settings = Settings(embedding_directory=tmp_path, download_embeddings=True)
    monkeypatch.setattr(
        embeddings, "urlopen", lambda *args, **kwargs: io.BytesIO(b"<html>denied</html>")
    )
    with pytest.raises(ValueError, match="not a Parquet"):
        embeddings.ensure_embeddings(settings)
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(
        embeddings, "urlopen", lambda *args, **kwargs: io.BytesIO(b"PAR1example dataPAR1")
    )
    assert len(embeddings.ensure_embeddings(settings)) == 2


def test_disabled_download_does_not_create_directory(tmp_path):
    directory = tmp_path / "unused"
    assert (
        embeddings.ensure_embeddings(
            Settings(embedding_directory=directory, download_embeddings=False)
        )
        == ()
    )
    assert not directory.exists()
