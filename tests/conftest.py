"""All persistence and UI tests use temporary SQLite databases."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bnivoice360.config import PROJECT_ROOT, Settings
from bnivoice360.container import create_services
from bnivoice360.data.demo import seed_records


@pytest.fixture(autouse=True)
def isolated_initialization(monkeypatch):
    # Tests never download private feedback or overwrite a runtime database.
    monkeypatch.setenv("BNIVOICE360_AUTO_INITIALIZE", "0")
    monkeypatch.setenv("BNIVOICE360_INIT_OVERWRITE", "0")
    monkeypatch.setenv("BNIVOICE360_DOWNLOAD_EMBEDDINGS", "0")
    monkeypatch.setenv("BNIVOICE360_COMPANION_REMOTE_ENABLED", "0")


@pytest.fixture
def synthetic_masks():
    """Explicit masks for existing synthetic fixtures; never used by application ingestion."""

    def fill(repository):
        with repository.database.transaction() as connection:
            connection.execute("UPDATE suara_nasabah SET masked_content=content")

    return fill


@pytest.fixture
def services(tmp_path):
    return create_services(Settings(tmp_path / "voices.db", seed_initial_data=False))


@pytest.fixture
def streamlit_app(tmp_path, monkeypatch):
    database_path = Path(tmp_path) / "ui.db"
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(database_path))
    create_services(Settings(database_path)).repository.seed_if_empty(seed_records())
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    return app
