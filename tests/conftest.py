"""All persistence and UI tests use temporary SQLite databases."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from bnivoice360.config import PROJECT_ROOT, Settings
from bnivoice360.container import create_services


@pytest.fixture
def services(tmp_path):
    return create_services(Settings(tmp_path / "voices.db", seed_demo_data=False))


@pytest.fixture
def streamlit_app(tmp_path, monkeypatch):
    database_path = Path(tmp_path) / "ui.db"
    monkeypatch.setenv("BNIVOICE360_DB_PATH", str(database_path))
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    return app
