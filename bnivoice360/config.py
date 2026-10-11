"""Central configuration; importing this module does not open a database."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES = (
    "X",
    "Threads",
    "Facebook",
    "Instagram",
    "TikTok",
    "PlayStore",
    "AppStore",
    "Laporan Nasabah",
)
SENTIMENTS = ("Negative", "Neutral", "Positive")
ESCALATION_STATUSES = ("Open", "Escalated", "Resolved")
CATEGORY_COLUMNS = {
    "IT": "is_IT",
    "Bisnis": "is_Bisnis",
    "Operasional": "is_Operasional",
}
PREDICTION_COLUMNS = (
    "sentiment",
    "is_IT",
    "is_Bisnis",
    "is_Operasional",
    "is_Urgent",
)
FEED_PAGE_SIZE = 12


@dataclass(frozen=True)
class Settings:
    """Runtime settings, with an overridable path for isolated tests."""

    database_path: Path = field(
        default_factory=lambda: (
            Path(os.environ.get("BNIVOICE360_DB_PATH", str(PROJECT_ROOT / "bnivoice360.db")))
            .expanduser()
            .resolve()
        )
    )
    seed_initial_data: bool = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_AUTO_INITIALIZE", "1") == "1"
    )
    # Consumed by the initialization CLI only, never by Streamlit startup.
    overwrite_initialization: bool = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_INIT_OVERWRITE", "0") == "1"
    )
    voice_csv: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_VOICE_CSV",
            "https://docs.google.com/spreadsheets/d/17EZg4weh8Xw8ggoogYxt7f19QXaSKFaUhzSoXEm2050/gviz/tq?tqx=out:csv&sheet=df_Clean",
        )
    )
    topic_csv: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_TOPIC_CSV",
            "https://drive.google.com/uc?export=download&id=1WacFf0Q0-jk8tCYTjE_T8w115thxmwPq",
        )
    )
    mapping_csv: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_MAPPING_CSV",
            "https://drive.google.com/uc?export=download&id=1l-7710H8kP5oWg8dWMFnui48-7aiSsrt",
        )
    )
