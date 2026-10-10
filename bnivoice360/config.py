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
    "topic_cluster",
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
    seed_demo_data: bool = True
