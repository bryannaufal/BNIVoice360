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
            "https://drive.google.com/uc?export=download&id=13JogqqXWxbVSuabTJjiCf4MeNCwUXthR",
        )
    )
    mapping_csv: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_MAPPING_CSV",
            "https://drive.google.com/uc?export=download&id=1FGyw1M9bYnZt9uR5wjKboMyhm49ZHN63",
        )
    )
    download_embeddings: bool = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_DOWNLOAD_EMBEDDINGS", "1") == "1"
    )
    embedding_directory: Path = field(
        default_factory=lambda: (
            Path(os.environ.get("BNIVOICE360_EMBEDDING_DIR", str(PROJECT_ROOT / ".embeddings")))
            .expanduser()
            .resolve()
        )
    )
    topic_embedding_url: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_TOPIC_EMBEDDING_URL",
            "https://drive.usercontent.google.com/download?id=1Gt3x4xIpcDZoQwzXgmgpg7nweNh-RSrs&export=download&confirm=t",
        )
    )
    data_embedding_url: str = field(
        default_factory=lambda: os.environ.get(
            "BNIVOICE360_DATA_EMBEDDING_URL",
            "https://drive.usercontent.google.com/download?id=16HeG768HT8B6fuupaoIV1DBWp2EDd9xx&export=download&confirm=t",
        )
    )
    agent_model: str = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_AGENT_MODEL", "openai/gpt-oss-20b")
    )
    agent_similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("BNIVOICE360_AGENT_SIMILARITY", "0.80"))
    )
    agent_candidate_limit: int = field(
        default_factory=lambda: int(os.environ.get("BNIVOICE360_AGENT_LIMIT", "20"))
    )
    agent_remote_enabled: bool = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_AGENT_REMOTE_ENABLED", "0") == "1"
    )
    companion_remote_enabled: bool = field(
        default_factory=lambda: os.environ.get("BNIVOICE360_COMPANION_REMOTE_ENABLED", "1") == "1"
    )
