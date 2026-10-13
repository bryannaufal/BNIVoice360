"""Typed values shared by persistence, services, and presentation."""

from dataclasses import dataclass
from datetime import date
from enum import Enum

from bnivoice360.config import CATEGORY_COLUMNS, SENTIMENTS, SOURCES


class Role(str, Enum):
    MANAGER = "Manager / Executive"
    DATA_ADMIN = "Data Admin"


@dataclass(frozen=True)
class Prediction:
    sentiment: str
    is_IT: int
    is_Bisnis: int
    is_Operasional: int
    is_Urgent: int
    topic_cluster: str


@dataclass(frozen=True)
class VoiceRecord:
    suara_nasabah_id: str
    content: str
    date_posted: str | None
    source: str
    sentiment: str | None = None
    is_IT: int | None = None
    is_Bisnis: int | None = None
    is_Operasional: int | None = None
    is_Urgent: int | None = None
    escalation_status: str = "Open"
    topic_cluster: str | None = None
    topic_id: int | None = None
    author_id_or_name: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    inserted: int
    skipped: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CrawlRequest:
    source: str
    limit: int = 10
    keywords: str = ""
    start_date: date | None = None
    end_date: date | None = None
    app_id: str = ""
    minimum_rating: int = 1
    branch_code: str = ""
    ticket_category: str = "All"


@dataclass(frozen=True)
class IssueFilters:
    start_date: date | None = None
    end_date: date | None = None
    sources: tuple[str, ...] = SOURCES
    topics: tuple[str, ...] | None = None
    sentiments: tuple[str, ...] = SENTIMENTS
    categories: tuple[str, ...] = tuple(CATEGORY_COLUMNS)
    escalation_status: str = "All"
    urgent_only: bool = False


@dataclass(frozen=True)
class OverviewMetrics:
    total_volume: int
    growth_percent: float
    net_sentiment: float
    urgent_open: int
    topic_count: int
    unescalated_topics: int = 0

