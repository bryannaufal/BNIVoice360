"""Session-owned conversation values; no credentials, SQL, or hidden ticket selection."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from bnivoice360.domain import IssueFilters
from bnivoice360.services.agent.workflow import Proposal


class Stage(str, Enum):
    IDLE = "idle"
    PERIOD = "period"
    OFFER = "offer"
    RESULTS = "results"
    SELECTED = "selected"
    CROSS_TOPIC = "cross_topic"
    DETAILS = "details"
    DRAFT = "draft"


@dataclass(frozen=True)
class ComplaintQuery:
    keyword: str
    start: date | None = None
    end: date | None = None
    sources: tuple[str, ...] = ()
    status: str = "All"
    categories: tuple[str, ...] = ()
    urgent_only: bool = False
    sentiments: tuple[str, ...] = ("Negative",)
    topic: str | None = None
    as_of: date | None = None

    def filters(self, *, similarity: bool = False) -> IssueFilters:
        from bnivoice360.config import CATEGORY_COLUMNS, SOURCES

        return IssueFilters(
            start_date=self.start,
            end_date=self.end,
            sources=self.sources or SOURCES,
            sentiments=("Negative",) if similarity else self.sentiments,
            escalation_status="Open" if similarity else self.status,
            categories=self.categories or tuple(CATEGORY_COLUMNS),
            topics=(self.topic,) if self.topic else None,
            urgent_only=self.urgent_only,
            as_of=self.as_of,
        )


@dataclass
class CompanionState:
    stage: Stage = Stage.IDLE
    query: ComplaintQuery | None = None
    result_ids: tuple[str, ...] = ()
    snapshots: dict[str, str] = field(default_factory=dict)
    total: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    page: int = 0
    selected_ids: tuple[str, ...] = ()
    result_kind: str = "keyword"
    scores: dict[str, float] = field(default_factory=dict)
    team: str = ""
    request: str = ""
    examination: str = ""
    draft_text: str = ""
    draft_mode: str = ""
    draft_reason: str = ""
    draft_revision: str = ""
    search_action: str = "display"
    proposal: Proposal | None = None
    role: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    analysis_sources: list[dict] = field(default_factory=list)
    analysis_scope: dict = field(default_factory=dict)
    analysis_question: str = ""
    analysis_waiting: bool = False
    analysis_evidence: dict = field(default_factory=dict)
    reply_kind: str = "workflow"
    recent_turns: list[dict] = field(default_factory=list)
    remote_content_version: int = 0

    @property
    def visible_ids(self) -> tuple[str, ...]:
        if self.stage == Stage.OFFER:
            return ()
        return self.result_ids[self.page * 10 : (self.page + 1) * 10]
