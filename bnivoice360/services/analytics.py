"""Pure data transformations used by pages and the companion."""

from datetime import timedelta

import pandas as pd

from bnivoice360.config import CATEGORY_COLUMNS, ESCALATION_STATUSES
from bnivoice360.domain import IssueFilters, OverviewMetrics


def open_urgent_count(frame: pd.DataFrame) -> int:
    return int((frame["is_Urgent"].eq(1) & frame["escalation_status"].eq("Open")).sum())


def assigned_topics(frame: pd.DataFrame) -> pd.DataFrame:
    """Outliers and unmapped voices contribute to totals, never topic rankings."""
    if "topic_id" in frame.columns:
        ids = pd.to_numeric(frame["topic_id"], errors="coerce")
        return frame.loc[ids.notna() & ids.ne(-1)]
    return frame.loc[
        frame["topic_cluster"].notna()
        & ~frame["topic_cluster"].isin(["Outlier", "Unlabeled", "out;ier"])
    ]


def topic_assignment_counts(frame: pd.DataFrame) -> dict[str, int]:
    ids = pd.to_numeric(frame["topic_id"], errors="coerce")
    return {
        "assigned": int((ids.notna() & ids.ne(-1)).sum()),
        "outlier": int(ids.eq(-1).sum()),
        "unlabeled": int(ids.isna().sum()),
    }


def overview_metrics(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> OverviewMetrics:
    now = now if now is not None else pd.Timestamp.now()
    recent_start = now.to_pydatetime() - timedelta(days=30)
    previous_start = now.to_pydatetime() - timedelta(days=60)
    current = frame[frame["date_posted"] >= recent_start]
    previous = frame[
        (frame["date_posted"] < recent_start) & (frame["date_posted"] >= previous_start)
    ]
    growth = (len(current) - len(previous)) / len(previous) * 100 if len(previous) else 0
    labeled = frame[frame["sentiment"].notna()]
    net = (
        (labeled["sentiment"].eq("Positive").mean() - labeled["sentiment"].eq("Negative").mean())
        * 100
        if len(labeled)
        else 0
    )
    return OverviewMetrics(
        len(frame),
        growth,
        net,
        open_urgent_count(frame),
        int(
            assigned_topics(frame)["topic_id" if "topic_id" in frame else "topic_cluster"].nunique()
        ),
    )


def channel_volume(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("source").size().reset_index(name="Volume")


def sentiment_trend(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["date_posted"].notna()]
    current_year = max(2026, pd.Timestamp.now().year)
    valid = valid[valid["date_posted"].dt.year <= current_year]
    return (
        valid.assign(month=valid["date_posted"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "sentiment"])
        .size()
        .reset_index(name="Volume")
    )


def top_topics(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return (
        assigned_topics(frame)
        .groupby("topic_cluster")
        .agg(
            Volume=("content", "size"),
            Negative=("sentiment", lambda values: values.eq("Negative").mean()),
        )
        .nlargest(limit, "Volume")
        .reset_index()
    )


def category_labels(row: pd.Series) -> list[str]:
    return [label for label, column in CATEGORY_COLUMNS.items() if row.get(column) == 1]


def category_topics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = assigned_topics(frame)
    if frame.empty:
        return pd.DataFrame(columns=["Category", "Topic", "Volume"])
    expanded = frame.assign(Category=frame.apply(category_labels, axis=1)).explode("Category")
    return (
        expanded.dropna(subset=["Category", "topic_cluster"])
        .groupby(["Category", "topic_cluster"])
        .size()
        .reset_index(name="Volume")
        .rename(columns={"topic_cluster": "Topic"})
    )


def topic_sentiments(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        assigned_topics(frame)
        .groupby(["topic_cluster", "sentiment"])
        .size()
        .reset_index(name="Volume")
    )


def filter_issues(frame: pd.DataFrame, filters: IssueFilters) -> pd.DataFrame:
    mask = frame["source"].isin(filters.sources) & frame["sentiment"].isin(filters.sentiments)
    if filters.start_date:
        mask &= frame["date_posted"].dt.date >= filters.start_date
    if filters.end_date:
        mask &= frame["date_posted"].dt.date <= filters.end_date
    if filters.topics is not None:
        mask &= frame["topic_cluster"].astype(str).isin(filters.topics)
    category_mask = pd.Series(False, index=frame.index)
    for category in filters.categories:
        column = CATEGORY_COLUMNS.get(category)
        if column:
            category_mask |= frame[column].eq(1)
    # Selecting every category means no category restriction, including all-zero voices.
    if set(filters.categories) != set(CATEGORY_COLUMNS):
        mask &= category_mask
    if filters.escalation_status in ESCALATION_STATUSES:
        mask &= frame["escalation_status"].eq(filters.escalation_status)
    elif filters.escalation_status != "All":
        raise ValueError("Unknown escalation status filter.")
    if filters.urgent_only:
        mask &= frame["is_Urgent"].eq(1)
    return frame.loc[mask].copy()


def voice_feed(frame: pd.DataFrame, search: str = "") -> pd.DataFrame:
    result = (
        frame[frame["content"].str.contains(search, case=False, na=False, regex=False)]
        if search
        else frame
    )
    return result.assign(
        priority=result["is_Urgent"].eq(1).astype(int),
        status_priority=result["escalation_status"].map({"Escalated": 0, "Open": 1, "Resolved": 2}),
    ).sort_values(
        ["priority", "status_priority", "date_posted"],
        ascending=[False, True, False],
        kind="stable",
    )
