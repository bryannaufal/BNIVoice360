"""Pure data transformations used by pages and the companion."""

from datetime import timedelta

import pandas as pd

from bnivoice360.config import CATEGORY_COLUMNS
from bnivoice360.domain import IssueFilters, OverviewMetrics


def pending_urgent_count(frame: pd.DataFrame) -> int:
    return int((frame["is_Urgent"].eq(1) & frame["is_escalated"].eq(0)).sum())


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
        pending_urgent_count(frame),
        int(frame["topic_cluster"].nunique()),
    )


def channel_volume(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("source").size().reset_index(name="Volume")


def sentiment_trend(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(day=frame["date_posted"].dt.floor("D"))
        .groupby(["day", "sentiment"])
        .size()
        .reset_index(name="Volume")
    )


def top_topics(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return (
        frame.groupby("topic_cluster")
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
    expanded = frame.assign(Category=frame.apply(category_labels, axis=1)).explode("Category")
    return (
        expanded.dropna(subset=["Category", "topic_cluster"])
        .groupby(["Category", "topic_cluster"])
        .size()
        .reset_index(name="Volume")
        .rename(columns={"topic_cluster": "Topic"})
    )


def topic_sentiments(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["topic_cluster", "sentiment"]).size().reset_index(name="Volume")


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
    mask &= category_mask
    if filters.escalation_status == "Pending Escalation":
        mask &= frame["is_escalated"].eq(0)
    elif filters.escalation_status == "Escalated":
        mask &= frame["is_escalated"].eq(1)
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
        priority=(result["is_Urgent"].eq(1) & result["is_escalated"].eq(0)).astype(int)
    ).sort_values(["priority", "date_posted"], ascending=[False, False])
