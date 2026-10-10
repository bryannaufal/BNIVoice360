"""Database-aware local companion; no UI or generated SQL."""

import re
from datetime import timedelta

import pandas as pd

from bnivoice360.config import SOURCES
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.services.analytics import pending_urgent_count

TOPIC_TERMS = {
    "mobile banking": "Login Aplikasi",
    "wondr": "Login Aplikasi",
    "qris": "QRIS",
    "transfer": "Gagal Transfer",
    "atm": "ATM & Tarik Tunai",
}


def _question_scope(question: str, frame: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Translate the supported question terms into deterministic data filters."""
    query = question.lower()
    sources = [
        source
        for source in SOURCES
        if re.search(r"(?<!\w)" + re.escape(source.lower()) + r"(?!\w)", query)
    ]
    subset = frame[frame["source"].isin(sources)] if sources else frame.copy()
    if any(term in query for term in ("minggu ini", "this week", "seminggu")):
        subset = subset[subset["date_posted"] >= now.to_pydatetime() - timedelta(days=7)]
    if "bulan ini" in query:
        subset = subset[subset["date_posted"] >= now.to_pydatetime() - timedelta(days=30)]
    topics = [topic for term, topic in TOPIC_TERMS.items() if term in query]
    if topics:
        subset = subset[subset["topic_cluster"].isin(topics)]
    if "urgent" in query:
        subset = subset[subset["is_Urgent"].eq(1)]
    if "it" in re.findall(r"\b\w+\b", query):
        subset = subset[subset["is_IT"].eq(1)]
    if any(term in query for term in ("pending", "belum", "unresolved")):
        subset = subset[subset["is_escalated"].eq(0)]
    if "negative" in query or "negatif" in query:
        subset = subset[subset["sentiment"].eq("Negative")]
    return subset


def answer_question(question: str, frame: pd.DataFrame, now: pd.Timestamp | None = None) -> str:
    query = question.lower()
    now = now if now is not None else pd.Timestamp.now()
    subset = _question_scope(question, frame, now)
    if any(term in query for term in ("berapa", "how many", "jumlah")):
        negative = int(subset["sentiment"].eq("Negative").sum())
        return (
            f"Terdapat **{len(subset)} voice** yang sesuai. Dari jumlah itu, "
            f"**{negative} negatif** dan **{pending_urgent_count(subset)} "
            "urgent belum dieskalasi**."
        )
    if "top" in query or "teratas" in query:
        match = re.search(r"(?:top|teratas)\s*(\d+)", query)
        limit = min(int(match.group(1)), 20) if match else 3
        top = subset.groupby("topic_cluster").size().nlargest(limit)
        if top.empty:
            return "Belum ada data yang sesuai dengan cakupan pertanyaan tersebut."
        lines = [
            f"{index}. **{topic}** — {count} voice"
            for index, (topic, count) in enumerate(top.items(), start=1)
        ]
        return "Top cluster berdasarkan data saat ini:\n\n" + "\n".join(lines)
    if subset.empty:
        return (
            "Belum ada data yang sesuai. Coba perluas periode atau "
            "hilangkan salah satu filter dalam pertanyaan."
        )
    distribution = subset["sentiment"].value_counts()
    top = subset["topic_cluster"].value_counts().head(3)
    themes = ", ".join(f"**{topic}** ({count})" for topic, count in top.items())
    return (
        f"Saya menemukan **{len(subset)} voice**. Sentimennya: "
        f"**{distribution.get('Positive', 0)} positif**, "
        f"**{distribution.get('Neutral', 0)} netral**, dan "
        f"**{distribution.get('Negative', 0)} negatif**. "
        f"Tema dominan: {themes}. "
        f"Ada **{pending_urgent_count(subset)}** isu urgent yang masih menunggu eskalasi."
    )


class CompanionService:
    def __init__(self, repository: VoiceRepository):
        self.repository = repository

    def answer(self, question: str) -> str:
        return answer_question(question, self.repository.load())
