"""Local language cues shared by search intent extraction and analytical routing."""

import re

from bnivoice360.config import CATEGORY_COLUMNS, SENTIMENTS, SOURCES

ANALYTICAL = re.compile(
    r"tren|trend|meningkat|peningkatan|penurunan|banding|compare|backlog|umur|usia|"
    r"prioritas|diprioritaskan|briefing|ringkas|pola|masalah spesifik|urai|konsisten|"
    r"salah kelompok|sejak kapan|lebih banyak|berapa|jumlah|teratas|\btop\b|analisis|analysis",
    re.I,
)
DISPLAY = re.compile(
    r"\b(?:tampilkan|tunjukkan|lihat|daftar|show|list|display|cari|carikan|search|find|apa saja|(?:issues?|isu|keluhan) apa)\b",
    re.I,
)
EXISTS = re.compile(
    r"\b(?:adakah|ada(?:kah)?|apakah\s+(?:terdapat|issues?|isu|keluhan)|any|are there|is there)\b",
    re.I,
)
URGENT = re.compile(
    r"\b(?:urgent|urgensi|mendesak|darurat|segera|secepatnya|prioritas tinggi)\b", re.I
)


def search_filters(text: str) -> dict:
    """None means unspecified; empty tuples explicitly clear a filter."""
    categories = tuple(
        name for name in CATEGORY_COLUMNS if re.search(r"\b" + name + r"\b", text, re.I)
    )
    sources = tuple(
        source
        for source in SOURCES
        if re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", text, re.I)
    )
    aliases = {"Twitter": "X", "Play Store": "PlayStore", "App Store": "AppStore"}
    sources = tuple(
        dict.fromkeys(
            (
                *sources,
                *(
                    value
                    for alias, value in aliases.items()
                    if re.search(r"\b" + alias + r"\b", text, re.I)
                ),
            )
        )
    )
    status = next(
        (
            name
            for name in ("Open", "Escalated", "Resolved")
            if re.search(r"\b" + name + r"\b", text, re.I)
        ),
        None,
    )
    if re.search(r"belum\s+(?:di)?eskalasi|belum\s+dieskalasi|not escalated", text, re.I):
        status = "Open"
    elif re.search(
        r"sudah\s+(?:di)?eskalasi|sudah\s+dieskalasi|masih\s+(?:di)?eskalasi", text, re.I
    ):
        status = "Escalated"
    elif re.search(r"(?:sudah\s+)?(?:selesai|terselesaikan)", text, re.I):
        status = "Resolved"
    urgent = True if URGENT.search(text) else None
    if re.search(
        r"semua\s+(?:urgensi|tingkat urgensi)|termasuk\s+(?:yang\s+)?(?:tidak|non)[ -]?urgent|all urgency",
        text,
        re.I,
    ):
        urgent = False
    sentiments = tuple(
        value
        for label, value in (
            ("negatif|negative", "Negative"),
            ("positif|positive", "Positive"),
            ("netral|neutral", "Neutral"),
        )
        if re.search(r"\b(?:" + label + r")\b", text, re.I)
    )
    return {
        "categories": ()
        if re.search(r"semua kategori|all categories", text, re.I)
        else categories or None,
        "sources": ()
        if re.search(r"semua kanal|semua sumber|all channels|all sources", text, re.I)
        else sources or None,
        "status": "All" if re.search(r"semua status|all status", text, re.I) else status,
        "urgent_only": urgent,
        "sentiments": SENTIMENTS
        if re.search(r"semua sentimen|all sentiments", text, re.I)
        else sentiments or None,
    }


def search_keyword(text: str) -> str | None:
    """Accept arbitrary explicit topic phrases, without guessing product synonyms."""
    if re.search(r"semua topik|semua produk|all topics|all products", text, re.I):
        return ""
    quoted = re.search(r'["“](.{1,100}?)["”]', text)
    if quoted:
        return quoted[1].strip()
    topic = re.search(r"\b(?:terkait|tentang|mengenai|seputar|about|related to)\s+(.+)", text, re.I)
    if topic:
        candidate = topic[1]
    else:
        topic = re.search(
            r"\b(?:issues?|isu|keluhan|komplain|complaints?|kendala|masalah)\s+(.+)", text, re.I
        )
        if not topic:
            topic = re.search(
                r"\b(?:tampilkan|tunjukkan|lihat|cari|carikan|show|list|display|search|find)\s+(.+)",
                text,
                re.I,
            )
        candidate = topic[1] if topic else ""
    candidate = re.split(
        r"\b(?:yang|dengan|dari|pada|selama|untuk|dalam|hari ini|minggu ini|bulan ini|"
        r"semua|akhir[- ]akhir|belakangan|recently|last|from|with|in|that|"
        r"IT|Operasional|Bisnis|urgent|mendesak|segera|open|escalated|resolved|"
        r"negatif|positif|netral|negative|positive|neutral|"
        r"X|Twitter|PlayStore|AppStore|Play Store|App Store|Laporan Nasabah|\d{4}-\d{2}-\d{2}|\d+\s*(?:hari|days))\b",
        candidate,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" ?.!,;:")
    if candidate and not re.fullmatch(
        r"tersebut|itu|ini|saja|masih|ada|perlu|detail(?:nya)?|hasil(?:nya)?|all|any",
        candidate,
        re.I,
    ):
        return candidate
    for term in ("QRIS", "transfer", "ATM", "login", "wondr", "mobile banking"):
        match = re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I)
        if match:
            return match[0]
    return None
