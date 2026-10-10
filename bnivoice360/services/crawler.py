"""Crawler adapter interface and the existing local simulator."""

import hashlib
import random
from dataclasses import asdict
from datetime import datetime, time, timedelta
from typing import Protocol

import pandas as pd

from bnivoice360.config import SOURCES
from bnivoice360.domain import CrawlRequest

TEMPLATES = (
    "Transfer saya gagal dan saldo sudah terpotong, mohon segera dibantu.",
    "Aplikasi wondr sangat mudah dipakai dan tampilannya bagus.",
    "QRIS tidak bisa digunakan sejak pagi, transaksi selalu pending.",
    "Antrean di cabang terlalu lama, layanan teller perlu ditambah.",
    "Promo kartu kredit BNI menarik dan prosesnya cepat.",
    "ATM sering tidak tersedia uang tunai di area saya.",
)


class Crawler(Protocol):
    def crawl(self, request: CrawlRequest) -> pd.DataFrame:
        """Collect voices using source-specific options."""


class SimulatedCrawler:
    """Produces repeatable text samples within the requested date range."""

    def crawl(self, request: CrawlRequest) -> pd.DataFrame:
        if request.source not in SOURCES:
            raise ValueError(f"Unknown source: {request.source}")
        if not 1 <= request.limit <= 100:
            raise ValueError("Limit must be between 1 and 100.")
        if not 1 <= request.minimum_rating <= 5:
            raise ValueError("Minimum rating must be between 1 and 5.")
        now = datetime.now()
        start = (
            datetime.combine(request.start_date, time.min)
            if request.start_date
            else now - timedelta(days=10)
        )
        end = datetime.combine(request.end_date, time.max) if request.end_date else now
        if end < start:
            raise ValueError("End date must be on or after start date.")
        digest = hashlib.sha256(repr(asdict(request)).encode()).hexdigest()
        rng = random.Random(int(digest[:8], 16))
        span = (end - start).total_seconds()
        return pd.DataFrame(
            [
                {
                    "content": rng.choice(TEMPLATES),
                    "date_posted": start + timedelta(seconds=rng.uniform(0, span)),
                    "source": request.source,
                }
                for _ in range(request.limit)
            ]
        )
