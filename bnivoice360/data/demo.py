"""Seed fixtures kept separate from production data access."""

import random
from datetime import datetime, timedelta

from bnivoice360.config import SOURCES
from bnivoice360.domain import VoiceRecord


def seed_records() -> list[VoiceRecord]:
    samples = [
        (
            "Transfer di wondr gagal terus tetapi saldo sudah terpotong. Mohon segera!",
            "Negative",
            1,
            0,
            0,
            1,
            "Gagal Transfer",
        ),
        ("QRIS BNI praktis dan transaksinya cepat, terima kasih.", "Positive", 0, 1, 0, 0, "QRIS"),
        (
            "Antrean teller cabang panjang sekali, sudah menunggu dua jam.",
            "Negative",
            0,
            0,
            1,
            1,
            "Antrean Cabang",
        ),
        (
            "Aplikasi mobile banking tidak bisa login setelah update.",
            "Negative",
            1,
            0,
            0,
            1,
            "Login Aplikasi",
        ),
        ("Promo kartu kredit bulan ini menarik.", "Positive", 0, 1, 0, 0, "Kartu Kredit & Promo"),
        (
            "Customer service membantu blokir kartu dengan ramah.",
            "Positive",
            0,
            0,
            1,
            0,
            "Layanan Customer Service",
        ),
        (
            "Top up e-wallet masih pending dari tadi pagi.",
            "Negative",
            1,
            1,
            0,
            1,
            "Top Up & E-Wallet",
        ),
        ("Informasi biaya admin tabungan kurang jelas.", "Neutral", 0, 1, 0, 0, "Biaya & Tarif"),
        (
            "ATM dekat rumah sering kehabisan uang tunai.",
            "Negative",
            0,
            0,
            1,
            0,
            "ATM & Tarik Tunai",
        ),
        (
            "Pembukaan rekening online mudah dan cepat.",
            "Positive",
            1,
            1,
            0,
            0,
            "Pembukaan Rekening",
        ),
        ("Notifikasi transaksi terlambat masuk.", "Neutral", 1, 0, 0, 0, "Notifikasi Transaksi"),
        (
            "Kartu debit tertelan ATM dan belum ada tindak lanjut. Urgent!",
            "Negative",
            0,
            0,
            1,
            1,
            "ATM & Tarik Tunai",
        ),
    ]
    rows, now = [], datetime.now()
    rng = random.Random(360)
    for i in range(96):
        content, sent, it, bisnis, ops, urgent, topic = samples[i % len(samples)]
        posted = now - timedelta(days=rng.randint(0, 74), hours=rng.randint(0, 23))
        source = SOURCES[i % len(SOURCES)]
        status = "Escalated" if urgent and i % 4 == 0 else "Open"
        rows.append(
            VoiceRecord(
                f"VOC-{i + 1:04d}",
                content,
                posted.isoformat(timespec="seconds"),
                source,
                sent,
                it,
                bisnis,
                ops,
                urgent,
                status,
                topic,
            )
        )
    return rows
