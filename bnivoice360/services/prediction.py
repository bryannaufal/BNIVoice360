"""Replaceable text classification strategy and enrichment orchestration."""

from typing import Protocol

from bnivoice360.data.repository import VoiceRepository
from bnivoice360.domain import Prediction

NEGATIVE_TERMS = (
    "gagal",
    "error",
    "lambat",
    "kecewa",
    "buruk",
    "pending",
    "tidak bisa",
    "terpotong",
    "hilang",
    "lama",
)
POSITIVE_TERMS = (
    "bagus",
    "cepat",
    "mudah",
    "ramah",
    "praktis",
    "terima kasih",
    "puas",
    "membantu",
)
URGENT_TERMS = ("segera", "urgent", "terpotong", "hilang", "blokir", "tertelan", "tidak bisa")
TOPIC_RULES = (
    (("transfer", "terpotong"), "Gagal Transfer", (1, 0, 0)),
    (("login", "aplikasi", "wondr", "mobile banking"), "Login Aplikasi", (1, 0, 0)),
    (("qris",), "QRIS", (1, 1, 0)),
    (("atm", "tarik tunai"), "ATM & Tarik Tunai", (0, 0, 1)),
    (("antrean", "cabang", "teller"), "Antrean Cabang", (0, 0, 1)),
    (("kartu kredit", "promo"), "Kartu Kredit & Promo", (0, 1, 0)),
    (("top up", "e-wallet"), "Top Up & E-Wallet", (1, 1, 0)),
    (("biaya", "tarif", "admin"), "Biaya & Tarif", (0, 1, 0)),
    (("customer service", "call center", "layanan"), "Layanan Customer Service", (0, 0, 1)),
    (("rekening",), "Pembukaan Rekening", (0, 1, 1)),
)


class VoiceClassifier(Protocol):
    def classify(self, text: str) -> Prediction:
        """Return sentiment, multi-label categories, urgency, and topic."""


class RuleBasedClassifier:
    """Local demo classifier; replace through the VoiceClassifier interface."""

    def classify(self, text: str) -> Prediction:
        lowered = text.lower()
        if any(term in lowered for term in NEGATIVE_TERMS):
            sentiment = "Negative"
        elif any(term in lowered for term in POSITIVE_TERMS):
            sentiment = "Positive"
        else:
            sentiment = "Neutral"

        topic, flags = "Lainnya", (0, 0, 1)
        for keywords, label, categories in TOPIC_RULES:
            if any(keyword in lowered for keyword in keywords):
                topic, flags = label, categories
                break
        urgent = int(sentiment == "Negative" and any(term in lowered for term in URGENT_TERMS))
        return Prediction(sentiment, *flags, urgent, topic)


class PredictionService:
    def __init__(self, repository: VoiceRepository, classifier: VoiceClassifier):
        self.repository = repository
        self.classifier = classifier

    def run(self, source: str) -> int:
        rows = self.repository.find_unpredicted(source)
        predictions = [
            (row["suara_nasabah_id"], self.classifier.classify(row["content"])) for row in rows
        ]
        self.repository.fill_predictions(predictions)
        return len(predictions)
