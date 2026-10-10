"""Composition root: dependencies are assembled in one place."""

from dataclasses import dataclass

from bnivoice360.config import Settings
from bnivoice360.data.database import Database
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.data.seed import seed_records
from bnivoice360.services.companion import CompanionService
from bnivoice360.services.crawler import Crawler, SimulatedCrawler
from bnivoice360.services.ingestion import IngestionService
from bnivoice360.services.prediction import PredictionService, RuleBasedClassifier


@dataclass(frozen=True)
class AppServices:
    repository: VoiceRepository
    ingestion: IngestionService
    prediction: PredictionService
    crawler: Crawler
    companion: CompanionService


def create_services(settings: Settings | None = None) -> AppServices:
    settings = settings or Settings()
    database = Database(settings.database_path)
    database.initialize()
    repository = VoiceRepository(database)
    if settings.seed_demo_data:
        repository.seed_if_empty(seed_records())
    return AppServices(
        repository=repository,
        ingestion=IngestionService(repository),
        prediction=PredictionService(repository, RuleBasedClassifier()),
        crawler=SimulatedCrawler(),
        companion=CompanionService(repository),
    )
