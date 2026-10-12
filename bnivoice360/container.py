"""Composition root: dependencies are assembled in one place."""

from dataclasses import dataclass

from bnivoice360.config import Settings
from bnivoice360.data.database import Database
from bnivoice360.data.repository import VoiceRepository
from bnivoice360.data.seed import initialize_data, load_seed
from bnivoice360.services.agent.workflow import BatchAgent
from bnivoice360.services.companion import CompanionService
from bnivoice360.services.companion_evaluation import CompanionEvaluation
from bnivoice360.services.companion_workflow import CompanionConversation
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
    batch_agent: BatchAgent
    conversation: CompanionConversation
    evaluation: CompanionEvaluation


def create_services(settings: Settings | None = None) -> AppServices:
    settings = settings or Settings()
    database = Database(settings.database_path)
    database.initialize()
    repository = VoiceRepository(database)
    if settings.seed_initial_data and repository.is_empty():
        initialize_data(database, load_seed(settings))
    batch_agent = BatchAgent(repository, settings)
    return AppServices(
        repository=repository,
        ingestion=IngestionService(repository),
        prediction=PredictionService(repository, RuleBasedClassifier()),
        crawler=SimulatedCrawler(),
        companion=CompanionService(repository),
        batch_agent=batch_agent,
        conversation=CompanionConversation(batch_agent),
        evaluation=CompanionEvaluation(settings),
    )
