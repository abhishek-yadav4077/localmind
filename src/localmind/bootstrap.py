"""Wire settings, database and services together.

Both the FastAPI app and the CLI scripts call ``build_context()`` so there is a
single, testable composition root instead of globals scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from localmind.arena import ArenaService
from localmind.chat import ChatService
from localmind.config import Settings, configure_logging, get_settings
from localmind.db import Database
from localmind.personas import PersonaService
from localmind.tracing import Tracer, TracingStatus, init_tracing
from localmind.tracing.feedback import FeedbackService


@dataclass
class AppContext:
    """Everything a request handler or script needs."""

    settings: Settings
    db: Database
    chat: ChatService
    personas: PersonaService
    tracing: TracingStatus
    tracer: Tracer
    feedback: FeedbackService
    arena: ArenaService


def build_context(settings: Settings | None = None) -> AppContext:
    """Create tables, seed personas, configure logging and tracing."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    tracing = init_tracing(settings)
    tracer = Tracer(tracing, settings)
    db = Database(settings.database_url)
    db.create_all()
    personas = PersonaService(db)
    personas.seed_defaults()
    return AppContext(
        settings=settings,
        db=db,
        chat=ChatService(db, settings, tracer=tracer),
        personas=personas,
        tracing=tracing,
        tracer=tracer,
        feedback=FeedbackService(db, tracer),
        arena=ArenaService(db, settings, tracer=tracer),
    )
