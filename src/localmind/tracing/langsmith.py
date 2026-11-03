"""LangSmith setup with graceful degradation.

Design: tracing is *optional*. If the user hasn't configured LangSmith, the app
must work exactly the same, minus traces. We therefore decide once at startup
and apply the decision globally via ``langsmith.configure`` — LangChain's tracer
reads the same global client/flag, so every chain, agent and arena run follows it.

Why not just rely on env vars? ``langsmith`` caches env lookups, and with
``LANGSMITH_TRACING=true`` but no API key it would log noisy 401/403 errors on
every run. Explicitly disabling avoids that.
"""

from __future__ import annotations

from dataclasses import dataclass

import langsmith as ls

from localmind.config import Settings, get_logger, get_settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class TracingStatus:
    """Outcome of ``init_tracing``."""

    enabled: bool
    project: str | None
    reason: str


def init_tracing(settings: Settings | None = None) -> TracingStatus:
    """Enable or disable LangSmith tracing for the whole process.

    Safe to call more than once (later calls overwrite the global config).
    """
    settings = settings or get_settings()

    if not settings.langsmith_tracing:
        ls.configure(enabled=False)
        status = TracingStatus(False, None, "LANGSMITH_TRACING is not 'true'")
        logger.info("langsmith tracing disabled", extra={"reason": status.reason})
        return status

    if not settings.langsmith_configured:
        ls.configure(enabled=False)
        status = TracingStatus(False, None, "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty")
        logger.warning(
            "langsmith tracing requested but not configured; continuing without traces",
            extra={"reason": status.reason},
        )
        return status

    assert settings.langsmith_api_key is not None  # guaranteed by langsmith_configured
    client = ls.Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key.get_secret_value(),
    )
    ls.configure(client=client, enabled=True, project_name=settings.langsmith_project)
    logger.info("langsmith tracing enabled", extra={"project": settings.langsmith_project})
    return TracingStatus(True, settings.langsmith_project, "configured")
