"""``Tracer``: the app's facade over LangSmith.

Services depend on this class, not on ``langsmith`` directly. When tracing is
disabled every method is a cheap no-op, so callers never need ``if enabled``
checks and the app behaves identically without LangSmith.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from langsmith.utils import LangSmithNotFoundError, get_host_url

from localmind.config import Settings, get_logger, get_settings
from localmind.tracing.langsmith import TracingStatus
from localmind.tracing.metadata import Mode, build_run_config

logger = get_logger(__name__)

FEEDBACK_KEY = "user-score"
LOOKUP_RETRY_S = 60.0  # after a failed project lookup, don't retry on every request


class Tracer:
    """Builds run configs, trace URLs and sends feedback.

    Args:
        status: Result of ``init_tracing`` (carries the client when enabled).
        settings: App settings (project name, endpoint, app version).
    """

    def __init__(self, status: TracingStatus, settings: Settings | None = None) -> None:
        self.status = status
        self.settings = settings or get_settings()
        self.client = status.client if status.enabled else None
        self._ids: tuple[str, str] | None = None  # (tenant_id, project_id), cached
        self._failed_at: float | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @property
    def project(self) -> str | None:
        return self.status.project

    # ----------------------------------------------------------- run config
    def run_config(
        self,
        mode: Mode,
        *,
        model: str,
        persona: str | None = None,
        conversation_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """``build_run_config`` with the app version filled in."""
        return build_run_config(
            mode,
            model=model,
            persona=persona,
            conversation_id=conversation_id,
            app_version=self.settings.app_version,
            **kwargs,
        )

    def stored_run_id(self, config: dict[str, Any]) -> str | None:
        """The run id worth persisting: only when a trace will actually exist."""
        return str(config["run_id"]) if self.enabled and "run_id" in config else None

    # ----------------------------------------------------------------- URLs
    @property
    def host_url(self) -> str:
        """Web UI host for the configured API endpoint (handles EU / self-hosted)."""
        return get_host_url(None, self.settings.langsmith_endpoint)

    def _resolve_ids(self) -> tuple[str, str] | None:
        """Look up (tenant_id, project_id) once. Creates the project if needed.

        One network call for the whole process instead of one per message. On
        failure we return None and only retry after ``LOOKUP_RETRY_S``, so a page
        listing 50 messages doesn't make 50 failing network calls.
        """
        if self._ids is not None or self.client is None:
            return self._ids
        if self._failed_at and time.monotonic() - self._failed_at < LOOKUP_RETRY_S:
            return None
        with self._lock:
            if self._ids is not None:
                return self._ids
            name = self.project or "default"
            try:
                try:
                    project = self.client.read_project(project_name=name)
                except LangSmithNotFoundError:
                    # Projects are created lazily by the first trace; create it now
                    # so links work even before that trace has been ingested.
                    project = self.client.create_project(name)
                self._ids = (str(project.tenant_id), str(project.id))
            except Exception:  # trace links are a nice-to-have; never break a request
                logger.warning("could not resolve LangSmith project ids", exc_info=True)
                self._failed_at = time.monotonic()
                return None
        return self._ids

    def tenant_id(self) -> str | None:
        """Workspace id, needed for deep links (None when unknown/disabled)."""
        ids = self._resolve_ids()
        return ids[0] if ids else None

    def project_url(self) -> str | None:
        ids = self._resolve_ids()
        if ids is None:
            return None
        tenant_id, project_id = ids
        return f"{self.host_url}/o/{tenant_id}/projects/p/{project_id}"

    def run_url(self, run_id: str | uuid.UUID | None) -> str | None:
        """Deep link to a trace. ``?poll=true`` makes the UI wait if not ingested yet."""
        if not run_id:
            return None
        base = self.project_url()
        return f"{base}/r/{run_id}?poll=true" if base else None

    # -------------------------------------------------------------- feedback
    def send_feedback(
        self,
        run_id: str,
        score: float,
        comment: str | None = None,
        feedback_id: str | None = None,
        key: str = FEEDBACK_KEY,
    ) -> str | None:
        """Create or update feedback on a run. Returns the feedback id, or None if disabled.

        Passing an existing ``feedback_id`` updates that feedback instead of
        adding a second one, so changing a vote doesn't double count.
        Raises on API errors; the caller decides how to degrade.
        """
        if self.client is None:
            return None
        if feedback_id:
            self.client.update_feedback(feedback_id, score=score, comment=comment)
            return feedback_id
        new_id = uuid.uuid4()
        self.client.create_feedback(run_id, key, score=score, comment=comment, feedback_id=new_id)
        return str(new_id)

    def flush(self) -> None:
        """Block until queued traces are sent (call before a script exits)."""
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:  # never fail a script's exit because of tracing
                logger.warning("langsmith flush failed", exc_info=True)


def disabled_tracer(settings: Settings | None = None) -> Tracer:
    """A no-op tracer (tests, or when LangSmith isn't configured)."""
    return Tracer(TracingStatus(False, None, "disabled"), settings)
