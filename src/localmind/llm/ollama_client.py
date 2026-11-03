"""Thin, typed client for the Ollama REST API.

LangChain's ``ChatOllama`` handles *inference*. Everything else the app needs
from Ollama — is the server up, which models are installed, pull a model with
progress, how big is a model's context window — goes through this class.

We talk to the REST API with ``httpx`` directly (instead of the ``ollama``
package) so timeouts are explicit and tests can swap in ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from localmind.config import Settings, get_logger, get_settings
from localmind.exceptions import ModelNotFoundError, ModelPullError, OllamaUnavailableError

logger = get_logger(__name__)

# Only transient transport problems are worth retrying; a 404 won't fix itself.
_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


def normalize_model_name(name: str) -> str:
    """Return the canonical ``name:tag`` form (Ollama implies ``:latest``)."""
    return name if ":" in name else f"{name}:latest"


def install_hint(base_url: str) -> str:
    """Actionable instructions shown when Ollama is not reachable."""
    return (
        f"Could not reach Ollama at {base_url}. To fix:\n"
        "  1. Install Ollama: https://ollama.com/download "
        "(Linux: curl -fsSL https://ollama.com/install.sh | sh)\n"
        "  2. Start the server: `ollama serve` (the desktop app starts it automatically)\n"
        "  3. Check it responds: curl http://localhost:11434/api/version\n"
        "  4. If Ollama runs elsewhere (e.g. Docker), set LOCALMIND_OLLAMA_BASE_URL in .env"
    )


@dataclass(frozen=True)
class ModelInfo:
    """An installed Ollama model."""

    name: str
    size_bytes: int = 0
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / 1e9, 2)


@dataclass(frozen=True)
class PullProgress:
    """One progress update while pulling a model."""

    status: str
    completed: int | None = None
    total: int | None = None

    @property
    def fraction(self) -> float | None:
        """Download progress in [0, 1] for the current layer, if known."""
        if self.total and self.completed is not None:
            return min(self.completed / self.total, 1.0)
        return None


@dataclass
class HealthReport:
    """Result of an Ollama health check, with fix instructions when unhealthy."""

    reachable: bool
    base_url: str
    version: str | None = None
    installed_models: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Healthy = server reachable and at least one chat model installed."""
        return self.reachable and bool(self.installed_models)


class OllamaClient:
    """Synchronous Ollama REST client with timeouts and retries.

    Args:
        settings: App settings (base URL, timeout, retry count).
        transport: Optional httpx transport, used by tests to mock the server.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")
        # Short connect timeout so a dead server fails fast; long read timeout
        # because the first request after startup may block while a model loads.
        timeout = httpx.Timeout(self.settings.ollama_timeout_s, connect=5.0)
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    # -- context manager -----------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level -----------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send a request with retry on transient errors; map failures to our exceptions."""

        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(self.settings.ollama_max_retries),
            wait=wait_exponential(multiplier=0.5, max=4),
            reraise=True,
        )
        def _send() -> httpx.Response:
            return self._http.request(method, path, **kwargs)

        try:
            response = _send()
        except _RETRYABLE as exc:
            raise OllamaUnavailableError(
                f"Ollama not reachable ({exc.__class__.__name__})",
                hint=install_hint(self.base_url),
            ) from exc
        return response

    # -- public API ----------------------------------------------------------
    def version(self) -> str:
        """Return the Ollama server version (raises if unreachable)."""
        response = self._request("GET", "/api/version")
        response.raise_for_status()
        return str(response.json().get("version", "unknown"))

    def list_models(self) -> list[ModelInfo]:
        """Return installed models, sorted by name."""
        response = self._request("GET", "/api/tags")
        response.raise_for_status()
        models = []
        for m in response.json().get("models", []):
            details = m.get("details") or {}
            models.append(
                ModelInfo(
                    name=m["name"],
                    size_bytes=int(m.get("size", 0)),
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
            )
        return sorted(models, key=lambda m: m.name)

    def is_installed(self, model: str) -> bool:
        """True if ``model`` (with or without tag) is installed."""
        wanted = normalize_model_name(model)
        return any(normalize_model_name(m.name) == wanted for m in self.list_models())

    def context_length(self, model: str) -> int | None:
        """Return the model's trained context length from ``/api/show``, if reported.

        Note: Ollama's *runtime* window is ``num_ctx`` (default 2048-4096 depending
        on version), which may be smaller than what the model was trained for.
        """
        response = self._request("POST", "/api/show", json={"model": model})
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model}' is not installed", hint=f"Run: ollama pull {model}"
            )
        response.raise_for_status()
        info: dict[str, Any] = response.json().get("model_info") or {}
        # Key is architecture-specific, e.g. "llama.context_length", "qwen2.context_length".
        for key, value in info.items():
            if key.endswith(".context_length"):
                return int(value)
        return None

    def pull(self, model: str) -> Iterator[PullProgress]:
        """Pull ``model`` and yield progress updates (streamed NDJSON).

        Not retried: a pull is long-running and Ollama resumes partial downloads
        on its own, so the caller can simply call ``pull`` again.
        """
        try:
            with self._http.stream(
                "POST", "/api/pull", json={"model": model, "stream": True}, timeout=None
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ModelPullError(f"Pull failed for '{model}': {response.text}")
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise ModelPullError(
                            f"Pull failed for '{model}': {data['error']}",
                            hint="Check the model name at https://ollama.com/library",
                        )
                    yield PullProgress(
                        status=data.get("status", ""),
                        completed=data.get("completed"),
                        total=data.get("total"),
                    )
        except _RETRYABLE as exc:
            raise OllamaUnavailableError(
                "Ollama not reachable during pull", hint=install_hint(self.base_url)
            ) from exc
        logger.info("model pulled", extra={"model": model})

    def health(self) -> HealthReport:
        """Check server reachability and installed models; never raises."""
        report = HealthReport(reachable=False, base_url=self.base_url)
        try:
            report.version = self.version()
            report.reachable = True
            report.installed_models = [m.name for m in self.list_models()]
        except OllamaUnavailableError as exc:
            report.hints.append(exc.hint or str(exc))
            return report
        except httpx.HTTPError as exc:  # reachable but misbehaving (e.g. 500)
            report.hints.append(f"Ollama responded with an error: {exc}")
            return report

        installed = {normalize_model_name(n) for n in report.installed_models}
        report.missing_preferred = [
            m for m in self.settings.preferred_models if normalize_model_name(m) not in installed
        ]
        if not report.installed_models:
            report.hints.append(
                f"No models installed. Pull one, e.g.: ollama pull {self.settings.default_model}"
            )
        elif normalize_model_name(self.settings.default_model) not in installed:
            report.hints.append(
                f"Default model '{self.settings.default_model}' is not installed. "
                f"Run: ollama pull {self.settings.default_model} "
                "(or change LOCALMIND_DEFAULT_MODEL)"
            )
        return report
