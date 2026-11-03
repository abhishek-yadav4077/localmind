"""Custom exception hierarchy.

Every error the app raises on purpose derives from ``LocalMindError`` so the API
layer can map them to HTTP responses in one place, and the UI can show the
human-readable ``hint`` instead of a stack trace.
"""

from __future__ import annotations


class LocalMindError(Exception):
    """Base class for all LocalMind errors.

    Args:
        message: What went wrong.
        hint: Optional actionable fix shown to the user.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  -> {self.hint}" if self.hint else self.message


class ConfigurationError(LocalMindError):
    """Invalid or missing configuration."""


class OllamaError(LocalMindError):
    """Base class for Ollama-related failures."""


class OllamaUnavailableError(OllamaError):
    """The Ollama server could not be reached."""


class ModelNotFoundError(OllamaError):  # see also NotFoundError mapping in the API
    """The requested model is not installed in Ollama."""


class ModelPullError(OllamaError):
    """Pulling a model from the Ollama registry failed."""


class NotFoundError(LocalMindError):
    """A requested resource does not exist (mapped to HTTP 404)."""


class FeatureUnavailableError(LocalMindError):
    """An optional feature's dependencies are not installed (HTTP 501)."""
