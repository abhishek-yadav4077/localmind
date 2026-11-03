"""Translate low-level Ollama/HTTP exceptions into user-facing messages + fixes."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from ollama import ResponseError

from localmind.llm.ollama_client import install_hint


@dataclass(frozen=True)
class FriendlyError:
    error: str
    hint: str | None = None


def describe_ollama_error(exc: Exception, model: str, base_url: str) -> FriendlyError:
    """Map an exception raised while calling a model to a message the user can act on."""
    if isinstance(exc, httpx.ConnectError | ConnectionError):
        return FriendlyError("Ollama is not reachable", install_hint(base_url))
    if isinstance(exc, ResponseError) and exc.status_code == 404:
        return FriendlyError(f"Model '{model}' is not installed", f"Run: ollama pull {model}")
    if isinstance(exc, httpx.TimeoutException):
        return FriendlyError(
            "Ollama timed out",
            "The model may still be loading; retry, or raise LOCALMIND_OLLAMA_TIMEOUT_S",
        )
    return FriendlyError(f"{type(exc).__name__}: {exc}")
