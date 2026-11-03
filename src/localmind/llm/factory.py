"""The single place where chat models are constructed.

Centralising construction keeps model parameters, timeouts and retry policy
consistent across chat, arena, agent and evaluation.
"""

from __future__ import annotations

from typing import TypeVar

import httpx
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from localmind.config import Settings, get_settings

R = TypeVar("R", bound=Runnable)


class ModelParams(BaseModel):
    """User-tunable generation parameters (mirrors Ollama's ``options``)."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0, description="Ollama num_predict")
    num_ctx: int = Field(default=4096, ge=512, description="Context window in tokens")


def get_chat_model(
    model: str | None = None,
    params: ModelParams | None = None,
    settings: Settings | None = None,
) -> ChatOllama:
    """Build a ``ChatOllama`` for ``model`` with the given parameters.

    Returns the raw chat model (not wrapped in retry) so callers can still use
    model-specific methods like ``bind_tools`` — see ``with_connection_retry``.
    """
    settings = settings or get_settings()
    params = params or ModelParams()
    return ChatOllama(
        model=model or settings.default_model,
        base_url=settings.ollama_base_url,
        temperature=params.temperature,
        top_p=params.top_p,
        num_predict=params.max_tokens,
        # Set num_ctx explicitly: Ollama's default is small and silently truncates
        # long prompts from the *front*, which looks like the model "forgetting".
        num_ctx=params.num_ctx,
        client_kwargs={"timeout": httpx.Timeout(settings.ollama_timeout_s, connect=5.0)},
    )


def with_connection_retry(runnable: R, settings: Settings | None = None) -> Runnable:
    """Wrap a runnable so it retries when Ollama can't be reached.

    Only connection-level errors are retried: they happen before any token is
    produced, so a retry can't duplicate streamed output.
    """
    settings = settings or get_settings()
    return runnable.with_retry(
        retry_if_exception_type=(httpx.ConnectError, httpx.ConnectTimeout),
        stop_after_attempt=settings.ollama_max_retries,
        wait_exponential_jitter=True,
    )
