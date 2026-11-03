"""Pydantic DTOs returned by the chat service (and reused by the API).

Services return these instead of ORM objects so callers never trigger lazy
loads on a closed session, and the API gets JSON-ready models for free.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageStats(BaseModel):
    """Performance numbers for one assistant reply. ``None`` = not measurable."""

    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    conversation_id: str
    role: str
    content: str
    model: str | None = None
    mode: str = "chat"
    created_at: datetime
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    memory_summarized: bool = False
    run_id: str | None = None


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    model: str
    persona_id: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str | None = None
    summarized_up_to: int | None = None
    created_at: datetime
    updated_at: datetime


# --- streaming events ----------------------------------------------------------
# A tagged union: the API forwards each one as an SSE event and the UI switches on
# ``type``. Keeping them typed makes the protocol self-documenting.


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class MemoryEvent(BaseModel):
    type: Literal["memory_summarized"] = "memory_summarized"
    summarized_messages: int


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    message: MessageOut


class TitleEvent(BaseModel):
    type: Literal["title"] = "title"
    title: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    error: str
    hint: str | None = None


ChatEvent = TokenEvent | MemoryEvent | DoneEvent | TitleEvent | ErrorEvent
