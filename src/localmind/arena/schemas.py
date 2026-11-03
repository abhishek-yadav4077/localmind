"""Arena DTOs and streaming events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Outcome = Literal["win", "tie", "all_bad"]


class ResponseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    position: int
    model: str
    content: str
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    run_id: str | None = None


class BattleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    prompt: str
    persona_id: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    parallel: bool
    outcome: str | None = None
    winner_model: str | None = None
    created_at: datetime
    responses: list[ResponseOut] = Field(default_factory=list)


# Events: tokens from different models interleave, so each carries ``model``.


class ArenaStartEvent(BaseModel):
    type: Literal["start"] = "start"
    battle_id: str
    models: list[str]


class ArenaTokenEvent(BaseModel):
    type: Literal["token"] = "token"
    model: str
    content: str


class ArenaResultEvent(BaseModel):
    """One model finished (successfully or with ``response.error`` set)."""

    type: Literal["result"] = "result"
    response: ResponseOut
    hint: str | None = None


class ArenaDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    battle: BattleOut


ArenaEvent = ArenaStartEvent | ArenaTokenEvent | ArenaResultEvent | ArenaDoneEvent
