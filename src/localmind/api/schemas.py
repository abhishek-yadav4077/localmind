"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from localmind.chat.schemas import MessageOut
from localmind.llm import ModelParams


class MessageView(MessageOut):
    """A message plus UI extras: its feedback and LangSmith link."""

    feedback_score: int | None = None
    feedback_comment: str | None = None
    trace_url: str | None = None


class ConversationCreate(BaseModel):
    model: str | None = None
    persona_id: int | None = None
    params: ModelParams | None = None
    title: str = "New chat"


class ConversationUpdate(BaseModel):
    """PATCH body: only fields that are sent are changed."""

    title: str | None = None
    model: str | None = None
    persona_id: int | None = None
    params: ModelParams | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    # Used only when creating a new conversation (no conversation_id).
    model: str | None = None
    persona_id: int | None = None
    params: ModelParams | None = None


class ArenaRequest(BaseModel):
    prompt: str = Field(min_length=1)
    models: list[str] = Field(min_length=2, max_length=3)
    persona_id: int | None = None
    params: ModelParams | None = None
    parallel: bool = True


class VoteRequest(BaseModel):
    outcome: Literal["win", "tie", "all_bad"]
    winner_model: str | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    model: str | None = None
    web_search: bool = False


class AskDocumentsRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] | None = None
    model: str | None = None
    conversation_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)


class FeedbackRequest(BaseModel):
    message_id: int
    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=2000)


class PersonaIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    system_prompt: str = Field(min_length=1)
    description: str = ""


class PersonaPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    system_prompt: str | None = None
    description: str | None = None


class PersonaOut(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    description: str
    system_prompt: str
    is_builtin: bool


class PullRequest(BaseModel):
    model: str = Field(min_length=1)
