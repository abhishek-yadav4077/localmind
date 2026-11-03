"""ORM models.

Schema overview::

    personas 1──* conversations 1──* messages

Per-message performance stats live on ``messages`` (not a separate table) because
they are always read together and it keeps the dashboard queries one-table simple.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class Persona(Base):
    """A reusable system-prompt preset."""

    __tablename__ = "personas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    description: Mapped[str] = mapped_column(String(300), default="")
    system_prompt: Mapped[str] = mapped_column(Text)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Conversation(Base):
    """A chat thread with its model, persona, generation params and rolling summary."""

    __tablename__ = "conversations"

    # UUID strings: safe to expose in URLs and as LangSmith metadata.
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    model: Mapped[str] = mapped_column(String(100))
    persona_id: Mapped[int | None] = mapped_column(
        ForeignKey("personas.id", ondelete="SET NULL"), nullable=True
    )
    # Generation params as JSON (temperature, top_p, max_tokens, num_ctx) so new
    # knobs don't need a schema migration.
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Rolling memory: summary of every message with id <= summarized_up_to.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summarized_up_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    persona: Mapped[Persona | None] = relationship(lazy="joined")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    """One chat message plus performance stats for assistant replies."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    mode: Mapped[str] = mapped_column(String(16), default="chat")  # chat/arena/agent/rag
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # --- stats (assistant messages only) ---
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # True when older turns were summarized right before this reply (UI indicator).
    memory_summarized: Mapped[bool] = mapped_column(Boolean, default=False)
    # LangSmith run id of the LLM call that produced this message (Phase 3).
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
