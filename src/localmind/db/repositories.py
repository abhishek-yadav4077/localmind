"""Data-access functions. Services call these; nothing else writes SQL.

Functions take an open ``Session`` so the caller controls the transaction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from localmind.db.models import Conversation, Message, Persona, utcnow

# --- personas ----------------------------------------------------------------


def list_personas(session: Session) -> Sequence[Persona]:
    return session.scalars(select(Persona).order_by(Persona.is_builtin.desc(), Persona.name)).all()


def get_persona(session: Session, persona_id: int) -> Persona | None:
    return session.get(Persona, persona_id)


def get_persona_by_name(session: Session, name: str) -> Persona | None:
    return session.scalar(select(Persona).where(Persona.name == name))


# --- conversations -------------------------------------------------------------


def create_conversation(
    session: Session,
    model: str,
    persona_id: int | None = None,
    params: dict[str, Any] | None = None,
    title: str = "New chat",
) -> Conversation:
    conv = Conversation(model=model, persona_id=persona_id, params=params or {}, title=title)
    session.add(conv)
    session.flush()  # assign id
    return conv


def get_conversation(session: Session, conversation_id: str) -> Conversation | None:
    return session.get(Conversation, conversation_id)


def list_conversations(
    session: Session, query: str | None = None, limit: int = 100
) -> Sequence[Conversation]:
    """List conversations, newest activity first.

    ``query`` matches titles *or* message content (case-insensitive LIKE). Fine
    for a personal app with thousands of messages; FTS5 would be the next step.
    """
    stmt = select(Conversation)
    if query:
        pattern = f"%{query.lower()}%"
        in_messages = select(Message.conversation_id).where(
            func.lower(Message.content).like(pattern)
        )
        stmt = stmt.where(
            or_(func.lower(Conversation.title).like(pattern), Conversation.id.in_(in_messages))
        )
    stmt = stmt.order_by(Conversation.updated_at.desc()).limit(limit)
    return session.scalars(stmt).unique().all()


def touch_conversation(session: Session, conv: Conversation) -> None:
    conv.updated_at = utcnow()


def delete_conversation(session: Session, conversation_id: str) -> bool:
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        return False
    session.delete(conv)
    return True


# --- messages -----------------------------------------------------------------


def add_message(
    session: Session, conversation_id: str, role: str, content: str, **fields
) -> Message:
    msg = Message(conversation_id=conversation_id, role=role, content=content, **fields)
    session.add(msg)
    session.flush()
    return msg


def list_messages(
    session: Session, conversation_id: str, after_id: int | None = None
) -> Sequence[Message]:
    stmt = select(Message).where(Message.conversation_id == conversation_id)
    if after_id is not None:
        stmt = stmt.where(Message.id > after_id)
    return session.scalars(stmt.order_by(Message.id)).all()


def search_messages(session: Session, query: str, limit: int = 10) -> Sequence[Message]:
    """Full-text-ish search over past messages (used by the agent tool)."""
    pattern = f"%{query.lower()}%"
    stmt = (
        select(Message)
        .where(func.lower(Message.content).like(pattern))
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    return session.scalars(stmt).all()
