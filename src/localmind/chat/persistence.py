"""Store single-shot exchanges (agent, RAG) as conversation messages.

Agent and document answers live in the same conversations/messages tables as
chat, so history, search, export, feedback, trace links and the dashboard work
for every mode without special cases.
"""

from __future__ import annotations

from typing import Any

from localmind.chat.schemas import MessageOut, MessageStats
from localmind.db import Database
from localmind.db import repositories as repo
from localmind.exceptions import LocalMindError


def ensure_conversation(db: Database, conversation_id: str | None, model: str, title: str) -> str:
    """Return ``conversation_id`` if it exists, else create a conversation."""
    with db.session() as s:
        if conversation_id:
            if repo.get_conversation(s, conversation_id) is None:
                raise LocalMindError(f"Conversation {conversation_id} not found")
            return conversation_id
        conv = repo.create_conversation(s, model=model, title=title[:80])
        conv.title_generated = True  # first question is a good enough title here
        return conv.id


def history(db: Database, conversation_id: str, limit: int) -> list[tuple[str, str]]:
    """Last ``limit`` (role, content) pairs of a conversation, oldest first."""
    with db.session() as s:
        rows = repo.list_messages(s, conversation_id)
        return [(m.role, m.content) for m in rows[-limit:]] if limit else []


def save_exchange(
    db: Database,
    conversation_id: str,
    *,
    user_text: str,
    answer: str,
    model: str,
    mode: str,
    stats: MessageStats | None = None,
    run_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> MessageOut:
    """Persist the user question + assistant answer; return the assistant message."""
    with db.session() as s:
        conv = repo.get_conversation(s, conversation_id)
        if conv is None:
            raise LocalMindError(f"Conversation {conversation_id} not found")
        repo.add_message(s, conversation_id, "user", user_text, mode=mode)
        msg = repo.add_message(
            s,
            conversation_id,
            "assistant",
            answer,
            model=model,
            mode=mode,
            run_id=run_id,
            details=details,
            **(stats.model_dump() if stats else {}),
        )
        repo.touch_conversation(s, conv)
        return MessageOut.model_validate(msg)
