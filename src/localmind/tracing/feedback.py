"""Thumbs up/down feedback: saved locally, mirrored to LangSmith.

Local-first on purpose: a LangSmith outage (or no key at all) must never lose
the user's vote, so the DB write happens first and LangSmith sync is best effort.
"""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import select

from localmind.config import get_logger
from localmind.db import Database, Feedback, Message
from localmind.exceptions import LocalMindError, NotFoundError
from localmind.tracing.tracer import Tracer

logger = get_logger(__name__)


class MessageNotFoundError(NotFoundError):
    """No message with that id."""


class FeedbackOut(BaseModel):
    message_id: int
    score: int
    comment: str | None
    synced_to_langsmith: bool
    langsmith_feedback_id: str | None = None
    sync_error: str | None = None


class FeedbackService:
    """Record feedback for assistant messages."""

    def __init__(self, db: Database, tracer: Tracer) -> None:
        self.db = db
        self.tracer = tracer

    def submit(self, message_id: int, thumbs_up: bool, comment: str | None = None) -> FeedbackOut:
        """Save (or change) feedback on a message and sync it to LangSmith.

        Args:
            message_id: An assistant message id.
            thumbs_up: True = good (score 1), False = bad (score 0).
            comment: Optional free-text explanation.
        """
        score = 1 if thumbs_up else 0
        comment = (comment or "").strip() or None

        with self.db.session() as s:
            message = s.get(Message, message_id)
            if message is None:
                raise MessageNotFoundError(f"Message {message_id} not found")
            if message.role != "assistant":
                raise LocalMindError("Feedback can only be given on assistant messages")
            fb = s.scalar(select(Feedback).where(Feedback.message_id == message_id))
            if fb is None:
                fb = Feedback(message_id=message_id, score=score, comment=comment)
                s.add(fb)
            else:
                fb.score, fb.comment = score, comment
            run_id, ls_id = message.run_id, fb.langsmith_feedback_id

        out = FeedbackOut(
            message_id=message_id,
            score=score,
            comment=comment,
            synced_to_langsmith=False,
            langsmith_feedback_id=ls_id,
        )
        if not (self.tracer.enabled and run_id):
            return out

        try:
            new_ls_id = self.tracer.send_feedback(run_id, score, comment, feedback_id=ls_id)
        except Exception as exc:  # keep the local vote, report the sync error
            logger.warning("feedback sync failed", extra={"message_id": message_id}, exc_info=True)
            out.sync_error = str(exc)
            return out

        if new_ls_id and new_ls_id != ls_id:
            with self.db.session() as s:
                fb = s.scalar(select(Feedback).where(Feedback.message_id == message_id))
                if fb is not None:
                    fb.langsmith_feedback_id = new_ls_id
        out.synced_to_langsmith = True
        out.langsmith_feedback_id = new_ls_id
        return out

    def get(self, message_id: int) -> Feedback | None:
        with self.db.session() as s:
            return s.scalar(select(Feedback).where(Feedback.message_id == message_id))
