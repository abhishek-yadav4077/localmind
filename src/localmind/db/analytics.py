"""Read-only aggregate queries for the Dashboard page.

Everything here is computed from the local SQLite database, so the dashboard
works with or without LangSmith. Results are plain dicts/lists ready for JSON.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import String, case, func, select

from localmind.db.models import ArenaBattle, ArenaResponse, Conversation, Feedback, Message
from localmind.db.session import Database


def _r(value: float | None, digits: int = 1) -> float | None:
    return round(float(value), digits) if value is not None else None


def dashboard_stats(db: Database) -> dict[str, Any]:
    """Totals, per-model performance, feedback, per-mode usage and daily series."""
    with db.session() as s:
        assistant = Message.role == "assistant"

        totals = {
            "conversations": s.scalar(select(func.count(Conversation.id))) or 0,
            "messages": s.scalar(select(func.count(Message.id))) or 0,
            "assistant_messages": s.scalar(select(func.count(Message.id)).where(assistant)) or 0,
            "arena_battles": s.scalar(select(func.count(ArenaBattle.id))) or 0,
            "arena_votes": s.scalar(
                select(func.count(ArenaBattle.id)).where(ArenaBattle.outcome.is_not(None))
            )
            or 0,
        }

        per_model = [
            {
                "model": model,
                "messages": n,
                "avg_latency_ms": _r(lat),
                "avg_ttft_ms": _r(ttft),
                "avg_tokens_per_sec": _r(tps),
                "total_completion_tokens": int(tokens or 0),
                "thumbs_up": int(up or 0),
                "thumbs_down": int(down or 0),
            }
            for model, n, lat, ttft, tps, tokens, up, down in s.execute(
                select(
                    Message.model,
                    func.count(Message.id),
                    func.avg(Message.latency_ms),
                    func.avg(Message.ttft_ms),
                    func.avg(Message.tokens_per_sec),
                    func.sum(Message.completion_tokens),
                    func.sum(case((Feedback.score == 1, 1), else_=0)),
                    func.sum(case((Feedback.score == 0, 1), else_=0)),
                )
                .outerjoin(Feedback, Feedback.message_id == Message.id)
                .where(assistant)
                .group_by(Message.model)
                .order_by(func.count(Message.id).desc())
            )
        ]

        up = s.scalar(select(func.count(Feedback.id)).where(Feedback.score == 1)) or 0
        down = s.scalar(select(func.count(Feedback.id)).where(Feedback.score == 0)) or 0
        feedback = {
            "thumbs_up": up,
            "thumbs_down": down,
            "positive_ratio": round(up / (up + down), 3) if up + down else None,
        }

        per_mode = {
            mode: n
            for mode, n in s.execute(
                select(Message.mode, func.count(Message.id)).where(assistant).group_by(Message.mode)
            )
        }
        arena_responses = s.scalar(select(func.count(ArenaResponse.id))) or 0
        if arena_responses:
            per_mode["arena"] = arena_responses

        # func.date -> 'YYYY-MM-DD' text in SQLite (and Postgres), portable and JSON-ready.
        day = func.date(Message.created_at, type_=String)
        daily = [
            {
                "date": str(d),
                "model": model,
                "messages": n,
                "avg_tokens_per_sec": _r(tps),
                "avg_latency_ms": _r(lat),
            }
            for d, model, n, tps, lat in s.execute(
                select(
                    day,
                    Message.model,
                    func.count(Message.id),
                    func.avg(Message.tokens_per_sec),
                    func.avg(Message.latency_ms),
                )
                .where(assistant)
                .group_by(day, Message.model)
                .order_by(day)
            )
        ]

    return {
        "totals": totals,
        "per_model": per_model,
        "feedback": feedback,
        "per_mode": per_mode,
        "daily": daily,
    }
