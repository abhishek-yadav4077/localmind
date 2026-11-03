"""Conversation CRUD, messages (with feedback + trace links) and export."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Response
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from localmind.api.deps import Ctx
from localmind.api.schemas import ConversationCreate, ConversationUpdate, MessageView
from localmind.chat import ConversationOut
from localmind.db import Feedback

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
def list_conversations(
    ctx: Ctx, q: str | None = None, limit: int = Query(100, le=500)
) -> list[ConversationOut]:
    return ctx.chat.list_conversations(query=q, limit=limit)


@router.post("", status_code=201)
def create_conversation(body: ConversationCreate, ctx: Ctx) -> ConversationOut:
    return ctx.chat.create_conversation(body.model, body.persona_id, body.params, body.title)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, ctx: Ctx) -> ConversationOut:
    return ctx.chat.get_conversation(conversation_id)


@router.patch("/{conversation_id}")
def update_conversation(
    conversation_id: str, body: ConversationUpdate, ctx: Ctx
) -> ConversationOut:
    sent = body.model_dump(exclude_unset=True)
    return ctx.chat.update_conversation(
        conversation_id,
        title=body.title,
        model=body.model,
        # Distinguish "not sent" (keep) from an explicit null (clear the persona).
        persona_id=body.persona_id if "persona_id" in sent else "unset",
        params=body.params,
    )


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, ctx: Ctx) -> Response:
    ctx.chat.delete_conversation(conversation_id)
    return Response(status_code=204)


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: str, ctx: Ctx) -> list[MessageView]:
    messages = await run_in_threadpool(ctx.chat.get_messages, conversation_id)
    ids = [m.id for m in messages]
    with ctx.db.session() as s:
        feedback = {
            f.message_id: f for f in s.scalars(select(Feedback).where(Feedback.message_id.in_(ids)))
        }
    views = []
    for m in messages:
        fb = feedback.get(m.id)
        views.append(
            MessageView(
                **m.model_dump(),
                feedback_score=fb.score if fb else None,
                feedback_comment=fb.comment if fb else None,
                trace_url=await run_in_threadpool(ctx.tracer.run_url, m.run_id),
            )
        )
    return views


@router.get("/{conversation_id}/export")
def export_conversation(
    conversation_id: str, ctx: Ctx, format: Literal["markdown", "json"] = "markdown"
) -> Response:
    body = ctx.chat.export(conversation_id, format)
    media, ext = ("text/markdown", "md") if format == "markdown" else ("application/json", "json")
    return Response(
        body,
        media_type=f"{media}; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="conversation-{conversation_id}.{ext}"'
        },
    )
