"""POST /chat: stream one chat turn as SSE."""

from __future__ import annotations

from fastapi import APIRouter

from localmind.api.deps import Ctx
from localmind.api.schemas import ChatRequest
from localmind.api.sse import sse_response

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(body: ChatRequest, ctx: Ctx):
    """Events: ``conversation`` (if created), ``token``*, ``memory_summarized``?,
    ``done``, ``title``?, or ``error``."""

    def events():
        conversation_id = body.conversation_id
        if conversation_id is None:
            conv = ctx.chat.create_conversation(body.model, body.persona_id, body.params)
            conversation_id = conv.id
            yield {"type": "conversation", "conversation": conv.model_dump(mode="json")}
        yield from ctx.chat.stream_reply(conversation_id, body.message)

    return await sse_response(events)
