"""POST /agent: stream the tool-calling agent's steps as SSE."""

from __future__ import annotations

from fastapi import APIRouter

from localmind.api.deps import Ctx
from localmind.api.schemas import AgentRequest
from localmind.api.sse import sse_response

router = APIRouter(tags=["agent"])


@router.post("/agent")
async def agent(body: AgentRequest, ctx: Ctx):
    """Events: ``start``, ``tool_call``/``tool_result``*, ``done`` or ``error``."""
    return await sse_response(
        lambda: ctx.agent.stream(
            body.question,
            conversation_id=body.conversation_id,
            model=body.model,
            web_search=body.web_search,
        )
    )
