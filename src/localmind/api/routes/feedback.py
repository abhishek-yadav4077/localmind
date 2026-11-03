"""POST /feedback: thumbs up/down on an assistant message."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from localmind.api.deps import Ctx
from localmind.api.schemas import FeedbackRequest
from localmind.tracing.feedback import FeedbackOut

router = APIRouter(tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, ctx: Ctx) -> FeedbackOut:
    # Off the event loop: the LangSmith call is a blocking network request.
    return await run_in_threadpool(
        ctx.feedback.submit, body.message_id, body.thumbs_up, body.comment
    )
