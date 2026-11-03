"""Installed models and model pulls."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from localmind.api.deps import Ctx
from localmind.api.schemas import PullRequest
from localmind.api.sse import sse_response
from localmind.exceptions import LocalMindError

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
async def list_models(ctx: Ctx) -> dict[str, Any]:
    models = await run_in_threadpool(ctx.ollama.list_models)
    installed = {m.name for m in models}
    return {
        "models": [asdict(m) | {"size_gb": m.size_gb} for m in models],
        "preferred": [
            {"name": p, "installed": p in installed or f"{p}:latest" in installed}
            for p in ctx.settings.preferred_models
        ],
        "default_model": ctx.settings.default_model,
    }


@router.post("/pull")
async def pull_model(body: PullRequest, ctx: Ctx):
    """Stream pull progress as SSE ``progress`` events, then ``done`` (or ``error``)."""

    def events():
        try:
            for p in ctx.ollama.pull(body.model):
                yield {
                    "type": "progress",
                    "status": p.status,
                    "completed": p.completed,
                    "total": p.total,
                    "fraction": p.fraction,
                }
        except LocalMindError as exc:
            yield {"type": "error", "error": exc.message, "hint": exc.hint}
            return
        yield {"type": "done", "model": body.model}

    return await sse_response(events)
