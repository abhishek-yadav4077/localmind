"""GET /health: Ollama, LangSmith and optional-feature status in one call."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from localmind.api.deps import Ctx

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(ctx: Ctx) -> dict[str, Any]:
    report = await run_in_threadpool(ctx.ollama.health)
    project_url = await run_in_threadpool(ctx.tracer.project_url)
    return {
        "status": "ok" if report.ok else "degraded",
        "version": ctx.settings.app_version,
        "ollama": asdict(report) | {"ok": report.ok},
        "langsmith": {
            "enabled": ctx.tracer.enabled,
            "project": ctx.tracer.project,
            "reason": ctx.tracing.reason,
            "project_url": project_url,
            # Lets clients build trace links for streamed results without a round trip.
            "trace_url_template": f"{project_url}/r/{{run_id}}?poll=true" if project_url else None,
        },
        "features": {
            "rag": importlib.util.find_spec("langchain_chroma") is not None,
            "web_search": importlib.util.find_spec("ddgs") is not None,
        },
        "defaults": {
            "model": ctx.settings.default_model,
            "agent_model": ctx.settings.agent_model,
            "embedding_model": ctx.settings.embedding_model,
        },
    }
