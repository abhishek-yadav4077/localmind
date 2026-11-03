"""Read-only access to saved evaluation runs (runs are started from the CLI)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from localmind.api.deps import Ctx
from localmind.evaluation.results import list_runs, load_run

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/runs")
def runs(ctx: Ctx) -> list[dict[str, Any]]:
    return list_runs(ctx.settings.eval_results_dir)


@router.get("/runs/{run_id}")
def run(run_id: str, ctx: Ctx) -> dict[str, Any]:
    return load_run(ctx.settings.eval_results_dir, run_id)
