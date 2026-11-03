"""GET /dashboard: local analytics for the Dashboard page."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from localmind.api.deps import Ctx
from localmind.arena import compute_leaderboard
from localmind.db.analytics import dashboard_stats

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard")
def dashboard(ctx: Ctx) -> dict[str, Any]:
    stats = dashboard_stats(ctx.db)
    stats["leaderboard"] = [r.model_dump() for r in compute_leaderboard(ctx.db)]
    return stats
