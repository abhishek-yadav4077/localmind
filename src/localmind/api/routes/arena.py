"""Model Arena endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from localmind.api.deps import Ctx
from localmind.api.schemas import ArenaRequest, VoteRequest
from localmind.api.sse import sse_response
from localmind.arena import BattleOut, LeaderboardRow, compute_leaderboard

router = APIRouter(prefix="/arena", tags=["arena"])


@router.post("")
async def run_battle(body: ArenaRequest, ctx: Ctx):
    """Events: ``start``, interleaved ``token`` (with ``model``), ``result`` per model, ``done``."""
    return await sse_response(
        lambda: ctx.arena.stream_battle(
            body.prompt,
            body.models,
            params=body.params,
            persona_id=body.persona_id,
            parallel=body.parallel,
        )
    )


@router.post("/{battle_id}/vote")
def vote(battle_id: str, body: VoteRequest, ctx: Ctx) -> BattleOut:
    return ctx.arena.vote(battle_id, body.outcome, body.winner_model)


@router.get("/battles")
def list_battles(ctx: Ctx, limit: int = Query(50, le=500)) -> list[BattleOut]:
    return ctx.arena.list_battles(limit)


@router.get("/battles/{battle_id}")
def get_battle(battle_id: str, ctx: Ctx) -> BattleOut:
    return ctx.arena.get_battle(battle_id)


@router.get("/leaderboard")
def leaderboard(ctx: Ctx, sequential_only: bool = False) -> list[LeaderboardRow]:
    return compute_leaderboard(ctx.db, sequential_only=sequential_only)
