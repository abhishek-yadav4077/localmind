"""Leaderboard: win rate and speed per model, from local arena data.

Definitions:
- battles:   voted battles in which the model produced an answer (errors excluded)
- win rate:  wins / battles. Ties and "all bad" count as battles without a win.
- speed:     averages over the model's successful responses. With
  ``sequential_only=True`` only battles run one model at a time are used,
  because parallel runs compete for the same hardware (see arena/service.py).

Win rate is shown together with the vote count: with few votes it is noise.
A Bradley-Terry/Elo rating would use pairwise information better; that is
listed as future work.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from localmind.db import ArenaBattle, Database


class LeaderboardRow(BaseModel):
    model: str
    battles: int
    wins: int
    ties: int
    win_rate: float | None
    responses: int
    avg_tokens_per_sec: float | None
    avg_latency_ms: float | None
    avg_ttft_ms: float | None


def _avg(values: list[float]) -> float | None:
    return round(mean(values), 1) if values else None


def compute_leaderboard(db: Database, sequential_only: bool = False) -> list[LeaderboardRow]:
    """Aggregate all battles into one row per model, best win rate first."""
    with db.session() as s:
        battles = s.scalars(select(ArenaBattle).options(selectinload(ArenaBattle.responses))).all()

        battles_n: dict[str, int] = defaultdict(int)
        wins: dict[str, int] = defaultdict(int)
        ties: dict[str, int] = defaultdict(int)
        speed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        responses_n: dict[str, int] = defaultdict(int)

        for battle in battles:
            for r in battle.responses:
                if r.error is not None:
                    continue
                responses_n[r.model] += 1
                if battle.outcome is not None:
                    battles_n[r.model] += 1
                    wins[r.model] += int(battle.outcome == "win" and battle.winner_model == r.model)
                    ties[r.model] += int(battle.outcome == "tie")
                if sequential_only and battle.parallel:
                    continue
                for field in ("tokens_per_sec", "latency_ms", "ttft_ms"):
                    value = getattr(r, field)
                    if value is not None:
                        speed[r.model][field].append(value)

    rows = [
        LeaderboardRow(
            model=model,
            battles=battles_n[model],
            wins=wins[model],
            ties=ties[model],
            win_rate=round(wins[model] / battles_n[model], 3) if battles_n[model] else None,
            responses=responses_n[model],
            avg_tokens_per_sec=_avg(speed[model]["tokens_per_sec"]),
            avg_latency_ms=_avg(speed[model]["latency_ms"]),
            avg_ttft_ms=_avg(speed[model]["ttft_ms"]),
        )
        for model in responses_n
    ]
    rows.sort(key=lambda r: (r.win_rate is not None, r.win_rate or 0, r.battles), reverse=True)
    return rows
