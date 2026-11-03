"""Arena: parallel/sequential battles, failures, votes, LangSmith feedback, leaderboard."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from localmind.arena import (
    ArenaDoneEvent,
    ArenaResultEvent,
    ArenaService,
    ArenaStartEvent,
    ArenaTokenEvent,
    compute_leaderboard,
)
from localmind.exceptions import LocalMindError
from localmind.personas import PersonaService
from localmind.tracing import Tracer, TracingStatus
from tests.fakes import FakeOllamaChat

# Different simulated decode times so speed stats differ per model.
EVAL_NS = {"fast": 250_000_000, "slow": 1_000_000_000, "mid": 500_000_000}


class ArenaFactory:
    """Each model answers '<model> answer number one' (4 tokens); 'broken' fails."""

    def __init__(self) -> None:
        self.prompts: dict[str, list] = {}

    def __call__(self, model: str, params) -> FakeOllamaChat:
        llm = FakeOllamaChat(
            responder=lambda msgs: f"{model} answer number one",
            eval_duration_ns=EVAL_NS.get(model, 500_000_000),
            error=httpx.ConnectError("refused") if model == "broken" else None,
        )
        self.prompts[model] = llm.calls
        return llm


@pytest.fixture
def factory() -> ArenaFactory:
    return ArenaFactory()


@pytest.fixture
def arena(db, settings, factory) -> ArenaService:
    return ArenaService(db, settings, llm_factory=factory)


def test_parallel_battle_streams_all_models(arena: ArenaService) -> None:
    events = list(arena.stream_battle("Explain RAG", ["fast", "slow"]))
    assert isinstance(events[0], ArenaStartEvent) and events[0].models == ["fast", "slow"]
    assert isinstance(events[-1], ArenaDoneEvent)

    for model in ("fast", "slow"):
        text = "".join(
            e.content for e in events if isinstance(e, ArenaTokenEvent) and e.model == model
        )
        assert text == f"{model} answer number one"

    battle = events[-1].battle
    assert [r.model for r in battle.responses] == ["fast", "slow"]  # ordered by position
    fast, slow = battle.responses
    assert fast.tokens_per_sec == 16.0 and slow.tokens_per_sec == 4.0
    assert battle.parallel is True and battle.outcome is None


def test_sequential_mode_runs_one_model_at_a_time(arena: ArenaService) -> None:
    events = list(arena.stream_battle("hi", ["fast", "slow", "mid"], parallel=False))
    order = [e.model for e in events if isinstance(e, ArenaTokenEvent)]
    # No interleaving: each model's tokens form one contiguous block.
    blocks = [m for i, m in enumerate(order) if i == 0 or order[i - 1] != m]
    assert blocks == ["fast", "slow", "mid"]
    assert events[-1].battle.parallel is False


@pytest.mark.parametrize(
    ("prompt", "models"),
    [("", ["a", "b"]), ("hi", ["a"]), ("hi", ["a", "a"]), ("hi", ["a", "b", "c", "d"])],
)
def test_validation(arena: ArenaService, prompt: str, models: list[str]) -> None:
    with pytest.raises(LocalMindError):
        list(arena.stream_battle(prompt, models))


def test_one_failing_model_does_not_stop_others(arena: ArenaService) -> None:
    events = list(arena.stream_battle("hi", ["fast", "broken"]))
    results = {e.response.model: e for e in events if isinstance(e, ArenaResultEvent)}
    assert results["fast"].response.error is None
    assert results["broken"].response.error == "Ollama is not reachable"
    assert "ollama serve" in (results["broken"].hint or "")
    battle_id = events[-1].battle.id
    with pytest.raises(LocalMindError):
        arena.vote(battle_id, "win", "broken")  # can't win without an answer


def test_persona_is_sent_to_every_model(arena: ArenaService, db, factory) -> None:
    personas = PersonaService(db)
    personas.seed_defaults()
    tutor = next(p for p in personas.list() if p.name == "Tutor")
    arena.run_battle("hi", ["fast", "slow"], persona_id=tutor.id)
    for model in ("fast", "slow"):
        prompt = factory.prompts[model][0]
        assert prompt[0].content == tutor.system_prompt and prompt[1].content == "hi"


def test_vote_and_change_vote(arena: ArenaService) -> None:
    battle = arena.run_battle("hi", ["fast", "slow"])
    voted = arena.vote(battle.id, "win", "slow")
    assert (voted.outcome, voted.winner_model) == ("win", "slow")
    changed = arena.vote(battle.id, "tie", "slow")
    assert (changed.outcome, changed.winner_model) == ("tie", None)


def test_vote_sends_arena_feedback_to_langsmith(db, settings, factory) -> None:
    client = MagicMock()
    tracer = Tracer(TracingStatus(True, "p", "ok", client=client), settings)
    arena = ArenaService(db, settings, llm_factory=factory, tracer=tracer)
    battle = arena.run_battle("hi", ["fast", "slow"])
    run_ids = {r.model: r.run_id for r in battle.responses}
    assert all(run_ids.values()) and len(set(run_ids.values())) == 2

    arena.vote(battle.id, "win", "fast")
    scores = {c.args[0]: c.kwargs["score"] for c in client.create_feedback.call_args_list}
    assert scores == {run_ids["fast"]: 1.0, run_ids["slow"]: 0.0}
    assert {c.args[1] for c in client.create_feedback.call_args_list} == {"arena-win"}

    arena.vote(battle.id, "tie")
    assert client.create_feedback.call_count == 2  # updated, not duplicated
    assert [c.kwargs["score"] for c in client.update_feedback.call_args_list] == [0.5, 0.5]


def test_leaderboard(arena: ArenaService, db) -> None:
    b1 = arena.run_battle("q1", ["fast", "slow"])  # parallel
    arena.vote(b1.id, "win", "fast")
    b2 = arena.run_battle("q2", ["fast", "slow"], parallel=False)
    arena.vote(b2.id, "win", "fast")
    b3 = arena.run_battle("q3", ["fast", "slow", "mid"], parallel=False)
    arena.vote(b3.id, "tie")
    arena.run_battle("q4", ["slow", "mid"])  # not voted

    rows = {r.model: r for r in compute_leaderboard(db)}
    assert (rows["fast"].wins, rows["fast"].battles, rows["fast"].win_rate) == (2, 3, 0.667)
    assert (rows["slow"].wins, rows["slow"].battles, rows["slow"].ties) == (0, 3, 1)
    assert rows["slow"].responses == 4
    assert (rows["mid"].battles, rows["mid"].win_rate) == (1, 0.0)
    assert rows["fast"].avg_tokens_per_sec == 16.0
    assert compute_leaderboard(db)[0].model == "fast"

    seq = {r.model: r for r in compute_leaderboard(db, sequential_only=True)}
    assert seq["mid"].avg_tokens_per_sec == 8.0
    assert seq["fast"].win_rate == rows["fast"].win_rate  # filter only affects speed


def test_leaderboard_empty(db) -> None:
    assert compute_leaderboard(db) == []
