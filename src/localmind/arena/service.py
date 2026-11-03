"""Run one prompt against 2-3 models and record the user's vote.

Concurrency: each model streams in its own worker thread and pushes events onto
a queue; the generator drains the queue so the UI sees all columns fill at once.
With ``parallel=False`` the same code runs with a single worker, i.e. one model
after another.

Fairness caveat (worth knowing in an interview): on one machine, parallel runs
share CPU/GPU/RAM bandwidth, and if the models don't all fit in memory Ollama
swaps them in and out. Parallel mode is great for *quality* comparison, but
speed numbers are only comparable from sequential runs; the leaderboard can
filter to those.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import select

from localmind.arena.schemas import (
    ArenaDoneEvent,
    ArenaEvent,
    ArenaResultEvent,
    ArenaStartEvent,
    ArenaTokenEvent,
    BattleOut,
    Outcome,
    ResponseOut,
)
from localmind.chat.streaming import CompletionStream, StreamResult
from localmind.config import Settings, get_logger, get_settings
from localmind.db import ArenaBattle, ArenaResponse, Database, Persona
from localmind.db.models import utcnow
from localmind.exceptions import LocalMindError
from localmind.llm import ModelParams, get_chat_model
from localmind.llm.errors import describe_ollama_error
from localmind.tracing import Tracer, disabled_tracer

logger = get_logger(__name__)

LLMFactory = Callable[[str, ModelParams], BaseChatModel]
MIN_MODELS, MAX_MODELS = 2, 3
ARENA_FEEDBACK_KEY = "arena-win"


class BattleNotFoundError(LocalMindError):
    """No battle with that id."""


@dataclass
class _Finished:
    """Internal queue message: a worker is done."""

    model: str
    position: int
    run_id: str | None
    result: StreamResult | None = None
    error: Exception | None = None


class ArenaService:
    """Create battles, stream answers side by side, record votes."""

    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        llm_factory: LLMFactory | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.llm_factory: LLMFactory = llm_factory or (
            lambda model, params: get_chat_model(model, params, self.settings)
        )
        self.tracer = tracer or disabled_tracer(self.settings)

    # ------------------------------------------------------------------ battle
    def stream_battle(
        self,
        prompt: str,
        models: list[str],
        params: ModelParams | None = None,
        persona_id: int | None = None,
        parallel: bool = True,
    ) -> Iterator[ArenaEvent]:
        """Send ``prompt`` to every model and yield interleaved events.

        A failing model doesn't stop the others: its result carries ``error``.
        """
        prompt = prompt.strip()
        models = list(dict.fromkeys(m.strip() for m in models if m.strip()))  # dedupe, keep order
        if not prompt:
            raise LocalMindError("Prompt is empty")
        if not MIN_MODELS <= len(models) <= MAX_MODELS:
            raise LocalMindError(f"Pick {MIN_MODELS}-{MAX_MODELS} different models")
        params = params or ModelParams()

        with self.db.session() as s:
            persona = s.get(Persona, persona_id) if persona_id else None
            system_prompt, persona_name = (
                (persona.system_prompt, persona.name) if persona else (None, None)
            )
            battle = ArenaBattle(
                prompt=prompt, persona_id=persona_id, params=params.model_dump(), parallel=parallel
            )
            s.add(battle)
            s.flush()
            battle_id = battle.id

        messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        if system_prompt:
            messages.insert(0, SystemMessage(content=system_prompt))

        yield ArenaStartEvent(battle_id=battle_id, models=models)

        events: queue.Queue[ArenaTokenEvent | _Finished] = queue.Queue()
        cancelled = threading.Event()

        def worker(position: int, model: str) -> None:
            config = self.tracer.run_config(
                "arena",
                model=model,
                persona=persona_name,
                run_name="arena",
                extra_metadata={"battle_id": battle_id, "parallel": parallel},
            )
            done = _Finished(model, position, self.tracer.stored_run_id(config))
            try:
                stream = CompletionStream(self.llm_factory(model, params), messages, config)
                for token in stream:
                    if cancelled.is_set():  # consumer went away (e.g. browser closed)
                        return
                    events.put(ArenaTokenEvent(model=model, content=token))
                done.result = stream.result
            except Exception as exc:
                done.error = exc
            events.put(done)

        executor = ThreadPoolExecutor(
            max_workers=len(models) if parallel else 1, thread_name_prefix="arena"
        )
        try:
            for position, model in enumerate(models):
                executor.submit(worker, position, model)
            remaining = len(models)
            while remaining:
                item = events.get()
                if isinstance(item, ArenaTokenEvent):
                    yield item
                    continue
                remaining -= 1
                yield self._save_response(battle_id, item)
        finally:
            cancelled.set()
            executor.shutdown(wait=False, cancel_futures=True)

        yield ArenaDoneEvent(battle=self.get_battle(battle_id))

    def run_battle(self, prompt: str, models: list[str], **kwargs: Any) -> BattleOut:
        """Blocking convenience wrapper around ``stream_battle``."""
        battle: BattleOut | None = None
        for event in self.stream_battle(prompt, models, **kwargs):
            if isinstance(event, ArenaDoneEvent):
                battle = event.battle
        assert battle is not None
        return battle

    def _save_response(self, battle_id: str, item: _Finished) -> ArenaResultEvent:
        hint = None
        fields: dict[str, Any] = {}
        if item.result is not None:
            fields = {"content": item.result.text, **item.result.stats.model_dump()}
        if item.error is not None:
            friendly = describe_ollama_error(item.error, item.model, self.settings.ollama_base_url)
            fields, hint = {"error": friendly.error}, friendly.hint
            logger.warning(
                "arena model failed", extra={"model": item.model, "error": friendly.error}
            )
        with self.db.session() as s:
            row = ArenaResponse(
                battle_id=battle_id,
                position=item.position,
                model=item.model,
                run_id=item.run_id if item.error is None else None,
                **fields,
            )
            s.add(row)
            s.flush()
            return ArenaResultEvent(response=ResponseOut.model_validate(row), hint=hint)

    # -------------------------------------------------------------------- vote
    def vote(self, battle_id: str, outcome: Outcome, winner_model: str | None = None) -> BattleOut:
        """Record (or change) the verdict and mirror it to LangSmith.

        Feedback key ``arena-win`` on each response's run: winner 1, others 0;
        tie 0.5 for everyone; all_bad 0 for everyone. Averaging that key per
        model in LangSmith reproduces the local win rate.
        """
        with self.db.session() as s:
            battle = self._require(s, battle_id)
            ok_models = {r.model for r in battle.responses if r.error is None}
            if outcome == "win":
                if winner_model not in ok_models:
                    raise LocalMindError(f"Winner must be one of: {', '.join(sorted(ok_models))}")
            else:
                winner_model = None
            battle.outcome, battle.winner_model, battle.voted_at = outcome, winner_model, utcnow()
            to_sync = [
                (
                    r.id,
                    r.run_id,
                    r.langsmith_feedback_id,
                    _arena_score(outcome, winner_model, r.model),
                )
                for r in battle.responses
                if r.run_id
            ]

        for response_id, run_id, fb_id, score in to_sync:
            try:
                new_id = self.tracer.send_feedback(
                    run_id, score, feedback_id=fb_id, key=ARENA_FEEDBACK_KEY
                )
            except Exception:
                logger.warning("arena feedback sync failed", exc_info=True)
                continue
            if new_id and new_id != fb_id:
                with self.db.session() as s:
                    row = s.get(ArenaResponse, response_id)
                    if row is not None:
                        row.langsmith_feedback_id = new_id
        return self.get_battle(battle_id)

    # ------------------------------------------------------------------ reads
    def get_battle(self, battle_id: str) -> BattleOut:
        with self.db.session() as s:
            return BattleOut.model_validate(self._require(s, battle_id))

    def list_battles(self, limit: int = 50) -> list[BattleOut]:
        with self.db.session() as s:
            rows = s.scalars(
                select(ArenaBattle).order_by(ArenaBattle.created_at.desc()).limit(limit)
            ).all()
            return [BattleOut.model_validate(b) for b in rows]

    @staticmethod
    def _require(session, battle_id: str) -> ArenaBattle:
        battle = session.get(ArenaBattle, battle_id)
        if battle is None:
            raise BattleNotFoundError(f"Battle {battle_id} not found")
        return battle


def _arena_score(outcome: str, winner: str | None, model: str) -> float:
    if outcome == "tie":
        return 0.5
    return 1.0 if outcome == "win" and model == winner else 0.0
