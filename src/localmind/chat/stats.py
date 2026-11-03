"""Per-message performance stats.

Definitions (what the UI shows):
- latency:          request sent -> last token received (wall clock)
- time to first token (TTFT): request sent -> first non-empty token. Includes
  model load time on a cold start, which is why the first message is slower.
- tokens/sec:       generation speed. Preferred source is Ollama's own
  ``eval_count / eval_duration`` (pure decode time, excludes prompt processing
  and network). Falls back to completion tokens / (end - first token).
- token counts:     Ollama's ``prompt_eval_count`` / ``eval_count`` via
  LangChain's ``usage_metadata``. Note: Ollama caches the prompt prefix, so
  prompt_tokens can be lower than the full prompt on follow-up turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from localmind.chat.schemas import MessageStats


@dataclass
class StreamTimer:
    """Collects timestamps while a response streams."""

    start: float = 0.0
    first_token: float | None = None
    end: float | None = None

    def begin(self) -> None:
        self.start = time.perf_counter()

    def on_token(self, text: str) -> None:
        if self.first_token is None and text:
            self.first_token = time.perf_counter()

    def finish(self) -> None:
        self.end = time.perf_counter()


def compute_stats(
    timer: StreamTimer,
    response_metadata: dict[str, Any] | None,
    usage_metadata: dict[str, Any] | None,
) -> MessageStats:
    """Combine wall-clock timings with Ollama-reported counters."""
    meta = response_metadata or {}
    usage = usage_metadata or {}

    latency_ms = (timer.end - timer.start) * 1000 if timer.end else None
    ttft_ms = (timer.first_token - timer.start) * 1000 if timer.first_token else None

    prompt_tokens = usage.get("input_tokens", meta.get("prompt_eval_count"))
    completion_tokens = usage.get("output_tokens", meta.get("eval_count"))

    tokens_per_sec: float | None = None
    eval_count, eval_duration_ns = meta.get("eval_count"), meta.get("eval_duration")
    if eval_count and eval_duration_ns:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
    elif completion_tokens and timer.first_token and timer.end and timer.end > timer.first_token:
        tokens_per_sec = completion_tokens / (timer.end - timer.first_token)

    return MessageStats(
        latency_ms=_round(latency_ms),
        ttft_ms=_round(ttft_ms),
        tokens_per_sec=_round(tokens_per_sec),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _round(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None
