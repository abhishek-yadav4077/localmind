"""Run metadata and tags attached to every traced LangChain run.

Tags vs metadata in LangSmith:
- **tags** are flat strings: quick filters in the UI (``mode:arena``) and easy
  to group by in charts.
- **metadata** is key/value: precise filters (``conversation_id = ...``) and it
  shows up structured on the trace. ``conversation_id`` is also one of the keys
  LangSmith uses to group runs into *Threads*, so a whole chat can be replayed
  as one thread in the UI.

We set both so traces are easy to slice either way.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

Mode = Literal["chat", "arena", "agent", "rag", "eval", "script"]


def slug(value: str) -> str:
    """``"Code Reviewer"`` -> ``"code-reviewer"`` (tag-friendly)."""
    return re.sub(r"[^a-z0-9.:_-]+", "-", value.lower()).strip("-")


def build_run_config(
    mode: Mode,
    *,
    model: str,
    app_version: str,
    persona: str | None = None,
    conversation_id: str | None = None,
    run_name: str | None = None,
    extra_tags: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Build a LangChain ``RunnableConfig`` with our tracing conventions.

    The ``run_id`` is generated *here*, before the call, so it can be stored on
    the message and later used for feedback and trace links — no need to fish
    it out of callbacks afterwards.
    """
    tags = ["localmind", f"mode:{mode}", f"model:{model}"]
    if persona:
        tags.append(f"persona:{slug(persona)}")
    tags.extend(extra_tags or [])

    metadata: dict[str, Any] = {
        "app": "localmind",
        "app_version": app_version,
        "mode": mode,
        "model": model,
        "persona": persona or "none",
    }
    if conversation_id:
        metadata["conversation_id"] = conversation_id
    metadata.update(extra_metadata or {})

    config: dict[str, Any] = {
        "run_id": run_id or uuid.uuid4(),
        "tags": tags,
        "metadata": metadata,
    }
    if run_name:
        config["run_name"] = run_name
    return config


def child_config(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    """Config for a secondary run (summary, title) that shares tags/metadata.

    Drops ``run_id``: reusing it would make LangSmith treat two runs as one.
    """
    child = {k: v for k, v in config.items() if k not in {"run_id", "run_name"}}
    if run_name:
        child["run_name"] = run_name
    return child


def merge_configs(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a caller's config into ours: tags concatenate, metadata merges, rest overrides."""
    if not extra:
        return dict(base)
    merged = {**base, **{k: v for k, v in extra.items() if k not in {"tags", "metadata"}}}
    merged["tags"] = [*base.get("tags", []), *extra.get("tags", [])]
    merged["metadata"] = {**base.get("metadata", {}), **extra.get("metadata", {})}
    return merged
