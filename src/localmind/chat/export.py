"""Export a conversation as Markdown or JSON."""

from __future__ import annotations

import json
from collections.abc import Sequence

from localmind.chat.schemas import ConversationOut, MessageOut


def _stats_line(m: MessageOut) -> str:
    parts = [m.model or ""]
    if m.latency_ms is not None:
        ms = m.latency_ms
        parts.append(f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s")
    if m.tokens_per_sec is not None:
        parts.append(f"{m.tokens_per_sec:.1f} tok/s")
    if m.completion_tokens is not None:
        parts.append(f"{m.completion_tokens} tokens")
    return " · ".join(p for p in parts if p)


def to_markdown(
    conv: ConversationOut, messages: Sequence[MessageOut], persona_name: str | None = None
) -> str:
    lines = [
        f"# {conv.title}",
        "",
        f"- **Model:** {conv.model}",
        f"- **Persona:** {persona_name or 'none'}",
        f"- **Created:** {conv.created_at:%Y-%m-%d %H:%M} UTC",
        "",
    ]
    if conv.summary:
        lines += ["> **Memory summary:** " + conv.summary.replace("\n", " "), ""]
    for m in messages:
        if m.role == "user":
            lines += ["### 🧑 User", "", m.content, ""]
        else:
            lines += [f"### 🤖 Assistant <sub>{_stats_line(m)}</sub>", "", m.content, ""]
    return "\n".join(lines).rstrip() + "\n"


def to_json(conv: ConversationOut, messages: Sequence[MessageOut]) -> str:
    payload = {
        "conversation": conv.model_dump(mode="json"),
        "messages": [m.model_dump(mode="json") for m in messages],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
