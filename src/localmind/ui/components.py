"""Shared Streamlit helpers: API access, status, stats lines, feedback, charts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import altair as alt
import pandas as pd
import streamlit as st

from localmind.ui.api_client import APIClient, APIError

T = TypeVar("T")

# Categorical palette in fixed order (validated for colour-vision deficiency;
# light/dark steps of the same hues). Colours follow the *model*, never its rank.
PALETTE = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
}


@st.cache_resource
def get_client() -> APIClient:
    return APIClient()


def call(fn: Callable[[], T], *, stop: bool = True) -> T | None:
    """Run an API call; show a friendly error (and optionally stop the page) on failure."""
    try:
        return fn()
    except APIError as exc:
        show_error(exc.error, exc.hint)
        if stop:
            st.stop()
        return None


def show_error(error: str, hint: str | None = None) -> None:
    st.error(error, icon="⚠️")
    if hint:
        st.caption(f"**How to fix:** {hint}")


@st.cache_data(ttl=10, show_spinner=False)
def health() -> dict[str, Any] | None:
    try:
        return get_client().get("/health")
    except APIError:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def installed_models() -> list[str]:
    try:
        return [m["name"] for m in get_client().get("/models")["models"]]
    except APIError:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def personas() -> list[dict[str, Any]]:
    try:
        return get_client().get("/personas")
    except APIError:
        return []


def model_picker(label: str, default: str | None, key: str, help: str | None = None) -> str | None:
    """Selectbox over installed models; falls back to a text box if none are listed."""
    models = installed_models()
    if not models:
        return st.text_input(label, value=default or "", key=key, help=help) or None
    norm = {m.removesuffix(":latest"): m for m in models}
    index = models.index(norm.get(default or "", models[0])) if default else 0
    return st.selectbox(label, models, index=index, key=key, help=help)


def stats_line(m: dict[str, Any]) -> str:
    """'llama3.2 · 1.4 s · TTFT 220 ms · 31.2 tok/s · 120 → 85 tokens'."""
    parts = [m.get("model") or ""]
    if (lat := m.get("latency_ms")) is not None:
        parts.append(f"{lat / 1000:.1f} s" if lat >= 1000 else f"{lat:.0f} ms")
    if (ttft := m.get("ttft_ms")) is not None:
        parts.append(f"TTFT {ttft:.0f} ms")
    if (tps := m.get("tokens_per_sec")) is not None:
        parts.append(f"{tps:.1f} tok/s")
    if m.get("prompt_tokens") is not None and m.get("completion_tokens") is not None:
        parts.append(f"{m['prompt_tokens']} → {m['completion_tokens']} tokens")
    return " · ".join(p for p in parts if p)


def trace_url(run_id: str | None) -> str | None:
    h = health()
    template = (h or {}).get("langsmith", {}).get("trace_url_template")
    return template.replace("{run_id}", run_id) if template and run_id else None


def message_footer(m: dict[str, Any]) -> None:
    """Stats, memory badge, trace link and feedback widget under an assistant message."""
    bits = [stats_line(m)]
    if m.get("memory_summarized"):
        bits.append("🧠 memory summarized")
    url = m.get("trace_url") or trace_url(m.get("run_id"))
    if url:
        bits.append(f"[🔍 LangSmith trace]({url})")
    st.caption(" · ".join(b for b in bits if b))
    feedback_widget(m)


def feedback_widget(m: dict[str, Any]) -> None:
    """👍/👎 (saved immediately) plus an optional comment."""
    mid = m["id"]

    def send(comment: str | None = None) -> None:
        value = st.session_state.get(f"fb_{mid}")
        if value is None:
            return
        try:
            out = get_client().post(
                "/feedback", {"message_id": mid, "thumbs_up": value == 1, "comment": comment}
            )
        except APIError as exc:
            st.toast(f"Feedback failed: {exc.error}", icon="⚠️")
            return
        where = "LangSmith + local" if out["synced_to_langsmith"] else "saved locally"
        st.toast(f"Feedback {where}", icon="✅")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.feedback("thumbs", key=f"fb_{mid}", default=m.get("feedback_score"), on_change=send)
        popover = st.popover("💬", help="Add a comment to your feedback")
    with popover:
        text = st.text_area(
            "Why?", value=m.get("feedback_comment") or "", key=f"fbc_{mid}", height=80
        )
        if st.button("Send", key=f"fbs_{mid}"):
            if st.session_state.get(f"fb_{mid}") is None:
                st.warning("Pick 👍 or 👎 first")
            else:
                send(text)


def sidebar_status() -> None:
    """Ollama / LangSmith status badges shown on every page."""
    h = health()
    with st.sidebar:
        st.divider()
        if h is None:
            st.markdown("🔴 **API offline**")
            st.caption("Run `uvicorn localmind.api.main:app`")
            return
        ollama = h["ollama"]
        icon = "🟢" if ollama["ok"] else ("🟠" if ollama["reachable"] else "🔴")
        st.markdown(f"{icon} **Ollama** {ollama.get('version') or 'offline'}")
        ls = h["langsmith"]
        if ls["enabled"]:
            link = f"[{ls['project']}]({ls['project_url']})" if ls["project_url"] else ls["project"]
            st.markdown(f"🟢 **LangSmith** → {link}")
        else:
            st.markdown("⚪ **LangSmith** off")
        st.caption("🔒 Inference is 100% local")


# ----------------------------------------------------------------- charts
def palette() -> list[str]:
    theme = getattr(st.context, "theme", None)
    return PALETTE["dark" if theme is not None and theme.type == "dark" else "light"]


def model_colors(models: list[str]) -> alt.Scale:
    """Stable colour per model: sorted names -> palette slots in fixed order."""
    domain = sorted(set(models))[: len(palette())]
    return alt.Scale(domain=domain, range=palette()[: len(domain)])


def bar(df: pd.DataFrame, x: str, y: str, title: str, fmt: str = ",.1f") -> alt.Chart:
    """Horizontal single-series bar chart (the axis names each bar, so one colour).

    Height is a per-row *step*, not pixels: a fixed pixel height gets squashed
    inside Streamlit columns, while a step grows with the number of bars.
    """
    integer = fmt.endswith("d")
    axis = alt.Axis(format=fmt, grid=True, tickCount=4, tickMinStep=1 if integer else alt.Undefined)
    return (
        alt.Chart(df, title=title)
        .mark_bar(cornerRadiusEnd=4, color=palette()[0])
        .encode(
            x=alt.X(f"{x}:Q", title=None, axis=axis),
            y=alt.Y(f"{y}:N", title=None, sort="-x", scale=alt.Scale(paddingInner=0.35)),
            tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)],
        )
        .properties(height=alt.Step(30))
    )


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def table_view(df: pd.DataFrame, label: str = "Show data table") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, use_container_width=True)
