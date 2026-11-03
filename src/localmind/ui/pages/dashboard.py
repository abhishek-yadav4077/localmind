"""Dashboard: local analytics (works without LangSmith)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from localmind.ui.components import (
    bar,
    call,
    get_client,
    model_colors,
    table_view,
)

api = get_client()
st.title("📊 Dashboard")
st.caption("Computed from the local SQLite database. Speed metrics depend on your hardware.")

data = call(lambda: api.get("/dashboard"))
totals, fb = data["totals"], data["feedback"]

if not totals["assistant_messages"] and not totals["arena_battles"]:
    st.info("No data yet. Chat, run the arena or the agent, then come back.")
    st.stop()

per_model = pd.DataFrame(data["per_model"])
c = st.columns(5)
c[0].metric("Conversations", totals["conversations"])
c[1].metric("Answers", totals["assistant_messages"])
if not per_model.empty and per_model["avg_tokens_per_sec"].notna().any():
    weighted = (per_model["avg_tokens_per_sec"] * per_model["messages"]).sum() / per_model[
        per_model["avg_tokens_per_sec"].notna()
    ]["messages"].sum()
    c[2].metric("Avg tokens/sec", f"{weighted:.1f}")
ratio = fb["positive_ratio"]
c[3].metric(
    "👍 ratio",
    f"{ratio * 100:.0f}%" if ratio is not None else "—",
    help=f"{fb['thumbs_up']} up / {fb['thumbs_down']} down",
)
c[4].metric("Arena votes", totals["arena_votes"], help=f"{totals['arena_battles']} battles")

# --------------------------------------------------------- per model
if not per_model.empty:
    st.subheader("Per model")
    left, mid, right = st.columns(3)
    left.altair_chart(bar(per_model, "messages", "model", "Answers", ",d"))
    speed = per_model.dropna(subset=["avg_tokens_per_sec"])
    if not speed.empty:
        mid.altair_chart(bar(speed, "avg_tokens_per_sec", "model", "Avg tokens/sec"))
    lat = per_model.dropna(subset=["avg_latency_ms"]).assign(
        avg_latency_s=lambda d: d["avg_latency_ms"] / 1000
    )
    if not lat.empty:
        right.altair_chart(bar(lat, "avg_latency_s", "model", "Avg latency (s)", ",.2f"))
    table_view(per_model)

# ---------------------------------------------------------- over time
daily = pd.DataFrame(data["daily"])
if not daily.empty and daily["date"].nunique() > 1:
    st.subheader("Over time")
    daily["date"] = pd.to_datetime(daily["date"])
    scale = model_colors(daily["model"].dropna().tolist())
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(daily).encode(
        # Data is aggregated per day, so the axis shows days (not hours).
        x=alt.X(
            "yearmonthdate(date):T", title=None, axis=alt.Axis(format="%b %d", tickCount="day")
        ),
        color=alt.Color("model:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
    )
    for field, title in [("messages", "Answers per day"), ("avg_tokens_per_sec", "Tokens/sec")]:
        lines = base.mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
            y=alt.Y(f"{field}:Q", title=None),
            tooltip=["date:T", "model:N", alt.Tooltip(f"{field}:Q", format=",.1f")],
        )
        rule = (
            alt.Chart(daily)
            .mark_rule(strokeDash=[4, 4], opacity=0.5)
            .encode(x="yearmonthdate(date):T")
            .add_params(hover)
            .transform_filter(hover)
        )
        st.altair_chart(
            (lines + rule).properties(title=title, height=220), use_container_width=True
        )
    table_view(daily)

# ------------------------------------------------------- modes / arena
left, right = st.columns(2)
with left:
    modes = pd.DataFrame(
        [{"mode": k, "answers": v} for k, v in data["per_mode"].items()]
    ).sort_values("answers", ascending=False)
    if not modes.empty:
        st.subheader("Usage by mode")
        st.altair_chart(bar(modes, "answers", "mode", "Answers", ",d"))
with right:
    board = pd.DataFrame(data["leaderboard"])
    if not board.empty:
        st.subheader("Arena leaderboard")
        st.dataframe(
            board[["model", "battles", "wins", "ties", "win_rate", "avg_tokens_per_sec"]].rename(
                columns={"battles": "votes", "win_rate": "win rate", "avg_tokens_per_sec": "tok/s"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={"win rate": st.column_config.NumberColumn(format="percent")},
        )
