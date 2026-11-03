"""Model Arena: one prompt, 2-3 models side by side, blind voting, leaderboard."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from localmind.ui.api_client import APIError
from localmind.ui.components import (
    bar,
    call,
    get_client,
    installed_models,
    palette,
    personas,
    show_error,
    stats_line,
    trace_url,
)

api = get_client()
state = st.session_state
state.setdefault("battle", None)

st.title("⚔️ Model Arena")
st.caption(
    "Send one prompt to several local models and vote for the better answer. "
    "Blind mode hides model names until you vote, to avoid brand bias."
)

models = installed_models()
persona_names = {p["id"]: p["name"] for p in personas()}

with st.form("arena"):
    chosen = st.multiselect(
        "Models (2-3)", models, default=models[:2], max_selections=3, placeholder="Pick models"
    )
    prompt = st.text_area("Prompt", placeholder="Explain overfitting to a 12-year-old.")
    c1, c2, c3, c4 = st.columns(4)
    persona_id = c1.selectbox(
        "Persona",
        [None, *persona_names],
        format_func=lambda i: "None" if i is None else persona_names[i],
    )
    temperature = c2.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    sequential = c3.toggle(
        "Sequential",
        help="Run models one at a time. Slower, but speed numbers become comparable: "
        "parallel runs share CPU/GPU/RAM.",
    )
    blind = c4.toggle("Blind voting", value=True)
    run = st.form_submit_button("⚔️ Fight", type="primary")

labels = "ABC"


def name_for(position: int, model: str, voted: bool) -> str:
    return f"Model {labels[position]}" if blind and not voted else model


if run:
    if not 2 <= len(chosen) <= 3 or not prompt.strip():
        st.warning("Pick 2-3 models and write a prompt.")
    else:
        state.battle = None
        cols = st.columns(len(chosen))
        slots, texts = {}, {m: "" for m in chosen}
        for i, (col, model) in enumerate(zip(cols, chosen, strict=True)):
            col.markdown(f"#### {name_for(i, model, False)}")
            slots[model] = col.empty()
        body = {
            "prompt": prompt,
            "models": chosen,
            "persona_id": persona_id,
            "params": {"temperature": temperature},
            "parallel": not sequential,
        }
        try:
            for event, data in api.stream("/arena", body):
                if event == "token":
                    texts[data["model"]] += data["content"]
                    slots[data["model"]].markdown(texts[data["model"]] + " ▌")
                elif event == "result":
                    r = data["response"]
                    slots[r["model"]].markdown(texts[r["model"]] or "")
                elif event == "done":
                    state.battle = data["battle"]
        except APIError as exc:
            show_error(exc.error, exc.hint)
        if state.battle:
            st.rerun()

# ---------------------------------------------------- result + vote
battle = state.battle
if battle:
    voted = battle["outcome"] is not None
    responses = battle["responses"]
    st.markdown(f"**Prompt:** {battle['prompt']}")
    cols = st.columns(len(responses))
    for col, r in zip(cols, responses, strict=True):
        with col:
            title = name_for(r["position"], r["model"], voted)
            winner = voted and battle["winner_model"] == r["model"]
            st.markdown(f"#### {'🏆 ' if winner else ''}{title}")
            with st.container(border=True):
                if r["error"]:
                    show_error(r["error"])
                else:
                    st.markdown(r["content"])
            if not r["error"] and (voted or not blind):
                url = trace_url(r.get("run_id"))
                st.caption(stats_line(r) + (f" · [🔍 trace]({url})" if url else ""))
            elif not r["error"]:
                # Speed is a tell (small models are faster), so hide it until the vote.
                st.caption("Speed stats appear after you vote.")

    ok = [r for r in responses if not r["error"]]
    vote_cols = st.columns(len(ok) + 2)
    for col, r in zip(vote_cols, ok, strict=False):
        label = f"👍 {name_for(r['position'], r['model'], voted)} wins"
        if col.button(label, key=f"win_{r['model']}", use_container_width=True):
            state.battle = call(
                lambda r=r: api.post(
                    f"/arena/{battle['id']}/vote", {"outcome": "win", "winner_model": r["model"]}
                )
            )
            st.rerun()
    if vote_cols[-2].button("🤝 Tie", use_container_width=True):
        state.battle = call(lambda: api.post(f"/arena/{battle['id']}/vote", {"outcome": "tie"}))
        st.rerun()
    if vote_cols[-1].button("👎 All bad", use_container_width=True):
        state.battle = call(lambda: api.post(f"/arena/{battle['id']}/vote", {"outcome": "all_bad"}))
        st.rerun()
    if voted:
        st.success(
            f"Vote recorded: {battle['winner_model'] or battle['outcome'].replace('_', ' ')}. "
            "You can change it; LangSmith feedback is updated, not duplicated."
        )

# ------------------------------------------------------- leaderboard
st.divider()
st.subheader("🏆 Leaderboard")
seq_only = st.toggle("Speed columns from sequential battles only (fair timing)")
rows = call(lambda: api.get("/arena/leaderboard", sequential_only=seq_only), stop=False) or []
if not rows:
    st.info("No battles yet. Run one above.")
else:
    df = pd.DataFrame(rows)
    voted_df = df[df["battles"] > 0].assign(win_pct=lambda d: d["win_rate"] * 100)
    left, right = st.columns(2)
    if not voted_df.empty:
        chart = (
            alt.Chart(voted_df, title="Win rate (%)")
            .mark_bar(cornerRadiusEnd=4, color=palette()[0])
            .encode(
                x=alt.X("win_pct:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("model:N", title=None, sort="-x", scale=alt.Scale(paddingInner=0.35)),
                tooltip=[
                    "model",
                    alt.Tooltip("win_pct:Q", format=".0f"),
                    "battles",
                    "wins",
                    "ties",
                ],
            )
            .properties(height=alt.Step(30))
        )
        left.altair_chart(chart)
        left.caption("Few votes = noisy win rate. Check the vote count.")
    speed = df.dropna(subset=["avg_tokens_per_sec"])
    if not speed.empty:
        right.altair_chart(bar(speed, "avg_tokens_per_sec", "model", "Avg tokens/sec"))
    st.dataframe(
        df.rename(
            columns={
                "battles": "votes",
                "win_rate": "win rate",
                "avg_tokens_per_sec": "tok/s",
                "avg_latency_ms": "latency ms",
                "avg_ttft_ms": "TTFT ms",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"win rate": st.column_config.NumberColumn(format="percent")},
    )
