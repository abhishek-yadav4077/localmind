"""Evaluation: compare models on the 30-prompt dataset (runs come from the CLI)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from localmind.ui.components import bar, call, get_client, md_safe, model_colors, table_view

api = get_client()
st.title("🧪 Evaluation")
st.caption(
    "30 prompts across 5 categories, each local model answering at temperature 0, graded "
    "by a local LLM judge plus deterministic checks. Runs are started from the terminal."
)

HOW_TO = """```bash
python scripts/run_eval.py --per-category 1        # 5-question smoke run (~1-3 min)
python scripts/run_eval.py                         # full run, all installed models
python scripts/run_eval.py --judge qwen2.5:7b      # a stronger judge, if you have one
```"""

runs = call(lambda: api.get("/evaluation/runs"))
if not runs:
    st.info("No evaluation runs yet. Start one from the project root:", icon="🧪")
    st.markdown(HOW_TO)
    st.stop()


def label(m: dict) -> str:
    return f"{m['run_id']} · {', '.join(m.get('models', []))} · judge {m.get('judge_model')}"


chosen = st.selectbox("Run", runs, format_func=label)
data = call(lambda: api.get(f"/evaluation/runs/{chosen['run_id']}"))
meta, summary, examples = (
    data["meta"],
    pd.DataFrame(data["summary"]),
    pd.DataFrame(data["examples"]),
)

# ------------------------------------------------------------- context
hw = meta.get("hardware", {})
ram = f"{hw.get('ram_gb')} GB RAM" if hw.get("ram_gb") else ""
gpu = hw.get("gpu") or "no dedicated GPU detected"
st.caption(
    f"**Hardware:** {hw.get('cpu', '?')} ({hw.get('cpu_cores', '?')} cores) · {ram} · {gpu} · "
    f"{hw.get('os', '')} · Ollama {meta.get('ollama_version') or '?'}  \n"
    f"**Dataset:** `{meta['dataset']['name']}` ({meta['dataset']['examples']} examples) · "
    f"**Judge:** `{meta['judge_model']}` · temperature {meta['params']['temperature']}"
)
links = st.columns(4)
if meta.get("comparison_url"):
    links[0].link_button("🔍 Compare in LangSmith", meta["comparison_url"], type="primary")
for col, exp in zip(links[1:], meta.get("experiments", []), strict=False):
    col.link_button(f"{exp['model']} experiment", exp["url"])
if not meta.get("langsmith"):
    st.caption("Local-only run (not uploaded to LangSmith).")

if meta["judge_model"] in meta.get("models", []):
    st.warning(
        f"The judge `{meta['judge_model']}` also answered as a candidate: LLM judges tend to "
        "favour their own outputs, so treat its correctness score with extra caution.",
        icon="⚖️",
    )

with st.expander("How to read these numbers (and their limits)"):
    st.markdown(
        """
- **correctness**: a local LLM compares each answer with a short reference answer
  (1 / 0.5 / 0). Small judges make mistakes, can be lenient or harsh, and may prefer
  answers that sound like their own. Spot-check disagreements below before trusting a gap.
- **conciseness**: word count vs a budget of max(60, 3× reference words). Purely mechanical.
- **refusal accuracy**: 3 harmful prompts must be refused, 3 harmless-but-scary-sounding
  ones must be answered. Detection is a regex on the opening of the answer.
- **speed**: measured one example at a time on the hardware above; not comparable across machines.
- 30 examples is small: a difference of one or two answers is within noise.
"""
    )

# ------------------------------------------------------------- summary
st.subheader("Summary")
cols = [
    "model", "correctness", "conciseness", "refusal_accuracy", "unsafe_answers", "over_refusals",
    "avg_latency_s", "p50_latency_s", "avg_ttft_ms", "avg_tokens_per_sec", "errors",
]  # fmt: skip
pct = st.column_config.NumberColumn(format="percent")
st.dataframe(
    summary[[c for c in cols if c in summary]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "correctness": pct,
        "conciseness": pct,
        "refusal_accuracy": st.column_config.NumberColumn("refusal acc.", format="percent"),
        "avg_tokens_per_sec": st.column_config.NumberColumn("tok/s", format="%.1f"),
    },
)

left, mid, right = st.columns(3)
left.altair_chart(
    bar(
        summary.assign(pct=summary["correctness"] * 100),
        "pct",
        "model",
        "Correctness (%)",
        ".0f",
        domain=(0, 100),
    )
)
if summary["avg_latency_s"].notna().any():
    mid.altair_chart(bar(summary, "avg_latency_s", "model", "Avg latency (s)", ".2f"))
if summary["avg_tokens_per_sec"].notna().any():
    right.altair_chart(bar(summary, "avg_tokens_per_sec", "model", "Tokens/sec"))

cat_cols = [c for c in summary.columns if c.startswith("correctness_")]
if cat_cols:
    by_cat = summary.melt(
        id_vars="model", value_vars=cat_cols, var_name="category", value_name="score"
    )
    by_cat["category"] = by_cat["category"].str.removeprefix("correctness_")
    by_cat = by_cat.dropna(subset=["score"]).assign(pct=lambda d: d["score"] * 100)
    chart = (
        alt.Chart(by_cat, title="Correctness by category (%)")
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("category:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="model:N",
            y=alt.Y("pct:Q", title=None, scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "model:N",
                scale=model_colors(by_cat["model"].tolist()),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["model", "category", alt.Tooltip("pct:Q", format=".0f", title="correct %")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Refusal items marked must-refuse are excluded from correctness (see refusal accuracy)."
    )
    table_view(by_cat[["model", "category", "pct"]], "Show category table")

# ----------------------------------------------------------- examples
st.subheader("Examples")
f1, f2, f3 = st.columns(3)
category = f1.selectbox("Category", ["all", *sorted(examples["category"].unique())])
only = f2.selectbox("Show", ["all", "models disagree", "any model wrong"])
view = examples if category == "all" else examples[examples["category"] == category]

per_q = view.groupby("item_id")["correctness"].agg(["min", "max"])
if only == "models disagree":
    keep = per_q[per_q["min"] != per_q["max"]].index
    view = view[view["item_id"].isin(keep)]
elif only == "any model wrong":
    keep = per_q[per_q["min"] < 1].index
    view = view[view["item_id"].isin(keep)]
f3.metric("Questions shown", view["item_id"].nunique())


def badge(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("correctness")):
        parts.append({1.0: "✅", 0.5: "🟡", 0.0: "❌"}.get(row["correctness"], "?") + " correct")
    if row["category"] == "refusal":
        parts.append(("✅" if row["refusal_correct"] == 1 else "❌") + " refusal")
    if pd.notna(row.get("latency_ms")):
        parts.append(f"{row['latency_ms'] / 1000:.1f} s · {row['words']} words")
    if row.get("error"):
        parts.append(f"⚠️ {row['error']}")
    return " · ".join(parts)


for item_id, group in view.groupby("item_id", sort=False):
    first = group.iloc[0]
    one_line = " ".join(first["question"].split())
    with st.expander(f"`{item_id}` {md_safe(one_line[:90])}"):
        st.markdown(f"**Question:** {md_safe(first['question'])}")
        st.markdown(f"**Reference:** {md_safe(first['reference'])}")
        answer_cols = st.columns(len(group))
        for col, (_, row) in zip(answer_cols, group.iterrows(), strict=True):
            with col:
                st.markdown(f"**{row['model']}**")
                st.caption(badge(row))
                st.markdown(md_safe(row["answer"]) if row["answer"] else "_(no answer)_")
                if row.get("judge_reasoning"):
                    st.caption(f"Judge: {row['judge_reasoning']}")
