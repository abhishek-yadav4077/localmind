"""Settings: Ollama health, installed models, model pulls, LangSmith status."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from localmind.ui.api_client import APIError
from localmind.ui.components import call, get_client, health, installed_models, show_error

api = get_client()
st.title("⚙️ Settings")

h = health()
if h is None:
    call(lambda: api.get("/health"))  # shows the "API not reachable" error + hint, then stops

# ------------------------------------------------------------- Ollama
st.subheader("Ollama")
ollama = h["ollama"]
c1, c2, c3 = st.columns(3)
c1.metric("Server", "running" if ollama["reachable"] else "offline")
c2.metric("Version", ollama.get("version") or "—")
c3.metric("Models", len(ollama["installed_models"]))
st.caption(f"URL: `{ollama['base_url']}`")
for hint in ollama["hints"]:
    st.warning(hint)

if ollama["reachable"]:
    models = call(lambda: api.get("/models"), stop=False)
    if models and models["models"]:
        df = pd.DataFrame(models["models"])[
            ["name", "parameter_size", "quantization", "family", "size_gb"]
        ]
        st.dataframe(
            df.rename(columns={"parameter_size": "params", "size_gb": "size (GB)"}),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("**Pull a model**")
    missing = [p["name"] for p in (models or {}).get("preferred", []) if not p["installed"]]
    if missing:
        st.caption("Suggested and not installed: " + ", ".join(f"`{m}`" for m in missing))
    with st.form("pull"):
        name = st.text_input(
            "Model name",
            value=missing[0] if missing else "",
            placeholder="e.g. qwen2.5:3b — see ollama.com/library",
        )
        pull = st.form_submit_button("⬇️ Pull", type="primary")
    if pull and name.strip():
        bar = st.progress(0.0, text="Starting…")
        try:
            for event, data in api.stream("/models/pull", {"model": name.strip()}):
                if event == "progress":
                    frac = data.get("fraction")
                    if frac is not None:
                        done_gb = (data["completed"] or 0) / 1e9
                        total_gb = (data["total"] or 0) / 1e9
                        bar.progress(
                            frac, text=f"{data['status']} · {done_gb:.2f}/{total_gb:.2f} GB"
                        )
                    else:
                        bar.progress(0.0, text=data["status"])
                elif event == "error":
                    show_error(data["error"], data.get("hint"))
                    break
                elif event == "done":
                    bar.progress(1.0, text="Done")
                    st.success(f"Pulled {data['model']}")
                    installed_models.clear()
                    health.clear()
        except APIError as exc:
            show_error(exc.error, exc.hint)

# ---------------------------------------------------------- LangSmith
st.subheader("LangSmith")
ls = h["langsmith"]
if ls["enabled"]:
    st.success(f"Tracing to project **{ls['project']}**")
    if ls["project_url"]:
        st.link_button("Open project in LangSmith", ls["project_url"])
else:
    st.info(f"Tracing disabled: {ls['reason']}. The app works fully without it.")
    st.code(
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=lsv2_...\nLANGSMITH_PROJECT=localmind", "bash"
    )
    st.caption("Add these to `.env` and restart the API.")

# ------------------------------------------------------------ features
st.subheader("Features & defaults")
f, d = h["features"], h["defaults"]
rag_status = "✅ installed" if f["rag"] else '❌ install with `pip install -e ".[rag]"`'
web_status = "✅ available (off by default)" if f["web_search"] else "❌ not installed"
st.markdown(
    f"- Document chat: {rag_status}\n"
    f"- Web search for the agent: {web_status}\n"
    f"- Default chat model: `{d['model']}` · agent: `{d['agent_model']}` · "
    f"embeddings: `{d['embedding_model']}`\n"
    f"- App version: `{h['version']}` · API: `{api.base_url}`"
)
