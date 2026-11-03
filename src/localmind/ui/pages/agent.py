"""Agent: a LangGraph tool-calling agent with visible intermediate steps."""

from __future__ import annotations

import json

import streamlit as st

from localmind.ui.api_client import APIError
from localmind.ui.components import (
    call,
    get_client,
    health,
    message_footer,
    model_picker,
    show_error,
)

api = get_client()
state = st.session_state
state.setdefault("agent_conv", None)

st.title("🛠️ Agent")
st.caption(
    "A LangGraph ReAct agent that can use tools: calculator, current date/time, "
    "search over your past conversations, and (optionally) web search."
)

h = health() or {}
with st.sidebar:
    if st.button("➕ New agent chat", use_container_width=True, type="primary"):
        state.agent_conv = None
        st.rerun()
    model = model_picker(
        "Agent model",
        h.get("defaults", {}).get("agent_model"),
        "agent_model",
        help="Needs a tool-capable model, e.g. qwen2.5:3b, llama3.2, mistral",
    )
    web = st.toggle(
        "🌐 Web search (online)",
        value=False,
        disabled=not h.get("features", {}).get("web_search", False),
        help="The only feature that leaves your machine: queries go to DuckDuckGo.",
    )
    if web:
        st.warning("Web search is ON: search queries are sent to DuckDuckGo.", icon="🌐")


def render_steps(steps: list[dict]) -> None:
    calls = [s for s in steps if s["kind"] == "call"]
    if not calls:
        return
    with st.expander(f"🛠 {len(calls)} tool call{'s' * (len(calls) != 1)}"):
        for s in steps:
            if s["kind"] == "call":
                st.markdown(f"**→ `{s['name']}`**")
                st.code(json.dumps(s["args"], indent=2), language="json")
            else:
                icon = "❌" if s.get("is_error") else "←"
                st.markdown(f"{icon} result")
                st.code(s["content"][:2000], language="text")


if state.agent_conv:
    for m in call(lambda: api.get(f"/conversations/{state.agent_conv}/messages")):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                render_steps((m.get("details") or {}).get("steps", []))
            st.markdown(m["content"])
            if m["role"] == "assistant":
                message_footer(m)
else:
    st.info("Try: *What is 17.5% of 2340, and what weekday is it today?*", icon="💡")

question = st.chat_input("Ask the agent…")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    body = {"question": question, "model": model, "web_search": web}
    if state.agent_conv:
        body["conversation_id"] = state.agent_conv
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        failed = False
        try:
            for event, data in api.stream("/agent", body):
                if event == "start":
                    state.agent_conv = data["conversation_id"]
                    status.write(f"Model `{data['model']}` · tools: {', '.join(data['tools'])}")
                elif event == "tool_call":
                    status.write(f"→ calling **{data['name']}** `{json.dumps(data['args'])}`")
                elif event == "tool_result":
                    preview = data["content"].replace("\n", " ")[:200]
                    status.write(f"{'❌' if data['is_error'] else '←'} {preview}")
                elif event == "error":
                    failed = True
                    status.update(label="Agent failed", state="error")
                    show_error(data["error"], data.get("hint"))
                elif event == "done":
                    status.update(label="Done", state="complete", expanded=False)
        except APIError as exc:
            failed = True
            status.update(label="Agent failed", state="error")
            show_error(exc.error, exc.hint)
    if not failed:
        st.rerun()
