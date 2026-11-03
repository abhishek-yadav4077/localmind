"""Chat: conversations, streaming answers, per-message stats, feedback, export."""

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
    personas,
    show_error,
)

api = get_client()
state = st.session_state
state.setdefault("conv_id", None)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        state.conv_id = None
        st.rerun()
    query = st.text_input(
        "Search", placeholder="Search titles and messages", label_visibility="collapsed"
    )
    convs = call(lambda: api.get("/conversations", q=query or None, limit=50), stop=False) or []
    for c in convs:
        label = ("▸ " if c["id"] == state.conv_id else "") + c["title"][:40]
        if st.button(label, key=f"conv_{c['id']}", use_container_width=True):
            state.conv_id = c["id"]
            st.rerun()
    if not convs:
        st.caption("No conversations yet." if not query else "No matches.")

conv = call(lambda: api.get(f"/conversations/{state.conv_id}")) if state.conv_id else None
persona_list = personas()
persona_names = {p["id"]: p["name"] for p in persona_list}

# ------------------------------------------------------- settings panel
with st.sidebar.expander("⚙️ Model & parameters", expanded=conv is None):
    defaults = (health() or {}).get("defaults", {})
    params = (conv or {}).get("params") or {}
    with st.form("conv_settings", border=False):
        model = model_picker(
            "Model", (conv or {}).get("model") or defaults.get("model"), "chat_model"
        )
        persona_ids = [None, *persona_names]
        persona_id = st.selectbox(
            "Persona",
            persona_ids,
            index=persona_ids.index((conv or {}).get("persona_id")),
            format_func=lambda i: "None" if i is None else persona_names[i],
        )
        temperature = st.slider("Temperature", 0.0, 2.0, params.get("temperature", 0.7), 0.05)
        top_p = st.slider("Top-p", 0.05, 1.0, params.get("top_p", 0.9), 0.05)
        max_tokens = st.number_input(
            "Max tokens (0 = no limit)", 0, 32768, params.get("max_tokens") or 0, 64
        )
        ctx_options = [2048, 4096, 8192, 16384, 32768]
        num_ctx = st.select_slider("Context window", ctx_options, params.get("num_ctx", 4096))
        new_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": int(max_tokens) or None,
            "num_ctx": num_ctx,
        }
        if st.form_submit_button("Apply", disabled=conv is None, use_container_width=True):
            call(
                lambda: api.patch(
                    f"/conversations/{conv['id']}",
                    {"model": model, "persona_id": persona_id, "params": new_params},
                )
            )
            st.toast("Settings saved — they apply from the next message", icon="✅")
    if conv is None:
        st.caption("Used for the new conversation you start below.")

# --------------------------------------------------------------- header
if conv is None:
    st.title("💬 Chat")
    st.caption("Everything runs on your machine through Ollama. Pick a model and start typing.")
else:
    head, actions = st.columns([3, 2])
    head.subheader(conv["title"])
    persona = persona_names.get(conv.get("persona_id"), "no persona")
    head.caption(f"{conv['model']} · {persona}")
    with actions:
        c1, c2, c3 = st.columns(3)
        with c1.popover("✏️", help="Rename"):
            title = st.text_input("Title", conv["title"])
            if st.button("Save title"):
                call(lambda: api.patch(f"/conversations/{conv['id']}", {"title": title}))
                st.rerun()
        with c2.popover("⬇️", help="Export"):
            md = call(lambda: api.get(f"/conversations/{conv['id']}/export", format="markdown"))
            js = call(lambda: api.get(f"/conversations/{conv['id']}/export", format="json"))
            st.download_button("Markdown", md, f"{conv['title']}.md", "text/markdown")
            st.download_button(
                "JSON",
                json.dumps(js, indent=2),
                f"{conv['title']}.json",
                "application/json",
            )
        with c3.popover("🗑️", help="Delete"):
            st.write("Delete this conversation?")
            if st.button("Delete", type="primary"):
                call(lambda: api.delete(f"/conversations/{conv['id']}"))
                state.conv_id = None
                st.rerun()
    if conv.get("summary"):
        with st.expander("🧠 Long-term memory (auto-summary of older turns)"):
            st.write(conv["summary"])

# ------------------------------------------------------------- messages
messages = call(lambda: api.get(f"/conversations/{conv['id']}/messages")) if conv else []
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("mode") in {"agent", "rag"}:
                st.caption(f"from {m['mode']} mode")
            message_footer(m)

# ---------------------------------------------------------------- input
prompt = st.chat_input("Message LocalMind…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    body = {"message": prompt}
    if conv:
        body["conversation_id"] = conv["id"]
    else:
        body |= {"model": model, "persona_id": persona_id, "params": new_params}

    side: dict = {}

    def tokens():
        for event, data in api.stream("/chat", body):
            if event == "token":
                yield data["content"]
            elif event == "conversation":
                state.conv_id = data["conversation"]["id"]
            elif event == "memory_summarized":
                st.toast(f"🧠 Memory summarized ({data['summarized_messages']} older messages)")
            elif event == "error":
                side["error"] = data
            elif event == "title":
                side["title"] = data["title"]

    with st.chat_message("assistant"):
        try:
            st.write_stream(tokens())
        except APIError as exc:
            side["error"] = {"error": exc.error, "hint": exc.hint}
    if "error" in side:
        show_error(side["error"]["error"], side["error"].get("hint"))
    else:
        st.rerun()  # re-render with stats, feedback buttons and the new title
