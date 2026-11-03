"""Documents: upload PDF/TXT, index locally, ask questions with citations."""

from __future__ import annotations

import streamlit as st

from localmind.ui.api_client import APIError
from localmind.ui.components import (
    call,
    get_client,
    health,
    message_footer,
    model_picker,
    plural,
    show_error,
)

api = get_client()
state = st.session_state
state.setdefault("docs_conv", None)

st.title("📄 Chat with documents")
h = health() or {}
if not h.get("features", {}).get("rag", False):
    st.warning("Document chat is an optional feature and is not installed.")
    st.code('pip install -e ".[rag]"\nollama pull nomic-embed-text', language="bash")
    st.stop()
emb = h.get("defaults", {}).get("embedding_model", "nomic-embed-text")
st.caption(f"Files are chunked and embedded locally with `{emb}` into a Chroma index on disk.")

# ------------------------------------------------------------ library
with st.expander("📚 Library", expanded=True):
    files = st.file_uploader("Upload", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if files and st.button("Index files", type="primary"):
        for f in files:
            with st.spinner(f"Embedding {f.name}…"):
                try:
                    info = api.upload("/documents", f.name, f.getvalue(), f.type or "text/plain")
                    st.success(
                        f"{info['filename']}: {plural(info['pages'], 'page')} → "
                        f"{plural(info['chunks'], 'chunk')}"
                    )
                except APIError as exc:
                    show_error(f"{f.name}: {exc.error}", exc.hint)
    docs = call(lambda: api.get("/documents"), stop=False) or []
    for d in docs:
        c1, c2 = st.columns([6, 1])
        c1.markdown(
            f"**{d['filename']}** · {plural(d['pages'], 'page')} · {plural(d['chunks'], 'chunk')}"
        )
        if c2.button("🗑️", key=f"del_{d['doc_id']}", help="Remove from index"):
            call(lambda d=d: api.delete(f"/documents/{d['doc_id']}"))
            st.rerun()
    if not docs:
        st.caption("No documents indexed yet.")

if not docs:
    st.stop()

# ------------------------------------------------------------ options
with st.sidebar:
    if st.button("➕ New document chat", use_container_width=True, type="primary"):
        state.docs_conv = None
        st.rerun()
    names = {d["doc_id"]: d["filename"] for d in docs}
    selected = st.multiselect(
        "Search in", list(names), format_func=names.get, placeholder="All documents"
    )
    model = model_picker("Answer model", h.get("defaults", {}).get("model"), "docs_model")
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)


def render_sources(sources: list[dict], cited: list[int] | None = None) -> None:
    if not sources:
        return
    with st.expander(f"📎 {plural(len(sources), 'source')}"):
        for s in sources:
            page = f", page {s['page']}" if s.get("page") else ""
            used = " ✅ cited" if cited and s["n"] in cited else ""
            st.markdown(f"**[{s['n']}] {s['filename']}{page}** · similarity {s['score']:.2f}{used}")
            st.caption(s["snippet"])


if state.docs_conv:
    for m in call(lambda: api.get(f"/conversations/{state.docs_conv}/messages")):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                details = m.get("details") or {}
                render_sources(details.get("sources", []), details.get("cited"))
                if details.get("invalid_citations"):
                    st.warning(
                        f"Cited non-existent sources {details['invalid_citations']}: "
                        "possible hallucination."
                    )
                message_footer(m)

question = st.chat_input("Ask about your documents…")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    body = {"question": question, "model": model, "k": k, "doc_ids": selected or None}
    if state.docs_conv:
        body["conversation_id"] = state.docs_conv
    side: dict = {}

    def tokens():
        for event, data in api.stream("/documents/ask", body):
            if event == "sources":
                state.docs_conv = data["conversation_id"]
                side["sources"] = data["sources"]
            elif event == "token":
                yield data["content"]
            elif event == "error":
                side["error"] = data

    with st.chat_message("assistant"):
        try:
            st.write_stream(tokens())
        except APIError as exc:
            side["error"] = {"error": exc.error, "hint": exc.hint}
    if "error" in side:
        show_error(side["error"]["error"], side["error"].get("hint"))
    else:
        st.rerun()
