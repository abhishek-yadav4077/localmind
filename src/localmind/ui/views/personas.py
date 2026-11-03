"""Personas: edit the system-prompt presets used by Chat and Arena."""

from __future__ import annotations

import streamlit as st

from localmind.ui.components import call, get_client, personas

api = get_client()

st.title("🎭 Personas")
st.caption("A persona is a reusable system prompt. Pick one per conversation in Chat.")

with st.expander("➕ New persona"), st.form("new_persona", clear_on_submit=True):
    name = st.text_input("Name")
    description = st.text_input("Short description")
    prompt = st.text_area("System prompt", height=150)
    if st.form_submit_button("Create", type="primary"):
        if name.strip() and prompt.strip():
            call(
                lambda: api.post(
                    "/personas",
                    {"name": name, "description": description, "system_prompt": prompt},
                )
            )
            personas.clear()
            st.rerun()
        st.warning("Name and system prompt are required.")

for p in personas():
    badge = " · built-in" if p["is_builtin"] else ""
    with (
        st.expander(f"**{p['name']}**{badge} — {p['description']}"),
        st.form(f"persona_{p['id']}"),
    ):
        name = st.text_input("Name", p["name"])
        description = st.text_input("Short description", p["description"])
        prompt = st.text_area("System prompt", p["system_prompt"], height=160)
        save, delete = st.columns(2)
        if save.form_submit_button("Save", type="primary", use_container_width=True):
            call(
                lambda p=p, n=name, d=description, sp=prompt: api.patch(
                    f"/personas/{p['id']}", {"name": n, "description": d, "system_prompt": sp}
                )
            )
            personas.clear()
            st.toast("Saved", icon="✅")
            st.rerun()
        if delete.form_submit_button("Delete", use_container_width=True):
            call(lambda p=p: api.delete(f"/personas/{p['id']}"))
            personas.clear()
            st.rerun()
