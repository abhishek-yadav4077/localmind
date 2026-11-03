"""Streamlit entry point: page navigation + shared sidebar status.

Run:  streamlit run src/localmind/ui/app.py
(the API must be running: uvicorn localmind.api.main:app)
"""

from pathlib import Path

import streamlit as st

from localmind.ui.components import sidebar_status

st.set_page_config(page_title="LocalMind", page_icon="🧠", layout="wide")

# Not named "pages/": Streamlit would auto-discover that folder and briefly show
# its own file-based menu before st.navigation takes over.
PAGES = Path(__file__).parent / "views"

nav = st.navigation(
    {
        "Talk": [
            st.Page(PAGES / "chat.py", title="Chat", icon="💬", default=True),
            st.Page(PAGES / "arena.py", title="Model Arena", icon="⚔️"),
            st.Page(PAGES / "agent.py", title="Agent", icon="🛠️"),
            st.Page(PAGES / "documents.py", title="Documents", icon="📄"),
        ],
        "Configure": [
            st.Page(PAGES / "personas.py", title="Personas", icon="🎭"),
            st.Page(PAGES / "settings.py", title="Settings", icon="⚙️"),
        ],
        "Observe": [
            st.Page(PAGES / "dashboard.py", title="Dashboard", icon="📊"),
            st.Page(PAGES / "evaluation.py", title="Evaluation", icon="🧪"),
        ],
    }
)
nav.run()
sidebar_status()  # after the page so the badges sit below the page's own sidebar controls
