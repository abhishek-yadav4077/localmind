"""Streamlit frontend. Run: ``streamlit run src/localmind/ui/app.py``.

The UI only talks to the FastAPI backend over HTTP (see ``api_client.py``); it
never imports services directly, so it could be swapped for any other client.
"""
