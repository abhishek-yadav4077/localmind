# LocalMind — Private LLM Chat Studio with Observability

> Work in progress. Full README lands in Phase 8. See [CLAUDE.md](CLAUDE.md) for the architecture and plan.

Quick start (Phase 1):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add your LANGSMITH_API_KEY
ollama pull llama3.2
python scripts/check_setup.py   # verifies Ollama + LangSmith
python scripts/basic_chat.py    # Assignment 24, Task 1
make dev                        # API :8000 + Streamlit UI :8501
```
