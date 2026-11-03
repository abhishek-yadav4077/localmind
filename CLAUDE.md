# CLAUDE.md — LocalMind

Guidance for anyone (human or AI) working in this repository.

## 1. Overview

**LocalMind** is a private LLM chat studio. All inference runs locally through
[Ollama](https://ollama.com); no prompt or response leaves the machine. The one
exception is LangSmith tracing (opt-in, sends traces to LangSmith) and the
agent's web-search tool (off by default, clearly labelled).

It started as "Assignment 24: Ollama Chatbot" (Ollama setup, LangChain Ollama
wrapper, LangSmith tracing). `scripts/basic_chat.py` is that assignment in its
smallest form; everything else grows the same idea into a portfolio project.

Features: streaming chat with persistent history, personas, automatic memory
summarization, per-message performance stats, a side-by-side model arena with
votes and a leaderboard, a LangGraph tool-calling agent, optional document chat
(RAG), LangSmith tracing + feedback + evaluation, and a local analytics dashboard.

## 2. Architecture

```
Streamlit UI (ui/)  ──HTTP/SSE──▶  FastAPI (api/)  ──▶  services
                                                     ├─ chat/       ChatService: prompt → stream → persist → stats
                                                     ├─ memory/     token budgeting + rolling summary
                                                     ├─ personas/   system-prompt presets (DB)
                                                     ├─ arena/      parallel multi-model runs + votes
                                                     ├─ agent/      LangGraph ReAct agent + tools
                                                     ├─ rag/        PDF/TXT → nomic-embed-text → Chroma
                                                     └─ evaluation/ LangSmith datasets, evaluators, experiments
            shared:  llm/ (ChatOllama factory, Ollama client, health, retries)
                     tracing/ (LangSmith config, metadata/tags, feedback, trace URLs)
                     db/ (SQLAlchemy models + repositories, SQLite)
                     config/ (pydantic-settings)
```

Rules:
- **UI talks only to the API.** Streamlit never imports services directly, so the
  backend is reusable and testable on its own.
- **Services don't know about HTTP.** API routes are thin; logic lives in services.
- **One place builds LLMs**: `llm/factory.py`. Everything else asks it for a
  `ChatOllama`, which keeps model params, timeouts and tracing tags consistent.
- **One place talks to LangSmith**: `tracing/`. If LangSmith isn't configured, the
  functions there become no-ops and log one warning.

## 3. Repository layout

```
src/localmind/
  config/      settings.py (pydantic-settings), logging.py
  llm/         ollama_client.py (REST: health, list, pull), factory.py, errors
  chat/        service.py, stats.py, titles.py
  memory/      summarizer.py, token budget
  personas/    defaults + service
  arena/       service.py, leaderboard.py
  agent/       graph.py, tools.py
  rag/         ingest.py, retriever.py, chain.py
  tracing/     langsmith.py (enable/disable, run metadata, feedback, URLs)
  evaluation/  dataset.py, evaluators.py, run_experiments.py
  db/          models.py, session.py, repositories.py
  api/         main.py, routes/*.py, schemas.py
  ui/          Home.py + pages/*.py, api_client.py
  exceptions.py
scripts/       basic_chat.py (assignment), pull/seed helpers
tests/         pytest, LLM calls mocked (no Ollama needed in CI)
eval/          dataset.json, results/*.csv
docs/          CONCEPTS.md, screenshots/ (+ CHECKLIST.md)
```

## 4. Conventions

- Python 3.11+, `src/` layout, packaged with `pyproject.toml` (installable via `pip install -e .`).
- **Current LangChain APIs only**: `from langchain_ollama import ChatOllama, OllamaEmbeddings`,
  LCEL (`prompt | llm | parser`), `langchain_core` messages/prompts, LangGraph
  `create_react_agent` / `StateGraph`. No `langchain.llms.Ollama`, no `LLMChain`,
  no `ConversationBufferMemory`. Versions are checked against what is installed.
- LangSmith via env vars `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`
  (+ optional `LANGSMITH_ENDPOINT`). Secrets only in `.env` (gitignored); `.env.example` is committed.
- Type hints everywhere, Google-style docstrings on public functions.
- Structured logging via the stdlib `logging` with a JSON-ish key=value formatter; no `print` outside scripts.
- Custom exceptions in `exceptions.py` (`OllamaUnavailableError`, `ModelNotFoundError`, …),
  mapped to HTTP errors in the API.
- Ollama calls: explicit timeouts + retries with backoff (`tenacity`) on connection errors only.
- Comments explain *why*, not *what*. Non-obvious decisions get a short comment so they can be
  explained in an interview.
- Tooling: `ruff` (lint + format), `pytest`. Tests never require a running Ollama or LangSmith.
- Git: small conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`).
- **No invented numbers.** Benchmarks/eval tables in the README are filled only from real runs
  on the author's machine; until then they show `TBD`.

## 5. Phase plan

Each phase ends with: code runs, tests pass, a summary, a manual test list
(+ which LangSmith screenshot to take), then **stop until "continue"**.

| # | Phase | Deliverables |
|---|-------|--------------|
| 1 | Scaffold & basics | pyproject, layout, settings + `.env.example`, logging, exceptions, Ollama client (health, list models, fix instructions), `scripts/basic_chat.py` traced in LangSmith, `scripts/check_setup.py` |
| 2 | Chat core | SQLite models/repos, ChatService with streaming, model params, personas (seeded), memory summarization, per-message stats (latency, TTFT, tok/s, token counts from Ollama metadata), auto titles, export MD/JSON |
| 3 | Observability | run metadata/tags (model, persona, conversation_id, mode, app_version), capture run_id per message, feedback API to LangSmith, trace URLs, graceful no-LangSmith mode |
| 4 | Arena | parallel runs over 2–3 models, stats, votes table, leaderboard (win rate, avg tok/s) |
| 5 | Agent + RAG | LangGraph agent (calculator, datetime, past-conversation search, opt-in DuckDuckGo), step capture; small isolated RAG module with citations |
| 6 | API + UI | FastAPI routes (/health, /models, /models/pull, /conversations, /chat SSE, /arena, /agent, /feedback, /personas), Streamlit pages: Chat, Arena, Agent, Documents, Personas, Dashboard, Evaluation, Settings |
| 7 | Evaluation | `eval/dataset.json` (30 items, 5 categories), upload to LangSmith, experiments per model, evaluators (LLM-judge correctness, conciseness, latency, refusal), CSV export, Evaluation page |
| 8 | Polish | test coverage gaps, Dockerfile + compose (ollama service + volume, host-Ollama option), GitHub Actions (ruff + pytest), README, CONCEPTS.md, screenshot checklist |

## 6. Running (filled in as phases land)

```bash
pip install -e ".[dev]"
cp .env.example .env          # add LANGSMITH_API_KEY
ollama pull llama3.2
python scripts/basic_chat.py
pytest
```
