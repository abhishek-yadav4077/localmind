"""HTTP API tests: real FastAPI app + routes, fake models and mocked Ollama."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from localmind.api.main import create_app
from localmind.bootstrap import build_context
from localmind.config import Settings
from localmind.llm import OllamaClient
from localmind.rag import DocumentStore, RagService
from tests.fakes import FakeFactory, FakeToolChat, tool_call
from tests.test_arena import ArenaFactory
from tests.test_rag import PYTHON_TXT, KeywordEmbeddings


def fake_ollama(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.12.0"})
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "llama3.2:latest", "size": 2e9}]})
    if request.url.path == "/api/pull":
        lines = [{"status": "pulling", "completed": 1, "total": 2}, {"status": "success"}]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
    return httpx.Response(404)


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = Settings(_env_file=None, database_url="sqlite://", ollama_max_retries=1)
    ctx = build_context(settings)
    ctx.chat.llm_factory = FakeFactory()
    ctx.arena.llm_factory = ArenaFactory()
    ctx.agent.llm_factory = lambda m, p: FakeToolChat(
        script=[tool_call("calculator", {"expression": "2+2"}), AIMessage(content="4")]
    )
    ctx.ollama = OllamaClient(settings, transport=httpx.MockTransport(fake_ollama))
    store = DocumentStore(
        settings, embeddings=KeywordEmbeddings(), persist=False, collection_name=uuid.uuid4().hex
    )
    ctx._rag = RagService(
        ctx.db, store, settings, llm_factory=FakeFactory(responder=lambda m: "Guido [1].")
    )
    with TestClient(create_app(ctx)) as c:
        yield c


def sse(response: httpx.Response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events, name, data = [], None, []
    for line in response.text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].strip())
        elif not line.strip() and name:
            events.append((name, json.loads("\n".join(data))))
            name, data = None, []
    return events


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ollama"]["installed_models"] == ["llama3.2:latest"]
    assert body["langsmith"]["enabled"] is False
    assert body["features"]["rag"] is True


def test_models_and_pull(client: TestClient) -> None:
    body = client.get("/models").json()
    assert body["models"][0]["name"] == "llama3.2:latest"
    assert {"name": "llama3.2", "installed": True} in body["preferred"]
    events = sse(client.post("/models/pull", json={"model": "llama3.2"}))
    assert [e for e, _ in events] == ["progress", "progress", "done"]
    assert events[0][1]["fraction"] == 0.5


def test_chat_flow_creates_streams_and_lists(client: TestClient) -> None:
    events = sse(client.post("/chat", json={"message": "hello", "model": "mistral"}))
    names = [e for e, _ in events]
    assert names[0] == "conversation" and "token" in names and names[-2:] == ["done", "title"]
    conv_id = events[0][1]["conversation"]["id"]
    text = "".join(d["content"] for e, d in events if e == "token")
    assert text == "Answer to: hello"

    # follow-up turn in the same conversation
    sse(client.post("/chat", json={"message": "again", "conversation_id": conv_id}))
    msgs = client.get(f"/conversations/{conv_id}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 2
    assert msgs[1]["tokens_per_sec"] is not None and msgs[1]["trace_url"] is None

    listed = client.get("/conversations", params={"q": "again"}).json()
    assert [c["id"] for c in listed] == [conv_id]


def test_conversation_crud_and_export(client: TestClient) -> None:
    conv = client.post("/conversations", json={"model": "llama3.2"}).json()
    cid = conv["id"]
    patched = client.patch(
        f"/conversations/{cid}", json={"title": "Renamed", "params": {"temperature": 0.2}}
    ).json()
    assert patched["title"] == "Renamed" and patched["params"]["temperature"] == 0.2
    sse(client.post("/chat", json={"message": "hi", "conversation_id": cid}))

    md = client.get(f"/conversations/{cid}/export", params={"format": "markdown"})
    assert md.text.startswith("# Renamed") and "attachment" in md.headers["content-disposition"]
    assert client.get(f"/conversations/{cid}/export", params={"format": "json"}).json()

    assert client.delete(f"/conversations/{cid}").status_code == 204
    missing = client.get(f"/conversations/{cid}")
    assert missing.status_code == 404 and "not found" in missing.json()["error"]


def test_persona_cleared_with_explicit_null(client: TestClient) -> None:
    persona_id = client.get("/personas").json()[0]["id"]
    cid = client.post("/conversations", json={"persona_id": persona_id}).json()["id"]
    assert (
        client.patch(f"/conversations/{cid}", json={"title": "x"}).json()["persona_id"]
        == persona_id
    )
    assert (
        client.patch(f"/conversations/{cid}", json={"persona_id": None}).json()["persona_id"]
        is None
    )


def test_feedback(client: TestClient) -> None:
    events = sse(client.post("/chat", json={"message": "hello"}))
    message_id = next(d for e, d in events if e == "done")["message"]["id"]
    body = client.post(
        "/feedback", json={"message_id": message_id, "thumbs_up": False, "comment": "meh"}
    ).json()
    assert body["score"] == 0 and body["synced_to_langsmith"] is False
    cid = events[0][1]["conversation"]["id"]
    msgs = client.get(f"/conversations/{cid}/messages").json()
    assert msgs[1]["feedback_score"] == 0 and msgs[1]["feedback_comment"] == "meh"
    assert client.post("/feedback", json={"message_id": 999, "thumbs_up": True}).status_code == 404


def test_arena_vote_and_leaderboard(client: TestClient) -> None:
    events = sse(client.post("/arena", json={"prompt": "hi", "models": ["fast", "slow"]}))
    assert events[0][0] == "start" and events[-1][0] == "done"
    battle_id = events[0][1]["battle_id"]
    voted = client.post(f"/arena/{battle_id}/vote", json={"outcome": "win", "winner_model": "fast"})
    assert voted.json()["winner_model"] == "fast"
    board = client.get("/arena/leaderboard").json()
    assert board[0]["model"] == "fast" and board[0]["win_rate"] == 1.0
    assert len(client.get("/arena/battles").json()) == 1


def test_arena_validation_is_http_400_not_broken_stream(client: TestClient) -> None:
    r = client.post("/arena", json={"prompt": "hi", "models": ["same", "same"]})
    assert r.status_code == 400 and "different models" in r.json()["error"]
    assert client.post("/arena", json={"prompt": "hi", "models": ["one"]}).status_code == 422


def test_agent_stream(client: TestClient) -> None:
    events = sse(client.post("/agent", json={"question": "2+2?"}))
    assert [e for e, _ in events] == ["start", "tool_call", "tool_result", "done"]
    assert events[2][1]["content"] == "2+2 = 4"
    assert events[-1][1]["message"]["details"]["steps"]


def test_documents_upload_list_ask_delete(client: TestClient) -> None:
    up = client.post("/documents", files={"file": ("facts.txt", PYTHON_TXT, "text/plain")})
    assert up.status_code == 201
    doc_id = up.json()["doc_id"]
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc_id]

    events = sse(client.post("/documents/ask", json={"question": "who created python", "k": 1}))
    assert events[0][0] == "sources" and "Guido" in events[0][1]["sources"][0]["snippet"]
    assert events[-1][1]["message"]["details"]["cited"] == [1]

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []
    bad = client.post("/documents", files={"file": ("x.docx", b"abc", "application/octet-stream")})
    assert bad.status_code == 400 and bad.json()["hint"]


def test_personas_crud_and_conflict(client: TestClient) -> None:
    created = client.post("/personas", json={"name": "Pirate", "system_prompt": "Arr"})
    assert created.status_code == 201
    pid = created.json()["id"]
    assert (
        client.patch(f"/personas/{pid}", json={"description": "sea"}).json()["description"] == "sea"
    )
    dup = client.post("/personas", json={"name": "Pirate", "system_prompt": "x"})
    assert dup.status_code == 409
    assert client.delete(f"/personas/{pid}").status_code == 204
    assert client.delete(f"/personas/{pid}").status_code == 404


def test_dashboard(client: TestClient) -> None:
    sse(client.post("/chat", json={"message": "hello"}))
    body = client.get("/dashboard").json()
    assert body["totals"]["assistant_messages"] == 1
    assert {"per_model", "feedback", "per_mode", "daily", "leaderboard"} <= body.keys()


def test_ollama_down_returns_503(client: TestClient) -> None:
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    ctx = client.app.state.ctx
    ctx.ollama = OllamaClient(ctx.settings, transport=httpx.MockTransport(refuse))
    r = client.get("/models")
    assert r.status_code == 503 and "ollama serve" in r.json()["hint"]
    assert client.get("/health").json()["status"] == "degraded"


def test_evaluation_endpoints(client: TestClient) -> None:
    from localmind.evaluation.runner import write_results

    settings = client.app.state.ctx.settings
    assert client.get("/evaluation/runs").json() == []
    write_results(
        Path(settings.eval_results_dir),
        "20260101-000000",
        [],
        [{"model": "m", "correctness": 0.9}],
        {"run_id": "20260101-000000", "models": ["m"]},
    )
    runs = client.get("/evaluation/runs").json()
    assert [r["run_id"] for r in runs] == ["20260101-000000"]
    detail = client.get("/evaluation/runs/20260101-000000").json()
    assert detail["summary"] == [{"model": "m", "correctness": 0.9}]
    assert client.get("/evaluation/runs/missing").status_code == 404
