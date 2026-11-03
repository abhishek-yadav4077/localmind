"""The UI's HTTP client: JSON calls, error mapping and SSE streaming."""

from __future__ import annotations

import httpx
import pytest

from localmind.ui.api_client import APIClient, APIError
from localmind.ui.components import stats_line


def client(handler) -> APIClient:
    return APIClient("http://api", transport=httpx.MockTransport(handler))


def test_json_get_and_204() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(204)
        assert request.url.params.get("q") == "x" and "limit" not in request.url.params
        return httpx.Response(200, json=[{"id": 1}])

    c = client(handler)
    assert c.get("/conversations", q="x", limit=None) == [{"id": 1}]
    assert c.delete("/conversations/1") is None


def test_error_body_becomes_api_error_with_hint() -> None:
    c = client(lambda r: httpx.Response(503, json={"error": "Ollama down", "hint": "ollama serve"}))
    with pytest.raises(APIError) as exc:
        c.get("/models")
    assert (exc.value.error, exc.value.hint, exc.value.status) == (
        "Ollama down",
        "ollama serve",
        503,
    )


def test_fastapi_validation_errors_are_readable() -> None:
    detail = {"detail": [{"loc": ["body", "models"], "msg": "too short"}]}
    with pytest.raises(APIError) as exc:
        client(lambda r: httpx.Response(422, json=detail)).post("/arena", {})
    assert exc.value.error == "body.models: too short"


def test_stream_parses_events_and_raises_on_http_error() -> None:
    body = 'event: token\ndata: {"content": "Hi"}\n\nevent: done\ndata: {"ok": true}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/bad":
            return httpx.Response(400, json={"error": "Prompt is empty"})
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    c = client(handler)
    assert list(c.stream("/chat", {})) == [("token", {"content": "Hi"}), ("done", {"ok": True})]
    with pytest.raises(APIError, match="Prompt is empty"):
        list(c.stream("/bad", {}))


def test_stats_line() -> None:
    m = {
        "model": "llama3.2",
        "latency_ms": 1450.0,
        "ttft_ms": 210.4,
        "tokens_per_sec": 31.25,
        "prompt_tokens": 120,
        "completion_tokens": 85,
    }
    assert stats_line(m) == "llama3.2 · 1.4 s · TTFT 210 ms · 31.2 tok/s · 120 → 85 tokens"
    assert stats_line({"model": "m", "latency_ms": 640.0}) == "m · 640 ms"
