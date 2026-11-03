"""UI smoke tests with Streamlit's AppTest (no browser, no running API)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from localmind.ui.api_client import APIClient, APIError, parse_sse

PAGES = sorted((Path(__file__).parents[1] / "src/localmind/ui/pages").glob("*.py"))


def test_parse_sse() -> None:
    lines = [
        ": ping",
        "event: token",
        'data: {"content": "Hi"}',
        "",
        "event: done",
        "data: {",
        'data: "ok": true}',
        "",
    ]
    assert list(parse_sse(iter(lines))) == [("token", {"content": "Hi"}), ("done", {"ok": True})]


def test_client_reports_unreachable_api() -> None:
    client = APIClient("http://127.0.0.1:9")
    with pytest.raises(APIError) as exc:
        client.get("/health")
    assert "uvicorn" in (exc.value.hint or "")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_renders_without_api(page: Path, monkeypatch) -> None:
    """With the API down every page must show a friendly error, never a traceback."""
    monkeypatch.setenv("LOCALMIND_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception, at.exception


def test_arena_blind_mode_hides_names_and_speed(monkeypatch) -> None:
    """After a battle in blind mode, columns say 'Model A/B' and speed stats are hidden."""
    battle = {
        "id": "b1",
        "prompt": "hi",
        "outcome": None,
        "winner_model": None,
        "responses": [
            {
                "position": i,
                "model": m,
                "content": f"answer {m}",
                "error": None,
                "latency_ms": 900.0,
                "ttft_ms": 100.0,
                "tokens_per_sec": 30.0,
                "prompt_tokens": 5,
                "completion_tokens": 9,
                "run_id": None,
            }
            for i, m in enumerate(["llama3.2:latest", "mistral:latest"])
        ],
    }
    monkeypatch.setenv("LOCALMIND_API_URL", "http://127.0.0.1:9")
    page = next(p for p in PAGES if p.stem == "arena")
    at = AppTest.from_file(str(page), default_timeout=30)
    at.session_state["battle"] = battle
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Model A" in text and "Model B" in text
    assert "llama3.2:latest" not in " ".join(m.value for m in at.markdown if "####" in m.value)
    assert "tok/s" not in " ".join(c.value for c in at.caption)
