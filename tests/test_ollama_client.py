"""OllamaClient against a fake Ollama server built with httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from localmind.config import Settings
from localmind.exceptions import ModelNotFoundError, ModelPullError, OllamaUnavailableError
from localmind.llm import OllamaClient
from localmind.llm.ollama_client import normalize_model_name

TAGS = {
    "models": [
        {
            "name": "llama3.2:latest",
            "size": 2_019_393_189,
            "details": {
                "family": "llama",
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
            },
        },
        {"name": "mistral:latest", "size": 4_113_301_824, "details": {}},
    ]
}


def make_client(settings: Settings, handler) -> OllamaClient:
    return OllamaClient(settings, transport=httpx.MockTransport(handler))


def healthy_handler(models: dict = TAGS):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=models)
        return httpx.Response(404)

    return handler


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection refused", request=request)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("llama3.2", "llama3.2:latest"), ("qwen2.5:3b", "qwen2.5:3b")],
)
def test_normalize_model_name(name: str, expected: str) -> None:
    assert normalize_model_name(name) == expected


def test_list_models_parses_details(settings: Settings) -> None:
    models = make_client(settings, healthy_handler()).list_models()
    assert [m.name for m in models] == ["llama3.2:latest", "mistral:latest"]
    assert models[0].quantization == "Q4_K_M"
    assert models[0].size_gb == 2.02


def test_is_installed_ignores_latest_tag(settings: Settings) -> None:
    client = make_client(settings, healthy_handler())
    assert client.is_installed("llama3.2")
    assert not client.is_installed("qwen2.5:3b")


def test_health_ok_reports_missing_preferred(settings: Settings) -> None:
    report = make_client(settings, healthy_handler()).health()
    assert report.ok
    assert report.version == "0.12.0"
    assert report.missing_preferred == ["qwen2.5:3b"]
    assert report.hints == []


def test_health_no_models_gives_pull_hint(settings: Settings) -> None:
    report = make_client(settings, healthy_handler({"models": []})).health()
    assert report.reachable and not report.ok
    assert "ollama pull llama3.2" in report.hints[0]


def test_health_default_model_missing(settings: Settings) -> None:
    only_mistral = {"models": [{"name": "mistral:latest"}]}
    report = make_client(settings, healthy_handler(only_mistral)).health()
    assert report.ok
    assert "Default model 'llama3.2' is not installed" in report.hints[0]


def test_health_unreachable_never_raises(settings: Settings) -> None:
    report = make_client(settings, refuse).health()
    assert not report.reachable and not report.ok
    assert "ollama serve" in report.hints[0]


def test_unreachable_raises_with_hint_after_retries() -> None:
    calls = 0

    def counting_refuse(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    s = Settings(_env_file=None, ollama_max_retries=2)
    with pytest.raises(OllamaUnavailableError) as exc:
        make_client(s, counting_refuse).version()
    assert calls == 2
    assert exc.value.hint and "ollama serve" in exc.value.hint


def test_context_length_from_model_info(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"model": "llama3.2"}
        return httpx.Response(200, json={"model_info": {"llama.context_length": 131072}})

    assert make_client(settings, handler).context_length("llama3.2") == 131072


def test_context_length_unknown_model(settings: Settings) -> None:
    client = make_client(settings, lambda r: httpx.Response(404, json={"error": "not found"}))
    with pytest.raises(ModelNotFoundError):
        client.context_length("nope")


def test_pull_streams_progress(settings: Settings) -> None:
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling abc", "completed": 50, "total": 100},
        {"status": "pulling abc", "completed": 100, "total": 100},
        {"status": "success"},
    ]
    body = "\n".join(json.dumps(x) for x in lines).encode()
    client = make_client(settings, lambda r: httpx.Response(200, content=body))
    progress = list(client.pull("llama3.2"))
    assert [p.status for p in progress][-1] == "success"
    assert progress[1].fraction == 0.5
    assert progress[0].fraction is None


def test_pull_error_line_raises(settings: Settings) -> None:
    body = json.dumps({"error": "pull model manifest: file does not exist"}).encode()
    client = make_client(settings, lambda r: httpx.Response(200, content=body))
    with pytest.raises(ModelPullError):
        list(client.pull("not-a-model"))
