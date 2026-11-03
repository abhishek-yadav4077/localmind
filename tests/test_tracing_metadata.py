import uuid

from localmind.tracing import build_run_config, child_config, merge_configs
from localmind.tracing.metadata import slug


def test_build_run_config_tags_and_metadata() -> None:
    cfg = build_run_config(
        "chat",
        model="llama3.2",
        app_version="0.1.0",
        persona="Code Reviewer",
        conversation_id="abc",
        run_name="chat",
    )
    assert isinstance(cfg["run_id"], uuid.UUID)
    assert cfg["tags"] == ["localmind", "mode:chat", "model:llama3.2", "persona:code-reviewer"]
    assert cfg["metadata"] == {
        "app": "localmind",
        "app_version": "0.1.0",
        "mode": "chat",
        "model": "llama3.2",
        "persona": "Code Reviewer",
        "conversation_id": "abc",
    }
    assert cfg["run_name"] == "chat"


def test_run_ids_are_unique() -> None:
    a = build_run_config("arena", model="m", app_version="x")
    b = build_run_config("arena", model="m", app_version="x")
    assert a["run_id"] != b["run_id"]
    assert a["metadata"]["persona"] == "none"


def test_child_config_drops_run_id() -> None:
    cfg = build_run_config("chat", model="m", app_version="x", run_name="chat")
    child = child_config(cfg, "generate_title")
    assert "run_id" not in child
    assert child["run_name"] == "generate_title"
    assert child["metadata"] == cfg["metadata"]


def test_merge_configs() -> None:
    base = {"tags": ["a"], "metadata": {"x": 1}, "run_name": "chat"}
    merged = merge_configs(base, {"tags": ["b"], "metadata": {"y": 2}, "callbacks": ["cb"]})
    assert merged == {
        "tags": ["a", "b"],
        "metadata": {"x": 1, "y": 2},
        "run_name": "chat",
        "callbacks": ["cb"],
    }


def test_slug() -> None:
    assert slug("Interview Coach") == "interview-coach"
    assert slug("qwen2.5:3b") == "qwen2.5:3b"
