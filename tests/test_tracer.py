"""Tracer facade with a mocked LangSmith client (no network)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

from langsmith.utils import LangSmithNotFoundError

from localmind.config import Settings
from localmind.tracing import Tracer, TracingStatus, disabled_tracer

TENANT = uuid.UUID(int=1)
PROJECT = uuid.UUID(int=2)


def enabled_tracer(client: MagicMock, endpoint: str = "https://api.smith.langchain.com") -> Tracer:
    settings = Settings(_env_file=None, langsmith_endpoint=endpoint)
    return Tracer(TracingStatus(True, "localmind", "configured", client=client), settings)


def fake_client() -> MagicMock:
    client = MagicMock()
    client.read_project.return_value = SimpleNamespace(id=PROJECT, tenant_id=TENANT)
    return client


def test_disabled_tracer_is_a_noop(settings) -> None:
    tracer = disabled_tracer(settings)
    assert not tracer.enabled
    assert tracer.run_url("123") is None
    assert tracer.project_url() is None
    assert tracer.send_feedback("123", 1) is None
    cfg = tracer.run_config("chat", model="m")
    assert tracer.stored_run_id(cfg) is None  # no dead trace links
    assert cfg["metadata"]["app_version"] == settings.app_version


def test_run_url_resolves_ids_once() -> None:
    client = fake_client()
    tracer = enabled_tracer(client)
    run_id = uuid.uuid4()
    url = tracer.run_url(run_id)
    assert url == (
        f"https://smith.langchain.com/o/{TENANT}/projects/p/{PROJECT}/r/{run_id}?poll=true"
    )
    tracer.run_url(uuid.uuid4())
    client.read_project.assert_called_once_with(project_name="localmind")


def test_eu_endpoint_maps_to_eu_web_host() -> None:
    tracer = enabled_tracer(fake_client(), endpoint="https://eu.api.smith.langchain.com")
    assert tracer.project_url().startswith("https://eu.smith.langchain.com/o/")


def test_project_created_when_missing() -> None:
    client = MagicMock()
    client.read_project.side_effect = LangSmithNotFoundError("nope")
    client.create_project.return_value = SimpleNamespace(id=PROJECT, tenant_id=TENANT)
    assert enabled_tracer(client).project_url().endswith(str(PROJECT))
    client.create_project.assert_called_once_with("localmind")


def test_lookup_failure_returns_none_and_retries_later() -> None:
    client = fake_client()
    client.read_project.side_effect = [ConnectionError("down"), client.read_project.return_value]
    tracer = enabled_tracer(client)
    assert tracer.run_url("r1") is None
    assert tracer.run_url("r1") is not None


def test_feedback_create_then_update() -> None:
    client = fake_client()
    tracer = enabled_tracer(client)
    fid = tracer.send_feedback("run-1", 1, "great")
    _, kwargs = client.create_feedback.call_args
    assert client.create_feedback.call_args.args == ("run-1", "user-score")
    assert kwargs["score"] == 1 and kwargs["comment"] == "great"
    assert str(kwargs["feedback_id"]) == fid

    assert tracer.send_feedback("run-1", 0, None, feedback_id=fid) == fid
    client.update_feedback.assert_called_once_with(fid, score=0, comment=None)
    assert client.create_feedback.call_count == 1
