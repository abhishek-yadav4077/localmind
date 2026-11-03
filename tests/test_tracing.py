"""init_tracing must flip LangSmith's global switch and never require a key."""

import pytest
from langsmith import utils as ls_utils
from pydantic import SecretStr

from localmind.config import Settings
from localmind.tracing import init_tracing


@pytest.fixture(autouse=True)
def reset_langsmith():
    yield
    import langsmith as ls

    ls.configure(enabled=None, client=None, project_name=None)


def test_disabled_by_default(settings: Settings) -> None:
    status = init_tracing(settings)
    assert not status.enabled
    assert ls_utils.tracing_is_enabled() is False


def test_requested_without_key_degrades_gracefully(caplog) -> None:
    s = Settings(_env_file=None, langsmith_tracing=True)
    status = init_tracing(s)
    assert not status.enabled
    assert "API_KEY" in status.reason
    assert ls_utils.tracing_is_enabled() is False


def test_enabled_with_key(monkeypatch) -> None:
    # Stub the client so the test never opens a network connection.
    monkeypatch.setattr("langsmith.Client", lambda **kwargs: object())
    s = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key=SecretStr("lsv2_fake"),
        langsmith_project="unit-test",
    )
    status = init_tracing(s)
    assert status.enabled and status.project == "unit-test"
    assert ls_utils.tracing_is_enabled() is True
