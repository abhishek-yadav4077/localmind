from pydantic import SecretStr

from localmind.config import Settings


def test_defaults(settings: Settings) -> None:
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.default_model == "llama3.2"
    assert settings.langsmith_tracing is False
    assert settings.langsmith_configured is False


def test_reads_native_langsmith_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "my-proj")
    s = Settings(_env_file=None)
    assert s.langsmith_configured is True
    assert s.langsmith_project == "my-proj"


def test_legacy_langchain_env_vars_still_work(monkeypatch) -> None:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
    assert Settings(_env_file=None).langsmith_configured is True


def test_tracing_without_key_is_not_configured() -> None:
    s = Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key=SecretStr("  "))
    assert s.langsmith_configured is False


def test_app_prefix_and_list_parsing(monkeypatch) -> None:
    monkeypatch.setenv("LOCALMIND_DEFAULT_MODEL", "mistral")
    monkeypatch.setenv("LOCALMIND_PREFERRED_MODELS", '["mistral","phi3"]')
    s = Settings(_env_file=None)
    assert s.default_model == "mistral"
    assert s.preferred_models == ["mistral", "phi3"]


def test_api_key_is_not_leaked_in_repr() -> None:
    s = Settings(_env_file=None, langsmith_api_key=SecretStr("super-secret"))
    assert "super-secret" not in repr(s)
