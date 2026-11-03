"""Shared fixtures. Tests never touch a real Ollama server or LangSmith."""

from __future__ import annotations

import pytest

from localmind.config import Settings, get_settings
from localmind.db import Database

_ENV_PREFIXES = ("LOCALMIND_", "LANGSMITH_", "LANGCHAIN_")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Strip app/LangSmith env vars and run from an empty dir so no real .env is read."""
    import os

    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings for tests (no .env, single retry for speed)."""
    return Settings(_env_file=None, ollama_max_retries=1)


@pytest.fixture
def db() -> Database:
    """Fresh in-memory SQLite database with tables created."""
    database = Database("sqlite://")
    database.create_all()
    return database
