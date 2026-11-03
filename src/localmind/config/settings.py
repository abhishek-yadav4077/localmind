"""Application settings loaded from environment variables and ``.env``.

App settings use the ``LOCALMIND_`` prefix. LangSmith settings deliberately use
LangSmith's *own* variable names (``LANGSMITH_*``, falling back to the legacy
``LANGCHAIN_*``) so the same ``.env`` works for this app and for any plain
LangChain script — no duplicated keys.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from localmind import __version__


class Settings(BaseSettings):
    """All runtime configuration for LocalMind."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LOCALMIND_",
        extra="ignore",
        populate_by_name=True,  # allow Settings(langsmith_api_key=...) in tests
    )

    # --- Ollama -----------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "llama3.2"
    preferred_models: list[str] = Field(
        default_factory=lambda: ["llama3.2", "mistral", "qwen2.5:3b"]
    )
    agent_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"
    ollama_timeout_s: float = Field(default=120.0, gt=0)
    ollama_max_retries: int = Field(default=3, ge=1, le=10)

    # --- App --------------------------------------------------------------
    database_url: str = "sqlite:///data/localmind.db"
    log_level: str = "INFO"
    app_version: str = __version__

    # --- LangSmith (native env var names, see module docstring) -----------
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
    )
    langsmith_project: str = Field(
        default="localmind",
        validation_alias=AliasChoices("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    )

    @property
    def langsmith_configured(self) -> bool:
        """True when tracing is requested *and* an API key is present."""
        key = self.langsmith_api_key.get_secret_value() if self.langsmith_api_key else ""
        return self.langsmith_tracing and bool(key.strip())


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call ``cache_clear`` in tests)."""
    return Settings()
