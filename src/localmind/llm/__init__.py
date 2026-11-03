"""Ollama access: REST client (health, models, pull) and the ChatOllama factory."""

from localmind.llm.factory import ModelParams, get_chat_model, with_connection_retry
from localmind.llm.ollama_client import HealthReport, ModelInfo, OllamaClient, PullProgress

__all__ = [
    "HealthReport",
    "ModelInfo",
    "ModelParams",
    "OllamaClient",
    "PullProgress",
    "get_chat_model",
    "with_connection_retry",
]
