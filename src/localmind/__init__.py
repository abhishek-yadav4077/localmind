"""LocalMind — private LLM chat studio on Ollama with LangSmith observability."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("localmind")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
