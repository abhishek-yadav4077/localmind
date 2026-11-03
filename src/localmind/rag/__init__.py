"""Optional "chat with a document" module (install with ``pip install -e ".[rag]"``).

Isolated on purpose: nothing outside this package imports Chroma, and the rest
of the app runs without these dependencies.
"""

from localmind.rag.chain import Citation, RagService, RagSourcesEvent
from localmind.rag.store import Chunk, DocumentInfo, DocumentStore

__all__ = ["Chunk", "Citation", "DocumentInfo", "DocumentStore", "RagService", "RagSourcesEvent"]
