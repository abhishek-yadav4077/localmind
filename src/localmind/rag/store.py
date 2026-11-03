"""Local vector store for uploaded documents (Chroma + Ollama embeddings).

Decisions:
- **Chroma, persistent on disk** (``data/chroma``): embedded, no server to run,
  survives restarts. Anonymous telemetry is switched off so nothing leaves the
  machine.
- **Cosine distance**: embedding models are trained for cosine similarity;
  Chroma's default is L2.
- **nomic-embed-text task prefixes**: the model was trained with
  ``search_document:`` / ``search_query:`` prefixes and retrieves noticeably
  better when they are used (see its model card).
- **Content-hash document ids**: uploading the same file twice is a no-op
  instead of duplicating every chunk.
- **One collection per embedding model**: vectors from different models are
  not comparable (and usually differ in dimension).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from localmind.config import Settings, get_logger, get_settings
from localmind.exceptions import FeatureUnavailableError, LocalMindError
from localmind.rag.loaders import load_bytes
from localmind.tracing.metadata import slug

logger = get_logger(__name__)


def _require_rag_deps() -> None:
    try:
        import chromadb  # noqa: F401
        import langchain_chroma  # noqa: F401
        import langchain_text_splitters  # noqa: F401
    except ImportError as exc:
        raise FeatureUnavailableError(
            "Document chat dependencies are not installed",
            hint='Run: pip install -e ".[rag]"',
        ) from exc


class NomicPrefixedEmbeddings(Embeddings):
    """Adds the task prefixes nomic-embed-text expects; passes through otherwise."""

    def __init__(self, inner: Embeddings, enabled: bool = True) -> None:
        self.inner = inner
        self.enabled = enabled

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.enabled:
            texts = [f"search_document: {t}" for t in texts]
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"search_query: {text}" if self.enabled else text)


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class Chunk:
    """A retrieved chunk with its similarity (cosine, higher is better)."""

    text: str
    filename: str
    page: int | None
    doc_id: str
    score: float


class DocumentStore:
    """Add, list, delete and search uploaded documents.

    Args:
        settings: App settings (Chroma path, chunking, embedding model).
        embeddings: Override the embedding model (tests use a fake).
        persist: False keeps the index in memory (tests).
        collection_name: Override the collection name (tests).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        persist: bool = True,
        collection_name: str | None = None,
    ) -> None:
        _require_rag_deps()
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.settings = settings or get_settings()
        if embeddings is None:
            from langchain_ollama import OllamaEmbeddings

            model = self.settings.embedding_model
            embeddings = NomicPrefixedEmbeddings(
                OllamaEmbeddings(model=model, base_url=self.settings.ollama_base_url),
                enabled=model.startswith("nomic-embed-text"),
            )
        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        client = (
            chromadb.PersistentClient(path=self.settings.chroma_dir, settings=chroma_settings)
            if persist
            else chromadb.EphemeralClient(settings=chroma_settings)
        )
        self.vs = Chroma(
            collection_name=collection_name or f"docs_{slug(self.settings.embedding_model)}",
            embedding_function=embeddings,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )

    def add_document(self, filename: str, data: bytes) -> DocumentInfo:
        """Parse, chunk, embed and store a file. Idempotent per file content."""
        max_bytes = self.settings.rag_max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise LocalMindError(f"File too large (max {self.settings.rag_max_upload_mb} MB)")
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        existing = self.vs.get(where={"doc_id": doc_id}, include=["metadatas"])
        if existing["ids"]:
            pages = {m.get("page") for m in existing["metadatas"]}
            return DocumentInfo(doc_id, filename, len(pages), len(existing["ids"]))

        pages = load_bytes(filename, data)
        chunks = self.splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"doc_id": doc_id, "chunk": i})
            if chunk.metadata.get("page") is None:
                chunk.metadata.pop("page", None)  # Chroma metadata can't hold None
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        self.vs.add_documents(chunks, ids=ids)
        logger.info("document indexed", extra={"file": filename, "chunks": len(chunks)})
        return DocumentInfo(doc_id, filename, len(pages), len(chunks))

    def list_documents(self) -> list[DocumentInfo]:
        data = self.vs.get(include=["metadatas"])
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for meta in data["metadatas"]:
            by_doc[meta["doc_id"]].append(meta)
        return sorted(
            (
                DocumentInfo(
                    doc_id=doc_id,
                    filename=metas[0]["source"],
                    pages=len({m.get("page") for m in metas}),
                    chunks=len(metas),
                )
                for doc_id, metas in by_doc.items()
            ),
            key=lambda d: d.filename.lower(),
        )

    def delete_document(self, doc_id: str) -> int:
        ids = self.vs.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.vs.delete(ids=ids)
        return len(ids)

    def search(
        self, query: str, k: int | None = None, doc_ids: list[str] | None = None
    ) -> list[Chunk]:
        """Top-k chunks for ``query``, optionally restricted to some documents."""
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        results = self.vs.similarity_search_with_score(
            query, k=k or self.settings.rag_top_k, filter=where
        )
        return [_to_chunk(doc, distance) for doc, distance in results]


def _to_chunk(doc: Document, distance: float) -> Chunk:
    return Chunk(
        text=doc.page_content,
        filename=doc.metadata.get("source", "?"),
        page=doc.metadata.get("page"),
        doc_id=doc.metadata.get("doc_id", ""),
        score=round(1 - distance, 3),  # cosine distance -> similarity
    )
