"""Document chat: loading, chunking, retrieval, citations (in-memory Chroma, fake models)."""

from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest
from langchain_core.embeddings import Embeddings

from localmind.chat.schemas import DoneEvent, ErrorEvent, TokenEvent
from localmind.config import Settings
from localmind.rag import DocumentStore, RagService, RagSourcesEvent
from localmind.rag.chain import cited_numbers
from localmind.rag.loaders import UnsupportedDocumentError, load_bytes
from localmind.rag.store import NomicPrefixedEmbeddings
from tests.fakes import FakeOllamaChat


class KeywordEmbeddings(Embeddings):
    """Bag-of-words hashed into 256 dims: similar words -> similar vectors."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in {"search", "document", "query"}:  # ignore nomic prefixes
                continue
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


FONT = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one line of text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        page_id, content_id = len(objs) + 1, len(objs) + 2
        kids.append(f"{page_id} 0 R")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {FONT} >> >> >>"
        )
        objs.append(f"<< /Length {len(content)} >>\nstream\n{content.decode()}\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


@pytest.fixture
def store(settings: Settings) -> DocumentStore:
    s = settings.model_copy(update={"rag_chunk_size": 100, "rag_chunk_overlap": 0})
    return DocumentStore(
        s, embeddings=KeywordEmbeddings(), persist=False, collection_name=f"t{uuid.uuid4().hex}"
    )


PYTHON_TXT = (
    b"Python was created by Guido van Rossum and first released in 1991.\n\n"
    b"Bananas are yellow fruits rich in potassium and grown in tropical climates.\n\n"
    b"Docker containers package an application with its dependencies."
)


def test_load_pdf_keeps_page_numbers() -> None:
    docs = load_bytes("report.pdf", make_pdf(["Intro page", "Results page"]))
    assert [(d.page_content, d.metadata["page"]) for d in docs] == [
        ("Intro page", 1),
        ("Results page", 2),
    ]


@pytest.mark.parametrize(
    ("name", "data"), [("x.docx", b"abc"), ("empty.txt", b"   "), ("blank.pdf", make_pdf([""]))]
)
def test_load_rejects_unsupported_or_empty(name: str, data: bytes) -> None:
    with pytest.raises(UnsupportedDocumentError):
        load_bytes(name, data)


def test_add_is_idempotent_and_listable(store: DocumentStore) -> None:
    info = store.add_document("facts.txt", PYTHON_TXT)
    assert info.chunks == 3
    again = store.add_document("facts-copy.txt", PYTHON_TXT)
    assert again.doc_id == info.doc_id and again.chunks == 3
    assert [d.filename for d in store.list_documents()] == ["facts.txt"]


def test_search_ranks_relevant_chunk_first(store: DocumentStore) -> None:
    store.add_document("facts.txt", PYTHON_TXT)
    store.add_document("paper.pdf", make_pdf(["Transformers use self attention", "Bananas"]))
    top = store.search("who created python", k=2)[0]
    assert "Guido" in top.text and top.filename == "facts.txt" and top.page is None
    pdf_hit = store.search("self attention transformers", k=1)[0]
    assert pdf_hit.filename == "paper.pdf" and pdf_hit.page == 1
    assert 0 < pdf_hit.score <= 1


def test_search_filter_and_delete(store: DocumentStore) -> None:
    a = store.add_document("facts.txt", PYTHON_TXT)
    b = store.add_document("other.txt", b"Python snakes live in Asia and Africa.")
    only_b = store.search("python", k=5, doc_ids=[b.doc_id])
    assert {c.doc_id for c in only_b} == {b.doc_id}
    assert store.delete_document(a.doc_id) == 3
    assert [d.doc_id for d in store.list_documents()] == [b.doc_id]


def test_nomic_prefixes() -> None:
    seen: list[str] = []

    class Spy(Embeddings):
        def embed_documents(self, texts):
            seen.extend(texts)
            return [[1.0] for _ in texts]

        def embed_query(self, text):
            seen.append(text)
            return [1.0]

    emb = NomicPrefixedEmbeddings(Spy())
    emb.embed_documents(["doc"])
    emb.embed_query("q")
    assert seen == ["search_document: doc", "search_query: q"]


def test_cited_numbers() -> None:
    assert cited_numbers("A [2] and B [1][2], see [10].") == [1, 2, 10]


def rag(db, settings, store, answer: str) -> tuple[RagService, list]:
    prompts: list = []

    def factory(model, params):
        llm = FakeOllamaChat(responder=lambda msgs: answer)
        prompts.append(llm.calls)
        return llm

    return RagService(db, store, settings, llm_factory=factory), prompts


def test_answer_with_citations(db, settings, store) -> None:
    store.add_document("facts.txt", PYTHON_TXT)
    service, prompts = rag(db, settings, store, "Guido van Rossum created it [1]. Also [7].")
    events = list(service.stream("Who created Python?", k=2))

    sources = events[0]
    assert isinstance(sources, RagSourcesEvent)
    assert "Guido" in sources.sources[0].snippet and sources.sources[0].n == 1
    assert "".join(e.content for e in events if isinstance(e, TokenEvent)).startswith("Guido")

    sent = prompts[0][0]
    assert "ONLY the numbered sources" in sent[0].content
    assert (
        "[1] (facts.txt)" in sent[1].content and "Question: Who created Python?" in sent[1].content
    )

    done = events[-1]
    assert isinstance(done, DoneEvent) and done.message.mode == "rag"
    assert done.message.details["cited"] == [1]
    assert done.message.details["invalid_citations"] == [7]
    assert len(done.message.details["sources"]) == 2


def test_no_documents_gives_helpful_error(db, settings, store) -> None:
    service, _ = rag(db, settings, store, "x")
    (event,) = list(service.stream("anything"))
    assert isinstance(event, ErrorEvent) and "Upload" in (event.hint or "")
