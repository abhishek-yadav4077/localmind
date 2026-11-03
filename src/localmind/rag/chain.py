"""Answer questions about uploaded documents, with numbered citations.

The whole answer is one LCEL chain::

    retrieve (top-k chunks) -> build prompt with [1]..[k] sources -> ChatOllama

so LangSmith shows a single ``rag`` trace with the retrieved chunks and the
exact prompt, which is what you need to debug "why did it answer that?".
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from localmind.chat import persistence
from localmind.chat.schemas import DoneEvent, ErrorEvent, TokenEvent
from localmind.chat.streaming import CompletionStream
from localmind.config import Settings, get_settings
from localmind.db import Database
from localmind.exceptions import LocalMindError
from localmind.llm import ModelParams, get_chat_model
from localmind.llm.errors import describe_ollama_error
from localmind.rag.store import Chunk, DocumentStore
from localmind.tracing import Tracer, disabled_tracer

LLMFactory = Callable[[str, ModelParams], BaseChatModel]

RAG_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. After every claim, "
    "cite the source number in square brackets, e.g. [1] or [2][3]. If the sources do "
    'not contain the answer, reply exactly: "I couldn\'t find that in the documents." '
    "Never cite a source number that was not provided."
)


class Citation(BaseModel):
    n: int
    filename: str
    page: int | None
    score: float
    snippet: str


class RagSourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    conversation_id: str
    sources: list[Citation]


RagEvent = RagSourcesEvent | TokenEvent | DoneEvent | ErrorEvent


def format_sources(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        where = f"{c.filename}, page {c.page}" if c.page else c.filename
        blocks.append(f"[{i}] ({where})\n{c.text}")
    return "\n\n".join(blocks)


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


class RagService:
    """Retrieval-augmented answers over the local document store."""

    def __init__(
        self,
        db: Database,
        store: DocumentStore,
        settings: Settings | None = None,
        llm_factory: LLMFactory | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.settings = settings or get_settings()
        self.llm_factory: LLMFactory = llm_factory or (
            lambda model, params: get_chat_model(model, params, self.settings)
        )
        self.tracer = tracer or disabled_tracer(self.settings)

    def stream(
        self,
        question: str,
        *,
        doc_ids: list[str] | None = None,
        model: str | None = None,
        conversation_id: str | None = None,
        k: int | None = None,
    ) -> Iterator[RagEvent]:
        question = question.strip()
        if not question:
            raise LocalMindError("Question is empty")
        if not self.store.list_documents():
            yield ErrorEvent(error="No documents uploaded yet", hint="Upload a PDF or TXT first")
            return
        model = model or self.settings.default_model
        conversation_id = persistence.ensure_conversation(
            self.db, conversation_id, model, title=f"📄 {question}"
        )

        retrieved: list[Chunk] = []

        def retrieve(q: str) -> dict[str, Any]:
            retrieved.extend(self.store.search(q, k=k, doc_ids=doc_ids))
            return {"question": q, "sources": format_sources(retrieved)}

        def build_prompt(inputs: dict[str, Any]) -> list:
            return [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Sources:\n{inputs['sources']}\n\nQuestion: {inputs['question']}"
                ),
            ]

        # Low temperature: we want faithful extraction, not creative writing.
        llm = self.llm_factory(model, ModelParams(temperature=0.1, num_ctx=8192))
        chain = (
            RunnableLambda(retrieve, name="retrieve")
            | RunnableLambda(build_prompt, name="build_prompt")
            | llm
        )
        config = self.tracer.run_config(
            "rag",
            model=model,
            conversation_id=conversation_id,
            run_name="rag",
            extra_metadata={"doc_ids": doc_ids or "all", "k": k or self.settings.rag_top_k},
        )

        stream = CompletionStream(chain, question, config)
        sources_sent = False
        try:
            for token in stream:
                if not sources_sent:  # retrieval has run by the time the first token arrives
                    sources_sent = True
                    yield RagSourcesEvent(
                        conversation_id=conversation_id, sources=_citations(retrieved)
                    )
                yield TokenEvent(content=token)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as an error event
            friendly = describe_ollama_error(exc, model, self.settings.ollama_base_url)
            yield ErrorEvent(error=friendly.error, hint=friendly.hint)
            return
        assert stream.result is not None
        if not sources_sent:
            yield RagSourcesEvent(conversation_id=conversation_id, sources=_citations(retrieved))

        citations = _citations(retrieved)
        cited = cited_numbers(stream.result.text)
        message = persistence.save_exchange(
            self.db,
            conversation_id,
            user_text=question,
            answer=stream.result.text,
            model=model,
            mode="rag",
            stats=stream.result.stats,
            run_id=self.tracer.stored_run_id(config),
            details={
                "sources": [c.model_dump() for c in citations],
                "cited": [n for n in cited if 1 <= n <= len(citations)],
                # Citations to sources that don't exist = a hallucination signal.
                "invalid_citations": [n for n in cited if not 1 <= n <= len(citations)],
            },
        )
        yield DoneEvent(message=message)


def _citations(chunks: list[Chunk]) -> list[Citation]:
    return [
        Citation(
            n=i,
            filename=c.filename,
            page=c.page,
            score=c.score,
            snippet=" ".join(c.text.split())[:300],
        )
        for i, c in enumerate(chunks, start=1)
    ]
