"""Test doubles that behave like ChatOllama without a server."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

Responder = Callable[[Sequence[BaseMessage]], str]


def default_responder(messages: Sequence[BaseMessage]) -> str:
    system = " ".join(str(m.content) for m in messages if m.type == "system")
    if "short title" in system:
        return 'Title: "Learning Python Basics".'
    if "long-term memory" in system:
        return "User is Ada, learning Python."
    return f"Answer to: {messages[-1].content}"


class FakeOllamaChat(BaseChatModel):
    """Streams the response word by word and attaches Ollama-like metadata."""

    responder: Responder = default_responder
    error: Exception | None = None
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    eval_duration_ns: int = 500_000_000  # 0.5 s of "decode time"

    @property
    def _llm_type(self) -> str:
        return "fake-ollama"

    def _meta(self, messages: Sequence[BaseMessage], text: str) -> tuple[dict, dict]:
        n_out = len(text.split())
        n_in = sum(len(str(m.content).split()) for m in messages)
        usage = {"input_tokens": n_in, "output_tokens": n_out, "total_tokens": n_in + n_out}
        meta = {
            "model": "fake",
            "done": True,
            "prompt_eval_count": n_in,
            "eval_count": n_out,
            "eval_duration": self.eval_duration_ns,
        }
        return usage, meta

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        text = self.responder(messages)
        usage, meta = self._meta(messages, text)
        msg = AIMessage(content=text, usage_metadata=usage, response_metadata=meta)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        text = self.responder(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=word if i == 0 else " " + word)
            )
        usage, meta = self._meta(messages, text)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="", usage_metadata=usage, response_metadata=meta)
        )


class FakeFactory:
    """``(model, params) -> FakeOllamaChat`` that records every model it built."""

    def __init__(self, responder: Responder = default_responder, error: Exception | None = None):
        self.responder = responder
        self.error = error
        self.built: list[tuple[str, Any, FakeOllamaChat]] = []

    def __call__(self, model: str, params: Any) -> FakeOllamaChat:
        llm = FakeOllamaChat(responder=self.responder, error=self.error)
        self.built.append((model, params, llm))
        return llm

    @property
    def all_calls(self) -> list[list[BaseMessage]]:
        return [c for _, _, llm in self.built for c in llm.calls]
