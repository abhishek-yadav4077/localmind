"""Stream a chat model while measuring it. Shared by chat, arena and RAG."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage

from localmind.chat.schemas import MessageStats
from localmind.chat.stats import StreamTimer, compute_stats


@dataclass(frozen=True)
class StreamResult:
    text: str
    stats: MessageStats


class CompletionStream:
    """Iterate to receive text tokens; afterwards ``result`` holds text + stats.

    Example::

        stream = CompletionStream(llm, messages, config)
        for token in stream:
            print(token, end="")
        print(stream.result.stats.tokens_per_sec)

    Exceptions from the model propagate to the caller unchanged.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        messages: Sequence[BaseMessage],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.messages = list(messages)
        self.config = config or {}
        self.result: StreamResult | None = None

    def __iter__(self) -> Iterator[str]:
        timer = StreamTimer()
        full: AIMessageChunk | None = None
        timer.begin()
        for chunk in self.llm.stream(self.messages, config=self.config):
            text = chunk.content if isinstance(chunk.content, str) else ""
            timer.on_token(text)
            # Adding chunks merges content *and* metadata; Ollama puts token
            # counts and timings on the final chunk.
            full = chunk if full is None else full + chunk
            if text:
                yield text
        timer.finish()
        stats = compute_stats(
            timer,
            full.response_metadata if full is not None else None,
            dict(full.usage_metadata) if full is not None and full.usage_metadata else None,
        )
        self.result = StreamResult(text=str(full.content) if full is not None else "", stats=stats)
