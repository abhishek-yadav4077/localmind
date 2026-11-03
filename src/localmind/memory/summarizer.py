"""Rolling-summary memory.

Strategy ("summary + recent window"):

    [system prompt] [summary of old turns] [last N turns verbatim] [new user msg]

When the estimated prompt size crosses ``trigger_ratio`` of the usable context
window, every turn except the most recent ``keep_recent`` is folded into the
summary (together with the previous summary, so nothing is lost twice).

Why not just truncate? Truncation silently drops facts the user stated early
("my name is...", "I'm using Python 3.11"). Why not summarize every turn? It
costs an extra LLM call per message. Summarizing only near the limit is the
usual middle ground.

Token counting: Ollama exposes no tokenizer endpoint, so we estimate with
~4 characters per token (a common rule of thumb for English BPE tokenizers)
and leave a safety margin via ``trigger_ratio``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from localmind.config import get_logger

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4
# Chat templates add a few tokens per message (role markers etc.).
PER_MESSAGE_OVERHEAD = 4

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You maintain the long-term memory of a chat. Merge the previous summary and "
            "the new conversation turns into one updated summary. Keep facts about the user "
            "(name, goals, preferences, constraints), decisions made, open questions and key "
            "technical details. Drop greetings and filler. Write in third person, at most "
            "{max_words} words. Output only the summary.",
        ),
        (
            "human",
            "Previous summary:\n{previous_summary}\n\nNew turns:\n{transcript}",
        ),
    ]
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Always >= 1 for non-empty text."""
    return 0 if not text else max(1, len(text) // CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    return sum(estimate_tokens(str(m.content)) + PER_MESSAGE_OVERHEAD for m in messages)


@dataclass(frozen=True)
class MemoryPolicy:
    """When and how much to summarize.

    Attributes:
        trigger_ratio: Summarize when the prompt exceeds this share of the budget.
        keep_recent: Number of most recent messages always kept verbatim.
        reserve_output_tokens: Tokens reserved for the model's reply when
            ``max_tokens`` is not set.
        summary_max_words: Length cap given to the summarizer.
    """

    trigger_ratio: float = 0.75
    keep_recent: int = 6
    reserve_output_tokens: int = 512
    summary_max_words: int = 200

    def budget(self, num_ctx: int, max_tokens: int | None) -> int:
        """Tokens available for the prompt = window minus room for the answer."""
        return max(num_ctx - (max_tokens or self.reserve_output_tokens), 256)


def summary_message(summary: str) -> SystemMessage:
    """How a stored summary is injected into the prompt."""
    return SystemMessage(content=f"Summary of the earlier conversation:\n{summary}")


def to_transcript(messages: Sequence[BaseMessage]) -> str:
    role = {"human": "User", "ai": "Assistant", "system": "System"}
    return "\n".join(f"{role.get(m.type, m.type)}: {m.content}" for m in messages)


class MemoryManager:
    """Decides when to summarize and produces the new summary.

    Args:
        llm_factory: Returns the chat model used for summarizing (by default the
            conversation's own model, so no extra model needs to be installed).
        policy: Thresholds; see ``MemoryPolicy``.
    """

    def __init__(
        self,
        llm_factory: Callable[[], BaseChatModel],
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.llm_factory = llm_factory
        self.policy = policy or MemoryPolicy()

    def needs_summary(
        self,
        prompt_messages: Sequence[BaseMessage],
        history_len: int,
        num_ctx: int,
        max_tokens: int | None = None,
    ) -> bool:
        """True when the full prompt is near the budget *and* there is old history to fold."""
        if history_len <= self.policy.keep_recent:
            return False
        used = estimate_messages_tokens(prompt_messages)
        limit = self.policy.trigger_ratio * self.policy.budget(num_ctx, max_tokens)
        return used > limit

    def split(self, history: Sequence[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split history into (to_summarize, keep_verbatim)."""
        cut = max(len(history) - self.policy.keep_recent, 0)
        return list(history[:cut]), list(history[cut:])

    def summarize(
        self,
        previous_summary: str | None,
        messages: Sequence[BaseMessage],
        config: dict | None = None,
    ) -> str:
        """Return a merged summary of ``previous_summary`` + ``messages``."""
        chain = SUMMARY_PROMPT | self.llm_factory() | StrOutputParser()
        summary = chain.invoke(
            {
                "previous_summary": previous_summary or "(none)",
                "transcript": to_transcript(messages),
                "max_words": self.policy.summary_max_words,
            },
            config={"run_name": "memory_summarize", **(config or {})},
        ).strip()
        logger.info(
            "memory summarized",
            extra={"messages": len(messages), "summary_chars": len(summary)},
        )
        return summary


def db_role_to_message(role: str, content: str) -> BaseMessage:
    """Map a stored role to a LangChain message."""
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return SystemMessage(content=content)
