"""ChatService: the use-case layer for chatting.

One user turn, in order:

1. persist the user message
2. build the prompt: persona system prompt + memory summary + recent turns
3. if the prompt nears the context window, fold old turns into the summary
4. stream the reply from Ollama, timing it
5. persist the assistant message with stats
6. on the first exchange, generate a title

The service is synchronous on purpose: SQLite and Ollama's client are sync, and
FastAPI runs sync generators in a thread pool, so async would add complexity
without a throughput benefit for a single-user local app.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from localmind.chat import export
from localmind.chat.schemas import (
    ChatEvent,
    ConversationOut,
    DoneEvent,
    ErrorEvent,
    MemoryEvent,
    MessageOut,
    TitleEvent,
    TokenEvent,
)
from localmind.chat.streaming import CompletionStream
from localmind.chat.titles import generate_title
from localmind.config import Settings, get_logger, get_settings
from localmind.db import Database
from localmind.db import repositories as repo
from localmind.exceptions import LocalMindError, NotFoundError
from localmind.llm import ModelParams, get_chat_model
from localmind.llm.errors import describe_ollama_error
from localmind.memory import MemoryManager, MemoryPolicy
from localmind.memory.summarizer import db_role_to_message, summary_message
from localmind.tracing import Tracer, child_config, disabled_tracer, merge_configs

logger = get_logger(__name__)

LLMFactory = Callable[[str, ModelParams], BaseChatModel]


class ConversationNotFoundError(NotFoundError):
    """No conversation with that id."""


class ChatService:
    """Conversation CRUD and streaming chat.

    Args:
        db: Database handle.
        settings: App settings.
        llm_factory: ``(model, params) -> BaseChatModel``. Injected so tests can
            pass fake models and never need a running Ollama.
        memory_policy: Summarization thresholds.
        tracer: LangSmith facade; defaults to a no-op tracer.
    """

    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        llm_factory: LLMFactory | None = None,
        memory_policy: MemoryPolicy | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.llm_factory: LLMFactory = llm_factory or (
            lambda model, params: get_chat_model(model, params, self.settings)
        )
        self.memory_policy = memory_policy or MemoryPolicy()
        self.tracer = tracer or disabled_tracer(self.settings)

    # ------------------------------------------------------------------ CRUD
    def create_conversation(
        self,
        model: str | None = None,
        persona_id: int | None = None,
        params: ModelParams | None = None,
        title: str = "New chat",
    ) -> ConversationOut:
        with self.db.session() as s:
            conv = repo.create_conversation(
                s,
                model=model or self.settings.default_model,
                persona_id=persona_id,
                params=(params or ModelParams()).model_dump(),
                title=title,
            )
            return ConversationOut.model_validate(conv)

    def get_conversation(self, conversation_id: str) -> ConversationOut:
        with self.db.session() as s:
            return ConversationOut.model_validate(self._require(s, conversation_id))

    def list_conversations(
        self, query: str | None = None, limit: int = 100
    ) -> list[ConversationOut]:
        with self.db.session() as s:
            return [
                ConversationOut.model_validate(c) for c in repo.list_conversations(s, query, limit)
            ]

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        model: str | None = None,
        persona_id: int | Literal["unset"] | None = "unset",
        params: ModelParams | None = None,
    ) -> ConversationOut:
        """Rename, switch model/persona or change params mid-conversation.

        ``persona_id`` uses a sentinel because ``None`` is meaningful ("no persona").
        """
        with self.db.session() as s:
            conv = self._require(s, conversation_id)
            if title is not None:
                conv.title = title.strip()[:200] or conv.title
                conv.title_generated = True  # a manual rename wins over auto-titles
            if model is not None:
                conv.model = model
            if persona_id != "unset":
                conv.persona_id = persona_id
            if params is not None:
                conv.params = params.model_dump()
            s.flush()
            return ConversationOut.model_validate(conv)

    def delete_conversation(self, conversation_id: str) -> None:
        with self.db.session() as s:
            if not repo.delete_conversation(s, conversation_id):
                raise ConversationNotFoundError(f"Conversation {conversation_id} not found")

    def get_messages(self, conversation_id: str) -> list[MessageOut]:
        with self.db.session() as s:
            self._require(s, conversation_id)
            return [MessageOut.model_validate(m) for m in repo.list_messages(s, conversation_id)]

    def export(self, conversation_id: str, fmt: Literal["markdown", "json"]) -> str:
        with self.db.session() as s:
            conv = self._require(s, conversation_id)
            persona_name = conv.persona.name if conv.persona else None
            conv_out = ConversationOut.model_validate(conv)
        messages = self.get_messages(conversation_id)
        if fmt == "markdown":
            return export.to_markdown(conv_out, messages, persona_name)
        return export.to_json(conv_out, messages)

    # ------------------------------------------------------------------ chat
    def stream_reply(
        self,
        conversation_id: str,
        user_text: str,
        run_config: dict[str, Any] | None = None,
    ) -> Iterator[ChatEvent]:
        """Run one user turn, yielding events as they happen.

        Errors from Ollama are yielded as ``ErrorEvent`` (not raised) so a
        streaming HTTP response can report them to the UI cleanly.

        Args:
            conversation_id: Target conversation.
            user_text: The user's message.
            run_config: Extra LangChain ``RunnableConfig`` (tags/metadata/callbacks).
        """
        user_text = user_text.strip()
        if not user_text:
            yield ErrorEvent(error="Message is empty")
            return

        # 1-2. Persist user message, load everything we need, then release the session:
        # never hold a DB session open across a multi-second LLM stream.
        with self.db.session() as s:
            conv = self._require(s, conversation_id)
            repo.add_message(s, conv.id, "user", user_text)
            model, params = conv.model, ModelParams(**(conv.params or {}))
            persona_prompt = conv.persona.system_prompt if conv.persona else None
            persona_name = conv.persona.name if conv.persona else None
            summary, summarized_up_to = conv.summary, conv.summarized_up_to
            needs_title = not conv.title_generated
            rows = repo.list_messages(s, conv.id, after_id=summarized_up_to)
            history = [(m.id, db_role_to_message(m.role, m.content)) for m in rows]

        # Tags/metadata (mode, model, persona, conversation_id, app version) and a
        # pre-generated run_id that we store on the assistant message for
        # feedback + trace links.
        config = merge_configs(
            self.tracer.run_config(
                "chat",
                model=model,
                persona=persona_name,
                conversation_id=conversation_id,
                run_name="chat",
            ),
            run_config,
        )

        # 3. Memory: summarize old turns if the prompt is getting too big.
        memory_summarized = False
        prompt = self._build_prompt(persona_prompt, summary, [m for _, m in history])
        memory = MemoryManager(lambda: self.llm_factory(model, params), self.memory_policy)
        # The new user message (last item) must always stay verbatim.
        if memory.needs_summary(prompt, len(history) - 1, params.num_ctx, params.max_tokens):
            older, recent = memory.split(history[:-1])
            try:
                summary = memory.summarize(
                    summary, [m for _, m in older], config=child_config(config, "memory_summarize")
                )
            except Exception as exc:  # degrade gracefully: answer with the full history
                logger.warning("memory summarization failed", exc_info=True)
                yield ErrorEvent(error=f"Memory summarization failed: {exc}")
            else:
                summarized_up_to = older[-1][0]
                history = [*recent, history[-1]]
                prompt = self._build_prompt(persona_prompt, summary, [m for _, m in history])
                with self.db.session() as s:
                    conv = self._require(s, conversation_id)
                    conv.summary, conv.summarized_up_to = summary, summarized_up_to
                memory_summarized = True
                yield MemoryEvent(summarized_messages=len(older))

        # 4. Stream the reply.
        stream = CompletionStream(self.llm_factory(model, params), prompt, config)
        try:
            for text in stream:
                yield TokenEvent(content=text)
        except Exception as exc:  # noqa: BLE001 - mapped to a user-facing error event
            yield self._error_event(exc, model)
            return
        assert stream.result is not None
        answer, stats = stream.result.text, stream.result.stats

        # 5. Persist assistant message.
        with self.db.session() as s:
            conv = self._require(s, conversation_id)
            msg = repo.add_message(
                s,
                conv.id,
                "assistant",
                answer,
                model=model,
                memory_summarized=memory_summarized,
                run_id=self.tracer.stored_run_id(config),
                **stats.model_dump(),
            )
            repo.touch_conversation(s, conv)
            done = DoneEvent(message=MessageOut.model_validate(msg))
        yield done

        # 6. Title after the first exchange (after "done" so it never delays the answer).
        if needs_title:
            title_llm = self.llm_factory(
                model, ModelParams(temperature=0.2, max_tokens=24, num_ctx=2048)
            )
            title = generate_title(
                title_llm, user_text, answer, config=child_config(config, "generate_title")
            )
            with self.db.session() as s:
                conv = self._require(s, conversation_id)
                conv.title, conv.title_generated = title, True
            yield TitleEvent(title=title)

    def reply(self, conversation_id: str, user_text: str, **kwargs: Any) -> MessageOut:
        """Non-streaming convenience wrapper (scripts, tests). Raises on error."""
        result: MessageOut | None = None
        for event in self.stream_reply(conversation_id, user_text, **kwargs):
            if isinstance(event, ErrorEvent) and result is None:
                raise LocalMindError(event.error, hint=event.hint)
            if isinstance(event, DoneEvent):
                result = event.message
        assert result is not None
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _build_prompt(
        persona_prompt: str | None, summary: str | None, history: Sequence[BaseMessage]
    ) -> list[BaseMessage]:
        prompt: list[BaseMessage] = []
        if persona_prompt:
            prompt.append(SystemMessage(content=persona_prompt))
        if summary:
            prompt.append(summary_message(summary))
        prompt.extend(history)
        if not prompt or not isinstance(prompt[-1], HumanMessage):  # pragma: no cover
            raise LocalMindError("Prompt must end with the user's message")
        return prompt

    def _error_event(self, exc: Exception, model: str) -> ErrorEvent:
        friendly = describe_ollama_error(exc, model, self.settings.ollama_base_url)
        logger.warning("chat stream failed", extra={"model": model, "error": friendly.error})
        return ErrorEvent(error=friendly.error, hint=friendly.hint)

    @staticmethod
    def _require(session, conversation_id: str):
        conv = repo.get_conversation(session, conversation_id)
        if conv is None:
            raise ConversationNotFoundError(f"Conversation {conversation_id} not found")
        return conv
