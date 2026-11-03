"""ChatService end to end against an in-memory DB and a fake model."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from ollama import ResponseError

from localmind.chat import (
    ChatService,
    ConversationNotFoundError,
    DoneEvent,
    ErrorEvent,
    MemoryEvent,
    TitleEvent,
    TokenEvent,
)
from localmind.db import Database
from localmind.llm import ModelParams
from localmind.memory import MemoryPolicy
from localmind.personas import PersonaService
from tests.fakes import FakeFactory


@pytest.fixture
def factory() -> FakeFactory:
    return FakeFactory()


@pytest.fixture
def chat(db: Database, settings, factory: FakeFactory) -> ChatService:
    return ChatService(db, settings, llm_factory=factory)


def run(chat: ChatService, cid: str, text: str) -> list:
    return list(chat.stream_reply(cid, text))


def test_stream_reply_emits_tokens_done_and_title(chat: ChatService) -> None:
    conv = chat.create_conversation(model="llama3.2")
    events = run(chat, conv.id, "hello there")

    tokens = "".join(e.content for e in events if isinstance(e, TokenEvent))
    assert tokens == "Answer to: hello there"
    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.message.content == tokens
    assert done.message.model == "llama3.2"
    # Stats come from the fake's Ollama-style metadata: 4 words / 0.5 s.
    assert done.message.tokens_per_sec == 8.0
    assert done.message.completion_tokens == 4
    assert done.message.latency_ms is not None and done.message.ttft_ms is not None
    # Title comes after done, cleaned up.
    assert isinstance(events[-1], TitleEvent)
    assert events[-1].title == "Learning Python Basics"
    assert chat.get_conversation(conv.id).title == "Learning Python Basics"
    assert [m.role for m in chat.get_messages(conv.id)] == ["user", "assistant"]


def test_title_generated_only_once(chat: ChatService) -> None:
    conv = chat.create_conversation()
    run(chat, conv.id, "first")
    second = run(chat, conv.id, "second")
    assert not any(isinstance(e, TitleEvent) for e in second)


def test_manual_rename_disables_auto_title(chat: ChatService) -> None:
    conv = chat.create_conversation()
    chat.update_conversation(conv.id, title="My title")
    events = run(chat, conv.id, "hi")
    assert not any(isinstance(e, TitleEvent) for e in events)
    assert chat.get_conversation(conv.id).title == "My title"


def test_persona_and_history_are_in_prompt(chat: ChatService, db, factory) -> None:
    personas = PersonaService(db)
    personas.seed_defaults()
    tutor = next(p for p in personas.list() if p.name == "Tutor")
    conv = chat.create_conversation(persona_id=tutor.id)
    run(chat, conv.id, "one")
    run(chat, conv.id, "two")

    chat_calls = [c for c in factory.all_calls if "short title" not in str(c[0].content)]
    last_prompt = chat_calls[-1]
    assert isinstance(last_prompt[0], SystemMessage)
    assert last_prompt[0].content == tutor.system_prompt
    assert [m.content for m in last_prompt[1:]] == ["one", "Answer to: one", "two"]


def test_model_switch_and_params_apply_to_next_turn(chat: ChatService, factory) -> None:
    conv = chat.create_conversation(model="llama3.2")
    chat.update_conversation(
        conv.id, model="mistral", params=ModelParams(temperature=0.1, num_ctx=8192)
    )
    run(chat, conv.id, "hi")
    model, params, _ = factory.built[0]
    assert model == "mistral"
    assert params.temperature == 0.1 and params.num_ctx == 8192


def test_memory_summarization_when_context_fills(db, settings, factory) -> None:
    chat = ChatService(db, settings, llm_factory=factory, memory_policy=MemoryPolicy(keep_recent=2))
    # Tiny window so a few long turns overflow it.
    conv = chat.create_conversation(params=ModelParams(num_ctx=512, max_tokens=64))
    long = "word " * 120  # ~150 tokens per message; budget is 0.75 * (512 - 64) = 336
    run(chat, conv.id, "My name is Ada. " + long)
    # Turn 2: only 2 older messages == keep_recent, so nothing can be folded yet.
    second = run(chat, conv.id, "Second. " + long)
    assert not any(isinstance(e, MemoryEvent) for e in second)
    # Turn 3: the first exchange (2 messages) gets folded into the summary.
    events = run(chat, conv.id, "Third. " + long)
    mem_events = [e for e in events if isinstance(e, MemoryEvent)]
    assert mem_events and mem_events[0].summarized_messages == 2

    saved = chat.get_conversation(conv.id)
    assert saved.summary == "User is Ada, learning Python."
    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.message.memory_summarized is True

    # Next prompt: summary injected, summarized turns no longer sent verbatim.
    run(chat, conv.id, "fourth")
    prompt = factory.all_calls[-1]
    assert any("Summary of the earlier conversation" in str(m.content) for m in prompt)
    assert not any("My name is Ada" in str(m.content) for m in prompt)
    assert isinstance(prompt[-1], HumanMessage) and prompt[-1].content == "fourth"
    # Full history is still stored for the user.
    assert len(chat.get_messages(conv.id)) == 8


def test_connection_error_becomes_error_event(db, settings) -> None:
    chat = ChatService(db, settings, llm_factory=FakeFactory(error=httpx.ConnectError("refused")))
    conv = chat.create_conversation()
    events = run(chat, conv.id, "hi")
    assert isinstance(events[-1], ErrorEvent)
    assert "ollama serve" in (events[-1].hint or "")
    # User message kept so it can be retried; no assistant message stored.
    assert [m.role for m in chat.get_messages(conv.id)] == ["user"]


def test_missing_model_error_event(db, settings) -> None:
    err = ResponseError("model 'nope' not found", status_code=404)
    chat = ChatService(db, settings, llm_factory=FakeFactory(error=err))
    conv = chat.create_conversation(model="nope")
    (event,) = run(chat, conv.id, "hi")
    assert event.hint == "Run: ollama pull nope"


def test_empty_message_rejected(chat: ChatService) -> None:
    conv = chat.create_conversation()
    assert isinstance(run(chat, conv.id, "   ")[0], ErrorEvent)


def test_unknown_conversation(chat: ChatService) -> None:
    with pytest.raises(ConversationNotFoundError):
        run(chat, "missing", "hi")


def test_list_search_delete(chat: ChatService) -> None:
    a = chat.create_conversation()
    chat.reply(a.id, "tell me about transformers")
    b = chat.create_conversation(title="Other")
    assert [c.id for c in chat.list_conversations(query="transformers")] == [a.id]
    chat.delete_conversation(b.id)
    assert [c.id for c in chat.list_conversations()] == [a.id]


def test_export_markdown_and_json(chat: ChatService) -> None:
    conv = chat.create_conversation(model="llama3.2")
    chat.reply(conv.id, "hello")
    md = chat.export(conv.id, "markdown")
    assert md.startswith("# Learning Python Basics")
    assert "### 🧑 User" in md and "tok/s" in md
    data = json.loads(chat.export(conv.id, "json"))
    assert data["conversation"]["id"] == conv.id
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
