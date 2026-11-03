"""Failure paths: timeouts, unknown errors, retries, summarization failures."""

from __future__ import annotations

import httpx
from langchain_core.runnables import RunnableLambda

from localmind.chat import ChatService, DoneEvent, ErrorEvent
from localmind.llm import ModelParams, with_connection_retry
from localmind.llm.errors import describe_ollama_error
from localmind.memory import MemoryPolicy
from tests.fakes import FakeOllamaChat


def test_timeout_and_unknown_errors_are_described() -> None:
    timeout = describe_ollama_error(httpx.ReadTimeout("slow"), "m", "http://x")
    assert timeout.error == "Ollama timed out" and "TIMEOUT" in timeout.hint
    other = describe_ollama_error(ValueError("weird"), "m", "http://x")
    assert other.error == "ValueError: weird" and other.hint is None


def test_with_connection_retry_retries_only_connection_errors(settings) -> None:
    calls = {"n": 0}

    def flaky(x: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        return x.upper()

    s = settings.model_copy(update={"ollama_max_retries": 3})
    assert with_connection_retry(RunnableLambda(flaky), s).invoke("ok") == "OK"
    assert calls["n"] == 2


def test_chat_continues_when_summarization_fails(db, settings) -> None:
    """If the summary call fails, answer with the full history and say so."""

    def factory(model: str, params: ModelParams) -> FakeOllamaChat:
        def respond(messages):
            if "long-term memory" in str(messages[0].content):
                raise RuntimeError("summarizer crashed")
            return "fine"

        return FakeOllamaChat(responder=respond)

    chat = ChatService(db, settings, llm_factory=factory, memory_policy=MemoryPolicy(keep_recent=2))
    conv = chat.create_conversation(params=ModelParams(num_ctx=512, max_tokens=64))
    long = "word " * 150
    for i in range(2):
        list(chat.stream_reply(conv.id, f"{i} {long}"))
    events = list(chat.stream_reply(conv.id, "third " + long))
    assert any(isinstance(e, ErrorEvent) and "summarization failed" in e.error for e in events)
    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.message.content == "fine" and done.message.memory_summarized is False
    assert chat.get_conversation(conv.id).summary is None
