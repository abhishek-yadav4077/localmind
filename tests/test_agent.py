"""AgentService with scripted fake tool-calling models (no Ollama)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage
from ollama import ResponseError

from localmind.agent import (
    AgentDoneEvent,
    AgentService,
    AgentStartEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from localmind.chat import ChatService
from localmind.chat.schemas import ErrorEvent
from localmind.tracing import Tracer, TracingStatus
from tests.fakes import FakeFactory, FakeToolChat, tool_call


def service(db, settings, llm: FakeToolChat, tracer=None) -> AgentService:
    return AgentService(db, settings, llm_factory=lambda m, p: llm, tracer=tracer, max_steps=3)


def test_tool_call_then_answer(db, settings) -> None:
    llm = FakeToolChat(
        script=[tool_call("calculator", {"expression": "23*19"}), AIMessage(content="It is 437.")]
    )
    events = list(service(db, settings, llm).stream("What is 23 times 19?"))

    assert isinstance(events[0], AgentStartEvent)
    assert events[0].tools == ["calculator", "current_datetime", "search_past_conversations"]
    assert events[0].model == settings.agent_model
    call, result, done = events[1:]
    assert isinstance(call, AgentToolCallEvent) and call.args == {"expression": "23*19"}
    assert isinstance(result, AgentToolResultEvent)
    assert result.content == "23*19 = 437" and not result.is_error
    assert isinstance(done, AgentDoneEvent)

    msg = done.message
    assert msg.content == "It is 437." and msg.mode == "agent"
    assert [s["kind"] for s in msg.details["steps"]] == ["call", "result"]
    assert msg.details["llm_calls"] == 2
    # Totals across both LLM calls: 2 x 5 tokens in 2 x 0.25 s.
    assert msg.completion_tokens == 10 and msg.tokens_per_sec == 20.0

    # The second LLM call saw the tool result.
    assert isinstance(llm.calls[1][-1], ToolMessage)
    # System prompt says there is no internet when web search is off.
    assert "NO internet" in llm.calls[0][0].content


def test_follow_up_sees_history(db, settings) -> None:
    llm = FakeToolChat(script=[AIMessage(content="Hi Ada."), AIMessage(content="Ada.")])
    agent = service(db, settings, llm)
    first = agent.run("I am Ada")
    agent.run("What is my name?", conversation_id=first.conversation_id)
    sent = [m.content for m in llm.calls[1][1:]]  # skip system prompt
    assert sent == ["I am Ada", "Hi Ada.", "What is my name?"]


def test_web_search_toggle_changes_tools_and_prompt(db, settings) -> None:
    llm = FakeToolChat(script=[AIMessage(content="ok")])
    list(service(db, settings, llm).stream("news?", web_search=True))
    assert "web_search" in llm.bound_tools
    assert "web_search for current events" in llm.calls[0][0].content


def test_unknown_tool_error_is_returned_to_model(db, settings) -> None:
    llm = FakeToolChat(script=[tool_call("rm_rf", {}), AIMessage(content="Sorry.")])
    events = list(service(db, settings, llm).stream("do it"))
    result = next(e for e in events if isinstance(e, AgentToolResultEvent))
    assert result.is_error and "rm_rf" in result.content
    assert isinstance(events[-1], AgentDoneEvent)


def test_tool_loop_is_bounded(db, settings) -> None:
    llm = FakeToolChat(script=[tool_call("current_datetime", {})], repeat_last=True)
    events = list(service(db, settings, llm).stream("loop forever"))
    assert isinstance(events[-1], ErrorEvent)
    assert "3 tool rounds" in events[-1].error
    assert len(llm.calls) <= 4


def test_model_without_tool_support(db, settings) -> None:
    llm = FakeToolChat(
        script=[], error=ResponseError("registry/gemma does not support tools", status_code=400)
    )
    events = list(service(db, settings, llm).stream("hi"))
    assert "does not support tool calling" in events[-1].error
    assert "qwen2.5" in events[-1].hint


def test_agent_run_is_traced_and_visible_in_chat_history(db, settings) -> None:
    tracer = Tracer(TracingStatus(True, "p", "ok", client=MagicMock()), settings)
    llm = FakeToolChat(script=[AIMessage(content="Done.")])
    msg = service(db, settings, llm, tracer).run("hello agent")
    assert msg.run_id is not None
    chat = ChatService(db, settings, llm_factory=FakeFactory())
    conv = chat.get_conversation(msg.conversation_id)
    assert conv.title.startswith("🛠 hello agent") and conv.model == settings.agent_model
    assert [m.mode for m in chat.get_messages(conv.id)] == ["agent", "agent"]
