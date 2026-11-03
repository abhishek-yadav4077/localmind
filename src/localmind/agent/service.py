"""Run the tool-calling agent and stream its intermediate steps.

We stream LangGraph *updates* (one event per node execution) rather than
tokens: the UI's main job here is to show the reasoning trail — which tool was
called with which arguments and what came back — as it happens.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from localmind.agent.graph import build_agent_graph, system_prompt
from localmind.agent.tools import build_tools
from localmind.chat import persistence
from localmind.chat.schemas import ErrorEvent, MessageOut, MessageStats
from localmind.config import Settings, get_logger, get_settings
from localmind.db import Database
from localmind.exceptions import LocalMindError
from localmind.llm import ModelParams, get_chat_model
from localmind.llm.errors import describe_ollama_error
from localmind.memory.summarizer import db_role_to_message
from localmind.tracing import Tracer, disabled_tracer

logger = get_logger(__name__)

LLMFactory = Callable[[str, ModelParams], BaseChatModel]
HISTORY_MESSAGES = 10  # earlier turns sent back to the agent for context


class AgentStartEvent(BaseModel):
    type: Literal["start"] = "start"
    conversation_id: str
    model: str
    tools: list[str]


class AgentToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    args: dict[str, Any]


class AgentToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    name: str
    content: str
    is_error: bool = False


class AgentDoneEvent(BaseModel):
    type: Literal["done"] = "done"
    message: MessageOut


AgentEvent = (
    AgentStartEvent | AgentToolCallEvent | AgentToolResultEvent | AgentDoneEvent | ErrorEvent
)


class AgentService:
    """Tool-using agent over a local model.

    Args:
        db: Database (also searched by the past-conversations tool).
        settings: App settings (default agent model).
        llm_factory: ``(model, params) -> chat model``; must support ``bind_tools``.
        tracer: LangSmith facade.
        max_steps: Max agent->tools round trips before giving up. Small local
            models sometimes loop calling the same tool; this bounds the cost.
    """

    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        llm_factory: LLMFactory | None = None,
        tracer: Tracer | None = None,
        max_steps: int = 6,
    ) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.llm_factory: LLMFactory = llm_factory or (
            lambda model, params: get_chat_model(model, params, self.settings)
        )
        self.tracer = tracer or disabled_tracer(self.settings)
        self.max_steps = max_steps

    def stream(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        model: str | None = None,
        web_search: bool = False,
    ) -> Iterator[AgentEvent]:
        """Answer ``question`` with tools, yielding each step."""
        question = question.strip()
        if not question:
            raise LocalMindError("Question is empty")
        model = model or self.settings.agent_model
        conversation_id = persistence.ensure_conversation(
            self.db, conversation_id, model, title=f"🛠 {question}"
        )
        prior: list[BaseMessage] = [
            db_role_to_message(role, content)
            for role, content in persistence.history(self.db, conversation_id, HISTORY_MESSAGES)
        ]

        tools = build_tools(
            self.db, web_search_enabled=web_search, exclude_conversation_id=conversation_id
        )
        tool_names = [t.name for t in tools]
        yield AgentStartEvent(conversation_id=conversation_id, model=model, tools=tool_names)

        # temperature 0: tool calling needs well-formed JSON arguments, not creativity.
        llm = self.llm_factory(model, ModelParams(temperature=0.0))
        graph = build_agent_graph(llm, tools, system_prompt(web_search))
        config = self.tracer.run_config(
            "agent",
            model=model,
            conversation_id=conversation_id,
            run_name="agent",
            extra_metadata={"web_search": web_search, "tools": tool_names},
        )
        # Each agent->tools round trip is 2 graph steps, plus the final answer.
        config["recursion_limit"] = 2 * self.max_steps + 1

        steps: list[dict[str, Any]] = []
        ai_messages: list[AIMessage] = []
        start = time.perf_counter()
        try:
            for update in graph.stream(
                {"messages": [*prior, HumanMessage(content=question)]},
                config=config,
                stream_mode="updates",
            ):
                for node, output in update.items():
                    for msg in (output or {}).get("messages", []):
                        if isinstance(msg, AIMessage):
                            ai_messages.append(msg)
                            for call in msg.tool_calls:
                                event = AgentToolCallEvent(
                                    call_id=call.get("id") or "",
                                    name=call["name"],
                                    args=call.get("args", {}),
                                )
                                steps.append(event.model_dump(exclude={"type"}) | {"kind": "call"})
                                yield event
                        elif isinstance(msg, ToolMessage):
                            event = AgentToolResultEvent(
                                call_id=msg.tool_call_id,
                                name=msg.name or node,
                                content=str(msg.content),
                                is_error=msg.status == "error",
                            )
                            steps.append(event.model_dump(exclude={"type"}) | {"kind": "result"})
                            yield event
        except GraphRecursionError:
            yield ErrorEvent(
                error=f"Agent stopped after {self.max_steps} tool rounds without a final answer",
                hint="Rephrase the question or try a larger model (e.g. qwen2.5:7b)",
            )
            return
        except Exception as exc:  # noqa: BLE001 - any model/graph failure becomes an error event
            friendly = describe_ollama_error(exc, model, self.settings.ollama_base_url)
            logger.warning("agent run failed", extra={"model": model, "error": friendly.error})
            yield ErrorEvent(error=friendly.error, hint=friendly.hint)
            return

        final = ai_messages[-1] if ai_messages else None
        answer = str(final.content).strip() if final is not None else ""
        message = persistence.save_exchange(
            self.db,
            conversation_id,
            user_text=question,
            answer=answer or "(The model returned an empty answer.)",
            model=model,
            mode="agent",
            stats=_agent_stats(ai_messages, (time.perf_counter() - start) * 1000),
            run_id=self.tracer.stored_run_id(config),
            details={"steps": steps, "web_search": web_search, "llm_calls": len(ai_messages)},
        )
        yield AgentDoneEvent(message=message)

    def run(self, question: str, **kwargs: Any) -> MessageOut:
        """Blocking wrapper; raises on error."""
        for event in self.stream(question, **kwargs):
            if isinstance(event, ErrorEvent):
                raise LocalMindError(event.error, hint=event.hint)
            if isinstance(event, AgentDoneEvent):
                return event.message
        raise LocalMindError("Agent finished without an answer")  # pragma: no cover


def _agent_stats(ai_messages: list[AIMessage], latency_ms: float) -> MessageStats:
    """Totals across every LLM call in the run (an agent answer = several calls)."""
    prompt_tokens = completion_tokens = eval_count = eval_ns = 0
    for m in ai_messages:
        usage = m.usage_metadata or {}
        prompt_tokens += usage.get("input_tokens", 0)
        completion_tokens += usage.get("output_tokens", 0)
        eval_count += m.response_metadata.get("eval_count") or 0
        eval_ns += m.response_metadata.get("eval_duration") or 0
    return MessageStats(
        latency_ms=round(latency_ms, 1),
        ttft_ms=None,  # not meaningful: the first tokens are usually a tool call
        tokens_per_sec=round(eval_count / (eval_ns / 1e9), 1) if eval_ns else None,
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
    )
