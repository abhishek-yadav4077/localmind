"""The agent as an explicit LangGraph state machine.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                   \\--(no tool calls)--> END

- ``agent`` node: the chat model with tools bound decides to answer or to call tools.
- ``tools`` node: LangGraph's ``ToolNode`` runs every requested tool call and
  appends ``ToolMessage`` results (errors are returned to the model as text so
  it can recover instead of crashing the run).
- ``tools_condition`` routes on whether the last AI message has ``tool_calls``.

This is the ReAct loop written out by hand rather than via the prebuilt
``create_react_agent`` (deprecated since LangGraph 1.0). Writing it explicitly
makes each step visible and easy to extend (e.g. a human-approval node).
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

AGENT_SYSTEM_PROMPT = """You are LocalMind Agent, a helpful assistant that can use tools.

Rules:
- Use the calculator for every calculation, even simple ones.
- Use current_datetime whenever the answer depends on today's date or time.
- Use search_past_conversations when the user refers to earlier chats.
{web_rule}
- Call a tool only when it helps. If you can answer directly, answer.
- After using tools, give a short final answer based on the tool results. Never invent tool results.
"""

WEB_ON = "- Use web_search for current events or facts you are unsure about, and cite URLs."
WEB_OFF = "- You have NO internet access. If a question needs live web data, say so."


def system_prompt(web_search_enabled: bool) -> str:
    return AGENT_SYSTEM_PROMPT.format(web_rule=WEB_ON if web_search_enabled else WEB_OFF)


def build_agent_graph(
    llm: BaseChatModel, tools: Sequence[BaseTool], prompt: str
) -> CompiledStateGraph:
    """Compile the agent graph for one run."""
    llm_with_tools = llm.bind_tools(list(tools))

    def agent(state: MessagesState) -> dict:
        # The system prompt is prepended at call time instead of being stored in
        # state, so it never gets duplicated across loop iterations.
        response = llm_with_tools.invoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
