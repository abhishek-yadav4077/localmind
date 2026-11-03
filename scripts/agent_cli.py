"""Tool-using agent in the terminal — for manual testing before the UI exists.

Run:
    python scripts/agent_cli.py "What is 17.5% of 2340, and what weekday is it today?"
    python scripts/agent_cli.py "What did we discuss about Docker?" --model llama3.2
    python scripts/agent_cli.py "Latest Ollama release?" --web    # ONLINE: uses DuckDuckGo
"""

from __future__ import annotations

import argparse
import json
import sys

from localmind.agent import (
    AgentDoneEvent,
    AgentStartEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
)
from localmind.bootstrap import build_context
from localmind.chat.schemas import ErrorEvent


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the LocalMind agent.")
    parser.add_argument("question")
    parser.add_argument("--model", help="tool-capable model (default: LOCALMIND_AGENT_MODEL)")
    parser.add_argument("--web", action="store_true", help="enable web search (goes online)")
    parser.add_argument("--conversation", help="continue an agent conversation id")
    args = parser.parse_args()

    ctx = build_context()
    try:
        for event in ctx.agent.stream(
            args.question, model=args.model, web_search=args.web, conversation_id=args.conversation
        ):
            if isinstance(event, AgentStartEvent):
                print(f"[agent] model={event.model} tools={', '.join(event.tools)}")
            elif isinstance(event, AgentToolCallEvent):
                print(f"  -> {event.name}({json.dumps(event.args)})")
            elif isinstance(event, AgentToolResultEvent):
                flag = " (error)" if event.is_error else ""
                print(f"  <- {event.name}{flag}: {event.content[:300]}")
            elif isinstance(event, ErrorEvent):
                print(f"[error] {event.error}" + (f"\n  -> {event.hint}" if event.hint else ""))
                return 1
            elif isinstance(event, AgentDoneEvent):
                m = event.message
                print(f"\n{m.content}\n")
                calls = m.details["llm_calls"]
                print(f"[{m.latency_ms} ms | {calls} LLM calls | conversation={m.conversation_id}]")
                if ctx.tracer.enabled:
                    print(f"[trace] {ctx.tracer.run_url(m.run_id)}")
    finally:
        ctx.tracer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
