"""Terminal chat on top of ChatService — for manual testing before the UI exists.

Run:
    python scripts/chat_cli.py
    python scripts/chat_cli.py --model mistral --persona "Code Reviewer" --num-ctx 2048
    python scripts/chat_cli.py --resume <conversation_id>

Commands inside the chat:  /export md | /export json | /quit
"""

from __future__ import annotations

import argparse
import sys

from localmind.bootstrap import build_context
from localmind.chat import DoneEvent, ErrorEvent, MemoryEvent, TitleEvent, TokenEvent
from localmind.llm import ModelParams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model")
    parser.add_argument("--persona", help="Persona name, e.g. Tutor")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--resume", help="Continue an existing conversation id")
    args = parser.parse_args()

    ctx = build_context()
    if args.resume:
        conv = ctx.chat.get_conversation(args.resume)
    else:
        persona_id = None
        if args.persona:
            match = [p for p in ctx.personas.list() if p.name.lower() == args.persona.lower()]
            if not match:
                names = ", ".join(p.name for p in ctx.personas.list())
                print(f"Unknown persona. Available: {names}")
                return 1
            persona_id = match[0].id
        params = ModelParams(
            temperature=args.temperature, num_ctx=args.num_ctx, max_tokens=args.max_tokens
        )
        conv = ctx.chat.create_conversation(args.model, persona_id, params)

    tracing = f"LangSmith project '{ctx.tracing.project}'" if ctx.tracing.enabled else "off"
    print(f"conversation={conv.id} model={conv.model} tracing={tracing}")
    print("Type /quit to exit, /export md|json to export.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text in {"/quit", "/exit"}:
            return 0
        if text.startswith("/export"):
            fmt = "json" if text.endswith("json") else "markdown"
            print(ctx.chat.export(conv.id, fmt))
            continue
        if not text:
            continue

        print("bot> ", end="", flush=True)
        for event in ctx.chat.stream_reply(conv.id, text):
            if isinstance(event, TokenEvent):
                print(event.content, end="", flush=True)
            elif isinstance(event, MemoryEvent):
                print(f"[memory summarized: {event.summarized_messages} messages] ", end="")
            elif isinstance(event, DoneEvent):
                m = event.message
                print(
                    f"\n     [{m.latency_ms} ms total | TTFT {m.ttft_ms} ms | "
                    f"{m.tokens_per_sec} tok/s | {m.prompt_tokens} in / "
                    f"{m.completion_tokens} out]"
                )
            elif isinstance(event, TitleEvent):
                print(f"     [title: {event.title}]")
            elif isinstance(event, ErrorEvent):
                print(f"\n[error] {event.error}" + (f"\n  -> {event.hint}" if event.hint else ""))
        print()


if __name__ == "__main__":
    sys.exit(main())
