"""Chat with documents in the terminal — for manual testing before the UI exists.

Needs:  pip install -e ".[rag]"   and   ollama pull nomic-embed-text

Run:
    python scripts/docs_cli.py add path/to/paper.pdf
    python scripts/docs_cli.py list
    python scripts/docs_cli.py ask "What dataset did the authors use?"
    python scripts/docs_cli.py delete <doc_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localmind.bootstrap import build_context
from localmind.chat.schemas import DoneEvent, ErrorEvent, TokenEvent
from localmind.rag import RagSourcesEvent


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with local documents.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("path", type=Path)
    sub.add_parser("list")
    sub.add_parser("delete").add_argument("doc_id")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--model")
    ask.add_argument("--k", type=int)
    args = parser.parse_args()

    ctx = build_context()
    rag = ctx.rag()
    if args.cmd == "add":
        info = rag.store.add_document(args.path.name, args.path.read_bytes())
        print(
            f"indexed {info.filename}: {info.pages} pages, {info.chunks} chunks (id {info.doc_id})"
        )
    elif args.cmd == "list":
        for d in rag.store.list_documents():
            print(f"{d.doc_id}  {d.filename}  ({d.pages} pages, {d.chunks} chunks)")
    elif args.cmd == "delete":
        print(f"deleted {rag.store.delete_document(args.doc_id)} chunks")
    else:
        for event in rag.stream(args.question, model=args.model, k=args.k):
            if isinstance(event, RagSourcesEvent):
                for s in event.sources:
                    page = f" p.{s.page}" if s.page else ""
                    print(f"  [{s.n}] {s.filename}{page} (similarity {s.score})")
                print()
            elif isinstance(event, TokenEvent):
                print(event.content, end="", flush=True)
            elif isinstance(event, ErrorEvent):
                print(f"[error] {event.error}" + (f"\n  -> {event.hint}" if event.hint else ""))
                return 1
            elif isinstance(event, DoneEvent):
                d = event.message.details
                print(f"\n\n[cited: {d['cited']}  invalid citations: {d['invalid_citations']}]")
        ctx.tracer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
