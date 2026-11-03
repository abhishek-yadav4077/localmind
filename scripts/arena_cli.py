"""Model Arena in the terminal — for manual testing before the UI exists.

Run:
    python scripts/arena_cli.py "Explain overfitting to a 12 year old" \
        --models llama3.2 mistral qwen2.5:3b
    python scripts/arena_cli.py "..." --models llama3.2 mistral --sequential   # fair speed stats
    python scripts/arena_cli.py --leaderboard
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from localmind.arena import ArenaResultEvent, ArenaTokenEvent, compute_leaderboard
from localmind.arena.schemas import ArenaDoneEvent
from localmind.bootstrap import AppContext, build_context


def print_leaderboard(ctx: AppContext, sequential_only: bool) -> None:
    rows = compute_leaderboard(ctx.db, sequential_only=sequential_only)
    if not rows:
        print("No arena data yet.")
        return
    print(f"\n{'model':<20}{'votes':>6}{'wins':>6}{'win%':>7}{'tok/s':>8}{'TTFT ms':>9}")
    for r in rows:
        win = f"{r.win_rate * 100:.0f}%" if r.win_rate is not None else "-"
        print(
            f"{r.model:<20}{r.battles:>6}{r.wins:>6}{win:>7}"
            f"{r.avg_tokens_per_sec or '-':>8}{r.avg_ttft_ms or '-':>9}"
        )
    if sequential_only:
        print("(speed columns from sequential battles only)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare local models side by side.")
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--models", nargs="+", default=["llama3.2", "mistral"])
    parser.add_argument("--sequential", action="store_true", help="one model at a time")
    parser.add_argument("--leaderboard", action="store_true")
    args = parser.parse_args()

    ctx = build_context()
    if args.leaderboard or not args.prompt:
        print_leaderboard(ctx, sequential_only=args.sequential)
        return 0

    battle = None
    for event in ctx.arena.stream_battle(args.prompt, args.models, parallel=not args.sequential):
        if isinstance(event, ArenaTokenEvent):
            print(".", end="", flush=True)  # progress; full answers printed below
        elif isinstance(event, ArenaResultEvent):
            r = event.response
            print(f"\n\n=== {r.position + 1}. {r.model} ===")
            if r.error:
                print(f"[error] {r.error}" + (f" -> {event.hint}" if event.hint else ""))
            else:
                print(textwrap.fill(r.content, 100, replace_whitespace=False))
                print(f"[{r.latency_ms} ms | TTFT {r.ttft_ms} ms | {r.tokens_per_sec} tok/s]")
        elif isinstance(event, ArenaDoneEvent):
            battle = event.battle
    ctx.tracer.flush()
    assert battle is not None

    # Numbers match the "=== N. model ===" headers; failed models can't be picked.
    votable = {str(r.position + 1): r.model for r in battle.responses if not r.error}
    options = "/".join(votable)
    choice = input(f"\nWinner? [{options}/t=tie/b=all bad/enter=skip] ").strip().lower()
    if choice in votable:
        ctx.arena.vote(battle.id, "win", votable[choice])
    elif choice in {"t", "b"}:
        ctx.arena.vote(battle.id, "tie" if choice == "t" else "all_bad")
    ctx.tracer.flush()
    print_leaderboard(ctx, sequential_only=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
