"""Run the evaluation dataset against local models (Assignment extension).

Examples:
    python scripts/run_eval.py                                  # all installed preferred models
    python scripts/run_eval.py --models llama3.2 mistral qwen2.5:3b --judge qwen2.5:7b
    python scripts/run_eval.py --per-category 1                 # 5-question smoke run
    python scripts/run_eval.py --local                          # no LangSmith upload
    python scripts/run_eval.py --upload-only                    # just create the LangSmith dataset

Results: eval/results/<run_id>_{examples,summary}.csv and _meta.json (incl. hardware).
With LangSmith configured, each model is an experiment and a comparison URL is printed.
"""

from __future__ import annotations

import argparse
import sys
import time

from localmind.bootstrap import build_context
from localmind.evaluation.dataset import ensure_langsmith_dataset, load_dataset
from localmind.evaluation.runner import EvalRunner, subset
from localmind.llm.ollama_client import normalize_model_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="default: installed preferred models")
    parser.add_argument("--judge", help="judge model (default: LOCALMIND_EVAL_JUDGE_MODEL)")
    parser.add_argument("--dataset", help="path to dataset JSON (default: eval/dataset.json)")
    parser.add_argument("--per-category", type=int, help="only the first N items per category")
    parser.add_argument("--local", action="store_true", help="don't upload to LangSmith")
    parser.add_argument("--upload-only", action="store_true", help="create the dataset and exit")
    args = parser.parse_args()

    ctx = build_context()
    dataset = subset(
        load_dataset(args.dataset or ctx.settings.eval_dataset_path), args.per_category
    )

    if args.upload_only:
        if not ctx.tracer.enabled:
            print("LangSmith is not configured (set LANGSMITH_API_KEY).")
            return 1
        print(f"dataset: {ensure_langsmith_dataset(ctx.tracer.client, dataset)}")
        return 0

    health = ctx.ollama.health()
    if not health.reachable:
        print("\n".join(health.hints))
        return 1
    installed = {normalize_model_name(m) for m in health.installed_models}
    models = args.models or [
        m for m in ctx.settings.preferred_models if normalize_model_name(m) in installed
    ]
    judge = args.judge or ctx.settings.eval_judge_model
    missing = [m for m in [*models, judge] if normalize_model_name(m) not in installed]
    if missing:
        print("Not installed: " + ", ".join(missing))
        print("Pull with: " + "; ".join(f"ollama pull {m}" for m in missing))
        return 1
    if judge in models:
        print(
            f"Note: judge '{judge}' is also a candidate. LLM judges tend to favour their own "
            "answers (self-preference bias); prefer a different, larger judge if you have one."
        )

    use_langsmith = ctx.tracer.enabled and not args.local
    print(
        f"{len(dataset.items)} examples x {len(models)} models, judge={judge}, "
        f"LangSmith={'on' if use_langsmith else 'off'}"
    )
    start = time.perf_counter()
    runner = EvalRunner(ctx.settings, ctx.tracer, ollama_version=health.version)
    result = runner.run(dataset, models, judge, use_langsmith=use_langsmith, progress=print)
    ctx.tracer.flush()

    print(f"\nDone in {(time.perf_counter() - start) / 60:.1f} min. Run id: {result.run_id}\n")
    header = f"{'model':<20}{'correct':>9}{'concise':>9}{'refusal':>9}{'lat s':>7}{'tok/s':>7}"
    print(header)
    for r in result.summary:
        print(
            f"{r['model']:<20}{_fmt(r['correctness'])}{_fmt(r['conciseness'])}"
            f"{_fmt(r['refusal_accuracy'])}{r['avg_latency_s'] or '-':>7}"
            f"{r['avg_tokens_per_sec'] or '-':>7}"
        )
    print("\nFiles:", *result.paths.values(), sep="\n  ")
    if result.meta["comparison_url"]:
        print(f"\nLangSmith comparison: {result.meta['comparison_url']}")
    return 0


def _fmt(value: float | None) -> str:
    return f"{value:>9.2f}" if value is not None else f"{'-':>9}"


if __name__ == "__main__":
    sys.exit(main())
