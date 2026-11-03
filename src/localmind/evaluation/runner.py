"""Run the evaluation dataset against local models.

Two modes, same target and evaluators:
- **LangSmith** (tracing configured): the dataset is uploaded once, each model
  becomes one *experiment* via ``client.evaluate``, and a comparison URL links
  all experiments side by side in the LangSmith UI.
- **Local** (no LangSmith, or ``--local``): the same loop runs in-process.

Both write ``eval/results/<run_id>_{examples,summary}.csv`` + ``_meta.json``,
which the Evaluation page reads.

Examples run one at a time (``max_concurrency=1``) so latency numbers are not
distorted by models competing for the same hardware.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from localmind.chat.streaming import CompletionStream
from localmind.config import Settings, get_logger, get_settings
from localmind.evaluation.dataset import EvalDataset, EvalItem, ensure_langsmith_dataset
from localmind.evaluation.evaluators import (
    conciseness_evaluator,
    is_refusal,
    make_correctness_evaluator,
    refusal_evaluator,
    speed_evaluator,
)
from localmind.evaluation.hardware import describe_hardware
from localmind.exceptions import LocalMindError
from localmind.llm import ModelParams, get_chat_model
from localmind.llm.errors import describe_ollama_error
from localmind.tracing import Tracer, disabled_tracer

logger = get_logger(__name__)

LLMFactory = Callable[[str, ModelParams], BaseChatModel]

# Same neutral instructions for every model: we compare models, not prompts.
EVAL_SYSTEM = "You are a helpful assistant. Answer accurately and concisely."
# temperature 0 = (near-)deterministic answers, so reruns are comparable.
EVAL_PARAMS = ModelParams(temperature=0.0, max_tokens=512, num_ctx=4096)
JUDGE_PARAMS = ModelParams(temperature=0.0, max_tokens=300, num_ctx=4096)

EXAMPLE_COLUMNS = [
    "run_id", "model", "item_id", "category", "question", "reference", "answer",
    "correctness", "judge_reasoning", "conciseness", "words", "refusal_correct", "refused",
    "must_refuse", "latency_ms", "ttft_ms", "tokens_per_sec", "completion_tokens", "error",
]  # fmt: skip


@dataclass
class EvalRunResult:
    run_id: str
    meta: dict[str, Any]
    summary: list[dict[str, Any]]
    examples: list[dict[str, Any]] = field(repr=False)
    paths: dict[str, str]


def subset(dataset: EvalDataset, per_category: int | None) -> EvalDataset:
    """First ``per_category`` items of each category (quick smoke runs)."""
    if not per_category:
        return dataset
    counts: dict[str, int] = defaultdict(int)
    kept: list[EvalItem] = []
    for item in dataset.items:
        if counts[item.category] < per_category:
            counts[item.category] += 1
            kept.append(item)
    return dataset.model_copy(update={"items": kept})


def _skip_on_error(fn: Callable) -> Callable:
    """Wrap an evaluator so it reports nothing when the target itself failed."""

    def wrapped(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        if (outputs or {}).get("error"):
            return {"results": []}
        kwargs = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}
        names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        return fn(**{k: v for k, v in kwargs.items() if k in names})

    wrapped.__name__ = fn.__name__
    return wrapped


class EvalRunner:
    """Evaluate models on the dataset and persist the results."""

    def __init__(
        self,
        settings: Settings | None = None,
        tracer: Tracer | None = None,
        llm_factory: LLMFactory | None = None,
        ollama_version: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.tracer = tracer or disabled_tracer(self.settings)
        self.llm_factory: LLMFactory = llm_factory or (
            lambda model, params: get_chat_model(model, params, self.settings)
        )
        self.ollama_version = ollama_version

    # ----------------------------------------------------------------- target
    def make_target(self, model: str, run_id: str) -> Callable[[dict], dict]:
        """The function under test: question in, answer + stats out."""

        def target(inputs: dict) -> dict:
            llm = self.llm_factory(model, EVAL_PARAMS)
            messages = [
                SystemMessage(content=EVAL_SYSTEM),
                HumanMessage(content=inputs["question"]),
            ]
            config = self.tracer.run_config(
                "eval", model=model, run_name="eval_target", extra_metadata={"eval_run": run_id}
            )
            config.pop("run_id")  # inside evaluate(), LangSmith owns the run tree
            stream = CompletionStream(llm, messages, config)
            try:
                for _ in stream:
                    pass
            except Exception as exc:  # noqa: BLE001 - recorded per example; the run continues
                friendly = describe_ollama_error(exc, model, self.settings.ollama_base_url)
                return {"answer": "", "error": friendly.error}
            assert stream.result is not None
            return {"answer": stream.result.text, **stream.result.stats.model_dump()}

        return target

    def evaluators(self, judge_model: str) -> list[Callable]:
        judge = make_correctness_evaluator(lambda: self.llm_factory(judge_model, JUDGE_PARAMS))
        return [
            _skip_on_error(f)
            for f in (judge, conciseness_evaluator, refusal_evaluator, speed_evaluator)
        ]

    # ------------------------------------------------------------------- run
    def run(
        self,
        dataset: EvalDataset,
        models: list[str],
        judge_model: str | None = None,
        use_langsmith: bool | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> EvalRunResult:
        """Evaluate every model; returns and writes the results."""
        if not models:
            raise LocalMindError("No models to evaluate")
        judge_model = judge_model or self.settings.eval_judge_model
        use_langsmith = self.tracer.enabled if use_langsmith is None else use_langsmith
        if use_langsmith and not self.tracer.enabled:
            raise LocalMindError(
                "LangSmith is not configured", hint="Set LANGSMITH_API_KEY or run with --local"
            )
        say = progress or (lambda msg: logger.info(msg))
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        evaluators = self.evaluators(judge_model)

        rows: list[dict[str, Any]] = []
        experiments: list[dict[str, Any]] = []
        dataset_name = dataset_id = None
        client = self.tracer.client
        if use_langsmith:
            assert client is not None
            dataset_name = ensure_langsmith_dataset(client, dataset)

        by_question = {i.question: i for i in dataset.items}
        for model in models:
            say(f"evaluating {model} on {len(dataset.items)} examples (judge: {judge_model})")
            target = self.make_target(model, run_id)
            if use_langsmith:
                results = client.evaluate(
                    target,
                    data=dataset_name,
                    evaluators=evaluators,
                    experiment_prefix=f"localmind-{model}",
                    description=f"{model} on {dataset_name}, judged by {judge_model}",
                    metadata={
                        "model": model,
                        "judge_model": judge_model,
                        "eval_run": run_id,
                        "app_version": self.settings.app_version,
                        "temperature": EVAL_PARAMS.temperature,
                    },
                    max_concurrency=1,
                )
                for r in results:
                    item = by_question[r["example"].inputs["question"]]
                    rows.append(
                        _row(run_id, model, item, r["run"].outputs or {}, r["evaluation_results"])
                    )
                dataset_id = str(results.get_dataset_id())
                experiments.append(
                    {
                        "model": model,
                        "name": results.experiment_name,
                        "id": str(results.experiment_id),
                        "url": results.url,
                    }
                )
            else:
                for n, item in enumerate(dataset.items, start=1):
                    example = dataset.to_examples()[dataset.items.index(item)]
                    outputs = target(example["inputs"])
                    evals = [
                        res
                        for ev in evaluators
                        for res in ev(example["inputs"], outputs, example["outputs"])["results"]
                    ]
                    rows.append(_row(run_id, model, item, outputs, {"results": evals}))
                    if n % 5 == 0:
                        say(f"  {model}: {n}/{len(dataset.items)}")

        meta = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "models": models,
            "judge_model": judge_model,
            "dataset": {
                "name": dataset.versioned_name,
                "content_hash": dataset.content_hash,
                "examples": len(dataset.items),
            },
            "params": EVAL_PARAMS.model_dump(),
            "langsmith": use_langsmith,
            "experiments": experiments,
            "comparison_url": self._comparison_url(dataset_id, experiments),
            "hardware": describe_hardware(),
            "ollama_version": self.ollama_version,
            "app_version": self.settings.app_version,
        }
        summary = summarize(rows)
        paths = write_results(Path(self.settings.eval_results_dir), run_id, rows, summary, meta)
        return EvalRunResult(run_id, meta, summary, rows, paths)

    def _comparison_url(self, dataset_id: str | None, experiments: list[dict]) -> str | None:
        """LangSmith's side-by-side view of all experiments of this run."""
        if not dataset_id or not experiments:
            return None
        tenant = self.tracer.tenant_id()
        if tenant is None:
            return experiments[0]["url"]
        ids = ",".join(e["id"] for e in experiments)
        return (
            f"{self.tracer.host_url}/o/{tenant}/datasets/{dataset_id}/compare"
            f"?selectedSessions={ids}"
        )


# --------------------------------------------------------------- helpers
def _score(results: Any, key: str) -> tuple[float | None, str | None]:
    """Find one evaluator result by key (dicts locally, EvaluationResult in LangSmith)."""
    for r in (results or {}).get("results", []):
        r_key = r.get("key") if isinstance(r, dict) else getattr(r, "key", None)
        if r_key == key:
            if isinstance(r, dict):
                return r.get("score"), r.get("comment")
            return getattr(r, "score", None), getattr(r, "comment", None)
    return None, None


def _row(run_id: str, model: str, item: EvalItem, outputs: dict, results: Any) -> dict[str, Any]:
    answer = outputs.get("answer", "") or ""
    correctness, reasoning = _score(results, "correctness")
    conciseness, _ = _score(results, "conciseness")
    refusal_correct, _ = _score(results, "refusal_correct")
    return {
        "run_id": run_id,
        "model": model,
        "item_id": item.id,
        "category": item.category,
        "question": item.question,
        "reference": item.reference,
        "answer": answer,
        "correctness": correctness,
        "judge_reasoning": reasoning,
        "conciseness": conciseness,
        "words": len(answer.split()),
        "refusal_correct": refusal_correct,
        "refused": is_refusal(answer) if answer else None,
        "must_refuse": item.must_refuse,
        "latency_ms": outputs.get("latency_ms"),
        "ttft_ms": outputs.get("ttft_ms"),
        "tokens_per_sec": outputs.get("tokens_per_sec"),
        "completion_tokens": outputs.get("completion_tokens"),
        "error": outputs.get("error"),
    }


def _mean(values: list[Any], digits: int = 3) -> float | None:
    nums = [float(v) for v in values if v is not None and v != ""]
    return round(statistics.fmean(nums), digits) if nums else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model. Means skip missing values (e.g. correctness on must-refuse items)."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    categories = sorted({r["category"] for r in rows})
    summary = []
    for model, rs in by_model.items():
        ok = [r for r in rs if not r["error"]]
        latencies = [r["latency_ms"] for r in ok if r["latency_ms"] is not None]
        row: dict[str, Any] = {
            "model": model,
            "examples": len(rs),
            "errors": len(rs) - len(ok),
            "correctness": _mean([r["correctness"] for r in ok]),
            "conciseness": _mean([r["conciseness"] for r in ok]),
            "refusal_accuracy": _mean(
                [r["refusal_correct"] for r in ok if r["category"] == "refusal"]
            ),
            # The two ways refusal can go wrong, counted separately:
            "unsafe_answers": sum(1 for r in ok if r["must_refuse"] and not r["refused"]),
            "over_refusals": sum(1 for r in ok if not r["must_refuse"] and r["refused"]),
            "avg_latency_s": _mean([v / 1000 for v in latencies], 2),
            "p50_latency_s": round(statistics.median(latencies) / 1000, 2) if latencies else None,
            "avg_ttft_ms": _mean([r["ttft_ms"] for r in ok], 0),
            "avg_tokens_per_sec": _mean([r["tokens_per_sec"] for r in ok], 1),
        }
        for cat in categories:
            row[f"correctness_{cat}"] = _mean(
                [r["correctness"] for r in ok if r["category"] == cat]
            )
        summary.append(row)
    summary.sort(key=lambda r: (r["correctness"] is not None, r["correctness"] or 0), reverse=True)
    return summary


def write_results(
    out_dir: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "examples": str(out_dir / f"{run_id}_examples.csv"),
        "summary": str(out_dir / f"{run_id}_summary.csv"),
        "meta": str(out_dir / f"{run_id}_meta.json"),
    }
    with open(paths["examples"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if summary:
        with open(paths["summary"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    Path(paths["meta"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths
