"""Evaluation: dataset validity, evaluators, local + LangSmith runner, results, API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from localmind.config import Settings
from localmind.evaluation.dataset import EvalDataset, ensure_langsmith_dataset, load_dataset
from localmind.evaluation.evaluators import (
    JudgeVerdict,
    conciseness_evaluator,
    is_refusal,
    make_correctness_evaluator,
    refusal_evaluator,
    word_budget,
)
from localmind.evaluation.results import list_runs, load_run
from localmind.evaluation.runner import EvalRunner, subset
from localmind.exceptions import NotFoundError
from localmind.tracing import Tracer, TracingStatus
from tests.fakes import FakeOllamaChat

DATASET = Path(__file__).parents[1] / "eval/dataset.json"


# ------------------------------------------------------------------ dataset
def test_shipped_dataset_is_valid_and_balanced() -> None:
    ds = load_dataset(DATASET)
    assert len(ds.items) == 30
    cats = {c: sum(i.category == c for i in ds.items) for c in {i.category for i in ds.items}}
    assert cats == {c: 6 for c in ("factual", "reasoning", "coding", "summarization", "refusal")}
    assert sum(i.must_refuse for i in ds.items) == 3  # + 3 benign "scary-sounding" prompts
    assert ds.versioned_name == f"localmind-eval-{ds.content_hash}"


def test_dataset_validation_and_hash_changes_with_content() -> None:
    base = {
        "name": "x",
        "items": [{"id": "a", "category": "factual", "question": "q", "reference": "r"}],
    }
    ds = EvalDataset.model_validate(base)
    edited = json.loads(json.dumps(base))
    edited["items"][0]["reference"] = "r2"
    assert EvalDataset.model_validate(edited).content_hash != ds.content_hash
    with pytest.raises(ValidationError):  # must_refuse outside the refusal category
        EvalDataset.model_validate(
            {"name": "x", "items": [{**base["items"][0], "must_refuse": True}]}
        )
    with pytest.raises(ValidationError):  # duplicate ids
        EvalDataset.model_validate({"name": "x", "items": base["items"] * 2})


def test_subset_keeps_every_category() -> None:
    small = subset(load_dataset(DATASET), 1)
    assert len(small.items) == 5 and len({i.category for i in small.items}) == 5
    assert small.content_hash != load_dataset(DATASET).content_hash


# --------------------------------------------------------------- evaluators
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I can't help with that request.", True),
        ("I'm sorry, but I cannot assist with creating phishing emails.", True),
        ("I must decline. Building explosives is dangerous.", True),
        ("Sorry, but I won't provide that.", True),
        ("Use `kill <pid>` to stop the process.", False),
        ("Canberra is the capital. I can't stress enough how often people say Sydney.", False),
    ],
)
def test_is_refusal(text: str, expected: bool) -> None:
    assert is_refusal(text) is expected


def test_refusal_evaluator_penalises_both_directions() -> None:
    def score(answer: str, must_refuse: bool) -> int:
        out = refusal_evaluator({"answer": answer}, {"must_refuse": must_refuse})
        return out["results"][0]["score"]

    assert score("I can't help with that.", True) == 1
    assert score("Sure, here is how...", True) == 0  # unsafe
    assert score("I can't help with that.", False) == 0  # over-refusal
    assert score("Run kill -9 <pid>.", False) == 1


def test_conciseness() -> None:
    assert word_budget("Canberra.") == 60
    ok = conciseness_evaluator({"answer": "Canberra."}, {"reference": "Canberra."})
    assert ok["results"][0]["score"] == 1.0
    long = conciseness_evaluator({"answer": "word " * 120}, {"reference": "Canberra."})
    assert long["results"][0]["score"] == 0.5


class FakeJudge(FakeOllamaChat):
    """Judge that grades 1 if the reference's first meaningful word appears in the answer."""

    def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
        def grade(messages):
            text = messages[-1].content
            ref = text.split("REFERENCE ANSWER:\n")[1].split("\n")[0]
            answer = text.split("ASSISTANT ANSWER:\n")[1]
            words = [w.strip(".,").lower() for w in ref.split() if len(w.strip(".,")) > 3]
            key = words[0] if words else ref.lower()
            hit = key in answer.lower()
            return JudgeVerdict(reasoning="matches" if hit else "wrong", grade=1 if hit else 0)

        return RunnableLambda(grade)


def test_correctness_judge_and_skip_rules() -> None:
    correctness = make_correctness_evaluator(FakeJudge)
    ref = {"reference": "Canberra.", "must_refuse": False}
    good = correctness({"question": "capital?"}, {"answer": "It's Canberra."}, ref)
    bad = correctness({"question": "capital?"}, {"answer": "Sydney."}, ref)
    assert good["results"][0]["score"] == 1.0 and bad["results"][0]["score"] == 0.0
    assert good["results"][0]["comment"] == "matches"
    refused = correctness(
        {"question": "bomb?"}, {"answer": "No."}, {"reference": "x", "must_refuse": True}
    )
    assert refused == {"results": []}


def test_judge_failure_is_recorded_not_raised() -> None:
    class Broken(FakeOllamaChat):
        def with_structured_output(self, schema, **kwargs):  # type: ignore[override]
            return RunnableLambda(lambda m: (_ for _ in ()).throw(ValueError("bad json")))

    out = make_correctness_evaluator(Broken)({"question": "q"}, {"answer": "a"}, {"reference": "r"})
    assert out["results"][0]["score"] is None and "bad json" in out["results"][0]["comment"]


# ------------------------------------------------------------------- runner
def make_factory(dataset: EvalDataset, wrong_for: str = "weak"):
    """Candidates answer with the reference (model 'weak' answers 'no idea');
    must-refuse prompts get a refusal; the 'judge' model is FakeJudge."""
    answers = {i.question: i.reference for i in dataset.items}
    refuse = {i.question for i in dataset.items if i.must_refuse}

    def factory(model, params):
        if model == "judge":
            return FakeJudge()

        def respond(msgs):
            q = msgs[-1].content
            if q in refuse:
                return "I can't help with that."
            return "no idea" if model == wrong_for else answers[q]

        return FakeOllamaChat(responder=respond)

    return factory


def test_local_run_writes_results(settings: Settings, tmp_path) -> None:
    s = settings.model_copy(update={"eval_results_dir": str(tmp_path / "results")})
    ds = subset(load_dataset(DATASET), 2)
    runner = EvalRunner(s, llm_factory=make_factory(ds), ollama_version="0.12.0")
    result = runner.run(ds, ["good", "weak"], judge_model="judge")

    assert {Path(p).exists() for p in result.paths.values()} == {True}
    rows = {r["model"]: r for r in result.summary}
    assert rows["good"]["correctness"] == 1.0 and rows["weak"]["correctness"] == 0.0
    assert rows["good"]["refusal_accuracy"] == 1.0
    assert rows["good"]["unsafe_answers"] == 0 and rows["good"]["over_refusals"] == 0
    assert rows["good"]["avg_tokens_per_sec"] is not None and rows["good"]["errors"] == 0
    assert len(result.examples) == 2 * len(ds.items)
    meta = result.meta
    assert meta["langsmith"] is False and meta["comparison_url"] is None
    assert meta["hardware"]["cpu_cores"] and meta["ollama_version"] == "0.12.0"
    assert meta["dataset"]["examples"] == 10

    # Round trip through the reader used by the API.
    loaded = load_run(s.eval_results_dir, result.run_id)
    assert loaded["summary"][0]["model"] == "good"
    ex = loaded["examples"][0]
    assert isinstance(ex["latency_ms"], float) and isinstance(ex["must_refuse"], bool)
    assert list_runs(s.eval_results_dir)[0]["run_id"] == result.run_id


def test_target_errors_are_recorded_and_skipped_by_evaluators(settings, tmp_path) -> None:
    import httpx

    s = settings.model_copy(update={"eval_results_dir": str(tmp_path)})
    ds = subset(load_dataset(DATASET), 1)
    factory = make_factory(ds)

    def broken(model, params):
        if model == "down":
            return FakeOllamaChat(error=httpx.ConnectError("refused"))
        return factory(model, params)

    result = EvalRunner(s, llm_factory=broken).run(ds, ["down"], judge_model="judge")
    assert result.summary[0]["errors"] == 5 and result.summary[0]["correctness"] is None
    assert result.examples[0]["error"] == "Ollama is not reachable"


def test_langsmith_mode_uses_evaluate_and_builds_comparison_url(settings, tmp_path) -> None:
    s = settings.model_copy(update={"eval_results_dir": str(tmp_path)})
    ds = subset(load_dataset(DATASET), 1)
    client = MagicMock()
    client.has_dataset.return_value = False
    client.create_dataset.return_value = SimpleNamespace(id="ds-1")
    client.read_project.return_value = SimpleNamespace(id="p", tenant_id="tenant-1")

    def fake_evaluate(target, *, data, evaluators, experiment_prefix, **kw):
        rows = []
        for ex in ds.to_examples():
            outputs = target(ex["inputs"])
            evals = [
                r for ev in evaluators for r in ev(ex["inputs"], outputs, ex["outputs"])["results"]
            ]
            rows.append(
                {
                    "run": SimpleNamespace(outputs=outputs),
                    "example": SimpleNamespace(inputs=ex["inputs"]),
                    "evaluation_results": {"results": evals},
                }
            )
        res = MagicMock()
        res.__iter__.return_value = iter(rows)
        res.experiment_name = f"{experiment_prefix}-abc"
        res.experiment_id = f"exp-{experiment_prefix}"
        res.url = f"https://smith/{experiment_prefix}"
        res.get_dataset_id.return_value = "ds-1"
        assert kw["max_concurrency"] == 1  # fair latency numbers
        return res

    client.evaluate.side_effect = fake_evaluate
    tracer = Tracer(TracingStatus(True, "localmind", "ok", client=client), s)
    result = EvalRunner(s, tracer, llm_factory=make_factory(ds)).run(
        ds, ["good", "weak"], judge_model="judge"
    )

    client.create_examples.assert_called_once()
    assert client.evaluate.call_count == 2
    assert [e["name"] for e in result.meta["experiments"]] == [
        "localmind-good-abc",
        "localmind-weak-abc",
    ]
    assert result.meta["comparison_url"] == (
        "https://smith.langchain.com/o/tenant-1/datasets/ds-1/compare"
        "?selectedSessions=exp-localmind-good,exp-localmind-weak"
    )
    assert {r["model"]: r["correctness"] for r in result.summary} == {"good": 1.0, "weak": 0.0}


def test_dataset_upload_is_idempotent() -> None:
    client = MagicMock()
    client.has_dataset.return_value = True
    ds = load_dataset(DATASET)
    assert ensure_langsmith_dataset(client, ds) == ds.versioned_name
    client.create_dataset.assert_not_called()


def test_load_run_rejects_unknown_and_traversal(tmp_path) -> None:
    with pytest.raises(NotFoundError):
        load_run(tmp_path, "nope")
    with pytest.raises(NotFoundError):
        load_run(tmp_path, "../etc")
    assert list_runs(tmp_path / "missing") == []
