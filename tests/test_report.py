import pytest

from localmind.evaluation.report import END, START, render_benchmark, replace_between_markers

RUN = {
    "meta": {
        "run_id": "20260101-000000",
        "dataset": {"name": "localmind-eval-abc", "examples": 30},
        "judge_model": "qwen2.5:7b",
        "params": {"temperature": 0.0},
        "hardware": {"cpu": "Test CPU", "cpu_cores": 8, "ram_gb": 16.0, "gpu": None, "os": "Linux"},
        "ollama_version": "0.12.0",
        "comparison_url": "https://smith/compare",
    },
    "summary": [
        {
            "model": "llama3.2", "correctness": 0.8, "correctness_coding": 0.5,
            "correctness_factual": 1.0, "conciseness": 0.95, "refusal_accuracy": 1.0,
            "unsafe_answers": 0, "over_refusals": 0, "avg_latency_s": 2.345,
            "p50_latency_s": 2.1, "avg_ttft_ms": 310.0, "avg_tokens_per_sec": 28.44,
        }
    ],
}  # fmt: skip


def test_render_benchmark_uses_run_values_only() -> None:
    md = render_benchmark(RUN)
    assert "| `llama3.2` | **80%** | 50% | 100% | 95% | 100% | 0 | 0 |" in md
    assert "| `llama3.2` | 2.35 s | 2.10 s | 310 ms | 28.4 |" in md
    assert "| CPU | Test CPU (8 cores) |" in md
    assert "none detected (CPU inference)" in md
    assert "https://smith/compare" in md and "20260101-000000" in md


def test_replace_between_markers() -> None:
    doc = f"intro\n{START}\nTBD\n{END}\noutro"
    out = replace_between_markers(doc, "NEW")
    assert out == f"intro\n{START}\nNEW\n{END}\noutro"
    with pytest.raises(ValueError):
        replace_between_markers("no markers", "x")
