"""Read saved evaluation runs (for the API / Evaluation page)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from localmind.exceptions import NotFoundError


def _convert(value: str) -> Any:
    """CSV gives strings back; restore numbers, booleans and blanks."""
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value else number


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _convert(v) for k, v in row.items()} for row in csv.DictReader(f)]


def list_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    """Metadata of every run, newest first."""
    metas = []
    for path in Path(results_dir).glob("*_meta.json"):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(metas, key=lambda m: m.get("run_id", ""), reverse=True)


def load_run(results_dir: str | Path, run_id: str) -> dict[str, Any]:
    base = Path(results_dir)
    meta_path = base / f"{run_id}_meta.json"
    if "/" in run_id or ".." in run_id or not meta_path.exists():
        raise NotFoundError(f"Evaluation run {run_id} not found")
    return {
        "meta": json.loads(meta_path.read_text(encoding="utf-8")),
        "summary": _read_csv(base / f"{run_id}_summary.csv"),
        "examples": _read_csv(base / f"{run_id}_examples.csv"),
    }
