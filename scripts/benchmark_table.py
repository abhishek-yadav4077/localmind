"""Turn a saved evaluation run into the README benchmark tables.

    python scripts/benchmark_table.py                 # print Markdown for the latest run
    python scripts/benchmark_table.py --run 20261103-101500
    python scripts/benchmark_table.py --write         # update README.md in place

Only real runs from eval/results are used, so README numbers are never hand-typed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from localmind.config import get_settings
from localmind.evaluation.report import render_benchmark, replace_between_markers
from localmind.evaluation.results import list_runs, load_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="run id (default: latest)")
    parser.add_argument("--write", action="store_true", help="update README.md")
    args = parser.parse_args()

    results_dir = get_settings().eval_results_dir
    runs = list_runs(results_dir)
    if not runs:
        print("No runs in eval/results. Run: python scripts/run_eval.py")
        return 1
    run = load_run(results_dir, args.run or runs[0]["run_id"])
    markdown = render_benchmark(run)
    if not args.write:
        print(markdown)
        return 0
    readme = Path("README.md")
    readme.write_text(
        replace_between_markers(readme.read_text(encoding="utf-8"), markdown), encoding="utf-8"
    )
    print(f"README.md updated from run {run['meta']['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
