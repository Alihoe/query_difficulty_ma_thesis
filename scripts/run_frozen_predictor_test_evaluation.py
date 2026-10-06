# scripts/run_frozen_predictor_test_evaluation.py
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from time import perf_counter

import typer

ROOT = Path(__file__).resolve().parents[1]

SCORE_LIGHTWEIGHT = ROOT / "scripts" / "score_frozen_lightweight_in_top_k_test.py"
SCORE_FULL = ROOT / "scripts" / "score_frozen_full_predictors_test.py"
BASELINES = ROOT / "scripts" / "compute_predictor_baselines.py"
EVALUATE = ROOT / "scripts" / "evaluate_frozen_predictors_test.py"
TEST_TABLES = ROOT / "latex" / "scripts" / "write_frozen_predictor_test_tables.py"
DEV_TABLES = ROOT / "latex" / "scripts" / "write_lightweight_tables.py"

STEPS = [
    ("Score missing lightweight in-top-k test predictions with frozen models (BM25, Qwen)", SCORE_LIGHTWEIGHT, []),
    ("Score missing full-predictor test predictions with frozen models (BM25, Qwen)", SCORE_FULL, ["--no-include-other-hit-at-1-retrievers"]),
    ("Compute majority-class and score-margin baselines from cached rankings", BASELINES, []),
    ("Evaluate frozen full and lightweight predictors on the test split", EVALUATE, []),
    ("Write LaTeX tables for the test-set evaluation", TEST_TABLES, []),
    ("Rewrite the dev lightweight tables with baseline columns", DEV_TABLES, []),
    ("Score full Hit@1 for the six other dense retrievers with frozen models", SCORE_FULL, []),
    ("Re-evaluate with the additional full Hit@1 predictors", EVALUATE, []),
    ("Rewrite the LaTeX tables for the test-set evaluation", TEST_TABLES, []),
]


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    overall_started_at = perf_counter()
    for index, (title, script, extra_arguments) in enumerate(STEPS, start=1):
        print(f"\n=== Step {index}/{len(STEPS)}: {title}")
        print(f"    {script.relative_to(ROOT)} {' '.join(extra_arguments)}".rstrip())
        started_at = perf_counter()
        result = subprocess.run(
            [sys.executable, str(script), "--language", language, "--config", config, *extra_arguments],
            cwd=ROOT,
        )
        elapsed = perf_counter() - started_at
        if result.returncode != 0:
            print(f"=== Step {index}/{len(STEPS)} failed with exit code {result.returncode} after {elapsed:.1f}s; stopping.")
            raise typer.Exit(code=result.returncode)
        print(f"=== Step {index}/{len(STEPS)} finished in {elapsed:.1f}s")
    print(f"\nAll {len(STEPS)} steps finished in {perf_counter() - overall_started_at:.1f}s")


if __name__ == "__main__":
    typer.run(main)
