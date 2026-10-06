from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.modeling import InTopMaxDepthPredictorRunner
from query_difficulty.runtime import build_runtime


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: list[str] = typer.Option(None, "--retriever", help="Run only the named retriever. Repeat for multiple retrievers."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("In-top-max-depth predictor")
    runtime.console.log(f"Loading project configuration from {config}")
    runtime.console.log(f"Running in-top-max-depth predictor for {language}")
    runtime.console.log(
        "Selecting threshold via cross-validation on train only: cutoff maximizing MCC (Matthews correlation "
        "coefficient) on a class_weight=balanced random forest, evaluated once on dev without further tuning. "
        "MCC is used instead of F1 because the positive class (in_top_max_depth=True) is highly imbalanced, "
        "which inflates F1 relative to actual discriminative skill on the minority class."
    )
    runner = InTopMaxDepthPredictorRunner(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features,
        runtime.retrieval,
        threshold_objective="mcc",
    )
    summary = runner.run(language, retrievers=retriever or None)
    summary_path = runtime.store.path("predictions", "in_top_max_depth", language, "summary.json")
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved in-top-max-depth summary to {summary_path}")
    runtime.console.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
