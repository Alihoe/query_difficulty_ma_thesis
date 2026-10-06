from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.modeling import InTopMaxDepthPredictorRunner
from query_difficulty.runtime import build_runtime

SWEEP_PREDICTOR_NAME = "in_top_k_sweep"
RETRIEVER_PRIORITY = ["bm25", "qwen3_embedding_0_6b", "bge_m3"]


def _retriever_major_order(retrievers: list[str]) -> list[str]:
    priority_index = {name: index for index, name in enumerate(RETRIEVER_PRIORITY)}
    return sorted(
        retrievers,
        key=lambda name: (priority_index.get(name, len(RETRIEVER_PRIORITY)), retrievers.index(name)),
    )


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: list[str] = typer.Option(None, "--retriever", help="Run only the named retriever. Repeat for multiple retrievers."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.config.predictor_feature_modes = ["local"]
    runtime.console.section("In-top-k sweep predictor")
    runtime.console.log(f"Loading project configuration from {config}")
    depths = sorted({int(size) for size in runtime.config.candidate_sizes})
    selected_retrievers = retriever or list(runtime.retrieval.enabled_retrievers())
    ordered_retrievers = _retriever_major_order(selected_retrievers)
    runtime.console.log(
        f"Running in-top-k predictors for {language} in retriever-major order {ordered_retrievers}, "
        f"each across candidate sizes {depths}, local feature mode only"
    )
    runtime.console.log(
        "Each depth reuses the same unconstrained-MCC threshold policy as the standalone in_top_max_depth "
        "predictor (5-fold cross-validation on train, class_weight=balanced random forest), just retargeted "
        "to predict gold_rank <= k instead of gold_rank <= max_ranking_depth."
    )
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for retriever_name in ordered_retrievers:
        for depth in depths:
            runtime.console.section(f"{retriever_name}: in-top-k={depth} predictor")
            runner = InTopMaxDepthPredictorRunner(
                runtime.config,
                runtime.store,
                runtime.console,
                runtime.features,
                runtime.retrieval,
                predictor_name=f"in_top_k{depth}",
                threshold_objective="mcc",
                target_depth=depth,
            )
            metrics = runner.run(language, retrievers=[retriever_name])
            summary.setdefault(str(depth), {}).update(metrics)
    summary_path = runtime.store.path("predictions", SWEEP_PREDICTOR_NAME, language, "summary.json")
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved in-top-k sweep summary to {summary_path}")
    runtime.console.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
