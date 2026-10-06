# scripts/train_full_predictors_without_consensus.py
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.lightweight_selection import no_consensus_predictor_name, uses_consensus_features
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    CalibratedOrdinalPredictorRunner,
    HitAtOnePredictorRunner,
    InTopMaxDepthPredictorRunner,
    TunedQppPredictorRunner,
)
from query_difficulty.retrieval import retriever_document_scope
from query_difficulty.runtime import build_runtime


def _families(runtime) -> list[tuple[str, str, object]]:
    families = [
        ("hit_at_1", "feature_importances.parquet", lambda: HitAtOnePredictorRunner(
            runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)),
        ("in_top_max_depth", "feature_importances.parquet", lambda: InTopMaxDepthPredictorRunner(
            runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval,
            predictor_name=no_consensus_predictor_name("in_top_max_depth"), threshold_objective="mcc")),
        ("calibrated_ordinal", "solvability_importances.parquet", lambda: CalibratedOrdinalPredictorRunner(
            runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)),
        ("tuned_qpp", "failure_importances.parquet", lambda: TunedQppPredictorRunner(
            runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)),
    ]
    for depth in sorted({int(size) for size in runtime.config.candidate_sizes}):
        families.append(
            (f"in_top_k{depth}", "feature_importances.parquet", lambda depth=depth: InTopMaxDepthPredictorRunner(
                runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval,
                predictor_name=no_consensus_predictor_name(f"in_top_k{depth}"), threshold_objective="mcc",
                target_depth=depth))
        )
    return families


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: list[str] = typer.Option(
        ["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to consider. Repeat for multiple."
    ),
    family: list[str] = typer.Option(
        None, "--family", help="Train only these predictor families (e.g. tuned_qpp, in_top_k5). Defaults to all."
    ),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.config.predictor_feature_modes = [PREDICTOR_MODE_LOCAL]
    console = runtime.console
    store_root = runtime.store.root_dir
    console.section("Full predictors without multi-retriever consensus features")
    console.log(
        f"The six multi-retriever consensus features {sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)} are removed "
        "from every feature frame before training. The resulting full models are the importance source for the "
        "lightweight feature selection, so the selection ranks only features computable from the target retriever "
        "and BM25. The reported full models (with consensus features) are not modified; the new models are written "
        "to '<family>_no_consensus' namespaces. Retrievers whose full model contains no consensus feature are "
        "skipped, since their full model already is consensus-free."
    )
    families = _families(runtime)
    if family:
        unknown = sorted(set(family) - {name for name, _, _ in families})
        if unknown:
            raise typer.BadParameter(f"Unknown families {unknown}; available: {[name for name, _, _ in families]}")
        families = [entry for entry in families if entry[0] in family]
    plan = []
    for family, importances_file, factory in families:
        for retriever_name in retriever:
            resolved = runtime.retrieval.resolve_retriever_name(retriever_name)
            full_path = store_root / "models" / family / resolved / importances_file
            if not full_path.exists():
                raise FileNotFoundError(f"Missing full {family} model importances for {resolved}: {full_path}")
            if uses_consensus_features(full_path):
                plan.append((family, importances_file, factory, resolved))
            else:
                console.log(f"{family} / {resolved}: full model has no consensus feature, nothing to retrain")
    console.log(f"Planned trainings ({len(plan)}): {[(family, name) for family, _, _, name in plan]}")

    summary: dict[str, dict] = {}
    for index, (family, importances_file, factory, retriever_name) in enumerate(plan, start=1):
        target = no_consensus_predictor_name(family)
        console.section(f"[{index}/{len(plan)}] {target} / {retriever_name}")
        started = time.perf_counter()
        runner = factory()
        runner.predictor_name = target
        runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
        if hasattr(runner, "_set_enabled_document_scopes"):
            runner._set_enabled_document_scopes([retriever_document_scope(retriever_name)])
        result = runner._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
        metrics = result[0] if isinstance(result, tuple) else result
        new_importances = store_root / "models" / target / retriever_name / importances_file
        if not new_importances.exists():
            raise FileNotFoundError(f"Training finished but {new_importances} was not written")
        if uses_consensus_features(new_importances):
            raise RuntimeError(f"{new_importances} still contains consensus features; the exclusion did not apply")
        elapsed = time.perf_counter() - started
        console.log(
            f"[{index}/{len(plan)}] {target} / {retriever_name} done in {elapsed / 60:.1f} min; importances at "
            f"{new_importances} contain no consensus feature; dev metrics: {json.dumps(metrics)}"
        )
        summary.setdefault(target, {})[retriever_name] = metrics
    summary_name = "summary.json" if not family else f"summary_{'_'.join(sorted(family))}.json"
    summary_path = runtime.store.path("predictions", "no_consensus_full_models", language, summary_name)
    runtime.store.write_json(summary_path, summary)
    console.log(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    typer.run(main)
