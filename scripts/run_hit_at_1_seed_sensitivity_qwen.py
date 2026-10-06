# scripts/run_hit_at_1_seed_sensitivity_qwen.py
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
)
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    _hit_at_1_metrics,
    _hit_at_1_target,
    _positive_class_probabilities,
    _prediction_output_paths,
    _prediction_split_output_paths,
)
from query_difficulty.modeling_hit_at_1_lightweight import (
    LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
    LightweightHitAtOnePredictorRunner,
)
from query_difficulty.retrieval import retriever_document_scope
from query_difficulty.runtime import build_runtime

REPORTED_METRICS = [
    "f0_5",
    "precision",
    "recall",
    "f1",
    "positive_predictions",
    "true_positives",
    "false_positives",
]


def _split_metrics(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    output = pd.DataFrame(
        {
            "hit_at_1": _hit_at_1_target(frame).astype(bool).to_numpy(),
            "predicted_hit_at_1": np.asarray(probabilities, dtype=np.float64) >= float(threshold),
        }
    )
    return _hit_at_1_metrics(output)


def _summary_row(values: pd.Series, deployed_value: float) -> dict[str, float]:
    std = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return {
        "mean": float(values.mean()),
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "deployed_seed_value": float(deployed_value),
        "deployed_minus_mean": float(deployed_value - values.mean()),
        "deployed_z": float((deployed_value - values.mean()) / std) if std and std > 0 else math.nan,
    }


def _same(left: float, right: float) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest: str = typer.Option(
        "config/lightweight_hit_at_1_features.json",
        help="Path to the frozen lightweight feature manifest.",
    ),
    retriever: str = typer.Option("qwen3_embedding_0_6b", help="Retriever whose lightweight hit@1 predictor is analysed."),
    seed_count: int = typer.Option(10, help="Number of consecutive random seeds, starting at runtime.random_seed."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.config.predictor_feature_modes = [PREDICTOR_MODE_LOCAL]
    console = runtime.console
    deployed_seed = int(runtime.config.runtime.random_seed)
    seeds = list(range(deployed_seed, deployed_seed + seed_count))
    console.section("Hit@1 seed sensitivity (lightweight)")
    console.log(f"Loading project configuration from {config}")
    console.log(f"Using frozen feature manifest at {manifest}")
    console.log(
        f"Seeds: {seeds} (deployed seed {deployed_seed} plus the next {seed_count - 1} consecutive seeds). "
        "Only the random forest random_state changes; the feature set, the sigmoid calibration folds and the "
        "dev F0.5 threshold policy are identical to the deployed predictor. The threshold is re-selected on dev "
        "for each seed and applied unchanged to test. Deployed models and predictions are not modified."
    )

    features = LightweightHitAtOnePredictorFeatureBuilder(
        runtime.config,
        runtime.store,
        console,
        runtime.features.bundle,
        runtime.retrieval,
        runtime.clustering,
        runtime.text_services,
        manifest_path=manifest,
        consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
    )
    runner = LightweightHitAtOnePredictorRunner(
        runtime.config,
        runtime.store,
        console,
        features,
        runtime.retrieval,
        predictor_name=LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
    )
    runner.compute_signal_summaries = False
    retriever_name = runtime.retrieval.resolve_retriever_name(retriever)
    if not features.has_manifest_entry(retriever_name):
        raise KeyError(f"No lightweight hit@1 manifest entry for {retriever_name} in {manifest}")
    qualifying = sorted(features.qualifying_features(retriever_name))
    console.log(f"{retriever_name}: {len(qualifying)} frozen lightweight features: {qualifying}")

    runner._set_enabled_document_scopes([retriever_document_scope(retriever_name)])
    feature_depth = runner._feature_depth()
    frames = {}
    for split in ("train", "dev", "test"):
        console.log(f"Building {language}_{split} lightweight hit@1 frame for {retriever_name} at depth {feature_depth}")
        frames[split] = runner.views.build(
            f"{language}_{split}",
            retriever_name,
            PREDICTOR_MODE_LOCAL,
            candidate_sizes=[feature_depth],
        )[feature_depth]
        positives = int(_hit_at_1_target(frames[split]).sum())
        console.log(f"    {split}: {len(frames[split])} queries, {positives} hit@1 positives")

    per_seed_rows = []
    importance_rows = []
    started_all = time.perf_counter()
    for index, seed in enumerate(seeds, start=1):
        started = time.perf_counter()
        console.section(f"[{index}/{len(seeds)}] seed {seed}")
        runtime.config.runtime.random_seed = seed
        _, dev_metrics, threshold, importances, probability_model, calibration_info = runner._fit_and_predict(
            f"{retriever_name} (seed {seed})",
            frames["train"],
            frames["dev"],
        )
        test_probabilities = _positive_class_probabilities(probability_model, frames["test"])
        test_metrics = _split_metrics(frames["test"], test_probabilities, threshold)
        row = {"seed": seed, "threshold": float(threshold), "calibration": calibration_info.get("method")}
        for metric in REPORTED_METRICS:
            row[f"dev_{metric}"] = float(dev_metrics[metric])
            row[f"test_{metric}"] = float(test_metrics[metric])
        per_seed_rows.append(row)
        ranked = importances.reset_index(drop=True)
        for rank, record in enumerate(ranked.itertuples(index=False), start=1):
            importance_rows.append({"seed": seed, "feature": record.feature, "importance": float(record.importance), "rank": rank})
        console.log(
            f"seed {seed}: threshold={threshold:.6f} | dev f0_5={dev_metrics['f0_5']:.4f} "
            f"precision={dev_metrics['precision']:.4f} recall={dev_metrics['recall']:.4f} | "
            f"test f0_5={test_metrics['f0_5']:.4f} precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} | {time.perf_counter() - started:.1f}s "
            f"(elapsed {(time.perf_counter() - started_all) / 60:.1f} min)"
        )
    runtime.config.runtime.random_seed = deployed_seed

    per_seed = pd.DataFrame(per_seed_rows)
    importance_frame = pd.DataFrame(importance_rows)
    deployed_row = per_seed.loc[per_seed["seed"].eq(deployed_seed)].iloc[0]

    console.section(f"Reproducibility check against the deployed seed {deployed_seed} artifacts")
    reproducibility = {}
    deployed_paths = _prediction_output_paths(runtime.store, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language, retriever_name)
    if deployed_paths["metadata"].exists():
        deployed_metadata = runtime.store.read_json(deployed_paths["metadata"])
        deployed_dev = deployed_metadata.get("dev_metrics", {})
        dev_matches = {metric: _same(deployed_row[f"dev_{metric}"], float(deployed_dev[metric])) for metric in REPORTED_METRICS if metric in deployed_dev}
        threshold_match = _same(deployed_row["threshold"], float(deployed_metadata.get("threshold", math.nan)))
        reproducibility["dev"] = {"metrics_match": dev_matches, "threshold_match": threshold_match}
        console.log(f"dev metrics reproduced: {all(dev_matches.values())} {dev_matches}; threshold reproduced: {threshold_match}")
    else:
        console.log(f"No deployed dev metadata at {deployed_paths['metadata']}; skipping dev reproducibility check")
    test_paths = _prediction_split_output_paths(runtime.store, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language, retriever_name, "test")
    if test_paths["prediction"].exists():
        deployed_test = _hit_at_1_metrics(runtime.store.read_parquet(test_paths["prediction"]))
        test_matches = {metric: _same(deployed_row[f"test_{metric}"], float(deployed_test[metric])) for metric in REPORTED_METRICS}
        reproducibility["test"] = {"metrics_match": test_matches}
        console.log(f"test metrics reproduced: {all(test_matches.values())} {test_matches}")
    else:
        console.log(f"No deployed test predictions at {test_paths['prediction']}; skipping test reproducibility check")

    console.section("Summary across seeds")
    summary_rows = []
    for column in ["threshold"] + [f"{split}_{metric}" for split in ("dev", "test") for metric in REPORTED_METRICS]:
        summary_rows.append({"quantity": column, **_summary_row(per_seed[column], float(deployed_row[column]))})
    summary = pd.DataFrame(summary_rows)
    for record in summary.itertuples(index=False):
        console.log(
            f"{record.quantity:28s} mean={record.mean:.4f} std={record.std:.4f} min={record.min:.4f} "
            f"max={record.max:.4f} range={record.range:.4f} seed{deployed_seed}={record.deployed_seed_value:.4f} "
            f"(z={record.deployed_z:.2f})"
        )

    console.section("Feature importance stability across seeds")
    importance_summary = (
        importance_frame.groupby("feature")
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", lambda values: float(values.std(ddof=1)) if len(values) > 1 else math.nan),
            rank_mean=("rank", "mean"),
            rank_min=("rank", "min"),
            rank_max=("rank", "max"),
        )
        .reset_index()
        .sort_values("importance_mean", ascending=False)
    )
    rank_matrix = importance_frame.pivot(index="feature", columns="seed", values="rank")
    rank_correlations = rank_matrix.corr(method="spearman").to_numpy()
    upper = rank_correlations[np.triu_indices_from(rank_correlations, k=1)]
    rank_stability = {
        "pairwise_spearman_mean": float(np.mean(upper)) if upper.size else math.nan,
        "pairwise_spearman_min": float(np.min(upper)) if upper.size else math.nan,
        "pairwise_spearman_max": float(np.max(upper)) if upper.size else math.nan,
    }
    for record in importance_summary.itertuples(index=False):
        console.log(
            f"{record.importance_mean:.4f} ± {record.importance_std:.4f}  rank {record.rank_mean:.1f} "
            f"[{record.rank_min}-{record.rank_max}]  {record.feature}"
        )
    console.log(
        f"Pairwise Spearman correlation of feature-importance ranks across seeds: "
        f"mean={rank_stability['pairwise_spearman_mean']:.4f} min={rank_stability['pairwise_spearman_min']:.4f} "
        f"max={rank_stability['pairwise_spearman_max']:.4f}"
    )

    output_dir = runtime.store.path("reports", "hit_at_1_seed_sensitivity", language, retriever_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_across_seeds.csv", index=False)
    importance_frame.to_csv(output_dir / "per_seed_feature_importances.csv", index=False)
    importance_summary.to_csv(output_dir / "feature_importance_summary.csv", index=False)
    runtime.store.write_json(
        output_dir / "summary.json",
        {
            "predictor": LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
            "retriever": retriever_name,
            "language": language,
            "manifest": manifest,
            "qualifying_features": qualifying,
            "feature_depth": feature_depth,
            "seeds": seeds,
            "deployed_seed": deployed_seed,
            "varied": "random forest random_state only (threshold re-selected on dev per seed)",
            "reproducibility": reproducibility,
            "summary": summary.to_dict(orient="records"),
            "feature_rank_stability": rank_stability,
        },
    )
    console.log(f"Saved seed-sensitivity outputs to {output_dir}")
    console.log(json.dumps(rank_stability, indent=2))


if __name__ == "__main__":
    typer.run(main)
