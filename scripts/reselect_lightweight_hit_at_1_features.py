# scripts/reselect_lightweight_hit_at_1_features.py
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _lightweight_tuned_qpp_common import (
    features_to_drop_by_step_fraction,
    load_checkpoint,
    prune_redundant_features,
    resolve_repo_relative_output_path,
    save_checkpoint,
)

from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES,
    LightweightHitAtOnePredictorFeatureBuilder,
)
from query_difficulty.lightweight_selection import base_feature_name, selection_source_predictor
from query_difficulty.modeling import IDENTIFIER_COLUMNS, HitAtOnePredictorRunner
from query_difficulty.modeling_hit_at_1_lightweight import LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, LightweightHitAtOnePredictorRunner
from query_difficulty.runtime import build_runtime

PREDICTOR_NAME = "hit_at_1"
REPORT_TAG = "hit_at_1_reselected"
DEPLOYED_MANIFEST = "config/lightweight_hit_at_1_features.json"
MANIFEST_PREFIX = "config/lightweight_hit_at_1_features.reselected"
BACKUP_DIRECTORY = "_backup_hit_at_1_pre_reselection"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _bootstrap_f0_5_standard_error(output: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> float:
    actual = output["hit_at_1"].to_numpy(dtype=bool)
    predicted = output["predicted_hit_at_1"].to_numpy(dtype=bool)
    n = len(actual)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual, resampled_predicted = actual[indices], predicted[indices]
    tp = np.sum(resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    fp = np.sum(~resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    fn = np.sum(resampled_actual & ~resampled_predicted, axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros(n_bootstrap), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(n_bootstrap), where=(tp + fn) > 0)
    denominator = 0.25 * precision + recall
    f0_5 = np.divide(1.25 * precision * recall, denominator, out=np.zeros(n_bootstrap), where=denominator > 0)
    valid = ((tp + fp) > 0) & ((tp + fn) > 0)
    values = f0_5[valid]
    return float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan


def _freeze(console, store_root: Path, retriever_name: str, threshold: float) -> dict:
    source = selection_source_predictor(store_root, PREDICTOR_NAME, retriever_name, "feature_importances.parquet")
    frame = pd.read_parquet(store_root / "models" / source / retriever_name / "feature_importances.parquet")
    frame = frame.loc[~frame["feature"].map(base_feature_name).isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
    frame = frame.sort_values("importance", ascending=False)
    qualifying = frame.loc[frame["importance"] > threshold]
    console.log(
        f"{retriever_name}: importance source {source}; {len(qualifying)}/{len(frame)} features above {threshold} "
        f"(coverage {float(qualifying['importance'].sum()):.4f})"
    )
    return {
        "importance_source": source,
        "qualifying_features": qualifying["feature"].tolist(),
        "importance": dict(zip(frame["feature"], frame["importance"].astype(float))),
        "source_feature_count": int(len(frame)),
        "qualifying_feature_count": int(len(qualifying)),
    }


def _correlation_round(console, train_frame, features, importance, threshold):
    present = [feature for feature in features if feature in train_frame.columns]
    corr = train_frame[present].corr(method="spearman")
    pairs, kept = prune_redundant_features(corr, pd.Series(importance), threshold)
    for record in pairs:
        if record["dropped"] is not None:
            console.log(
                f"    rho={record['spearman_rho']:+.3f}: {record['feature_a']} ({record['importance_a']:.4f}) vs "
                f"{record['feature_b']} ({record['importance_b']:.4f}) -> drop {record['dropped']}"
            )
    console.log(f"    |rho|>={threshold:.2f}: {len(pairs)} pair(s), kept {len(kept)}/{len(present)}")
    return kept, pairs


def _rfe(runner, console, name, train_frame, dev_frame, features, rng, n_bootstrap, checkpoint_path, checkpoint, step_fraction):
    existing = checkpoint.get(name)
    if existing and existing.get("step_fraction") != step_fraction:
        console.log(f"  {name}: checkpoint was computed with a different RFE step rule, recomputing this path")
        existing = None
    if existing and existing.get("complete"):
        console.log(f"  {name}: already complete in checkpoint ({len(existing['path'])} steps)")
        return list(existing["path"])
    if existing and existing.get("path"):
        path, remaining = list(existing["path"]), list(existing["next_remaining"])
        console.log(f"  {name}: resuming at {len(remaining)} features ({len(path)} step(s) recorded)")
    else:
        path, remaining = [], list(features)
    while True:
        keep_columns = [column for column in train_frame.columns if column in IDENTIFIER_COLUMNS or column in remaining]
        output, metrics, threshold, importances, _model, _calibration = runner._fit_and_predict(
            f"{name} [RFE step {len(path) + 1}, {len(remaining)} features]",
            train_frame[keep_columns].copy(),
            dev_frame[keep_columns].copy(),
        )
        se = _bootstrap_f0_5_standard_error(output, rng, n_bootstrap)
        console.log(
            f"  [step {len(path) + 1}] {len(remaining)} features -> f0_5={metrics['f0_5']:.4f} (se={se:.4f}), "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, threshold={threshold:.4f}"
        )
        path.append(
            {
                "feature_count": len(remaining),
                "features": list(remaining),
                "f0_5": metrics["f0_5"],
                "f0_5_bootstrap_se": se,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "threshold": float(threshold),
            }
        )
        if len(remaining) <= 1:
            checkpoint[name] = {"path": path, "complete": True, "step_fraction": step_fraction}
            save_checkpoint(checkpoint_path, checkpoint)
            return path
        lookup = importances.set_index("feature")["importance"]
        weakest = features_to_drop_by_step_fraction(remaining, lookup, step_fraction)
        console.log(f"    dropping {len(weakest)} weakest remaining feature(s): {weakest}")
        remaining = [feature for feature in remaining if feature not in weakest]
        checkpoint[name] = {"path": path, "complete": False, "next_remaining": remaining, "step_fraction": step_fraction}
        save_checkpoint(checkpoint_path, checkpoint)


def _one_standard_error(path):
    best = max(path, key=lambda entry: entry["f0_5"])
    se = best["f0_5_bootstrap_se"]
    tolerance = best["f0_5"] - (se if math.isfinite(se) else 0.0)
    chosen = min((entry for entry in path if entry["f0_5"] >= tolerance), key=lambda entry: entry["feature_count"])
    return chosen, best, tolerance


def _dev_metrics(store_root: Path, predictor_name: str, language: str, retriever_name: str) -> dict | None:
    path = store_root / "predictions" / predictor_name / language / retriever_name / "metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("dev_metrics")


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    threshold: float = typer.Option(0.005, help="Minimum full-model feature importance for a feature to qualify."),
    correlation_threshold: list[float] = typer.Option(
        [0.80, 0.70], "--correlation-threshold", help="Absolute Spearman rho threshold per correlation round, in order."
    ),
    n_bootstrap: int = typer.Option(2000, help="Bootstrap resamples for the dev F0.5 standard error at every RFE step."),
    step_fraction: float = typer.Option(0.10, help="Fraction of the remaining features dropped per RFE step (minimum 1)."),
    retriever: list[str] = typer.Option(["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to process."),
    train_final_model: bool = typer.Option(True, help="Adopt the selected features and train the lightweight predictor."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    store_root = runtime.store.root_dir
    resolved = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
    console.section("Hit@1 lightweight feature reselection")
    console.log(
        "(1) importance freeze on the consensus-free full hit@1 model (importance > threshold), (2) Spearman "
        f"correlation pruning in rounds {correlation_threshold}, keeping the more important feature of each pair, "
        f"(3) RFE dropping {step_fraction:.0%} of the remaining features per step (minimum 1) with the one-standard-error rule on dev F0.5 (the objective of the "
        f"hit@1 threshold, {n_bootstrap} bootstrap resamples), (4) adopting the selection in {DEPLOYED_MANIFEST} for "
        f"{resolved} only and training the lightweight predictor."
    )
    previous = {name: _dev_metrics(store_root, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language, name) for name in resolved}
    full = {name: _dev_metrics(store_root, PREDICTOR_NAME, language, name) for name in resolved}

    console.section("Step 1: importance freeze")
    freeze = {name: _freeze(console, store_root, name, threshold) for name in resolved}
    runner = HitAtOnePredictorRunner(runtime.config, runtime.store, console, runtime.features, runtime.retrieval)
    runner.compute_signal_summaries = False
    feature_depth = runner._feature_depth()
    frames = {}
    for name in resolved:
        console.log(f"Building {language}_train and {language}_dev hit@1 frames for {name} at depth {feature_depth}")
        frames[name] = tuple(
            runtime.features.build_features(f"{language}_{split}", name, candidate_sizes=[feature_depth])[feature_depth]
            for split in ("train", "dev")
        )

    stages = {"freeze": {name: {"qualifying_features": freeze[name]["qualifying_features"], "importance_source": freeze[name]["importance_source"]} for name in resolved}}
    current = {name: list(freeze[name]["qualifying_features"]) for name in resolved}
    correlation_report = {name: {} for name in resolved}
    for round_index, round_threshold in enumerate(correlation_threshold, start=1):
        console.section(f"Step 2.{round_index}: correlation pruning |rho|>={round_threshold:.2f}")
        for name in resolved:
            console.log(f"  {name}: {len(current[name])} features")
            kept, pairs = _correlation_round(console, frames[name][0], current[name], freeze[name]["importance"], round_threshold)
            correlation_report[name][f"round{round_index}"] = {"threshold": round_threshold, "pairs": pairs, "kept": kept}
            current[name] = kept
        stages["correlation_pruned" if round_index == 1 else "correlation_pruned_round2"] = {
            name: {"qualifying_features": current[name]} for name in resolved
        }
    correlation_report_path = runtime.store.path("predictions", "lightweight_feature_correlation", REPORT_TAG, language, "correlation_report.json")
    if correlation_report_path.exists():
        correlation_report = {**runtime.store.read_json(correlation_report_path), **correlation_report}
    runtime.store.write_json(correlation_report_path, correlation_report)

    console.section(f"Step 3: RFE, {step_fraction:.0%} of the remaining features per step, one-standard-error rule on dev F0.5")
    rng = np.random.default_rng(runtime.config.runtime.random_seed)
    checkpoint_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    rfe_report = {}
    for name in resolved:
        console.section(f"RFE for {name}: {len(current[name])} starting features")
        path = _rfe(runner, console, name, frames[name][0], frames[name][1], current[name], rng, n_bootstrap, checkpoint_path, checkpoint, step_fraction)
        chosen, best, tolerance = _one_standard_error(path)
        console.log(
            f"{name}: best f0_5={best['f0_5']:.4f} at {best['feature_count']} features (tolerance>={tolerance:.4f}); "
            f"chosen {chosen['feature_count']}: {chosen['features']}"
        )
        rfe_report[name] = {"starting_feature_count": len(current[name]), "path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}
    rfe_report_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_report.json")
    stored_rfe_report = runtime.store.read_json(rfe_report_path) if rfe_report_path.exists() else {}
    runtime.store.write_json(rfe_report_path, {**stored_rfe_report, **rfe_report})
    stages["rfe_pruned"] = {
        name: {
            "qualifying_features": rfe_report[name]["chosen_entry"]["features"],
            "qualifying_feature_count": rfe_report[name]["chosen_entry"]["feature_count"],
            "importance_source": freeze[name]["importance_source"],
            "pipeline_stage": "rfe_pruned",
        }
        for name in resolved
    }
    for stage, retrievers in stages.items():
        stage_path = resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.{stage}.json")
        if stage_path.exists():
            retrievers = {**json.loads(stage_path.read_text(encoding="utf-8")).get("retrievers", {}), **retrievers}
        _write_json(
            stage_path,
            {
                "metadata_version": 1,
                "pipeline_stage": stage,
                "importance_threshold": threshold,
                "correlation_thresholds": correlation_threshold,
                "rfe_step_fraction": step_fraction,
                "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
                "retrievers": retrievers,
            },
        )

    if train_final_model:
        console.section(f"Step 4: adopt the selection in {DEPLOYED_MANIFEST} and train the lightweight predictor")
        deployed_path = resolve_repo_relative_output_path(DEPLOYED_MANIFEST)
        backup_root = store_root / BACKUP_DIRECTORY
        backup_manifest = backup_root / "config" / deployed_path.name
        if not backup_manifest.exists():
            backup_manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(deployed_path, backup_manifest)
            console.log(f"Backed up {deployed_path} -> {backup_manifest}")
        deployed = json.loads(deployed_path.read_text(encoding="utf-8"))
        for name in resolved:
            deployed["retrievers"][name] = {
                **stages["rfe_pruned"][name],
                "adopted_from": f"{MANIFEST_PREFIX}.rfe_pruned.json",
                "rfe_objective": "f0_5",
                "correlation_thresholds": correlation_threshold,
            }
            console.log(f"{name}: deployed lightweight features -> {stages['rfe_pruned'][name]['qualifying_features']}")
        _write_json(deployed_path, deployed)
        for kind in ("models", "predictions", "reports"):
            for name in resolved:
                source = store_root / kind / LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME / (name if kind == "models" else Path(language) / name)
                destination = backup_root / source.relative_to(store_root)
                if source.exists() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, destination)
                    console.log(f"Backed up {source} -> {destination}")
        features = LightweightHitAtOnePredictorFeatureBuilder(
            runtime.config, runtime.store, console, runtime.features.bundle, runtime.retrieval,
            runtime.clustering, runtime.text_services, manifest_path=str(deployed_path), consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
        )
        lightweight_runner = LightweightHitAtOnePredictorRunner(runtime.config, runtime.store, console, features, runtime.retrieval)
        lightweight_runner.run(language, retrievers=resolved)

    console.section("Summary (dev)")
    for name in resolved:
        new = _dev_metrics(store_root, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language, name) if train_final_model else None
        console.log(f"{name}: {rfe_report[name]['chosen_entry']['feature_count']} features {rfe_report[name]['chosen_entry']['features']}")
        for label, metrics in (("full model", full[name]), ("previous lightweight", previous[name]), ("new lightweight", new)):
            if metrics:
                console.log(
                    f"    {label:22s} f0_5={metrics['f0_5']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
                )


if __name__ == "__main__":
    typer.run(main)
