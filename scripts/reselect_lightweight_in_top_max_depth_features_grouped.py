# scripts/reselect_lightweight_in_top_max_depth_features_grouped.py
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

from _lightweight_in_top_max_depth_common import (
    bootstrap_mcc_standard_error,
    build_merged_frame,
    features_to_drop_by_step_fraction,
    load_checkpoint,
    prune_redundant_features,
    resolve_repo_relative_output_path,
    save_checkpoint,
    strip_depth_prefix,
)

from query_difficulty.analysis import in_top_max_depth_predictor_metric_summary
from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.lightweight_selection import selection_source_predictor
from query_difficulty.modeling import (
    IDENTIFIER_COLUMNS,
    IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
    IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
    InTopMaxDepthPredictorRunner,
)
from query_difficulty.modeling_in_top_max_depth_lightweight import (
    LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
    LightweightInTopMaxDepthPredictorRunner,
)
from query_difficulty.runtime import build_runtime

REPORT_TAG = "in_top_max_depth_grouped"
MANIFEST_METADATA_VERSION = 2
FREEZE_MANIFEST = "config/lightweight_in_top_max_depth_features.grouped.json"
CORRELATION_ROUND_1_MANIFEST = "config/lightweight_in_top_max_depth_features.grouped.correlation_pruned.json"
CORRELATION_ROUND_2_MANIFEST = "config/lightweight_in_top_max_depth_features.grouped.correlation_pruned_round2.json"
RFE_MANIFEST = "config/lightweight_in_top_max_depth_features.grouped.rfe_pruned.json"
BACKUP_DIRECTORY = "_backup_in_top_max_depth_pre_grouped"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _grouped_freeze(console, store_root: Path, retriever_name: str, threshold: float, predictor_name: str = IN_TOP_MAX_DEPTH_PREDICTOR_NAME) -> dict:
    source = selection_source_predictor(store_root, predictor_name, retriever_name, "feature_importances.parquet")
    importances_path = store_root / "models" / source / retriever_name / "feature_importances.parquet"
    console.log(f"{retriever_name}: importance source {source}")
    frame = pd.read_parquet(importances_path)
    frame["base_feature"] = frame["feature"].apply(strip_depth_prefix)
    excluded = sorted(set(frame.loc[frame["base_feature"].isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES), "base_feature"]))
    frame = frame.loc[~frame["base_feature"].isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
    grouped = frame.groupby("base_feature")["importance"].sum().sort_values(ascending=False)
    representative = (
        frame.sort_values("importance", ascending=False).drop_duplicates("base_feature").set_index("base_feature")["feature"]
    )
    qualifying_groups = grouped.loc[grouped > threshold]
    individually_qualifying = int((frame["importance"] > threshold).sum())
    console.log(
        f"{retriever_name}: {len(frame)} depth-specific columns in {len(grouped)} feature groups after excluding "
        f"{len(excluded)} consensus feature(s) {excluded}"
    )
    console.log(
        f"{retriever_name}: {len(qualifying_groups)}/{len(grouped)} groups have summed importance > {threshold} "
        f"(importance coverage {float(qualifying_groups.sum()):.4f}); for comparison, the ungrouped threshold keeps "
        f"{individually_qualifying} single columns with coverage "
        f"{float(frame.loc[frame['importance'] > threshold, 'importance'].sum()):.4f}"
    )
    for base_feature, group_importance in qualifying_groups.items():
        console.log(
            f"    {base_feature:55s} group importance {group_importance:.4f} -> representative "
            f"{representative[base_feature]} ({float(frame.set_index('feature').loc[representative[base_feature], 'importance']):.4f})"
        )
    features = [representative[base_feature] for base_feature in qualifying_groups.index]
    return {
        "importance_source": source,
        "qualifying_features": features,
        "group_importance": {representative[name]: float(value) for name, value in qualifying_groups.items()},
        "source_feature_count": int(len(frame)),
        "source_group_count": int(len(grouped)),
        "qualifying_feature_count": len(features),
        "qualifying_importance_sum": float(qualifying_groups.sum()),
        "excluded_consensus_features": excluded,
    }


def _correlation_round(console, train_frame: pd.DataFrame, features: list[str], group_importance: dict, threshold: float) -> tuple[list[str], list[dict]]:
    present = [feature for feature in features if feature in train_frame.columns]
    missing = sorted(set(features) - set(present))
    if missing:
        console.log(f"    {len(missing)} feature(s) missing from the train frame: {missing}")
    corr = train_frame[present].corr(method="spearman")
    pair_records, kept = prune_redundant_features(corr, pd.Series(group_importance), threshold)
    for record in pair_records:
        if record["dropped"] is not None:
            console.log(
                f"    rho={record['spearman_rho']:+.3f}: {record['feature_a']} ({record['importance_a']:.4f}) vs "
                f"{record['feature_b']} ({record['importance_b']:.4f}) -> drop {record['dropped']}"
            )
    console.log(f"    |rho|>={threshold:.2f}: {len(pair_records)} pair(s), kept {len(kept)}/{len(present)}")
    return kept, pair_records


def _rfe_path(runner, console, display_name, train_frame, dev_frame, features, rng, n_bootstrap, checkpoint_path, checkpoint, step_fraction):
    existing = checkpoint.get(display_name)
    if existing and existing.get("step_fraction") != step_fraction:
        console.log(f"  {display_name}: checkpoint was computed with a different RFE step rule, recomputing this path")
        existing = None
    if existing and existing.get("complete"):
        console.log(f"  {display_name}: RFE already complete in checkpoint ({len(existing['path'])} steps)")
        return list(existing["path"])
    if existing and existing.get("path"):
        path = list(existing["path"])
        remaining = list(existing["next_remaining"])
        console.log(f"  {display_name}: resuming RFE at {len(remaining)} features ({len(path)} step(s) recorded)")
    else:
        path = []
        remaining = list(features)
    while True:
        keep_columns = [column for column in train_frame.columns if column in IDENTIFIER_COLUMNS or column in remaining]
        output, _metrics, threshold, importances, _model, _calibration, threshold_selection = runner._fit_and_predict(
            f"{display_name} [RFE step {len(path) + 1}, {len(remaining)} features]",
            train_frame[keep_columns].copy(),
            dev_frame[keep_columns].copy(),
        )
        dev_metrics = in_top_max_depth_predictor_metric_summary(output)
        se = bootstrap_mcc_standard_error(output, rng, n_bootstrap)
        console.log(
            f"  [step {len(path) + 1}] {len(remaining)} features -> mcc={dev_metrics['mcc']:.4f} "
            f"(se={se:.4f}), precision={dev_metrics['precision']:.4f}, recall={dev_metrics['recall']:.4f}, "
            f"threshold={threshold:.4f}, cv_recall={threshold_selection['cv_recall']:.4f}"
        )
        path.append(
            {
                "feature_count": len(remaining),
                "features": list(remaining),
                "mcc": dev_metrics["mcc"],
                "mcc_bootstrap_se": se,
                "precision": dev_metrics["precision"],
                "recall": dev_metrics["recall"],
                "balanced_accuracy": dev_metrics["balanced_accuracy"],
                "threshold": float(threshold),
            }
        )
        if len(remaining) <= 1:
            checkpoint[display_name] = {"path": path, "complete": True, "step_fraction": step_fraction}
            save_checkpoint(checkpoint_path, checkpoint)
            break
        importance_lookup = importances.set_index("feature")["importance"]
        weakest = features_to_drop_by_step_fraction(remaining, importance_lookup, step_fraction)
        console.log(f"    dropping {len(weakest)} weakest remaining feature(s): {weakest}")
        remaining = [feature for feature in remaining if feature not in weakest]
        checkpoint[display_name] = {"path": path, "complete": False, "next_remaining": remaining, "step_fraction": step_fraction}
        save_checkpoint(checkpoint_path, checkpoint)
    return path


def _select_one_standard_error(path: list[dict]) -> tuple[dict, dict, float]:
    best = max(path, key=lambda entry: entry["mcc"])
    se = best["mcc_bootstrap_se"]
    tolerance = best["mcc"] - (se if math.isfinite(se) else 0.0)
    chosen = min((entry for entry in path if entry["mcc"] >= tolerance), key=lambda entry: entry["feature_count"])
    return chosen, best, tolerance


def _dev_metrics_from_metadata(store_root: Path, predictor_name: str, language: str, retriever_name: str) -> dict | None:
    path = store_root / "predictions" / predictor_name / language / retriever_name / "metadata.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metrics = metadata.get("dev_metrics", {})
    tp, fp = metrics.get("true_positives", math.nan), metrics.get("false_positives", math.nan)
    tn, fn = metrics.get("true_negatives", math.nan), metrics.get("false_negatives", math.nan)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if all(map(math.isfinite, [tp, fp, tn, fn])) else 0.0
    mcc = (tp * tn - fp * fn) / denominator if denominator else math.nan
    return {"mcc": mcc, "precision": metrics.get("precision"), "recall": metrics.get("recall"), "false_negatives": fn}


def _backup_previous_lightweight(console, store_root: Path, language: str, retrievers: list[str]) -> None:
    backup_root = store_root / BACKUP_DIRECTORY
    for kind in ("models", "predictions", "reports"):
        for retriever_name in retrievers:
            if kind == "models":
                source = store_root / kind / LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME / retriever_name
            else:
                source = store_root / kind / LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME / language / retriever_name
            if not source.exists():
                continue
            destination = backup_root / source.relative_to(store_root)
            if destination.exists():
                console.log(f"Backup already present, not overwritten: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            console.log(f"Backed up previous lightweight output {source} -> {destination}")


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    threshold: float = typer.Option(0.005, help="Minimum summed (grouped) feature importance for a feature group to qualify."),
    correlation_threshold: list[float] = typer.Option(
        [0.80, 0.70], "--correlation-threshold", help="Absolute Spearman rho threshold per correlation round, in order."
    ),
    n_bootstrap: int = typer.Option(2000, help="Bootstrap resamples for the dev MCC standard error at every RFE step."),
    step_fraction: float = typer.Option(0.10, help="Fraction of the remaining features dropped per RFE step (minimum 1)."),
    retriever: list[str] = typer.Option(["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to process."),
    train_final_model: bool = typer.Option(True, help="Train and persist the lightweight predictor on the selected features."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    store_root = runtime.store.root_dir
    console.section("In-top-max-depth lightweight feature reselection (grouped importance, Hit@1 procedure)")
    console.log(
        "Every feature exists once per feature depth (depth_N::feature) and the copies are near-identical, so the "
        "random forest splits a feature's importance across them and single copies fall below the importance "
        "threshold. This run therefore (1) sums importances per feature over its depth copies (grouped variable "
        "importance, Gregorutti et al. 2015), keeps groups above the threshold and represents each by its most "
        "important depth copy, then applies the Hit@1 procedure: (2) Spearman correlation pruning in rounds "
        f"{correlation_threshold}, (3) RFE dropping {step_fraction:.0%} of the remaining features per step (minimum 1) with the one-standard-error rule on dev MCC "
        f"({n_bootstrap} bootstrap resamples), (4) training the lightweight predictor on the chosen features."
    )
    resolved = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
    previous = {name: _dev_metrics_from_metadata(store_root, LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME, language, name) for name in resolved}
    full = {name: _dev_metrics_from_metadata(store_root, IN_TOP_MAX_DEPTH_PREDICTOR_NAME, language, name) for name in resolved}

    console.section("Step 1: grouped importance freeze")
    freeze = {name: _grouped_freeze(console, store_root, name, threshold) for name in resolved}
    _write_json(
        resolve_repo_relative_output_path(FREEZE_MANIFEST),
        {
            "metadata_version": MANIFEST_METADATA_VERSION,
            "predictor_name": IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
            "pipeline_stage": "grouped_freeze",
            "importance_threshold": threshold,
            "importance_grouping": "sum over depth copies of the same feature; representative = most important copy",
            "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
            "source": "artifacts/models/<importance_source>/<retriever>/feature_importances.parquet (consensus-free full model)",
            "retrievers": freeze,
        },
    )
    console.log(f"Wrote {FREEZE_MANIFEST}")

    runner = InTopMaxDepthPredictorRunner(
        runtime.config, runtime.store, console, runtime.features, runtime.retrieval, threshold_objective="mcc",
    )
    runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
    runner.compute_signal_summaries = False
    frames = {}
    for name in resolved:
        console.log(f"Building merged multi-depth train and dev frames for {name}")
        frames[name] = (build_merged_frame(runner, f"{language}_train", name), build_merged_frame(runner, f"{language}_dev", name))

    current = {name: list(freeze[name]["qualifying_features"]) for name in resolved}
    correlation_report: dict[str, dict] = {name: {} for name in resolved}
    for round_index, round_threshold in enumerate(correlation_threshold, start=1):
        console.section(f"Step 2.{round_index}: correlation pruning |rho|>={round_threshold:.2f}")
        for name in resolved:
            console.log(f"  {name}: {len(current[name])} features")
            kept, pairs = _correlation_round(console, frames[name][0], current[name], freeze[name]["group_importance"], round_threshold)
            correlation_report[name][f"round{round_index}"] = {"threshold": round_threshold, "pairs": pairs, "kept": kept}
            current[name] = kept
        output_manifest = CORRELATION_ROUND_1_MANIFEST if round_index == 1 else CORRELATION_ROUND_2_MANIFEST
        _write_json(
            resolve_repo_relative_output_path(output_manifest),
            {
                "metadata_version": MANIFEST_METADATA_VERSION,
                "predictor_name": IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
                "pipeline_stage": f"correlation_pruned_round{round_index}",
                "importance_threshold": threshold,
                "correlation_thresholds": correlation_threshold[:round_index],
                "retrievers": {
                    name: {
                        "qualifying_features": current[name],
                        "source_feature_count": freeze[name]["qualifying_feature_count"],
                        "qualifying_feature_count": len(current[name]),
                    }
                    for name in resolved
                },
            },
        )
        console.log(f"Wrote {output_manifest}")
    runtime.store.write_json(
        runtime.store.path("predictions", "lightweight_feature_correlation", REPORT_TAG, language, "correlation_report.json"),
        correlation_report,
    )

    console.section(f"Step 3: RFE, {step_fraction:.0%} of the remaining features per step, one-standard-error rule on dev MCC")
    rng = np.random.default_rng(runtime.config.runtime.random_seed)
    checkpoint_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    rfe_report: dict[str, dict] = {}
    for name in resolved:
        console.section(f"RFE for {name}: {len(current[name])} starting features")
        path = _rfe_path(runner, console, name, frames[name][0], frames[name][1], current[name], rng, n_bootstrap, checkpoint_path, checkpoint, step_fraction)
        chosen, best, tolerance = _select_one_standard_error(path)
        console.log(
            f"{name}: best mcc={best['mcc']:.4f} at {best['feature_count']} features (se={best['mcc_bootstrap_se']:.4f}, "
            f"tolerance>={tolerance:.4f}); chosen {chosen['feature_count']} features with mcc={chosen['mcc']:.4f}: {chosen['features']}"
        )
        rfe_report[name] = {"starting_feature_count": len(current[name]), "path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}
    runtime.store.write_json(
        runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_report.json"), rfe_report
    )
    rfe_manifest_path = resolve_repo_relative_output_path(RFE_MANIFEST)
    _write_json(
        rfe_manifest_path,
        {
            "metadata_version": MANIFEST_METADATA_VERSION,
            "predictor_name": IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
            "pipeline_stage": "rfe_pruned",
            "importance_threshold": threshold,
            "importance_grouping": "sum over depth copies of the same feature; representative = most important copy",
            "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
            "correlation_thresholds": correlation_threshold,
            "rfe_step_fraction": step_fraction,
            "rfe_selection_rule": "smallest feature count within one bootstrap standard error of best dev mcc",
            "rfe_bootstrap_samples": n_bootstrap,
            "source": f"grouped freeze -> correlation {correlation_threshold} -> RFE",
            "retrievers": {
                name: {
                    "qualifying_features": rfe_report[name]["chosen_entry"]["features"],
                    "source_feature_count": rfe_report[name]["starting_feature_count"],
                    "qualifying_feature_count": rfe_report[name]["chosen_entry"]["feature_count"],
                }
                for name in resolved
            },
        },
    )
    console.log(f"Wrote {RFE_MANIFEST}")

    summary = {}
    if train_final_model:
        console.section("Step 4: train the lightweight predictor on the selected features")
        _backup_previous_lightweight(console, store_root, language, resolved)
        lightweight_runner = LightweightInTopMaxDepthPredictorRunner(
            runtime.config, runtime.store, console, runtime.features, runtime.retrieval,
            manifest_path=str(rfe_manifest_path),
            predictor_name=LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
            threshold_policy_name=IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
            threshold_objective="mcc",
        )
        summary = lightweight_runner.run(language, retrievers=resolved)
        runtime.store.write_json(
            runtime.store.path("predictions", LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME, language, "summary.json"), summary
        )

    console.section("Summary: dev MCC per retriever")
    for name in resolved:
        new = _dev_metrics_from_metadata(store_root, LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME, language, name) if train_final_model else None
        chosen = rfe_report[name]["chosen_entry"]
        rows = [
            ("full model", full[name]),
            ("previous lightweight", previous[name]),
            ("new lightweight (RFE step)", {"mcc": chosen["mcc"], "precision": chosen["precision"], "recall": chosen["recall"], "false_negatives": math.nan}),
            ("new lightweight (trained)", new),
        ]
        console.log(
            f"{name}: freeze {freeze[name]['qualifying_feature_count']} groups -> correlation {len(current[name])} -> "
            f"RFE {chosen['feature_count']} features {chosen['features']}"
        )
        for label, metrics in rows:
            if metrics is None:
                continue
            console.log(
                f"    {label:28s} mcc={metrics['mcc']:.4f} precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} false_negatives={metrics['false_negatives']}"
            )
    console.log(
        "The previous manifests (config/lightweight_in_top_max_depth_features*.json) are unchanged. Previous "
        f"lightweight outputs were copied to artifacts/{BACKUP_DIRECTORY}/ before retraining."
    )


if __name__ == "__main__":
    typer.run(main)
