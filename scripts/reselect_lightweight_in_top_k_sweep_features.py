# scripts/reselect_lightweight_in_top_k_sweep_features.py
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys

import joblib

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _lightweight_in_top_max_depth_common import (
    bootstrap_mcc_standard_error,
    build_merged_frame,
    features_to_drop_by_step_fraction,
    load_checkpoint,
    resolve_repo_relative_output_path,
    save_checkpoint,
)
from reselect_lightweight_in_top_max_depth_features_grouped import (
    _correlation_round,
    _grouped_freeze,
    _select_one_standard_error,
    _write_json,
)

from query_difficulty.analysis import in_top_max_depth_predictor_metric_summary
from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.modeling import (
    IDENTIFIER_COLUMNS,
    IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
    InTopMaxDepthPredictorRunner,
)
from query_difficulty.modeling_in_top_k_sweep_lightweight import LIGHTWEIGHT_IN_TOP_K_SWEEP_MANIFEST_METADATA_VERSION
from query_difficulty.runtime import build_runtime

REPORT_TAG = "in_top_k_sweep_reselected"
MANIFEST_PREFIX = "config/lightweight_in_top_k_sweep_features.reselected"
BACKUP_DIRECTORY = "_backup_in_top_k_pre_reselection"


def _rfe_path(runner, console, key, train_frame, dev_frame, features, step_fraction, rng, n_bootstrap, checkpoint_path, checkpoint):
    existing = checkpoint.get(key)
    if existing and existing.get("step_fraction") != step_fraction:
        console.log(f"  {key}: checkpoint was computed with a different RFE step rule, recomputing this path")
        existing = None
    if existing and existing.get("complete"):
        console.log(f"  {key}: RFE already complete in checkpoint ({len(existing['path'])} steps)")
        return list(existing["path"])
    if existing and existing.get("path"):
        path, remaining = list(existing["path"]), list(existing["next_remaining"])
        console.log(f"  {key}: resuming RFE at {len(remaining)} features ({len(path)} step(s) recorded)")
    else:
        path, remaining = [], list(features)
    while True:
        keep_columns = [column for column in train_frame.columns if column in IDENTIFIER_COLUMNS or column in remaining]
        output, _metrics, threshold, importances, _model, _calibration, threshold_selection = runner._fit_and_predict(
            f"{key} [RFE, {len(remaining)} features]", train_frame[keep_columns].copy(), dev_frame[keep_columns].copy(),
        )
        dev_metrics = in_top_max_depth_predictor_metric_summary(output)
        se = bootstrap_mcc_standard_error(output, rng, n_bootstrap)
        console.log(
            f"  [step {len(path) + 1}] {len(remaining)} features -> mcc={dev_metrics['mcc']:.4f} (se={se:.4f}), "
            f"precision={dev_metrics['precision']:.4f}, recall={dev_metrics['recall']:.4f}, threshold={threshold:.4f}"
        )
        path.append(
            {
                "feature_count": len(remaining),
                "features": list(remaining),
                "mcc": dev_metrics["mcc"],
                "mcc_bootstrap_se": se,
                "precision": dev_metrics["precision"],
                "recall": dev_metrics["recall"],
                "threshold": float(threshold),
            }
        )
        if len(remaining) <= 1:
            checkpoint[key] = {"path": path, "complete": True, "step_fraction": step_fraction}
            save_checkpoint(checkpoint_path, checkpoint)
            return path
        lookup = importances.set_index("feature")["importance"]
        weakest = features_to_drop_by_step_fraction(remaining, lookup, step_fraction)
        console.log(f"    dropping {len(weakest)} weakest remaining feature(s): {weakest}")
        remaining = [feature for feature in remaining if feature not in weakest]
        checkpoint[key] = {"path": path, "complete": False, "next_remaining": remaining, "step_fraction": step_fraction}
        save_checkpoint(checkpoint_path, checkpoint)


def _dev_mcc(store_root: Path, predictor_name: str, language: str, retriever_name: str) -> float | None:
    path = store_root / "predictions" / predictor_name / language / retriever_name / "metadata.json"
    if not path.exists():
        return None
    return _mcc_from_metadata(path)


def _previous_dev_mcc(store_root: Path, depth: int, language: str, retriever_name: str) -> float | None:
    relative = Path("predictions") / "lightweight" / f"in_top_k{depth}" / language / retriever_name / "metadata.json"
    backup = store_root / BACKUP_DIRECTORY / relative
    if backup.exists():
        return _mcc_from_metadata(backup)
    current = store_root / relative
    return _mcc_from_metadata(current) if current.exists() else None


def _mcc_from_metadata(path: Path) -> float | None:
    metrics = json.loads(path.read_text(encoding="utf-8")).get("dev_metrics", {})
    tp, fp = metrics.get("true_positives"), metrics.get("false_positives")
    tn, fn = metrics.get("true_negatives"), metrics.get("false_negatives")
    if None in (tp, fp, tn, fn):
        return None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else math.nan


def _model_feature_set(model_path: Path) -> set[str] | None:
    if not model_path.exists():
        return None
    pipeline = joblib.load(model_path)["pipeline"]
    columns = getattr(pipeline, "feature_names_in_", None)
    if columns is None:
        return None
    return {str(column) for column in columns} - set(IDENTIFIER_COLUMNS)


def _move_outdated_output(console, store_root: Path, source: Path) -> None:
    destination = store_root / BACKUP_DIRECTORY / source.relative_to(store_root)
    if destination.exists():
        destination = destination.parent / f"{destination.name}__replaced_{datetime.now():%Y%m%d_%H%M%S}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    console.log(f"Moved outdated lightweight output {source} -> {destination}")


def _train_final_models(runtime, console, store_root: Path, language: str, depths, resolved, frames, rfe_report) -> dict:
    summary: dict[str, dict[str, dict]] = {}
    total = len(depths) * len(resolved)
    counter = 0
    for name in resolved:
        train_frame, dev_frame = frames[name]
        for depth in depths:
            counter += 1
            chosen = rfe_report[str(depth)][name]["chosen_entry"]
            features = list(chosen["features"])
            predictor_name = f"lightweight/in_top_k{depth}"
            model_path = store_root / "models" / predictor_name / name / "model.joblib"
            console.section(f"[{counter}/{total}] Train lightweight in-top-k={depth} for {name} on {len(features)} features")
            if _model_feature_set(model_path) == set(features):
                console.log(f"{name} k={depth}: existing model at {model_path} already uses exactly the selected features, skipping")
                continue
            for kind in ("models", "predictions", "reports"):
                source = store_root / kind / predictor_name / (name if kind == "models" else Path(language) / name)
                if source.exists():
                    _move_outdated_output(console, store_root, source)
            runner = InTopMaxDepthPredictorRunner(
                runtime.config, runtime.store, console, runtime.features, runtime.retrieval,
                predictor_name=predictor_name, threshold_policy_name=IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
                threshold_objective="mcc", target_depth=depth,
            )
            keep_columns = [column for column in train_frame.columns if column in IDENTIFIER_COLUMNS or column in features]
            display_name = f"{name} (k={depth})"
            output, metrics, threshold, importances, model, calibration_info, threshold_selection = runner._fit_and_predict(
                display_name, train_frame[keep_columns].copy(), dev_frame[keep_columns].copy(),
            )
            runner._write_outputs(
                language, name, display_name, output, metrics, threshold, importances, model, runner._feature_depths(),
                calibration_info=calibration_info, threshold_selection=threshold_selection,
            )
            trained_mcc = in_top_max_depth_predictor_metric_summary(output)["mcc"]
            matches = math.isclose(trained_mcc, chosen["mcc"], rel_tol=0.0, abs_tol=1e-9)
            console.log(
                f"{name} k={depth}: trained dev mcc={trained_mcc:.4f}, selection step recorded mcc={chosen['mcc']:.4f} "
                f"({'identical' if matches else 'DIFFERENT - check the log'}); threshold={threshold:.4f}; written to {model_path}"
            )
            summary.setdefault(name, {})[str(depth)] = {**metrics, "matches_selection_step": matches}
    return summary


def _manifest(stage: str, entries: dict, extra: dict) -> dict:
    return {
        "metadata_version": LIGHTWEIGHT_IN_TOP_K_SWEEP_MANIFEST_METADATA_VERSION,
        "predictor_name": "in_top_k_sweep",
        "pipeline_stage": stage,
        **extra,
        "depths": {str(depth): {"retrievers": retrievers} for depth, retrievers in sorted(entries.items())},
    }


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    threshold: float = typer.Option(0.005, help="Minimum summed (grouped) feature importance for a feature group to qualify."),
    correlation_threshold: list[float] = typer.Option(
        [0.80, 0.70], "--correlation-threshold", help="Absolute Spearman rho threshold per correlation round, in order."
    ),
    step_fraction: float = typer.Option(0.10, help="Fraction of the remaining features dropped per RFE step (minimum 1)."),
    n_bootstrap: int = typer.Option(2000, help="Bootstrap resamples for the dev MCC standard error at every RFE step."),
    retriever: list[str] = typer.Option(["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to process."),
    train_final_model: bool = typer.Option(True, help="Train and persist the lightweight in-top-k predictors on the selected features."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    store_root = runtime.store.root_dir
    depths = sorted({int(size) for size in runtime.config.candidate_sizes})
    resolved = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
    console.section("In-top-k sweep (binary predictor) lightweight feature reselection")
    console.log(
        f"Per retriever and per k in {depths}: (1) grouped importance freeze on the consensus-free full in_top_k<k> "
        "model (importances summed over the depth copies of each feature, Gregorutti et al. 2015; each kept group "
        f"represented by its most important depth copy), (2) Spearman correlation pruning in rounds {correlation_threshold}, "
        f"(3) RFE dropping {step_fraction:.0%} of the remaining features per step with the one-standard-error rule on dev MCC "
        f"({n_bootstrap} bootstrap resamples), (4) training the lightweight predictors on the selected features."
    )
    previous = {name: {depth: _previous_dev_mcc(store_root, depth, language, name) for depth in depths} for name in resolved}
    full = {name: {depth: _dev_mcc(store_root, f"in_top_k{depth}", language, name) for depth in depths} for name in resolved}

    console.section("Step 1: grouped importance freeze per k")
    freeze = {depth: {name: _grouped_freeze(console, store_root, name, threshold, predictor_name=f"in_top_k{depth}") for name in resolved} for depth in depths}
    _write_json(
        resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.freeze.json"),
        _manifest("grouped_freeze", {depth: freeze[depth] for depth in depths}, {"importance_threshold": threshold}),
    )

    runner = InTopMaxDepthPredictorRunner(
        runtime.config, runtime.store, console, runtime.features, runtime.retrieval, threshold_objective="mcc",
    )
    runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
    runner.compute_signal_summaries = False
    frames = {}
    for name in resolved:
        console.log(f"Building merged multi-depth train and dev frames for {name}")
        frames[name] = (build_merged_frame(runner, f"{language}_train", name), build_merged_frame(runner, f"{language}_dev", name))

    current = {depth: {name: list(freeze[depth][name]["qualifying_features"]) for name in resolved} for depth in depths}
    correlation_report = {str(depth): {name: {} for name in resolved} for depth in depths}
    for round_index, round_threshold in enumerate(correlation_threshold, start=1):
        console.section(f"Step 2.{round_index}: correlation pruning |rho|>={round_threshold:.2f}")
        for depth in depths:
            for name in resolved:
                console.log(f"  k={depth} {name}: {len(current[depth][name])} features")
                kept, pairs = _correlation_round(console, frames[name][0], current[depth][name], freeze[depth][name]["group_importance"], round_threshold)
                correlation_report[str(depth)][name][f"round{round_index}"] = {"threshold": round_threshold, "pairs": pairs, "kept": kept}
                current[depth][name] = kept
        stage = "correlation_pruned" if round_index == 1 else "correlation_pruned_round2"
        _write_json(
            resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.{stage}.json"),
            _manifest(
                stage,
                {depth: {name: {"qualifying_features": current[depth][name], "qualifying_feature_count": len(current[depth][name])} for name in resolved} for depth in depths},
                {"importance_threshold": threshold, "correlation_thresholds": correlation_threshold[:round_index]},
            ),
        )
    runtime.store.write_json(
        runtime.store.path("predictions", "lightweight_feature_correlation", REPORT_TAG, language, "correlation_report.json"),
        correlation_report,
    )

    console.section(f"Step 3: RFE, {step_fraction:.0%} of the remaining features per step, one-standard-error rule on dev MCC")
    rng = np.random.default_rng(runtime.config.runtime.random_seed)
    checkpoint_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    rfe_report = {str(depth): {} for depth in depths}
    total = len(depths) * len(resolved)
    counter = 0
    for name in resolved:
        for depth in depths:
            counter += 1
            console.section(f"[{counter}/{total}] RFE for {name} k={depth}: {len(current[depth][name])} starting features")
            runner.target_depth = depth
            path = _rfe_path(
                runner, console, f"{name}::k{depth}", frames[name][0], frames[name][1], current[depth][name],
                step_fraction, rng, n_bootstrap, checkpoint_path, checkpoint,
            )
            chosen, best, tolerance = _select_one_standard_error(path)
            console.log(
                f"{name} k={depth}: best mcc={best['mcc']:.4f} at {best['feature_count']} features "
                f"(tolerance>={tolerance:.4f}); chosen {chosen['feature_count']}: {chosen['features']}"
            )
            rfe_report[str(depth)][name] = {"starting_feature_count": len(current[depth][name]), "path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}
    runtime.store.write_json(runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_report.json"), rfe_report)
    final_path = resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.rfe_pruned.json")
    _write_json(
        final_path,
        _manifest(
            "rfe_pruned",
            {
                depth: {
                    name: {
                        "qualifying_features": rfe_report[str(depth)][name]["chosen_entry"]["features"],
                        "importance_source": freeze[depth][name]["importance_source"],
                        "source_feature_count": rfe_report[str(depth)][name]["starting_feature_count"],
                        "qualifying_feature_count": rfe_report[str(depth)][name]["chosen_entry"]["feature_count"],
                    }
                    for name in resolved
                }
                for depth in depths
            },
            {
                "importance_threshold": threshold,
                "importance_grouping": "sum over depth copies of the same feature; representative = most important copy",
                "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
                "correlation_thresholds": correlation_threshold,
                "rfe_step_fraction": step_fraction,
                "rfe_selection_rule": "smallest feature count within one bootstrap standard error of best dev mcc",
                "rfe_bootstrap_samples": n_bootstrap,
            },
        ),
    )
    console.log(f"Wrote {final_path}")

    if train_final_model:
        console.section("Step 4: train the lightweight in-top-k predictors on the selected features")
        console.log(
            "Each model is fit on the cached full feature tables (consensus features removed) restricted to its selected "
            "features, the same frames the selection used and the same feature builder the experiments use at test time. "
            "Pairs whose existing model already uses exactly the selected features are skipped; outdated outputs are moved "
            f"to artifacts/{BACKUP_DIRECTORY}/, never deleted."
        )
        summary = _train_final_models(runtime, console, store_root, language, depths, resolved, frames, rfe_report)
        runtime.store.write_json(runtime.store.path("predictions", "lightweight_in_top_k_sweep", language, "summary.json"), summary)

    console.section("Summary: dev MCC per retriever and k (full model / previous lightweight / new lightweight)")
    for name in resolved:
        for depth in depths:
            new = _dev_mcc(store_root, f"lightweight/in_top_k{depth}", language, name) if train_final_model else None
            chosen = rfe_report[str(depth)][name]["chosen_entry"]
            values = [full[name][depth], previous[name][depth], new]
            text = " / ".join("n/a" if value is None else f"{value:.4f}" for value in values)
            console.log(f"  {name} k={depth}: {text}  ({chosen['feature_count']} features: {chosen['features']})")


if __name__ == "__main__":
    typer.run(main)
