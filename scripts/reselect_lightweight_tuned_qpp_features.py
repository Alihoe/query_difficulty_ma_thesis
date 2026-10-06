# scripts/reselect_lightweight_tuned_qpp_features.py
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
    FAILURE_TARGET_COLUMN,
    REGRESSOR_TARGET_COLUMN,
    bootstrap_r2_standard_error,
    build_qpp_frame,
    features_to_drop_by_step_fraction,
    compute_r2,
    fit_regressor_subset,
    load_checkpoint,
    prune_redundant_features,
    resolve_repo_relative_output_path,
    save_checkpoint,
)

from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.lightweight_selection import base_feature_name, selection_source_predictor
from query_difficulty.modeling import TunedQppPredictorRunner
from query_difficulty.modeling_tuned_qpp_lightweight import (
    LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME,
    LightweightTunedQppPredictorRunner,
)
from query_difficulty.runtime import build_runtime

PREDICTOR_NAME = "tuned_qpp"
REPORT_TAG = "tuned_qpp_reselected"
MANIFEST_METADATA_VERSION = 2
BACKUP_DIRECTORY = "_backup_tuned_qpp_pre_reselection"
HEADS = {
    "failure_classifier": {
        "importances_file": "failure_importances.parquet",
        "manifest_prefix": "config/lightweight_tuned_qpp_failure_features.reselected",
        "objective": "failure_f1",
    },
    "score_regressor": {
        "importances_file": "score_importances.parquet",
        "manifest_prefix": "config/lightweight_tuned_qpp_regressor_features.reselected",
        "objective": "r2",
    },
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _failure_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    tp = float(np.sum(actual & predicted))
    fp = float(np.sum(~actual & predicted))
    fn = float(np.sum(actual & ~predicted))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else math.nan


def _bootstrap_failure_f1_standard_error(actual: np.ndarray, predicted: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> float:
    n = len(actual)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual = actual[indices]
    resampled_predicted = predicted[indices]
    tp = np.sum(resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    fp = np.sum(~resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    fn = np.sum(resampled_actual & ~resampled_predicted, axis=1).astype(np.float64)
    denominator = 2 * tp + fp + fn
    valid = denominator > 0
    values = (2 * tp[valid]) / denominator[valid]
    return float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan


def _freeze(console, store_root: Path, retriever_name: str, head: str, threshold: float) -> dict:
    importances_file = HEADS[head]["importances_file"]
    source = selection_source_predictor(store_root, PREDICTOR_NAME, retriever_name, importances_file)
    frame = pd.read_parquet(store_root / "models" / source / retriever_name / importances_file)
    is_consensus = frame["feature"].map(base_feature_name).isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)
    frame = frame.loc[~is_consensus].sort_values("importance", ascending=False)
    qualifying = frame.loc[frame["importance"] > threshold]
    console.log(
        f"{retriever_name} {head}: importance source {source}; {len(qualifying)}/{len(frame)} features above "
        f"{threshold} (coverage {float(qualifying['importance'].sum()):.4f})"
    )
    return {
        "importance_source": source,
        "qualifying_features": qualifying["feature"].tolist(),
        "importance": dict(zip(frame["feature"], frame["importance"].astype(float))),
        "source_feature_count": int(len(frame)),
        "qualifying_feature_count": int(len(qualifying)),
        "qualifying_importance_sum": float(qualifying["importance"].sum()),
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


def _elimination(console, key, features, step, checkpoint_path, checkpoint, objective, step_fraction):
    existing = checkpoint.get(key)
    if existing and existing.get("step_fraction") != step_fraction:
        console.log(f"  {key}: checkpoint was computed with a different RFE step rule, recomputing this path")
        existing = None
    if existing and existing.get("complete"):
        console.log(f"  {key}: already complete in checkpoint ({len(existing['path'])} steps)")
        return list(existing["path"])
    if existing and existing.get("path"):
        path, remaining = list(existing["path"]), list(existing["next_remaining"])
        console.log(f"  {key}: resuming at {len(remaining)} features ({len(path)} step(s) recorded)")
    else:
        path, remaining = [], list(features)
    while True:
        value, se, importances, extras = step(remaining)
        console.log(
            f"  [step {len(path) + 1}] {len(remaining)} features -> {objective}={value:.4f} (se={se:.4f})"
            + "".join(f", {name}={extra:.4f}" for name, extra in extras.items())
        )
        path.append({"feature_count": len(remaining), "features": list(remaining), objective: value, f"{objective}_bootstrap_se": se, **extras})
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


def _one_standard_error(path, objective):
    best = max(path, key=lambda entry: entry[objective])
    se = best[f"{objective}_bootstrap_se"]
    tolerance = best[objective] - (se if math.isfinite(se) else 0.0)
    chosen = min((entry for entry in path if entry[objective] >= tolerance), key=lambda entry: entry["feature_count"])
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
    n_bootstrap: int = typer.Option(2000, help="Bootstrap resamples for the dev standard error at every RFE step."),
    step_fraction: float = typer.Option(0.10, help="Fraction of the remaining features dropped per RFE step (minimum 1)."),
    retriever: list[str] = typer.Option(["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to process."),
    train_final_model: bool = typer.Option(True, help="Train and persist the lightweight predictor on the selected features."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    store_root = runtime.store.root_dir
    console.section("Tuned QPP lightweight feature reselection (Hit@1 procedure)")
    console.log(
        "Per head: (1) importance freeze on the consensus-free full model (importance > threshold), (2) Spearman "
        f"correlation pruning in rounds {correlation_threshold}, keeping the more important feature of each pair, "
        f"(3) RFE dropping {step_fraction:.0%} of the remaining features per step (minimum 1) with the one-standard-error rule. The failure classifier is scored by "
        "dev failure F1 at its tuned threshold (the objective its threshold maximises); the score regressor by dev "
        f"R^2 of {REGRESSOR_TARGET_COLUMN}. (4) The lightweight predictor is trained on both selected sets."
    )
    resolved = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
    previous = {name: _dev_metrics(store_root, LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME, language, name) for name in resolved}
    full = {name: _dev_metrics(store_root, PREDICTOR_NAME, language, name) for name in resolved}

    console.section("Step 1: importance freeze")
    freeze = {name: {head: _freeze(console, store_root, name, head, threshold) for head in HEADS} for name in resolved}

    runner = TunedQppPredictorRunner(runtime.config, runtime.store, console, runtime.features, runtime.retrieval)
    runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
    runner.compute_signal_summaries = False
    frames = {}
    for name in resolved:
        console.log(f"Building {language}_train and {language}_dev QPP frames for {name}")
        frames[name] = (build_qpp_frame(runner, f"{language}_train", name), build_qpp_frame(runner, f"{language}_dev", name))

    current = {name: {head: list(freeze[name][head]["qualifying_features"]) for head in HEADS} for name in resolved}
    stage_manifests = {head: {"freeze": {}} for head in HEADS}
    for name in resolved:
        for head in HEADS:
            stage_manifests[head]["freeze"][name] = {
                "qualifying_features": current[name][head],
                "importance_source": freeze[name][head]["importance_source"],
                "source_feature_count": freeze[name][head]["source_feature_count"],
                "qualifying_feature_count": freeze[name][head]["qualifying_feature_count"],
                "qualifying_importance_sum": freeze[name][head]["qualifying_importance_sum"],
            }
    correlation_report = {name: {head: {} for head in HEADS} for name in resolved}
    for round_index, round_threshold in enumerate(correlation_threshold, start=1):
        console.section(f"Step 2.{round_index}: correlation pruning |rho|>={round_threshold:.2f}")
        for name in resolved:
            for head in HEADS:
                console.log(f"  {name} {head}: {len(current[name][head])} features")
                kept, pairs = _correlation_round(console, frames[name][0], current[name][head], freeze[name][head]["importance"], round_threshold)
                correlation_report[name][head][f"round{round_index}"] = {"threshold": round_threshold, "pairs": pairs, "kept": kept}
                current[name][head] = kept
        for head in HEADS:
            stage_manifests[head][f"correlation_pruned{'' if round_index == 1 else '_round2'}"] = {
                name: {"qualifying_features": current[name][head], "qualifying_feature_count": len(current[name][head])}
                for name in resolved
            }
    runtime.store.write_json(
        runtime.store.path("predictions", "lightweight_feature_correlation", REPORT_TAG, language, "correlation_report.json"),
        correlation_report,
    )
    for head, stages in stage_manifests.items():
        for stage, retrievers in stages.items():
            _write_json(
                resolve_repo_relative_output_path(f"{HEADS[head]['manifest_prefix']}.{stage}.json"),
                {
                    "metadata_version": MANIFEST_METADATA_VERSION,
                    "predictor_name": PREDICTOR_NAME,
                    "head": head,
                    "pipeline_stage": stage,
                    "importance_threshold": threshold,
                    "correlation_thresholds": correlation_threshold,
                    "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
                    "retrievers": retrievers,
                },
            )

    console.section(f"Step 3: RFE, {step_fraction:.0%} of the remaining features per step, one-standard-error rule")
    rng = np.random.default_rng(runtime.config.runtime.random_seed)
    checkpoint_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    rfe_report: dict[str, dict] = {name: {} for name in resolved}
    for name in resolved:
        train_frame, dev_frame = frames[name]
        console.section(f"{name} failure classifier: {len(current[name]['failure_classifier'])} starting features")
        console.log("  fitting the score regressor and size mapping once (the failure-head RFE does not change them)")
        score_context = runner._fit_score_context(train_frame, dev_frame, name)
        actual_failure = dev_frame[FAILURE_TARGET_COLUMN].astype(bool).to_numpy()

        def failure_step(features, train_frame=train_frame, dev_frame=dev_frame, score_context=score_context, actual_failure=actual_failure, name=name):
            output, failure_threshold, _, _, failure_importances, _ = runner._fit_and_predict_frame(
                train_frame, dev_frame, f"{name} [failure RFE, {len(features)} features]",
                evaluation_split="dev", failure_feature_columns=features, precomputed_score_context=score_context,
            )
            predicted = output["predicted_failure"].to_numpy(dtype=bool)
            return (
                _failure_f1(actual_failure, predicted),
                _bootstrap_failure_f1_standard_error(actual_failure, predicted, rng, n_bootstrap),
                failure_importances,
                {"threshold": float(failure_threshold), "failure_rate": float(predicted.mean())},
            )

        path = _elimination(console, f"{name}::failure_classifier", current[name]["failure_classifier"], failure_step, checkpoint_path, checkpoint, "failure_f1", step_fraction)
        chosen, best, tolerance = _one_standard_error(path, "failure_f1")
        console.log(
            f"{name} failure classifier: best failure_f1={best['failure_f1']:.4f} at {best['feature_count']} features "
            f"(tolerance>={tolerance:.4f}); chosen {chosen['feature_count']}: {chosen['features']}"
        )
        rfe_report[name]["failure_classifier"] = {"path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}

        console.section(f"{name} score regressor: {len(current[name]['score_regressor'])} starting features")

        def regressor_step(features, train_frame=train_frame, dev_frame=dev_frame):
            actual, predicted, importances = fit_regressor_subset(
                train_frame[features + [REGRESSOR_TARGET_COLUMN]].copy(),
                dev_frame[features + [REGRESSOR_TARGET_COLUMN]].copy(),
                features, runner.qpp_config.regressor, runtime.config.runtime.random_seed,
            )
            return compute_r2(actual, predicted), bootstrap_r2_standard_error(actual, predicted, rng, n_bootstrap), importances, {}

        path = _elimination(console, f"{name}::score_regressor", current[name]["score_regressor"], regressor_step, checkpoint_path, checkpoint, "r2", step_fraction)
        chosen, best, tolerance = _one_standard_error(path, "r2")
        console.log(
            f"{name} score regressor: best r2={best['r2']:.4f} at {best['feature_count']} features "
            f"(tolerance>={tolerance:.4f}); chosen {chosen['feature_count']}: {chosen['features']}"
        )
        rfe_report[name]["score_regressor"] = {"path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}
    runtime.store.write_json(runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_report.json"), rfe_report)

    final_paths = {}
    for head in HEADS:
        final_paths[head] = resolve_repo_relative_output_path(f"{HEADS[head]['manifest_prefix']}.rfe_pruned.json")
        _write_json(
            final_paths[head],
            {
                "metadata_version": MANIFEST_METADATA_VERSION,
                "predictor_name": PREDICTOR_NAME,
                "head": head,
                "pipeline_stage": "rfe_pruned",
                "importance_threshold": threshold,
                "correlation_thresholds": correlation_threshold,
                "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
                "rfe_step_fraction": step_fraction,
                "rfe_objective": HEADS[head]["objective"],
                "rfe_selection_rule": "smallest feature count within one bootstrap standard error of the best dev objective",
                "rfe_bootstrap_samples": n_bootstrap,
                "retrievers": {
                    name: {
                        "qualifying_features": rfe_report[name][head]["chosen_entry"]["features"],
                        "importance_source": freeze[name][head]["importance_source"],
                        "qualifying_feature_count": rfe_report[name][head]["chosen_entry"]["feature_count"],
                    }
                    for name in resolved
                },
            },
        )
        console.log(f"Wrote {final_paths[head]}")

    if train_final_model:
        console.section("Step 4: train the lightweight tuned QPP predictor on the selected features")
        backup_root = store_root / BACKUP_DIRECTORY
        for kind in ("models", "predictions", "reports"):
            for name in resolved:
                source = store_root / kind / LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME / (name if kind == "models" else Path(language) / name)
                destination = backup_root / source.relative_to(store_root)
                if source.exists() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, destination)
                    console.log(f"Backed up {source} -> {destination}")
                if kind != "models" and source.exists():
                    for cached in ("metadata.json",):
                        if (source / cached).exists():
                            (source / cached).unlink()
                            console.log(f"Removed cached {source / cached} so the predictor is retrained on the new features")
        lightweight_runner = LightweightTunedQppPredictorRunner(
            runtime.config, runtime.store, console, runtime.features, runtime.retrieval,
            failure_manifest_path=str(final_paths["failure_classifier"]),
            regressor_manifest_path=str(final_paths["score_regressor"]),
        )
        summary = lightweight_runner.run(language, retrievers=resolved)
        runtime.store.write_json(runtime.store.path("predictions", LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME, language, "summary.json"), summary)

    console.section("Summary (dev)")
    for name in resolved:
        new = _dev_metrics(store_root, LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME, language, name) if train_final_model else None
        console.log(
            f"{name}: failure head {len(rfe_report[name]['failure_classifier']['chosen_entry']['features'])} features, "
            f"score regressor {len(rfe_report[name]['score_regressor']['chosen_entry']['features'])} features"
        )
        for label, metrics in (("full model", full[name]), ("previous lightweight", previous[name]), ("new lightweight", new)):
            if metrics:
                console.log(
                    f"    {label:22s} candidate_recall={metrics['candidate_recall']:.4f} "
                    f"mean_selected_size={metrics['mean_selected_size']:.2f} "
                    f"median_selected_size={metrics['median_selected_size']:.1f} failure_rate={metrics['failure_rate']:.4f}"
                )


if __name__ == "__main__":
    typer.run(main)
