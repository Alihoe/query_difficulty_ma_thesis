# scripts/reselect_lightweight_calibrated_ordinal_features.py
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

from _lightweight_calibrated_ordinal_common import (
    features_to_drop_by_step_fraction,
    load_checkpoint,
    prune_redundant_features,
    resolve_repo_relative_output_path,
    save_checkpoint,
)
from _lightweight_calibrated_ordinal_per_head_common import (
    bootstrap_mean_standard_error,
    build_frames_by_depth,
    fit_head,
    head_keys,
    head_label,
    head_target,
    head_training_rows,
    is_constant_head,
    positive_probabilities,
    write_manifest,
)

from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.lightweight_selection import base_feature_name, selection_source_predictor
from query_difficulty.modeling import CalibratedOrdinalPredictorRunner
from query_difficulty.modeling_calibrated_ordinal_lightweight import (
    LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME,
    SOLVABILITY_HEAD,
    LightweightCalibratedOrdinalPredictorRunner,
)
from query_difficulty.runtime import build_runtime

PREDICTOR_NAME = "calibrated_ordinal"
REPORT_TAG = "calibrated_ordinal_reselected"
MANIFEST_PREFIX = "config/lightweight_calibrated_ordinal_features.reselected"
BACKUP_DIRECTORY = "_backup_calibrated_ordinal_pre_reselection"
OBJECTIVE = "brier"


def _importances_file(head: str) -> str:
    return "solvability_importances.parquet" if head == SOLVABILITY_HEAD else f"depth_{int(head):04d}_importances.parquet"


def _freeze(console, store_root: Path, retriever_name: str, head: str, threshold: float) -> dict:
    importances_file = _importances_file(head)
    source = selection_source_predictor(store_root, PREDICTOR_NAME, retriever_name, importances_file)
    frame = pd.read_parquet(store_root / "models" / source / retriever_name / importances_file)
    frame = frame.loc[~frame["feature"].map(base_feature_name).isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
    frame = frame.sort_values("importance", ascending=False)
    qualifying = frame.loc[frame["importance"] > threshold]
    console.log(
        f"{retriever_name} {head_label(head)}: importance source {source}; {len(qualifying)}/{len(frame)} features "
        f"above {threshold} (coverage {float(qualifying['importance'].sum()):.4f})"
    )
    return {
        "importance_source": source,
        "qualifying_features": qualifying["feature"].tolist(),
        "importance": dict(zip(frame["feature"], frame["importance"].astype(float))),
        "source_feature_count": int(len(frame)),
    }


def _brier_step(runner, train_rows, dev_rows, head, features, name, rng, n_bootstrap):
    model, importances, _ = fit_head(runner, train_rows, head, features, f"RFE {head_label(head)} for {name} [{len(features)} features]")
    probabilities = positive_probabilities(model, dev_rows)
    truth = head_target(dev_rows, head).to_numpy(dtype=np.float64)
    squared_errors = (probabilities - truth) ** 2
    return float(np.mean(squared_errors)), bootstrap_mean_standard_error(squared_errors, rng, n_bootstrap), importances


def _rfe(console, key, features, step, checkpoint_path, checkpoint, step_fraction):
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
        value, se, importances = step(remaining)
        console.log(f"  [step {len(path) + 1}] {len(remaining)} features -> {OBJECTIVE}={value:.5f} (se={se:.5f})")
        path.append({"feature_count": len(remaining), "features": list(remaining), OBJECTIVE: value, f"{OBJECTIVE}_bootstrap_se": se})
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


def _one_standard_error(path):
    best = min(path, key=lambda entry: entry[OBJECTIVE])
    se = best[f"{OBJECTIVE}_bootstrap_se"]
    tolerance = best[OBJECTIVE] + (se if math.isfinite(se) else 0.0)
    chosen = min((entry for entry in path if entry[OBJECTIVE] <= tolerance), key=lambda entry: entry["feature_count"])
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
    n_bootstrap: int = typer.Option(2000, help="Bootstrap resamples for the dev Brier standard error at every RFE step."),
    step_fraction: float = typer.Option(0.10, help="Fraction of the remaining features dropped per RFE step (minimum 1)."),
    retriever: list[str] = typer.Option(["bm25", "qwen3_embedding_0_6b"], "--retriever", help="Retrievers to process."),
    train_final_model: bool = typer.Option(True, help="Train and persist the lightweight predictor on the selected features."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    store_root = runtime.store.root_dir
    resolved = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
    runner = CalibratedOrdinalPredictorRunner(runtime.config, runtime.store, console, runtime.features, runtime.retrieval)
    runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
    candidate_sizes = runner._candidate_sizes()
    heads = head_keys(candidate_sizes)
    console.section("Calibrated ordinal lightweight feature reselection (one feature set per head)")
    console.log(
        f"Heads {heads}. Per head: (1) importance freeze on the consensus-free full calibrated ordinal model "
        f"(importance > {threshold}), (2) Spearman correlation pruning in rounds {correlation_threshold} on the rows the "
        "head is trained on (all train queries for the solvability head, solvable train queries for stage heads), "
        f"(3) RFE dropping {step_fraction:.0%} of the remaining features per step (minimum 1) with the one-standard-error rule on the dev Brier score of the head's own "
        f"calibrated probabilities (lower is better, {n_bootstrap} bootstrap resamples), (4) training the lightweight "
        "predictor with each head fit on its own features. Heads with a single-class training target are constant and "
        "not pruned; they receive the solvability head's features, which cannot change their output."
    )
    previous = {name: _dev_metrics(store_root, LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME, language, name) for name in resolved}
    full = {name: _dev_metrics(store_root, PREDICTOR_NAME, language, name) for name in resolved}

    frames = {}
    for name in resolved:
        console.log(f"Building {language}_train and {language}_dev frames for {name} at depths {candidate_sizes}")
        frames[name] = (build_frames_by_depth(runner, f"{language}_train", name), build_frames_by_depth(runner, f"{language}_dev", name))

    console.section("Step 1: importance freeze per head")
    freeze, constant = {}, {}
    for name in resolved:
        freeze[name], constant[name] = {}, []
        for head in heads:
            if is_constant_head(head_training_rows(frames[name][0], head), head):
                console.log(f"{name} {head_label(head)}: constant head (single-class training target), not selected")
                constant[name].append(head)
                continue
            freeze[name][head] = _freeze(console, store_root, name, head, threshold)

    current = {name: {head: list(entry["qualifying_features"]) for head, entry in freeze[name].items()} for name in resolved}
    stages = {"freeze": {name: {"heads": {head: {"qualifying_features": list(features)} for head, features in current[name].items()}} for name in resolved}}
    correlation_report = {name: {head: {} for head in current[name]} for name in resolved}
    for round_index, round_threshold in enumerate(correlation_threshold, start=1):
        console.section(f"Step 2.{round_index}: correlation pruning |rho|>={round_threshold:.2f}")
        for name in resolved:
            for head in current[name]:
                rows = head_training_rows(frames[name][0], head)
                present = [feature for feature in current[name][head] if feature in rows.columns]
                corr = rows[present].corr(method="spearman")
                pairs, kept = prune_redundant_features(corr, pd.Series(freeze[name][head]["importance"]), round_threshold)
                console.log(f"  {name} {head_label(head)}: {len(pairs)} pair(s) with |rho|>={round_threshold:.2f}, kept {len(kept)}/{len(present)}")
                correlation_report[name][head][f"round{round_index}"] = {"threshold": round_threshold, "pairs": pairs, "kept": kept}
                current[name][head] = kept
        stages["correlation_pruned" if round_index == 1 else "correlation_pruned_round2"] = {
            name: {"heads": {head: {"qualifying_features": list(features)} for head, features in current[name].items()}} for name in resolved
        }
    runtime.store.write_json(
        runtime.store.path("predictions", "lightweight_feature_correlation", REPORT_TAG, language, "correlation_report.json"),
        correlation_report,
    )

    console.section(f"Step 3: RFE per head, {step_fraction:.0%} of the remaining features per step, one-standard-error rule on dev Brier")
    rng = np.random.default_rng(runtime.config.runtime.random_seed)
    checkpoint_path = runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_path)
    rfe_report = {name: {} for name in resolved}
    chosen_features = {name: {} for name in resolved}
    for name in resolved:
        for head in current[name]:
            train_rows = head_training_rows(frames[name][0], head)
            dev_rows = head_training_rows(frames[name][1], head)
            console.section(f"{name} {head_label(head)}: {len(current[name][head])} starting features, {len(train_rows)} train / {len(dev_rows)} dev rows")
            path = _rfe(
                console, f"{name}::{head}", current[name][head],
                lambda features, tr=train_rows, dv=dev_rows, hd=head: _brier_step(runner, tr, dv, hd, features, name, rng, n_bootstrap),
                checkpoint_path, checkpoint, step_fraction,
            )
            chosen, best, tolerance = _one_standard_error(path)
            console.log(
                f"{name} {head_label(head)}: best brier={best[OBJECTIVE]:.5f} at {best['feature_count']} features "
                f"(tolerance<={tolerance:.5f}); chosen {chosen['feature_count']}: {chosen['features']}"
            )
            chosen_features[name][head] = list(chosen["features"])
            rfe_report[name][head] = {"starting_feature_count": len(current[name][head]), "path": path, "best_entry": best, "one_standard_error_tolerance": tolerance, "chosen_entry": chosen}
    runtime.store.write_json(runtime.store.path("predictions", "lightweight_feature_rfe", REPORT_TAG, language, "rfe_report.json"), rfe_report)

    final_retrievers = {}
    for name in resolved:
        head_entries = {}
        for head in heads:
            if head in constant[name]:
                head_entries[head] = {
                    "qualifying_features": chosen_features[name][SOLVABILITY_HEAD],
                    "qualifying_feature_count": len(chosen_features[name][SOLVABILITY_HEAD]),
                    "constant_head": True,
                    "feature_source": SOLVABILITY_HEAD,
                }
            else:
                head_entries[head] = {
                    "qualifying_features": chosen_features[name][head],
                    "qualifying_feature_count": len(chosen_features[name][head]),
                    "importance_source": freeze[name][head]["importance_source"],
                    "starting_feature_count": rfe_report[name][head]["starting_feature_count"],
                }
        final_retrievers[name] = {"heads": head_entries}
        union = sorted({feature for entry in head_entries.values() for feature in entry["qualifying_features"]})
        console.log(f"{name}: features per head {({head: entry['qualifying_feature_count'] for head, entry in head_entries.items()})}; {len(union)} distinct features overall")
    stages["rfe_pruned"] = final_retrievers
    for stage, retrievers in stages.items():
        write_manifest(
            resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.{stage}.json"),
            {
                "metadata_version": 2,
                "predictor_name": PREDICTOR_NAME,
                "pipeline_stage": stage,
                "heads": heads,
                "importance_threshold": threshold,
                "correlation_thresholds": correlation_threshold,
                "excluded_multi_retriever_consensus_features": sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES),
                "rfe_step_fraction": step_fraction,
                "rfe_objective": f"{OBJECTIVE} of each head's calibrated probabilities on dev (lower is better)",
                "rfe_selection_rule": "smallest feature count within one bootstrap standard error of the best dev objective",
                "rfe_bootstrap_samples": n_bootstrap,
                "retrievers": retrievers,
            },
        )
    final_path = resolve_repo_relative_output_path(f"{MANIFEST_PREFIX}.rfe_pruned.json")
    console.log(f"Wrote {final_path}")

    if train_final_model:
        console.section("Step 4: train the lightweight calibrated ordinal predictor with per-head features")
        backup_root = store_root / BACKUP_DIRECTORY
        for kind in ("models", "predictions", "reports"):
            for name in resolved:
                source = store_root / kind / LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME / (name if kind == "models" else Path(language) / name)
                destination = backup_root / source.relative_to(store_root)
                if source.exists() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source, destination)
                    console.log(f"Backed up {source} -> {destination}")
        lightweight_runner = LightweightCalibratedOrdinalPredictorRunner(
            runtime.config, runtime.store, console, runtime.features, runtime.retrieval, manifest_path=str(final_path),
        )
        lightweight_runner.views.excluded_features = MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
        summary = lightweight_runner.run(language, retrievers=resolved)
        runtime.store.write_json(runtime.store.path("predictions", LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME, language, "summary.json"), summary)

    console.section("Summary (dev)")
    for name in resolved:
        new = _dev_metrics(store_root, LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME, language, name) if train_final_model else None
        for label, metrics in (("full model", full[name]), ("previous lightweight", previous[name]), ("new lightweight", new)):
            if metrics:
                console.log(
                    f"  {name} {label:22s} candidate_recall={metrics['candidate_recall']:.4f} "
                    f"mean_selected_size={metrics['mean_selected_size']:.2f} failure_rate={metrics['failure_rate']:.4f}"
                )


if __name__ == "__main__":
    typer.run(main)
