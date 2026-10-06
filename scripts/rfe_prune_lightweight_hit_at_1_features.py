# scripts/rfe_prune_lightweight_hit_at_1_features.py
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.config import REPO_ROOT
from query_difficulty.modeling import IDENTIFIER_COLUMNS, HitAtOnePredictorRunner
from query_difficulty.runtime import build_runtime


def _resolve_repo_relative_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    direct_path = path.resolve()
    if direct_path.exists():
        return direct_path
    repo_relative_path = (REPO_ROOT / path).resolve()
    if repo_relative_path.exists():
        return repo_relative_path
    raise FileNotFoundError(
        f"Could not find '{raw_path}'. Tried: {direct_path}, {repo_relative_path}"
    )


def _bootstrap_f0_5_standard_error(output: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> float:
    actual = output["hit_at_1"].to_numpy(dtype=bool)
    predicted = output["predicted_hit_at_1"].to_numpy(dtype=bool)
    n = len(actual)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual = actual[indices]
    resampled_predicted = predicted[indices]

    true_positives = np.sum(resampled_actual & resampled_predicted, axis=1)
    false_positives = np.sum(~resampled_actual & resampled_predicted, axis=1)
    false_negatives = np.sum(resampled_actual & ~resampled_predicted, axis=1)
    positive_predictions = true_positives + false_positives
    actual_positives = true_positives + false_negatives
    valid = (positive_predictions > 0) & (actual_positives > 0)

    precision = np.divide(true_positives, positive_predictions, out=np.zeros(n_bootstrap), where=positive_predictions > 0)
    recall = np.divide(true_positives, actual_positives, out=np.zeros(n_bootstrap), where=actual_positives > 0)
    beta_squared = 0.5**2
    denominator = beta_squared * precision + recall
    f0_5 = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros(n_bootstrap),
        where=denominator > 0,
    )

    f0_5_values = f0_5[valid]
    if len(f0_5_values) < 2:
        return math.nan
    return float(np.std(f0_5_values, ddof=1))


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2)


def _run_elimination_path(
    runner: HitAtOnePredictorRunner,
    console,
    display_name: str,
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
    checkpoint_path: Path,
    checkpoint: dict,
    retriever_key: str,
) -> list[dict]:
    existing_entry = checkpoint.get(retriever_key)
    if existing_entry and existing_entry.get("complete"):
        path = list(existing_entry["path"])
        console.log(f"  {retriever_key}: already complete in checkpoint ({len(path)} steps), skipping refits")
        return path
    if existing_entry and existing_entry.get("path"):
        path = list(existing_entry["path"])
        remaining = list(existing_entry["next_remaining"])
        console.log(
            f"  {retriever_key}: resuming from checkpoint at {len(remaining)} features "
            f"({len(path)} step(s) already recorded)"
        )
    else:
        path = []
        remaining = list(features)
    while True:
        keep_columns = [
            column for column in train_frame.columns
            if column in IDENTIFIER_COLUMNS or column in remaining
        ]
        train_subset = train_frame[keep_columns].copy()
        dev_subset = dev_frame[keep_columns].copy()
        output, metrics, _threshold, importances, _model, _calibration_info = runner._fit_and_predict(
            f"{display_name} [RFE step, {len(remaining)} features]", train_subset, dev_subset
        )
        se = _bootstrap_f0_5_standard_error(output, rng, n_bootstrap)
        console.log(
            f"  {len(remaining)} features -> f0_5={metrics['f0_5']:.4f} (se={se:.4f}), "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}"
        )
        path.append(
            {
                "feature_count": len(remaining),
                "features": list(remaining),
                "f0_5": metrics["f0_5"],
                "f0_5_bootstrap_se": se,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )
        if len(remaining) <= 1:
            checkpoint[retriever_key] = {"path": path, "complete": True}
            _save_checkpoint(checkpoint_path, checkpoint)
            break
        importance_lookup = importances.set_index("feature")["importance"]
        weakest_feature = min(remaining, key=lambda feature: importance_lookup.get(feature, 0.0))
        console.log(f"  dropping weakest remaining feature: {weakest_feature}")
        remaining = [feature for feature in remaining if feature != weakest_feature]
        checkpoint[retriever_key] = {"path": path, "complete": False, "next_remaining": remaining}
        _save_checkpoint(checkpoint_path, checkpoint)
    return path


def _select_one_standard_error_subset(path: list[dict]) -> tuple[dict, dict, float]:
    best_entry = max(path, key=lambda entry: entry["f0_5"])
    se = best_entry["f0_5_bootstrap_se"]
    tolerance = best_entry["f0_5"] - (se if math.isfinite(se) else 0.0)
    eligible = [entry for entry in path if entry["f0_5"] >= tolerance]
    chosen_entry = min(eligible, key=lambda entry: entry["feature_count"])
    return chosen_entry, best_entry, tolerance


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest: str = typer.Option(
        "config/lightweight_hit_at_1_features.correlation_pruned_round2.json",
        help="Path to the feature manifest to prune further.",
    ),
    n_bootstrap: int = typer.Option(
        2000,
        help="Number of bootstrap resamples used to estimate the standard error of dev F0.5 at each step.",
    ),
    retriever: list[str] = typer.Option(None, "--retriever", help="Analyze only the named retriever. Repeat for multiple retrievers."),
    output: str = typer.Option(
        "config/lightweight_hit_at_1_features.rfe_pruned.json",
        help="Where to write the RFE-pruned candidate manifest (does not overwrite the input manifest).",
    ),
    report_tag: str = typer.Option(
        "",
        help="Optional subdirectory to keep this run's rfe_report.json separate from other feature-selection "
        "tracks (e.g. 'no_multi_consensus'). Empty keeps the default, untagged path.",
    ),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Hit@1 lightweight RFE-style feature elimination")
    runtime.console.log(
        "For each retriever, repeatedly refits on the current feature set, drops the lowest-importance "
        "feature, and tracks dev F0.5 down to a single feature. A bootstrap standard error of dev F0.5 is "
        "computed at each step; the final recommendation is the smallest feature set whose F0.5 is within "
        "one standard error of the best F0.5 seen anywhere along the path (Breiman et al. one-standard-error "
        f"rule). This does not modify the input manifest at {manifest}."
    )
    manifest_path = _resolve_repo_relative_path(manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)

    runner = HitAtOnePredictorRunner(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features,
        runtime.retrieval,
    )
    feature_depth = runner._feature_depth()
    selected = runner._ordered_retrievers(retriever or None)
    rng = np.random.default_rng(runtime.config.runtime.random_seed)

    checkpoint_path_segments = ["predictions", "lightweight_feature_rfe"]
    if report_tag:
        checkpoint_path_segments.append(report_tag)
    checkpoint_path_segments += ["hit_at_1", language, "rfe_checkpoint.json"]
    checkpoint_path = runtime.store.path(*checkpoint_path_segments)
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint:
        runtime.console.log(f"Resuming from checkpoint at {checkpoint_path} ({len(checkpoint)} retriever(s) recorded)")

    train_key = f"{language}_train"
    dev_key = f"{language}_dev"
    report: dict[str, dict] = {}
    for retriever_name in selected:
        resolved_name = runtime.retrieval.resolve_retriever_name(retriever_name)
        entry = manifest_data.get("retrievers", {}).get(resolved_name)
        if entry is None:
            runtime.console.log(f"Skipping {resolved_name}: no manifest entry for it")
            continue
        qualifying_features = entry["qualifying_features"]
        runtime.console.section(f"RFE elimination path for {resolved_name}")
        train_frame = runtime.features.build_features(train_key, resolved_name, candidate_sizes=[feature_depth])[feature_depth]
        dev_frame = runtime.features.build_features(dev_key, resolved_name, candidate_sizes=[feature_depth])[feature_depth]
        present_features = [f for f in qualifying_features if f in train_frame.columns]

        path = _run_elimination_path(
            runner, runtime.console, resolved_name, train_frame, dev_frame, present_features, rng, n_bootstrap,
            checkpoint_path, checkpoint, resolved_name,
        )
        chosen_entry, best_entry, tolerance = _select_one_standard_error_subset(path)
        runtime.console.log(
            f"{resolved_name}: best f0_5={best_entry['f0_5']:.4f} at {best_entry['feature_count']} features "
            f"(se={best_entry['f0_5_bootstrap_se']:.4f}, tolerance>={tolerance:.4f}); "
            f"chosen {chosen_entry['feature_count']}/{len(present_features)} features "
            f"with f0_5={chosen_entry['f0_5']:.4f}"
        )
        report[resolved_name] = {
            "starting_feature_count": len(present_features),
            "path": path,
            "best_entry": best_entry,
            "one_standard_error_tolerance": tolerance,
            "chosen_entry": chosen_entry,
        }

    report_path_segments = ["predictions", "lightweight_feature_rfe"]
    if report_tag:
        report_path_segments.append(report_tag)
    report_path_segments += ["hit_at_1", language, "rfe_report.json"]
    report_path = runtime.store.path(*report_path_segments)
    runtime.store.write_json(report_path, report)
    runtime.console.log(f"Saved RFE elimination report to {report_path}")

    pruned_manifest = {
        "metadata_version": manifest_data.get("metadata_version"),
        "importance_threshold": manifest_data.get("importance_threshold"),
        "correlation_threshold": manifest_data.get("correlation_threshold"),
        "rfe_selection_rule": "smallest feature count within one bootstrap standard error of best dev f0_5",
        "rfe_bootstrap_samples": n_bootstrap,
        "source": f"RFE-pruned from {manifest}; review before adopting",
        "retrievers": {
            retriever_name: {
                "qualifying_features": data["chosen_entry"]["features"],
                "source_feature_count": data["starting_feature_count"],
                "qualifying_feature_count": data["chosen_entry"]["feature_count"],
            }
            for retriever_name, data in report.items()
        },
    }
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(pruned_manifest, handle, ensure_ascii=False, indent=2)
    runtime.console.log(f"Wrote RFE-pruned candidate manifest to {output_path}")
    runtime.console.log(
        "This candidate manifest is a recommendation only. Review artifacts/predictions/"
        "lightweight_feature_rfe/hit_at_1/en/rfe_report.json before replacing the frozen manifest."
    )


if __name__ == "__main__":
    typer.run(main)
