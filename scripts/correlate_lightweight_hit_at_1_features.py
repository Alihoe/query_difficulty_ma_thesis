# scripts/correlate_lightweight_hit_at_1_features.py
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.config import REPO_ROOT
from query_difficulty.modeling import HitAtOnePredictorRunner
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


def _prune_redundant_features(
    corr: pd.DataFrame,
    importances: pd.Series,
    threshold: float,
) -> tuple[list[dict], list[str]]:
    features = list(corr.columns)
    pairs = []
    for i, feature_i in enumerate(features):
        for feature_j in features[i + 1 :]:
            rho = corr.loc[feature_i, feature_j]
            if pd.isna(rho):
                continue
            if abs(rho) >= threshold:
                pairs.append((abs(rho), feature_i, feature_j, rho))
    pairs.sort(key=lambda item: item[0], reverse=True)

    removed: set[str] = set()
    pair_records = []
    for _, feature_i, feature_j, rho in pairs:
        importance_i = float(importances.get(feature_i, 0.0))
        importance_j = float(importances.get(feature_j, 0.0))
        already_resolved = feature_i in removed or feature_j in removed
        drop_feature = None
        if not already_resolved:
            drop_feature = feature_j if importance_i >= importance_j else feature_i
            removed.add(drop_feature)
        pair_records.append(
            {
                "feature_a": feature_i,
                "feature_b": feature_j,
                "spearman_rho": float(rho),
                "importance_a": importance_i,
                "importance_b": importance_j,
                "dropped": drop_feature,
                "already_resolved_by_earlier_pair": already_resolved,
            }
        )
    kept = [feature for feature in features if feature not in removed]
    return pair_records, kept


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest: str = typer.Option(
        "config/lightweight_hit_at_1_features.correlation_pruned.json",
        help="Path to the feature manifest to prune further.",
    ),
    threshold: float = typer.Option(
        0.70,
        help="Minimum absolute Spearman rho for a feature pair to be treated as redundant.",
    ),
    retriever: list[str] = typer.Option(None, "--retriever", help="Analyze only the named retriever. Repeat for multiple retrievers."),
    output: str = typer.Option(
        "config/lightweight_hit_at_1_features.correlation_pruned_round2.json",
        help="Where to write the correlation-pruned candidate manifest (does not overwrite the input manifest).",
    ),
    report_tag: str = typer.Option(
        "",
        help="Optional subdirectory to keep this run's correlation_report.json separate from other feature-"
        "selection tracks (e.g. 'no_multi_consensus'). Empty keeps the default, untagged path.",
    ),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Hit@1 lightweight feature correlation analysis")
    runtime.console.log(
        f"Computing pairwise Spearman correlation among the currently qualifying features per retriever, "
        f"flagging pairs with |rho| >= {threshold:.2f}, and recommending the lower-importance member of "
        f"each flagged pair for removal. This does not modify the frozen manifest at {manifest}."
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

    train_key = f"{language}_train"
    report: dict[str, dict] = {}
    for retriever_name in selected:
        resolved_name = runtime.retrieval.resolve_retriever_name(retriever_name)
        entry = manifest_data.get("retrievers", {}).get(resolved_name)
        if entry is None:
            runtime.console.log(f"Skipping {resolved_name}: no manifest entry for it")
            continue
        qualifying_features = entry["qualifying_features"]
        runtime.console.section(f"Correlation scan for {resolved_name}")
        runtime.console.log(f"Loading train features for {resolved_name} at depth {feature_depth}")
        train_frame = runtime.features.build_features(train_key, resolved_name, candidate_sizes=[feature_depth])[feature_depth]
        present_features = [f for f in qualifying_features if f in train_frame.columns]
        missing = sorted(set(qualifying_features) - set(present_features))
        if missing:
            runtime.console.log(f"{resolved_name}: {len(missing)} manifest features not found in feature table: {missing}")

        importances_path = runtime.store.root_dir / "models" / "hit_at_1" / resolved_name / "feature_importances.parquet"
        importances_frame = pd.read_parquet(importances_path)
        importances = importances_frame.set_index("feature")["importance"]

        corr = train_frame[present_features].corr(method="spearman")
        pair_records, kept = _prune_redundant_features(corr, importances, threshold)
        dropped = sorted(set(present_features) - set(kept))

        runtime.console.log(
            f"{resolved_name}: {len(pair_records)} pairs with |rho|>={threshold:.2f}, "
            f"recommending {len(dropped)} drops -> {len(kept)}/{len(present_features)} features kept"
        )
        for record in pair_records:
            if record["dropped"] is not None:
                counterpart = record["feature_b"] if record["dropped"] == record["feature_a"] else record["feature_a"]
                runtime.console.log(
                    f"  drop {record['dropped']} (rho={record['spearman_rho']:.3f} with {counterpart}, "
                    f"importance {record['importance_a']:.4f} vs {record['importance_b']:.4f})"
                )

        report[resolved_name] = {
            "threshold": threshold,
            "source_feature_count": entry["source_feature_count"],
            "qualifying_feature_count": len(present_features),
            "correlated_pairs": pair_records,
            "recommended_kept_features": kept,
            "recommended_dropped_features": dropped,
        }

    report_path_segments = ["predictions", "lightweight_feature_correlation"]
    if report_tag:
        report_path_segments.append(report_tag)
    report_path_segments += ["hit_at_1", language, "correlation_report.json"]
    report_path = runtime.store.path(*report_path_segments)
    runtime.store.write_json(report_path, report)
    runtime.console.log(f"Saved correlation report to {report_path}")

    pruned_manifest = {
        "metadata_version": manifest_data.get("metadata_version"),
        "importance_threshold": manifest_data.get("importance_threshold"),
        "correlation_threshold": threshold,
        "source": "correlation-pruned from config/lightweight_hit_at_1_features.json; review before adopting",
        "retrievers": {
            retriever_name: {
                "qualifying_features": data["recommended_kept_features"],
                "source_feature_count": data["source_feature_count"],
                "qualifying_feature_count": len(data["recommended_kept_features"]),
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
    runtime.console.log(f"Wrote correlation-pruned candidate manifest to {output_path}")
    runtime.console.log(
        "This candidate manifest is a recommendation only. Review artifacts/predictions/"
        "lightweight_feature_correlation/hit_at_1/en/correlation_report.json before replacing the frozen manifest."
    )


if __name__ == "__main__":
    typer.run(main)
