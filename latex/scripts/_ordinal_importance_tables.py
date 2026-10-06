from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ORDINAL_MODEL_DIRECTORY = "calibrated_ordinal"


def ordinal_importances_by_depth(
    store_root: Path, retriever: str, model_directory: str = ORDINAL_MODEL_DIRECTORY
) -> dict[int, pd.DataFrame]:
    model_root = store_root / "models" / model_directory / retriever
    manifest_path = model_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Ordinal model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_sizes = sorted(int(size) for size in manifest["candidate_sizes"])
    max_depth = candidate_sizes[-1]

    importances: dict[int, pd.DataFrame] = {}
    for depth in candidate_sizes:
        if depth == max_depth:
            imp_path = model_root / "solvability_importances.parquet"
        else:
            imp_path = model_root / f"depth_{depth:04d}_importances.parquet"
        if not imp_path.exists():
            raise FileNotFoundError(f"Importances file not found: {imp_path}")
        importances[depth] = pd.read_parquet(imp_path)
    return importances


def ordinal_training_target(training_frame: pd.DataFrame, depth: int, max_depth: int) -> tuple[pd.DataFrame, pd.Series]:
    if depth == max_depth:
        return training_frame, training_frame["smallest_sufficient_size"].notna()
    solvable = training_frame.loc[training_frame["smallest_sufficient_size"].notna()]
    return solvable, solvable["gold_in_top_k"].astype(bool)


def mean_importance_across_depths(importances_by_depth: dict[int, pd.DataFrame]) -> pd.Series:
    per_depth = [frame.set_index("feature")["importance"] for frame in importances_by_depth.values()]
    all_features = sorted(set().union(*(series.index for series in per_depth)))
    aligned = pd.DataFrame({i: series.reindex(all_features).fillna(0.0) for i, series in enumerate(per_depth)})
    return aligned.mean(axis=1)
