from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from query_difficulty.config import REPO_ROOT
from query_difficulty.modeling import PREDICTOR_MODE_LOCAL, InTopMaxDepthPredictorRunner


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2)


def resolve_repo_relative_path(raw_path: str) -> Path:
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


def resolve_repo_relative_output_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def strip_depth_prefix(feature: str) -> str:
    if "::" in feature:
        return feature.split("::", 1)[1]
    return feature


def features_to_drop_by_step_fraction(
    remaining: list[str],
    importance_lookup: pd.Series,
    step_fraction: float,
) -> list[str]:
    drop_count = max(1, round(step_fraction * len(remaining)))
    drop_count = min(drop_count, len(remaining))
    return sorted(remaining, key=lambda feature: importance_lookup.get(feature, 0.0))[:drop_count]


def build_merged_frame(runner: InTopMaxDepthPredictorRunner, split_key: str, retriever_name: str) -> pd.DataFrame:
    depths = runner._feature_depths()
    frames_by_depth = runner.views.build(split_key, retriever_name, PREDICTOR_MODE_LOCAL, candidate_sizes=depths)
    return runner._merge_multi_depth_frames(frames_by_depth)


def bootstrap_mcc_standard_error(output: pd.DataFrame, rng: np.random.Generator, n_bootstrap: int) -> float:
    actual = output["in_top_max_depth"].to_numpy(dtype=bool)
    predicted = output["predicted_in_top_max_depth"].to_numpy(dtype=bool)
    n = len(actual)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual = actual[indices]
    resampled_predicted = predicted[indices]

    true_positives = np.sum(resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    false_positives = np.sum(~resampled_actual & resampled_predicted, axis=1).astype(np.float64)
    true_negatives = np.sum(~resampled_actual & ~resampled_predicted, axis=1).astype(np.float64)
    false_negatives = np.sum(resampled_actual & ~resampled_predicted, axis=1).astype(np.float64)

    numerator = true_positives * true_negatives - false_positives * false_negatives
    denominator = np.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )
    valid = denominator > 0
    mcc = np.divide(numerator, denominator, out=np.zeros(n_bootstrap), where=valid)

    mcc_values = mcc[valid]
    if len(mcc_values) < 2:
        return math.nan
    return float(np.std(mcc_values, ddof=1))


def prune_redundant_features(
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
