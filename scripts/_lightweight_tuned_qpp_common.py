from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from query_difficulty.config import REPO_ROOT
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    TunedQppPredictorRunner,
    _fit_pipeline,
    _make_regressor,
    _feature_columns,
    _feature_importances,
)

FAILURE_TARGET_COLUMN = "retrieval_failure_within_max_depth"
REGRESSOR_TARGET_COLUMN = "rr_at_max_depth"


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


def features_to_drop_by_step_fraction(
    remaining: list[str],
    importance_lookup: pd.Series,
    step_fraction: float,
) -> list[str]:
    drop_count = max(1, round(step_fraction * len(remaining)))
    drop_count = min(drop_count, len(remaining))
    return sorted(remaining, key=lambda feature: importance_lookup.get(feature, 0.0))[:drop_count]


def build_qpp_frame(runner: TunedQppPredictorRunner, split_key: str, retriever_name: str) -> pd.DataFrame:
    depth = runner.config.max_ranking_depth
    return runner.views.build(split_key, retriever_name, PREDICTOR_MODE_LOCAL)[depth]


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


def bootstrap_mcc_standard_error(
    actual_failure: np.ndarray,
    predicted_failure: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> float:
    n = len(actual_failure)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual = actual_failure[indices]
    resampled_predicted = predicted_failure[indices]

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


def compute_mcc(actual_failure: np.ndarray, predicted_failure: np.ndarray) -> float:
    tp = float(np.sum(actual_failure & predicted_failure))
    fp = float(np.sum(~actual_failure & predicted_failure))
    tn = float(np.sum(~actual_failure & ~predicted_failure))
    fn = float(np.sum(actual_failure & ~predicted_failure))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return math.nan
    return (tp * tn - fp * fn) / denominator


def fit_regressor_subset(
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    feature_columns: list[str],
    regressor_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pipeline = _fit_pipeline(
        _make_regressor(regressor_name, seed),
        feature_columns,
        train_frame,
        train_frame[REGRESSOR_TARGET_COLUMN],
    )
    importances = _feature_importances(pipeline, feature_columns)
    predicted = np.asarray(pipeline.predict(dev_frame), dtype=np.float64)
    actual = dev_frame[REGRESSOR_TARGET_COLUMN].to_numpy(dtype=np.float64)
    return actual, predicted, importances


def compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum_squares = float(np.sum((actual - predicted) ** 2))
    total_sum_squares = float(np.sum((actual - actual.mean()) ** 2))
    if total_sum_squares == 0:
        return math.nan
    return 1.0 - residual_sum_squares / total_sum_squares


def bootstrap_r2_standard_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> float:
    n = len(actual)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_actual = actual[indices]
    resampled_predicted = predicted[indices]
    residual_sum_squares = np.sum((resampled_actual - resampled_predicted) ** 2, axis=1)
    means = resampled_actual.mean(axis=1, keepdims=True)
    total_sum_squares = np.sum((resampled_actual - means) ** 2, axis=1)
    valid = total_sum_squares > 0
    r2 = np.full(n_bootstrap, math.nan)
    r2[valid] = 1.0 - residual_sum_squares[valid] / total_sum_squares[valid]
    r2_values = r2[valid]
    if len(r2_values) < 2:
        return math.nan
    return float(np.std(r2_values, ddof=1))
