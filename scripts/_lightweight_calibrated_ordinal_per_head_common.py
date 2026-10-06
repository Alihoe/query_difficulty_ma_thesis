from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from query_difficulty.modeling import (
    IDENTIFIER_COLUMNS,
    PREDICTOR_MODE_LOCAL,
    CalibratedOrdinalPredictorRunner,
    _fit_calibrated_classifier,
    _positive_class_probabilities,
)
from query_difficulty.modeling_calibrated_ordinal_lightweight import SOLVABILITY_HEAD, stage_head_key

PREDICTOR_NAME = "calibrated_ordinal"
REPORT_TAG = "calibrated_ordinal_per_head"
DEFAULT_RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]
FREEZE_MANIFEST = "config/lightweight_calibrated_ordinal_features.per_head.json"
CORRELATION_ROUND_1_MANIFEST = "config/lightweight_calibrated_ordinal_features.per_head.correlation_pruned.json"
CORRELATION_ROUND_2_MANIFEST = "config/lightweight_calibrated_ordinal_features.per_head.correlation_pruned_round2.json"
RFE_MANIFEST = "config/lightweight_calibrated_ordinal_features.per_head.rfe_pruned.json"


def head_keys(candidate_sizes: list[int]) -> list[str]:
    return [SOLVABILITY_HEAD] + [stage_head_key(depth) for depth in sorted(candidate_sizes)]


def head_label(head: str) -> str:
    return "solvability head" if head == SOLVABILITY_HEAD else f"stage head k={head}"


def head_importances_path(store_root: Path, retriever_name: str, head: str) -> Path:
    model_root = store_root / "models" / PREDICTOR_NAME / retriever_name
    if head == SOLVABILITY_HEAD:
        return model_root / "solvability_importances.parquet"
    return model_root / f"depth_{int(head):04d}_importances.parquet"


def load_head_importances(store_root: Path, retriever_name: str, head: str) -> pd.Series:
    return pd.read_parquet(head_importances_path(store_root, retriever_name, head)).set_index("feature")["importance"]


def build_frames_by_depth(runner: CalibratedOrdinalPredictorRunner, split_key: str, retriever_name: str) -> dict[int, pd.DataFrame]:
    candidate_sizes = runner._candidate_sizes()
    return runner.views.build(split_key, retriever_name, PREDICTOR_MODE_LOCAL, candidate_sizes=candidate_sizes)


def head_training_rows(frames_by_depth: dict[int, pd.DataFrame], head: str) -> pd.DataFrame:
    if head == SOLVABILITY_HEAD:
        return frames_by_depth[max(frames_by_depth)]
    frame = frames_by_depth[int(head)]
    return frame.loc[frame["smallest_sufficient_size"].notna()]


def head_target(rows: pd.DataFrame, head: str) -> pd.Series:
    if head == SOLVABILITY_HEAD:
        return rows["smallest_sufficient_size"].notna().astype(int)
    return rows["gold_in_top_k"].astype(int)


def is_constant_head(train_rows: pd.DataFrame, head: str) -> bool:
    return head_target(train_rows, head).nunique() < 2


def restrict_columns(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    keep_columns = [column for column in frame.columns if column in IDENTIFIER_COLUMNS or column in features]
    return frame[keep_columns].copy()


def fit_head(runner: CalibratedOrdinalPredictorRunner, train_rows: pd.DataFrame, head: str, features: list[str], purpose: str):
    subset = restrict_columns(train_rows, features)
    return _fit_calibrated_classifier(
        console=runner.console,
        train_frame=subset,
        feature_columns=[feature for feature in features if feature in subset.columns],
        target=head_target(subset, head),
        base_model_name=runner.config.predictors.ordinal.base_model,
        seed=runner.config.runtime.random_seed,
        purpose=purpose,
        method="isotonic",
    )


def positive_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(_positive_class_probabilities(model, frame), dtype=np.float64)


def bootstrap_mean_standard_error(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return math.nan
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    return float(np.std(values[indices].mean(axis=1), ddof=1))


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)
