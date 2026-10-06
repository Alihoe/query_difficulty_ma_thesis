# src/query_difficulty/lightweight_selection.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES

NO_CONSENSUS_PREDICTOR_SUFFIX = "_no_consensus"


def no_consensus_predictor_name(predictor_name: str) -> str:
    return f"{predictor_name}{NO_CONSENSUS_PREDICTOR_SUFFIX}"


def base_feature_name(feature: str) -> str:
    return feature.split("::")[-1]


def uses_consensus_features(importances_path: Path) -> bool:
    frame = pd.read_parquet(importances_path)
    return bool(frame["feature"].map(base_feature_name).isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES).any())


def selection_source_predictor(store_root: Path, predictor_name: str, retriever_name: str, importances_file: str) -> str:
    full_path = store_root / "models" / predictor_name / retriever_name / importances_file
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full-model importances for {predictor_name}/{retriever_name}: {full_path}")
    if not uses_consensus_features(full_path):
        return predictor_name
    source = no_consensus_predictor_name(predictor_name)
    source_path = store_root / "models" / source / retriever_name / importances_file
    if not source_path.exists():
        raise FileNotFoundError(
            f"The full {predictor_name} model for {retriever_name} uses multi-retriever consensus features, so the "
            f"lightweight selection must start from the consensus-free full model, but {source_path} is missing. "
            "Run scripts/train_full_predictors_without_consensus.py first."
        )
    return source
