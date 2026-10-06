# src/query_difficulty/modeling_in_top_k_sweep_lightweight.py
from __future__ import annotations

import pandas as pd

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.console import Console
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    FeatureBuilder,
    load_or_create_in_top_k_test_predictions,
)
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX = "lightweight/in_top_k"
LIGHTWEIGHT_IN_TOP_K_SWEEP_MANIFEST_METADATA_VERSION = 2


def load_or_create_lightweight_in_top_k_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    target_depth: int,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    return load_or_create_in_top_k_test_predictions(
        store=store,
        console=console,
        features=features,
        retrieval=retrieval,
        language=language,
        retriever_name=retriever_name,
        target_depth=target_depth,
        mode=mode,
        predictor_name_prefix=LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX,
    )
