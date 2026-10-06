# src/query_difficulty/modeling_in_top_max_depth_lightweight.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.modeling import (
    IDENTIFIER_COLUMNS,
    PREDICTOR_MODE_LOCAL,
    FeatureBuilder,
    InTopMaxDepthPredictorRunner,
    load_or_create_max_depth_test_predictions,
)
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME = "lightweight/in_top_max_depth"


def load_or_create_lightweight_max_depth_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    return load_or_create_max_depth_test_predictions(
        store=store,
        console=console,
        features=features,
        retrieval=retrieval,
        language=language,
        retriever_name=retriever_name,
        mode=mode,
        predictor_name=LIGHTWEIGHT_IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
    )


class LightweightInTopMaxDepthPredictorRunner(InTopMaxDepthPredictorRunner):
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
        manifest_path: str,
        predictor_name: str,
        threshold_policy_name: str,
        threshold_objective: str,
    ) -> None:
        super().__init__(
            config,
            store,
            console,
            features,
            retrieval,
            predictor_name=predictor_name,
            threshold_policy_name=threshold_policy_name,
            threshold_objective=threshold_objective,
        )
        self.manifest_path = manifest_path
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            self._manifest = json.load(handle)

    def _qualifying_features(self, retriever_name: str) -> list[str] | None:
        entry = self._manifest.get("retrievers", {}).get(retriever_name)
        if entry is None:
            return None
        return list(entry["qualifying_features"])

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        selected = self._ordered_retrievers(retrievers)
        for retriever_name in selected:
            resolved_name = self.retrieval.resolve_retriever_name(retriever_name)
            qualifying_features = self._qualifying_features(resolved_name)
            if qualifying_features is None:
                self.console.log(
                    f"Skipping {resolved_name}: no lightweight feature manifest entry at {self.manifest_path}"
                )
                continue
            summary[resolved_name] = self._run_lightweight_retriever(language, resolved_name, qualifying_features)
        return summary

    def _run_lightweight_retriever(
        self,
        language: str,
        retriever_name: str,
        qualifying_features: list[str],
    ) -> dict[str, float]:
        display_name = retriever_name
        self.console.section(f"{self._depth_label()} lightweight predictor for {display_name}")
        self.console.log(f"Using {len(qualifying_features)} qualifying feature(s): {qualifying_features}")
        feature_depths = self._feature_depths()
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        train_frames_by_depth = self.views.build(
            train_key, retriever_name, PREDICTOR_MODE_LOCAL, candidate_sizes=feature_depths
        )
        dev_frames_by_depth = self.views.build(
            dev_key, retriever_name, PREDICTOR_MODE_LOCAL, candidate_sizes=feature_depths
        )
        train_frame = self._merge_multi_depth_frames(train_frames_by_depth)
        dev_frame = self._merge_multi_depth_frames(dev_frames_by_depth)
        keep_columns = [
            column for column in train_frame.columns
            if column in IDENTIFIER_COLUMNS or column in qualifying_features
        ]
        train_subset = train_frame[keep_columns].copy()
        dev_subset = dev_frame[keep_columns].copy()
        output, metrics, threshold, importances, model, calibration_info, threshold_selection = self._fit_and_predict(
            display_name, train_subset, dev_subset
        )
        self.console.log(f"Writing lightweight in-top-max-depth outputs for {display_name}")
        self._write_outputs(
            language,
            retriever_name,
            display_name,
            output,
            metrics,
            threshold,
            importances,
            model,
            feature_depths,
            calibration_info=calibration_info,
            threshold_selection=threshold_selection,
        )
        return metrics
