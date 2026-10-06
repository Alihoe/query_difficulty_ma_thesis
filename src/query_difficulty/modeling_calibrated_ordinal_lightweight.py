# src/query_difficulty/modeling_calibrated_ordinal_lightweight.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.console import Console
from query_difficulty.modeling import (
    CalibratedOrdinalPredictorRunner,
    FeatureBuilder,
    IDENTIFIER_COLUMNS,
    PREDICTOR_MODE_LOCAL,
    PredictorContext,
    _prediction_output_paths,
    load_or_create_calibrated_ordinal_test_predictions,
)
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME = "lightweight/calibrated_ordinal"
SOLVABILITY_HEAD = "solvability"
MANIFEST_SIGNATURE_KEY = "lightweight_manifest_signature"


def stage_head_key(depth: int) -> str:
    return str(int(depth))


def manifest_entry_signature(entry: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()


def load_or_create_lightweight_calibrated_ordinal_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    return load_or_create_calibrated_ordinal_test_predictions(
        store=store,
        console=console,
        features=features,
        retrieval=retrieval,
        language=language,
        retriever_name=retriever_name,
        mode=mode,
        predictor_name=LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME,
    )


class LightweightCalibratedOrdinalPredictorRunner(CalibratedOrdinalPredictorRunner):
    predictor_name = LIGHTWEIGHT_CALIBRATED_ORDINAL_PREDICTOR_NAME

    def __init__(self, *args, manifest_path: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.manifest_path = manifest_path
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            self._manifest = json.load(handle)
        self._active_retriever: str | None = None

    def _entry(self, retriever_name: str) -> dict[str, Any] | None:
        return self._manifest.get("retrievers", {}).get(retriever_name)

    def _is_per_head(self, retriever_name: str | None) -> bool:
        entry = self._entry(retriever_name) if retriever_name is not None else None
        return entry is not None and "heads" in entry

    def _qualifying_features(self, retriever_name: str) -> list[str] | None:
        entry = self._entry(retriever_name)
        if entry is None:
            return None
        if "heads" in entry:
            return sorted({feature for head in entry["heads"].values() for feature in head["qualifying_features"]})
        return list(entry["qualifying_features"])

    def _head_features(self, retriever_name: str, head: str) -> list[str]:
        heads = self._entry(retriever_name)["heads"]
        if head not in heads:
            raise ValueError(
                f"Per-head lightweight manifest {self.manifest_path} has no '{head}' head for {retriever_name}; "
                f"available heads: {sorted(heads)}"
            )
        return list(heads[head]["qualifying_features"])

    @staticmethod
    def _restrict_columns(frame: pd.DataFrame, qualifying_features: list[str]) -> pd.DataFrame:
        keep_columns = [
            column for column in frame.columns
            if column in IDENTIFIER_COLUMNS or column in qualifying_features
        ]
        return frame[keep_columns].copy()

    def _context(self, language: str, retriever_name: str, mode: str, source_retrievers=None) -> PredictorContext:
        context = super()._context(language, retriever_name, mode, source_retrievers=source_retrievers)
        if self._entry(retriever_name) is None or self._is_per_head(retriever_name):
            return context
        qualifying_features = self._qualifying_features(retriever_name)
        return PredictorContext(
            train_frames={
                depth: self._restrict_columns(frame, qualifying_features)
                for depth, frame in context.train_frames.items()
            },
            dev_frames={
                depth: self._restrict_columns(frame, qualifying_features)
                for depth, frame in context.dev_frames.items()
            },
            test_frames={},
        )

    def _run_retriever(self, language: str, retriever_name: str, mode: str, source_retrievers=None):
        self._active_retriever = retriever_name
        return super()._run_retriever(language, retriever_name, mode, source_retrievers=source_retrievers)

    def _fit_solvability_model(self, training_frame: pd.DataFrame, display_name: str):
        if not self._is_per_head(self._active_retriever):
            return super()._fit_solvability_model(training_frame, display_name)
        features = self._head_features(self._active_retriever, SOLVABILITY_HEAD)
        self.console.log(
            f"Fitting solvability head for {display_name} on its {len(features)} per-head features: {features}"
        )
        return super()._fit_solvability_model(self._restrict_columns(training_frame, features), display_name)

    def _fit_cumulative_models(self, training_frames: dict[int, pd.DataFrame], display_name: str):
        if not self._is_per_head(self._active_retriever):
            return super()._fit_cumulative_models(training_frames, display_name)
        stage_models = {}
        for depth in sorted(training_frames):
            features = self._head_features(self._active_retriever, stage_head_key(depth))
            self.console.log(
                f"Fitting stage head k={int(depth)} for {display_name} on its {len(features)} per-head features: "
                f"{features}"
            )
            restricted = {int(depth): self._restrict_columns(training_frames[int(depth)], features)}
            stage_models.update(super()._fit_cumulative_models(restricted, display_name))
        return stage_models

    def _metadata_path(self, language: str, retriever_name: str, mode: str) -> Path:
        return _prediction_output_paths(self.store, self.predictor_name, language, retriever_name, mode=mode)["metadata"]

    def _cache_status(self, language: str, retriever_name: str, mode: str, candidate_sizes: list[int]):
        metrics, selection_metrics, reason = super()._cache_status(language, retriever_name, mode, candidate_sizes)
        if metrics is None:
            return metrics, selection_metrics, reason
        metadata_path = self._metadata_path(language, retriever_name, mode)
        cached_signature = json.loads(metadata_path.read_text(encoding="utf-8")).get(MANIFEST_SIGNATURE_KEY)
        current_signature = manifest_entry_signature(self._entry(retriever_name))
        if cached_signature != current_signature:
            return None, None, (
                f"cached outputs were trained with a different lightweight manifest entry than the one in "
                f"{self.manifest_path} (cached signature {cached_signature or '<missing>'}, "
                f"current {current_signature})"
            )
        return metrics, selection_metrics, None

    def _write_outputs(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        output: pd.DataFrame,
        mode: str,
        *args,
        **kwargs,
    ) -> None:
        super()._write_outputs(language, retriever_name, display_name, output, mode, *args, **kwargs)
        metadata_path = self._metadata_path(language, retriever_name, mode)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata[MANIFEST_SIGNATURE_KEY] = manifest_entry_signature(self._entry(retriever_name))
        metadata["lightweight_manifest_path"] = str(self.manifest_path)
        metadata["lightweight_manifest_per_head"] = self._is_per_head(retriever_name)
        self.store.write_json(metadata_path, metadata)

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        selected = self._ordered_retrievers(retrievers)
        for retriever_name in selected:
            resolved_name = self.retrieval.resolve_retriever_name(retriever_name)
            if self._entry(resolved_name) is None:
                self.console.log(
                    f"Skipping {resolved_name}: no lightweight feature manifest entry at {self.manifest_path}"
                )
                continue
            layout = "per-head" if self._is_per_head(resolved_name) else "shared across heads"
            self.console.log(f"{resolved_name}: using {layout} lightweight feature sets from {self.manifest_path}")
            metrics, _ = self._run_retriever(language, resolved_name, PREDICTOR_MODE_LOCAL)
            summary[resolved_name] = metrics
        return summary
