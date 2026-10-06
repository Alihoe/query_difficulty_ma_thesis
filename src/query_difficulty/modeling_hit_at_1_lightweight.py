# src/query_difficulty/modeling_hit_at_1_lightweight.py
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.features_hit_at_1_lightweight import LightweightHitAtOnePredictorFeatureBuilder
from query_difficulty.modeling import (
    HitAtOnePredictorRunner,
    PREDICTOR_MODE_LOCAL,
    PredictorFeatureViews,
    _display_retriever_name,
    _hit_at_1_target,
    _hit_at_1_threshold_policy,
    _positive_class_probabilities,
    _prediction_model_dir,
    _prediction_model_root,
    _prediction_output_paths,
    _prediction_split_output_paths,
    _sha256_path,
)
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME = "lightweight/hit_at_1"
LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION = 2


class LightweightHitAtOnePredictorRunner(HitAtOnePredictorRunner):
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: LightweightHitAtOnePredictorFeatureBuilder,
        retrieval: RetrievalManager,
        predictor_name: str = LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.views = PredictorFeatureViews(store, console, features, retrieval)
        self.predictor_name = predictor_name

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        selected = self._ordered_retrievers(retrievers)
        original_scopes = self._configured_document_scopes()
        try:
            for document_scope, scope_retrievers in self._scope_phases(selected):
                self.console.log(
                    f"Running lightweight hit@1 predictor scope phase {document_scope} "
                    f"for {len(scope_retrievers)} retrievers"
                )
                self._set_enabled_document_scopes([document_scope])
                for retriever_name in scope_retrievers:
                    resolved_name = self.retrieval.resolve_retriever_name(retriever_name)
                    if not self.features.has_manifest_entry(resolved_name):
                        self.console.log(
                            f"Skipping {resolved_name}: no frozen lightweight feature manifest entry. "
                            "Train the full hit@1 predictor for it, then regenerate the manifest with "
                            "scripts/freeze_lightweight_hit_at_1_feature_manifest.py."
                        )
                        continue
                    metrics = self._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
                    summary[retriever_name] = metrics
        finally:
            self._set_enabled_document_scopes(original_scopes)
        return summary

    def _run_retriever(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None = None,
    ) -> dict[str, float]:
        display_name = _display_retriever_name(retriever_name, mode)
        self.console.section(f"Lightweight hit@1 predictor for {display_name}")
        prediction_paths = _prediction_output_paths(
            self.store, self.predictor_name, language, retriever_name, mode=mode
        )
        cached_metrics = self._cached_lightweight_metrics(prediction_paths, retriever_name)
        if cached_metrics is not None:
            self.console.log(f"Using cached lightweight hit@1 outputs for {display_name}")
            return cached_metrics
        feature_depth = self._feature_depth()
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        self.console.log(f"Preparing lightweight hit@1 feature context for {display_name} at depth {feature_depth}")
        requested_candidate_sizes = [feature_depth]
        train_frame = self.views.build(
            train_key,
            retriever_name,
            mode,
            source_retrievers=source_retrievers,
            candidate_sizes=requested_candidate_sizes,
        )[feature_depth]
        dev_frame = self.views.build(
            dev_key,
            retriever_name,
            mode,
            source_retrievers=source_retrievers,
            candidate_sizes=requested_candidate_sizes,
        )[feature_depth]
        output, metrics, threshold, importances, model, calibration_info = self._fit_and_predict(
            display_name,
            train_frame,
            dev_frame,
        )
        self.console.log(f"Writing lightweight hit@1 outputs for {display_name}")
        self._write_outputs(
            language,
            retriever_name,
            display_name,
            output,
            metrics,
            threshold,
            importances,
            model,
            feature_depth,
            mode=mode,
            source_retrievers=source_retrievers,
            calibration_info=calibration_info,
        )
        return metrics

    def _cached_lightweight_metrics(self, prediction_paths: dict[str, Any], retriever_name: str) -> dict[str, float] | None:
        metadata_path = prediction_paths["metadata"]
        if not metadata_path.exists():
            return None
        metadata = self.store.read_json(metadata_path)
        if int(metadata.get("metadata_version", 0)) != LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION:
            return None
        if int(metadata.get("feature_manifest_metadata_version", -1)) != self.features.manifest_metadata_version:
            return None
        if metadata.get("consensus_mode") != self.features.consensus_mode:
            return None
        if sorted(metadata.get("qualifying_features", [])) != sorted(self.features.qualifying_features(retriever_name)):
            return None
        dev_metrics = metadata.get("dev_metrics")
        if not isinstance(dev_metrics, dict):
            return None
        return {str(key): float(value) for key, value in dev_metrics.items()}

    def _write_outputs(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        output,
        metrics: dict[str, float],
        threshold: float,
        importances,
        model: Any,
        feature_depth: int,
        mode: str = PREDICTOR_MODE_LOCAL,
        source_retrievers: list[str] | None = None,
        calibration_info: dict[str, Any] | None = None,
    ) -> None:
        prediction_paths = _prediction_output_paths(
            self.store, self.predictor_name, language, retriever_name, mode=mode
        )
        prediction_paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
        prediction_paths["report"].parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(prediction_paths["prediction"], index=False)
        qualifying_features = sorted(self.features.qualifying_features(retriever_name))
        metadata = {
            "metadata_version": LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": list(source_retrievers or [retriever_name]),
            "evaluation_split": "dev",
            "feature_depth": feature_depth,
            "threshold": threshold,
            "threshold_policy": _hit_at_1_threshold_policy(),
            "qualifying_features": qualifying_features,
            "feature_manifest_path": str(self.features.manifest_path),
            "feature_manifest_metadata_version": self.features.manifest_metadata_version,
            "consensus_mode": self.features.consensus_mode,
            "calibration": calibration_info or {"method": "none"},
            "dev_metrics": metrics,
        }
        self.store.write_json(prediction_paths["metadata"], metadata)
        prediction_paths["report"].write_text(
            self._prediction_report(display_name, metadata, output),
            encoding="utf-8",
        )
        model_root = _prediction_model_root(self.store, self.predictor_name, retriever_name, mode=mode)
        joblib.dump(
            {
                "predictor_metadata_version": LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION,
                "predictor_feature_mode": mode,
                "target_retriever": retriever_name,
                "source_retrievers": list(source_retrievers or [retriever_name]),
                "pipeline": model,
                "threshold": threshold,
                "feature_depth": feature_depth,
                "threshold_policy": _hit_at_1_threshold_policy(),
                "qualifying_features": qualifying_features,
                "feature_manifest_path": str(self.features.manifest_path),
                "feature_manifest_metadata_version": self.features.manifest_metadata_version,
                "consensus_mode": self.features.consensus_mode,
                "calibration": calibration_info or {"method": "none"},
            },
            model_root / "model.joblib",
        )
        importances.to_parquet(model_root / "feature_importances.parquet", index=False)


def _lightweight_hit_at_1_model_artifact_status(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    features: LightweightHitAtOnePredictorFeatureBuilder | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    model_path = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "model.joblib"
    if not model_path.exists():
        return None, f"cached {predictor_name} model is missing at {model_path}"
    model_payload = joblib.load(model_path)
    if not isinstance(model_payload, dict):
        return None, f"cached {predictor_name} model at {model_path} does not contain a valid payload"
    if int(model_payload.get("predictor_metadata_version", -1)) != LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION:
        return None, (
            f"cached {predictor_name} model uses predictor metadata version "
            f"{model_payload.get('predictor_metadata_version')}, but version "
            f"{LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION} is required"
        )
    if features is not None:
        model_qualifying_features = sorted(model_payload.get("qualifying_features", []))
        current_qualifying_features = sorted(features.qualifying_features(retriever_name))
        if model_qualifying_features != current_qualifying_features:
            return None, (
                f"cached {predictor_name} model for {retriever_name} was trained on "
                f"{len(model_qualifying_features)} feature(s), but the current manifest now qualifies "
                f"{len(current_qualifying_features)} feature(s) for it (manifest changed since training)"
            )
    return model_payload, None


def lightweight_hit_at_1_model_is_current(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    features: LightweightHitAtOnePredictorFeatureBuilder | None = None,
) -> bool:
    _, reason = _lightweight_hit_at_1_model_artifact_status(
        store, predictor_name, retriever_name, mode=mode, features=features
    )
    return reason is None


def load_or_create_lightweight_hit_at_1_test_predictions(
    config: ProjectConfig,
    store: ArtifactStore,
    console: Console,
    features: LightweightHitAtOnePredictorFeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    predictor_name: str = LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    model_payload, reason = _lightweight_hit_at_1_model_artifact_status(
        store, predictor_name, retriever_name, mode=mode, features=features
    )
    if model_payload is None:
        console.log(f"Refreshing {predictor_name} model artifacts for {retriever_name}: {reason}")
        runner = LightweightHitAtOnePredictorRunner(config, store, console, features, retrieval, predictor_name=predictor_name)
        runner.run(language, retrievers=[retriever_name])
        model_payload, reason = _lightweight_hit_at_1_model_artifact_status(
            store, predictor_name, retriever_name, mode=mode, features=features
        )
        if model_payload is None:
            raise FileNotFoundError(
                f"Unable to prepare current {predictor_name} model for {language}/{retriever_name}: {reason}"
            )
    model_path = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "model.joblib"
    current_model_signature = _sha256_path(model_path)
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    cached = None
    if paths["prediction"].exists() and paths["metadata"].exists():
        metadata = store.read_json(paths["metadata"])
        if (
            int(metadata.get("metadata_version", 0)) == LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION
            and metadata.get("evaluation_split") == "test"
            and metadata.get("model_signature") == current_model_signature
        ):
            cached = store.read_parquet(paths["prediction"])
    if cached is not None:
        console.log(f"Using cached {predictor_name} test predictions for {retriever_name}")
        return cached
    feature_depth = int(model_payload["feature_depth"])
    threshold = float(model_payload["threshold"])
    model = model_payload["pipeline"]
    source_retrievers = [str(name) for name in model_payload.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring {predictor_name} test split for {retriever_name}")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frame = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=[feature_depth],
    )[feature_depth]
    probabilities = _positive_class_probabilities(model, test_frame)
    output = test_frame[["query_id", "document_id", "language", "split_name", "retriever_name", "gold_rank"]].copy()
    output["hit_at_1"] = _hit_at_1_target(test_frame).astype(bool)
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output["predicted_hit_at_1"] = output["confidence"] >= threshold
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], output)
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": LIGHTWEIGHT_HIT_AT_1_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "evaluation_split": "test",
            "feature_depth": feature_depth,
            "threshold": threshold,
            "threshold_policy": dict(model_payload.get("threshold_policy", _hit_at_1_threshold_policy())),
            "model_signature": current_model_signature,
        },
    )
    return output
