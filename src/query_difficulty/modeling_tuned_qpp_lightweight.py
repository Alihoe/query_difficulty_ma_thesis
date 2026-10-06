# src/query_difficulty/modeling_tuned_qpp_lightweight.py
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    TUNED_QPP_METADATA_VERSION,
    TUNED_QPP_MODEL_METADATA_VERSION,
    FeatureBuilder,
    PredictorFeatureViews,
    TunedQppPredictorRunner,
    _normalize_qpp_score,
    _positive_class_probabilities,
    _prediction_model_dir,
    _prediction_split_output_paths,
    _qpp_model_artifact_status,
    _safe_prediction_frame,
    _sha256_path,
)
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME = "lightweight/tuned_qpp"
LIGHTWEIGHT_TUNED_QPP_SPLIT_METADATA_VERSION = 1


class LightweightTunedQppPredictorRunner(TunedQppPredictorRunner):
    predictor_name = LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME

    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
        failure_manifest_path: str,
        regressor_manifest_path: str,
    ) -> None:
        super().__init__(config, store, console, features, retrieval)
        self.failure_manifest_path = failure_manifest_path
        self.regressor_manifest_path = regressor_manifest_path
        with Path(failure_manifest_path).open("r", encoding="utf-8") as handle:
            self._failure_manifest = json.load(handle)
        with Path(regressor_manifest_path).open("r", encoding="utf-8") as handle:
            self._regressor_manifest = json.load(handle)

    def _qualifying_features(self, manifest: dict, retriever_name: str) -> list[str] | None:
        entry = manifest.get("retrievers", {}).get(retriever_name)
        if entry is None:
            return None
        return list(entry["qualifying_features"])

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary: dict[str, dict[str, float]] = {}
        selected = [self.retrieval.resolve_retriever_name(name) for name in (retrievers or self.retrieval.enabled_retrievers())]
        for retriever_name in selected:
            failure_columns = self._qualifying_features(self._failure_manifest, retriever_name)
            score_columns = self._qualifying_features(self._regressor_manifest, retriever_name)
            if failure_columns is None or score_columns is None:
                self.console.log(
                    f"Skipping {retriever_name}: missing lightweight tuned QPP manifest entry "
                    f"(failure: {self.failure_manifest_path}, regressor: {self.regressor_manifest_path})"
                )
                continue
            self.console.log(
                f"Using {len(failure_columns)} failure-classifier feature(s) and {len(score_columns)} "
                f"score-regressor feature(s) for {retriever_name}"
            )
            metrics, _selection_metrics = self._run_retriever(
                language,
                retriever_name,
                PREDICTOR_MODE_LOCAL,
                failure_feature_columns=failure_columns,
                score_feature_columns=score_columns,
            )
            summary[retriever_name] = metrics
        return summary


def _lightweight_tuned_qpp_model_signature(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    manifest: dict[str, Any],
    mode: str = PREDICTOR_MODE_LOCAL,
) -> str:
    model_dir = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    failure_model_path = model_dir / "failure_model.joblib"
    if not failure_model_path.exists():
        raise FileNotFoundError(f"Missing {predictor_name} failure model for {retriever_name}: {failure_model_path}")
    digest.update(_sha256_path(failure_model_path).encode("utf-8"))
    if manifest.get("score_source") == "regression_full":
        score_model_path = model_dir / "score_model.joblib"
        if not score_model_path.exists():
            raise FileNotFoundError(f"Missing {predictor_name} score model for {retriever_name}: {score_model_path}")
        digest.update(_sha256_path(score_model_path).encode("utf-8"))
    return digest.hexdigest()


def _lightweight_tuned_qpp_cached_test_predictions(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    manifest: dict[str, Any],
    current_model_signature: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame | None:
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    if not paths["prediction"].exists() or not paths["metadata"].exists():
        return None
    metadata = store.read_json(paths["metadata"])
    if int(metadata.get("metadata_version", 0)) != LIGHTWEIGHT_TUNED_QPP_SPLIT_METADATA_VERSION:
        return None
    if metadata.get("evaluation_split") != "test":
        return None
    if int(metadata.get("predictor_metadata_version", 0)) != TUNED_QPP_METADATA_VERSION:
        return None
    if metadata.get("input_compatibility") != manifest.get("input_compatibility"):
        return None
    if metadata.get("source_retrievers") != manifest.get("source_retrievers"):
        return None
    if metadata.get("score_source") != manifest.get("score_source"):
        return None
    if metadata.get("mapping") != manifest.get("mapping"):
        return None
    for key in ("feature_depth", "minimum_size", "maximum_size"):
        cached_value = metadata.get(key)
        manifest_value = manifest.get(key)
        if cached_value is None or manifest_value is None or int(cached_value) != int(manifest_value):
            return None
    for key in ("train_min", "train_max", "failure_threshold"):
        cached_value = metadata.get(key)
        manifest_value = manifest.get(key)
        if cached_value is None or manifest_value is None:
            return None
        if not math.isclose(float(cached_value), float(manifest_value), rel_tol=0.0, abs_tol=1e-12):
            return None
    if metadata.get("model_signature") != current_model_signature:
        return None
    return store.read_parquet(paths["prediction"])


def load_or_create_lightweight_tuned_qpp_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = LIGHTWEIGHT_TUNED_QPP_PREDICTOR_NAME,
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    manifest, reason = _qpp_model_artifact_status(
        store,
        retriever_name,
        mode=mode,
        predictor_name=predictor_name,
        metadata_version=TUNED_QPP_METADATA_VERSION,
        model_metadata_version=TUNED_QPP_MODEL_METADATA_VERSION,
    )
    if manifest is None:
        raise FileNotFoundError(
            f"Missing {predictor_name} model artifacts for {language}/{retriever_name}: {reason}. "
            f"Run the lightweight tuned QPP predictor first to create train-only model artifacts."
        )
    if manifest.get("mapping") != "isotonic":
        raise ValueError(
            f"{predictor_name} model manifest for {language}/{retriever_name} uses mapping "
            f"{manifest.get('mapping')!r}; this loader only supports the isotonic mapping used by tuned QPP."
        )
    model_signature = _lightweight_tuned_qpp_model_signature(
        store, predictor_name, retriever_name, manifest, mode=mode
    )
    cached = _lightweight_tuned_qpp_cached_test_predictions(
        store, predictor_name, language, retriever_name, manifest, model_signature, mode=mode,
    )
    if cached is not None:
        console.log(f"Using cached {predictor_name} test predictions for {retriever_name}")
        return cached
    feature_depth = manifest.get("feature_depth")
    if feature_depth is None:
        raise ValueError(f"{predictor_name} model manifest for {language}/{retriever_name} is missing feature_depth")
    feature_depth = int(feature_depth)
    minimum_size = int(manifest["minimum_size"])
    maximum_size = int(manifest["maximum_size"])
    train_min = float(manifest["train_min"])
    train_max = float(manifest["train_max"])
    failure_threshold = float(manifest["failure_threshold"])
    source_retrievers = [str(name) for name in manifest.get("source_retrievers") or [retriever_name]]
    console.log(
        f"Scoring {predictor_name} test split for {retriever_name}: refitting the isotonic score-to-size "
        "mapping from persisted train-split artifacts (the fitted calibrator itself is not persisted at "
        "training time, only the score model and failure model it is derived from)"
    )
    views = PredictorFeatureViews(store, console, features, retrieval)
    train_frame = views.build(
        f"{language}_train",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=[feature_depth],
    )[feature_depth]
    test_frame = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=[feature_depth],
    )[feature_depth]
    model_dir = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    failure_model = joblib.load(model_dir / "failure_model.joblib")
    score_source = str(manifest["score_source"])
    if score_source == "regression_full":
        score_pipeline = joblib.load(model_dir / "score_model.joblib")
        train_scores = score_pipeline.predict(train_frame)
        test_scores = score_pipeline.predict(test_frame)
    else:
        train_scores = train_frame[score_source].to_numpy(dtype=np.float64)
        test_scores = test_frame[score_source].to_numpy(dtype=np.float64)
    train_normalized_scores = np.asarray(
        [_normalize_qpp_score(value, train_min, train_max) for value in train_scores],
        dtype=np.float64,
    )
    non_failure_mask = ~train_frame["retrieval_failure_within_max_depth"].astype(bool).to_numpy()
    calibrator = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=minimum_size, y_max=maximum_size)
    calibrator.fit(
        train_normalized_scores[non_failure_mask],
        np.clip(
            train_frame.loc[non_failure_mask, "gold_rank"].astype(float).to_numpy(),
            minimum_size,
            maximum_size,
        ),
    )
    test_normalized_scores = np.asarray(
        [_normalize_qpp_score(value, train_min, train_max) for value in test_scores],
        dtype=np.float64,
    )
    predicted_sizes = calibrator.predict(test_normalized_scores)
    selected_sizes = [
        int(min(max(math.ceil(value - 1e-12), minimum_size), maximum_size)) for value in predicted_sizes
    ]
    hit_probabilities = _positive_class_probabilities(failure_model, test_frame)
    output = test_frame[["query_id", "document_id", "gold_rank", "smallest_sufficient_size"]].copy()
    output["confidence"] = np.asarray(hit_probabilities, dtype=np.float64)
    output["predicted_failure"] = output["confidence"] < failure_threshold
    output["normalized_qpp_score"] = test_normalized_scores
    output["selected_size"] = [
        math.nan if failure else float(size)
        for failure, size in zip(output["predicted_failure"].tolist(), selected_sizes, strict=True)
    ]
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], _safe_prediction_frame(output))
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": LIGHTWEIGHT_TUNED_QPP_SPLIT_METADATA_VERSION,
            "predictor_metadata_version": TUNED_QPP_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "feature_depth": feature_depth,
            "score_source": score_source,
            "mapping": "isotonic",
            "minimum_size": minimum_size,
            "maximum_size": maximum_size,
            "train_min": train_min,
            "train_max": train_max,
            "failure_threshold": failure_threshold,
            "training_split": manifest.get("training_split", "train"),
            "evaluation_split": "test",
            "input_compatibility": manifest.get("input_compatibility", {}),
            "model_signature": model_signature,
        },
    )
    return output
