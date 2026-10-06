from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from query_difficulty.analysis import ordinal_predictor_metric_summary
from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.features import DESCRIPTIVE_FEATURE_SCHEMA_VERSION, FeatureBuilder, QUERY_SIGNAL_SCHEMA_VERSION
from query_difficulty.retrieval import LEXICAL_RETRIEVERS, RetrievalManager, base_retriever_name, retriever_document_scope


IDENTIFIER_COLUMNS = {
    "query_id",
    "document_id",
    "language",
    "split_name",
    "retriever_name",
    "candidate_size",
    "gold_rank",
    "smallest_sufficient_size",
    "retrieval_failure_within_max_depth",
    "rr_at_max_depth",
    "gold_in_top_k",
}

PREDICTOR_MODE_LOCAL = "local"
PREDICTOR_MODE_ALL_MEANINGFUL = "all_meaningful"
PREDICTOR_MODES = {PREDICTOR_MODE_LOCAL, PREDICTOR_MODE_ALL_MEANINGFUL}
PREDICTOR_VIEW_JOIN_COLUMNS = ["query_id", "document_id", "candidate_size"]
PREDICTOR_VIEW_METADATA_VERSION = 2
PREDICTOR_FEATURE_DEPENDENCY_VERSION = DESCRIPTIVE_FEATURE_SCHEMA_VERSION
HIT_AT_1_METADATA_VERSION = 5
HIT_AT_1_FEATURE_DEPENDENCY_VERSION = PREDICTOR_FEATURE_DEPENDENCY_VERSION
QPP_METADATA_VERSION = 4
CALIBRATED_ORDINAL_METADATA_VERSION = 6
CALIBRATED_ORDINAL_DECODING_NAME = "frank_hall_argmax"
QPP_MODEL_METADATA_VERSION = 2
QPP_SPLIT_METADATA_VERSION = 2
TUNED_QPP_METADATA_VERSION = 1
TUNED_QPP_MODEL_METADATA_VERSION = 1
CALIBRATED_ORDINAL_MODEL_METADATA_VERSION = 2
CALIBRATED_ORDINAL_SPLIT_METADATA_VERSION = 2
HIT_AT_1_THRESHOLD_POLICY_NAME = "maximize_f0_5"
QPP_FAILURE_THRESHOLD_POLICY_NAME = "maximize_failure_f1"
IN_TOP_MAX_DEPTH_METADATA_VERSION = 3
IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION = PREDICTOR_FEATURE_DEPENDENCY_VERSION
IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME = "cv_maximize_mcc_balanced_rf"
IN_TOP_MAX_DEPTH_CV_FOLDS = 5
IN_TOP_MAX_DEPTH_CLASS_WEIGHT = "balanced"
IN_TOP_MAX_DEPTH_PREDICTOR_NAME = "in_top_max_depth"
IN_TOP_K_SWEEP_SPLIT_METADATA_VERSION = 1


def _selection_prediction_report(
    predictor_name: str,
    retriever_name: str,
    metadata: dict[str, Any],
    most_confident: pd.DataFrame,
    least_confident: pd.DataFrame,
) -> str:
    metrics = metadata["dev_metrics"]
    lines = [
        f"# {predictor_name} predictor report for {retriever_name}",
        "",
        "## Metrics",
        "",
        f"- Evaluation split: {metadata.get('evaluation_split', 'dev')}",
        f"- Candidate recall: {metrics['candidate_recall']:.4f}",
        f"- Mean selected size: {metrics['mean_selected_size']:.2f}",
        f"- Median selected size: {metrics['median_selected_size']:.2f}",
        f"- Failure rate: {metrics['failure_rate']:.4f}",
    ]
    if "dfbeta_1" in metrics:
        lines.append(f"- DFβ (β=1.0): {metrics['dfbeta_1']:.4f}")
    lines.extend([
        "",
        "## Most confident predictions",
        "",
    ])
    for row in most_confident.itertuples(index=False):
        lines.append(
            f"- Query {row.query_id}: confidence={row.confidence:.4f}, selected_size={row.selected_size}, "
            f"failure={row.predicted_failure}, signals={row.strong_feature_signals}"
        )
    lines.extend(["", "## Least confident predictions", ""])
    for row in least_confident.itertuples(index=False):
        lines.append(
            f"- Query {row.query_id}: confidence={row.confidence:.4f}, selected_size={row.selected_size}, "
            f"failure={row.predicted_failure}, signals={row.strong_feature_signals}"
        )
    return "\n".join(lines)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in IDENTIFIER_COLUMNS]


def _default_metrics() -> dict[str, float]:
    return {
        "candidate_recall": math.nan,
        "mean_selected_size": math.nan,
        "median_selected_size": math.nan,
        "failure_rate": math.nan,
    }


def _fbeta_score(precision: float, recall: float, beta: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall):
        return math.nan
    beta_squared = float(beta) ** 2
    denominator = (beta_squared * precision) + recall
    if denominator == 0.0:
        return 0.0
    return float(((1.0 + beta_squared) * precision * recall) / denominator)


def _normalize_predictor_modes(config: ProjectConfig | Any) -> list[str]:
    configured = getattr(config, "predictor_feature_modes", None) or [PREDICTOR_MODE_LOCAL]
    modes: list[str] = []
    for raw_mode in configured:
        mode = str(raw_mode)
        if mode not in PREDICTOR_MODES:
            raise ValueError(f"Unsupported predictor feature mode: {mode}")
        if mode not in modes:
            modes.append(mode)
    return modes or [PREDICTOR_MODE_LOCAL]


def _display_retriever_name(retriever_name: str, mode: str) -> str:
    if mode == PREDICTOR_MODE_LOCAL:
        return retriever_name
    return f"{retriever_name} [{mode}]"


def _prediction_output_root(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> tuple[Path, Path]:
    if mode == PREDICTOR_MODE_LOCAL:
        prediction_root = store.root_dir / "predictions" / predictor_name / language / retriever_name
        report_root = store.root_dir / "reports" / predictor_name / language / retriever_name
    else:
        prediction_root = store.root_dir / "predictions" / predictor_name / language / mode / retriever_name
        report_root = store.root_dir / "reports" / predictor_name / language / mode / retriever_name
    return prediction_root, report_root


def _prediction_output_paths(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, Path]:
    prediction_root, report_root = _prediction_output_root(
        store,
        predictor_name,
        language,
        retriever_name,
        mode=mode,
    )
    return {
        "prediction": prediction_root / "predictions.parquet",
        "metadata": prediction_root / "metadata.json",
        "report": report_root / "prediction_summary.md",
    }


def _predictor_completed_retrievers(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> set[str]:
    if mode == PREDICTOR_MODE_LOCAL:
        root = store.root_dir / "predictions" / predictor_name / language
    else:
        root = store.root_dir / "predictions" / predictor_name / language / mode
    if not root.exists():
        return set()
    completed = set()
    for child in root.iterdir():
        if child.is_dir() and (child / "metadata.json").exists() and (child / "predictions.parquet").exists():
            completed.add(child.name)
    return completed


def retrievers_ready_for_predictors(
    store: ArtifactStore,
    retrieval: RetrievalManager,
    language: str,
    predictor_names: list[str],
    mode: str = PREDICTOR_MODE_LOCAL,
    candidates: list[str] | None = None,
) -> list[str]:
    candidate_list = [
        retrieval.resolve_retriever_name(name)
        for name in (candidates if candidates is not None else retrieval.enabled_retrievers())
    ]
    ready = set(candidate_list)
    for predictor_name in predictor_names:
        ready &= _predictor_completed_retrievers(store, predictor_name, language, mode=mode)
    return [name for name in candidate_list if name in ready]


def _prediction_split_output_paths(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    split_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, Path]:
    prediction_root, _ = _prediction_output_root(
        store,
        predictor_name,
        language,
        retriever_name,
        mode=mode,
    )
    return {
        "prediction": prediction_root / f"{split_name}_predictions.parquet",
        "metadata": prediction_root / f"{split_name}_metadata.json",
    }


def _prediction_model_dir(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> Path:
    if mode == PREDICTOR_MODE_LOCAL:
        return store.root_dir / "models" / predictor_name / retriever_name
    return store.root_dir / "models" / predictor_name / mode / retriever_name


def _prediction_model_root(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> Path:
    root = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coerce_metrics(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    return {
        "candidate_recall": float(value.get("candidate_recall", math.nan)),
        "mean_selected_size": float(value.get("mean_selected_size", math.nan)),
        "median_selected_size": float(value.get("median_selected_size", math.nan)),
        "failure_rate": float(value.get("failure_rate", math.nan)),
    }


def _cached_prediction_metadata(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, Any] | None:
    paths = _prediction_output_paths(store, predictor_name, language, retriever_name, mode=mode)
    if paths["metadata"].exists():
        return store.read_json(paths["metadata"])
    return None


def _cached_prediction_metadata_status(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    required_metadata_version: int | None = None,
    required_feature_dependency_version: int | None = None,
    required_query_signal_schema_version: int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None, None
    if required_metadata_version is not None and int(metadata.get("metadata_version", 0)) != int(required_metadata_version):
        return None, (
            f"cached {predictor_name} outputs use metadata version {int(metadata.get('metadata_version', 0))}, "
            f"but version {int(required_metadata_version)} is required"
        )
    if required_feature_dependency_version is not None:
        compatibility = metadata.get("input_compatibility")
        if not isinstance(compatibility, dict):
            return None, f"cached {predictor_name} outputs are missing input compatibility metadata"
        cached_dependency_version = compatibility.get("feature_dependency_version")
        if cached_dependency_version is None or int(cached_dependency_version) != int(required_feature_dependency_version):
            return None, (
                f"cached {predictor_name} outputs use feature dependency version {cached_dependency_version}, "
                f"but version {int(required_feature_dependency_version)} is required"
            )
        if required_query_signal_schema_version is not None:
            cached_query_signal_schema = compatibility.get("query_signal_schema_version")
            if cached_query_signal_schema is None or int(cached_query_signal_schema) != int(required_query_signal_schema_version):
                return None, (
                    f"cached {predictor_name} outputs use query-signal schema version {cached_query_signal_schema}, "
                    f"but version {int(required_query_signal_schema_version)} is required"
                )
    return metadata, None


def _cached_prediction_cache_status(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    required_metadata_version: int | None = None,
    required_feature_dependency_version: int | None = None,
    required_query_signal_schema_version: int | None = None,
    evaluation_metric_key: str = "test_metrics",
    selection_metric_key: str | None = "dev_metrics",
) -> tuple[dict[str, float] | None, dict[str, float] | None, str | None]:
    metadata, reason = _cached_prediction_metadata_status(
        store,
        predictor_name,
        language,
        retriever_name,
        mode=mode,
        required_metadata_version=required_metadata_version,
        required_feature_dependency_version=required_feature_dependency_version,
        required_query_signal_schema_version=required_query_signal_schema_version,
    )
    if metadata is None:
        return None, None, reason
    metrics = _coerce_metrics(metadata.get(evaluation_metric_key))
    if metrics is None:
        missing_label = evaluation_metric_key.replace("_", " ")
        return None, None, f"cached {predictor_name} outputs are missing {missing_label}"
    if selection_metric_key is None or selection_metric_key == evaluation_metric_key:
        selection_metrics = metrics
    else:
        selection_metrics = _coerce_metrics(metadata.get(selection_metric_key)) or metrics
    return metrics, selection_metrics, None


def _cached_selection_metrics(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, float] | None:
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None
    return _coerce_metrics(metadata.get("dev_metrics")) or _coerce_metrics(metadata.get("test_metrics"))


def _cached_prediction_metrics(
    store: ArtifactStore,
    predictor_name: str,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, float] | None:
    paths = _prediction_output_paths(store, predictor_name, language, retriever_name, mode=mode)
    if paths["metadata"].exists():
        metadata = store.read_json(paths["metadata"])
        metrics = _coerce_metrics(metadata.get("dev_metrics")) or _coerce_metrics(metadata.get("test_metrics"))
        if metrics is not None:
            return metrics
    if not paths["prediction"].exists():
        return None
    frame = store.read_parquet(paths["prediction"])
    return _selection_metrics(frame)


def _safe_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe_frame = frame.copy()
    if "stage_probabilities" in safe_frame.columns:
        safe_frame["stage_probabilities"] = safe_frame["stage_probabilities"].map(
            lambda value: json.dumps({str(key): float(score) for key, score in value.items()}) if isinstance(value, dict) else value
        )
    return safe_frame


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if chunk:
                digest.update(chunk)
    return digest.hexdigest()


def _metrics_sort_key(metrics: dict[str, float] | None) -> tuple[float, float]:
    if metrics is None:
        return (float("-inf"), float("-inf"))
    recall = float(metrics.get("candidate_recall", math.nan))
    mean_selected_size = float(metrics.get("mean_selected_size", math.nan))
    recall_key = recall if np.isfinite(recall) else float("-inf")
    mean_key = -mean_selected_size if np.isfinite(mean_selected_size) else float("-inf")
    return recall_key, mean_key



def _coarsen_candidate_sizes(candidate_sizes: list[int] | tuple[int, ...]) -> list[int]:
    ordered = sorted({int(size) for size in candidate_sizes if int(size) > 0})
    if len(ordered) <= 5:
        return ordered
    coarsened = [ordered[0]]
    for size in ordered[1:-1]:
        previous = coarsened[-1]
        if size <= 20 or float(size) >= float(previous) * 1.6:
            coarsened.append(size)
    if ordered[-1] not in coarsened:
        coarsened.append(ordered[-1])
    return coarsened


def _map_smallest_sufficient_sizes_to_grid(
    values: pd.Series | np.ndarray,
    candidate_sizes: list[int] | tuple[int, ...],
) -> np.ndarray:
    ordered_sizes = np.asarray(sorted({int(size) for size in candidate_sizes}), dtype=np.int64)
    array = np.asarray(values, dtype=np.float64)
    output = np.full(len(array), np.nan, dtype=np.float64)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return output
    positions = np.searchsorted(ordered_sizes, array[finite_mask], side="left")
    positions = np.clip(positions, 0, len(ordered_sizes) - 1)
    output[finite_mask] = ordered_sizes[positions].astype(np.float64)
    return output


def _monotone_cumulative_probabilities(probability_by_depth: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    ordered_depths = sorted(probability_by_depth)
    matrix = np.column_stack(
        [np.asarray(probability_by_depth[int(depth)], dtype=np.float64) for depth in ordered_depths]
    )
    matrix = np.maximum.accumulate(matrix, axis=1)
    return {
        int(depth): matrix[:, index].astype(np.float64, copy=False)
        for index, depth in enumerate(ordered_depths)
    }


def _safe_sort_float(value: float, *, nan_value: float = float("-inf"), inf_value: float = float("-inf")) -> float:
    numeric = float(value)
    if math.isnan(numeric):
        return nan_value
    if math.isinf(numeric):
        return inf_value if numeric < 0.0 else numeric
    return numeric


def _binary_classification_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(actual, dtype=bool)
    guess = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    tn = int(np.sum(~truth & ~guess))
    fn = int(np.sum(truth & ~guess))
    precision = float(tp / (tp + fp)) if tp + fp else math.nan
    recall = float(tp / (tp + fn)) if tp + fn else math.nan
    return {
        "precision": precision,
        "recall": recall,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
        "positive_predictions": float(tp + fp),
    }


def _qpp_failure_threshold_policy() -> dict[str, Any]:
    return {
        "name": QPP_FAILURE_THRESHOLD_POLICY_NAME,
        "objective": "failure_f1",
        "secondary_metric": "balanced_accuracy",
        "tie_breakers": ["candidate_recall", "smaller_mean_selected_size", "smaller_threshold"],
        "prediction_rule": "predict_failure_when_success_probability_below_threshold",
    }


def _normalize_qpp_score(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if minimum == maximum:
        return 0.5
    clipped = min(max(value, minimum), maximum)
    return float((clipped - minimum) / (maximum - minimum))


def _map_qpp_score(normalized_score: float, mapping: str, lower: int, upper: int) -> int:
    if mapping == "inverse_linear":
        mapped = upper - normalized_score * (upper - lower)
    else:
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        mapped = math.exp(log_upper - normalized_score * (log_upper - log_lower))
    return int(min(max(math.ceil(mapped - 1e-12), lower), upper))


def _select_best_variants(retrievers: list[str], metrics_by_retriever: dict[str, dict[str, float] | None]) -> list[str]:
    winners: list[str] = []
    grouped: dict[str, list[str]] = {}
    for retriever_name in retrievers:
        grouped.setdefault(base_retriever_name(retriever_name), []).append(retriever_name)
    for variants in grouped.values():
        best_name = None
        best_key = (float("-inf"), float("-inf"))
        for retriever_name in variants:
            key = _metrics_sort_key(metrics_by_retriever.get(retriever_name))
            if key > best_key:
                best_name = retriever_name
                best_key = key
        if best_name is not None:
            winners.append(best_name)
    return winners


_RANDOM_FOREST_N_JOBS = 1


def configure_random_forest_n_jobs(n_jobs: int) -> None:
    global _RANDOM_FOREST_N_JOBS
    _RANDOM_FOREST_N_JOBS = max(int(n_jobs), 1)


def _make_classifier(name: str, seed: int, class_weight: str | None = None):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed)
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=_RANDOM_FOREST_N_JOBS,
        class_weight=class_weight,
    )


def _make_regressor(name: str, seed: int):
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=seed)
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=_RANDOM_FOREST_N_JOBS,
    )


def _pipeline_template(model, feature_columns: list[str], frame: pd.DataFrame) -> Pipeline:
    usable_columns = [column for column in feature_columns if not frame[column].isna().all()]
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                usable_columns,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("transformer", transformer),
            ("model", model),
        ]
    )


def _fit_pipeline(model, feature_columns: list[str], frame: pd.DataFrame, target: pd.Series) -> Pipeline:
    unique_targets = pd.Series(target).dropna().unique()
    if len(unique_targets) == 1:
        if isinstance(model, ClassifierMixin):
            model = DummyClassifier(strategy="constant", constant=int(unique_targets[0]))
        elif isinstance(model, RegressorMixin):
            model = DummyRegressor(strategy="constant", constant=float(unique_targets[0]))
    pipeline = _pipeline_template(model, feature_columns, frame)
    pipeline.fit(frame, target)
    return pipeline


def _fit_calibrated_classifier(
    *,
    console: Console,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target: pd.Series,
    base_model_name: str,
    seed: int,
    purpose: str,
    class_weight: str | None = None,
    method: str = "sigmoid",
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    importance_model = _fit_pipeline(
        _make_classifier(base_model_name, seed, class_weight=class_weight),
        feature_columns,
        train_frame,
        target,
    )
    importances = _feature_importances(importance_model, feature_columns)
    target_series = pd.Series(target).astype(int)
    positive_count = int(target_series.sum())
    negative_count = int(len(target_series) - positive_count)
    if positive_count == 0 or negative_count == 0:
        console.log(f"Skipping {purpose} probability calibration because the training split contains only one class")
        return importance_model, importances, {"method": "none", "reason": "single_class_training_data"}
    cv_folds = min(3, positive_count, negative_count)
    if cv_folds < 2:
        console.log(
            f"Skipping {purpose} probability calibration because the class counts are too small for cross-validation"
        )
        return importance_model, importances, {"method": "none", "reason": "insufficient_class_counts_for_calibration"}
    console.log(f"Calibrating {purpose} probabilities with {method} calibration over {cv_folds} folds")
    calibrated_model = CalibratedClassifierCV(
        estimator=_pipeline_template(
            _make_classifier(base_model_name, seed, class_weight=class_weight),
            feature_columns,
            train_frame,
        ),
        method=method,
        cv=cv_folds,
    )
    calibrated_model.fit(train_frame, target_series)
    return calibrated_model, importances, {"method": method, "cv_folds": int(cv_folds)}


def _positive_class_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline):
        inner_model = model.named_steps["model"]
        classes = getattr(inner_model, "classes_", None)
    if classes is None:
        return probabilities.reshape(-1)
    labels = list(classes)
    if 1 in labels:
        return probabilities[:, labels.index(1)]
    positive = 1.0 if labels and int(labels[0]) == 1 else 0.0
    return np.full(len(frame), positive, dtype=np.float64)


def _feature_importances(pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    transformer = pipeline.named_steps["transformer"]
    transformed_columns = [name.split("__", 1)[-1] for name in transformer.get_feature_names_out()]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        values = np.zeros(len(transformed_columns), dtype=np.float64)
    usable = min(len(transformed_columns), len(values))
    return pd.DataFrame({"feature": transformed_columns[:usable], "importance": values[:usable]}).sort_values("importance", ascending=False)


def _local_signal_summary(
    row: pd.Series,
    training_frame: pd.DataFrame,
    importances: pd.DataFrame,
    limit: int = 5,
) -> str:
    signals = []
    for item in importances.head(40).itertuples(index=False):
        column = item.feature
        series = training_frame[column]
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std == 0:
            continue
        value = float(row[column])
        if not np.isfinite(value):
            continue
        strength = abs((value - mean) / std) * float(item.importance)
        direction = "high" if value >= mean else "low"
        signals.append((strength, f"{column}={value:.4f} ({direction} vs train mean {mean:.4f})"))
    signals.sort(reverse=True)
    return "; ".join(text for _, text in signals[:limit])


def _candidate_recall(frame: pd.DataFrame) -> float:
    hits = []
    for row in frame.itertuples(index=False):
        if math.isnan(row.gold_rank):
            hits.append(float(row.predicted_failure))
        else:
            hits.append(float((not row.predicted_failure) and row.selected_size >= row.gold_rank))
    return float(np.mean(hits)) if hits else 0.0


def _vectorized_candidate_recall(
    gold_rank: np.ndarray,
    predicted_failure: np.ndarray,
    selected_size: np.ndarray,
) -> float:
    gold_rank = np.asarray(gold_rank, dtype=np.float64)
    predicted_failure = np.asarray(predicted_failure, dtype=bool)
    selected_size = np.asarray(selected_size, dtype=np.float64)
    if gold_rank.size == 0:
        return 0.0
    is_unsolvable = np.isnan(gold_rank)
    hit_solvable = (~predicted_failure) & (selected_size >= gold_rank)
    hits = np.where(is_unsolvable, predicted_failure, hit_solvable)
    return float(np.mean(hits))


def _dfbeta_scores(frame: pd.DataFrame, max_ranking_depth: int, beta: float = 1.0) -> np.ndarray:
    beta_squared = beta * beta
    scores = []
    for row in frame.itertuples(index=False):
        if math.isnan(row.gold_rank):
            hit = bool(row.predicted_failure)
            space_saved = 1.0
        else:
            hit = (not row.predicted_failure) and row.selected_size >= row.gold_rank
            space_saved = 0.0 if row.predicted_failure else 1.0 - float(row.selected_size) / float(max_ranking_depth)
        if not hit:
            scores.append(0.0)
        else:
            scores.append((1.0 + beta_squared) * space_saved / (beta_squared * space_saved + 1.0))
    return np.asarray(scores, dtype=np.float64)


def _selection_metrics(frame: pd.DataFrame, max_ranking_depth: int | None = None) -> dict[str, float]:
    non_failure = frame.loc[~frame["predicted_failure"], "selected_size"]
    metrics = {
        "candidate_recall": _candidate_recall(frame),
        "mean_selected_size": float(non_failure.mean()) if not non_failure.empty else math.nan,
        "median_selected_size": float(non_failure.median()) if not non_failure.empty else math.nan,
        "failure_rate": float(frame["predicted_failure"].mean()),
    }
    if max_ranking_depth is not None:
        scores = _dfbeta_scores(frame, max_ranking_depth)
        metrics["dfbeta_1"] = float(np.mean(scores)) if len(scores) else math.nan
    return metrics


def _hit_at_1_target(frame: pd.DataFrame) -> pd.Series:
    return frame["gold_rank"].fillna(-1.0).astype(float).eq(1.0)


def _hit_at_1_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["hit_at_1"].astype(bool)
    predicted = frame["predicted_hit_at_1"].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
    positive_predictions = tp + fp
    total = tp + fp + tn + fn
    precision = float(tp / positive_predictions) if positive_predictions else math.nan
    recall = float(tp / (tp + fn)) if tp + fn else math.nan
    f0_5 = _fbeta_score(precision, recall, beta=0.5)
    f1 = _fbeta_score(precision, recall, beta=1.0)
    return {
        "f0_5": f0_5,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else math.nan,
        "positive_predictions": float(positive_predictions),
        "predicted_positive_rate": float(positive_predictions / total) if total else math.nan,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def _hit_at_1_threshold_policy() -> dict[str, Any]:
    return {"name": HIT_AT_1_THRESHOLD_POLICY_NAME}


def _hit_at_1_model_artifact_status(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> tuple[dict[str, Any] | None, str | None]:
    model_path = _prediction_model_dir(store, "hit_at_1", retriever_name, mode=mode) / "model.joblib"
    if not model_path.exists():
        return None, f"cached hit@1 model is missing at {model_path}"
    model_payload = joblib.load(model_path)
    if not isinstance(model_payload, dict):
        return None, f"cached hit@1 model at {model_path} does not contain a valid payload"
    cached_metadata_version = model_payload.get("predictor_metadata_version")
    if cached_metadata_version is None or int(cached_metadata_version) != HIT_AT_1_METADATA_VERSION:
        return None, (
            f"cached hit@1 model uses predictor metadata version {cached_metadata_version}, "
            f"but version {HIT_AT_1_METADATA_VERSION} is required"
        )
    threshold_policy = model_payload.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        return None, f"cached hit@1 model at {model_path} is missing threshold policy metadata"
    cached_policy_name = str(threshold_policy.get("name", ""))
    if cached_policy_name != HIT_AT_1_THRESHOLD_POLICY_NAME:
        return None, (
            f"cached hit@1 model was tuned with threshold policy {cached_policy_name or '<missing>'}, "
            f"but {HIT_AT_1_THRESHOLD_POLICY_NAME} is now required"
        )
    cached_dependency_version = model_payload.get("feature_dependency_version")
    if cached_dependency_version is None or int(cached_dependency_version) != HIT_AT_1_FEATURE_DEPENDENCY_VERSION:
        return None, (
            f"cached hit@1 model uses feature dependency version {cached_dependency_version}, "
            f"but version {HIT_AT_1_FEATURE_DEPENDENCY_VERSION} is required"
        )
    cached_query_signal_schema = model_payload.get("query_signal_schema_version")
    if cached_query_signal_schema is None or int(cached_query_signal_schema) != QUERY_SIGNAL_SCHEMA_VERSION:
        return None, (
            f"cached hit@1 model uses query-signal schema version {cached_query_signal_schema}, "
            f"but version {QUERY_SIGNAL_SCHEMA_VERSION} is required"
        )
    for key in ("feature_depth", "threshold", "pipeline"):
        if key not in model_payload:
            return None, f"cached hit@1 model at {model_path} is missing {key}"
    return model_payload, None


def hit_at_1_model_is_current(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> bool:
    _, reason = _hit_at_1_model_artifact_status(store, retriever_name, mode=mode)
    return reason is None


def _cached_hit_at_1_cache_status(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "hit_at_1",
) -> tuple[dict[str, float] | None, str | None]:
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None, None
    if int(metadata.get("metadata_version", 0)) != HIT_AT_1_METADATA_VERSION:
        return None, (
            f"cached hit@1 outputs use metadata version {int(metadata.get('metadata_version', 0))}, "
            f"but version {HIT_AT_1_METADATA_VERSION} is required"
        )
    compatibility = metadata.get("input_compatibility")
    if not isinstance(compatibility, dict):
        return None, "cached hit@1 outputs do not record the current feature compatibility metadata"
    cached_dependency_version = compatibility.get("feature_dependency_version")
    if cached_dependency_version is None or int(cached_dependency_version) != HIT_AT_1_FEATURE_DEPENDENCY_VERSION:
        return None, (
            f"cached hit@1 outputs use feature dependency version {cached_dependency_version}, "
            f"but version {HIT_AT_1_FEATURE_DEPENDENCY_VERSION} is required"
        )
    cached_query_signal_schema = compatibility.get("query_signal_schema_version")
    if cached_query_signal_schema is None or int(cached_query_signal_schema) != QUERY_SIGNAL_SCHEMA_VERSION:
        return None, (
            f"cached hit@1 outputs use query-signal schema version {cached_query_signal_schema}, "
            f"but version {QUERY_SIGNAL_SCHEMA_VERSION} is required"
        )
    threshold_policy = metadata.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        return None, "cached hit@1 outputs are missing threshold policy metadata"
    cached_policy_name = str(threshold_policy.get("name", ""))
    if cached_policy_name != HIT_AT_1_THRESHOLD_POLICY_NAME:
        return None, (
            f"cached hit@1 outputs were tuned with threshold policy {cached_policy_name or '<missing>'}, "
            f"but {HIT_AT_1_THRESHOLD_POLICY_NAME} is now required"
        )
    _, model_reason = _hit_at_1_model_artifact_status(store, retriever_name, mode=mode)
    if model_reason is not None:
        return None, model_reason
    if not isinstance(metadata.get("dev_metrics"), dict):
        return None, "cached hit@1 outputs are missing dev metrics"
    return {str(key): float(value) for key, value in metadata["dev_metrics"].items()}, None


def _cached_hit_at_1_metrics(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, float] | None:
    metrics, _ = _cached_hit_at_1_cache_status(
        store,
        language,
        retriever_name,
        mode=mode,
    )
    return metrics


def _ensure_current_hit_at_1_model(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> dict[str, Any]:
    model_payload, reason = _hit_at_1_model_artifact_status(store, retriever_name, mode=mode)
    if model_payload is not None:
        return model_payload
    console.log(f"Refreshing hit@1 model artifacts for {retriever_name}: {reason}")
    runner = HitAtOnePredictorRunner(features.config, store, console, features, retrieval)
    runner.run(language, retrievers=[retriever_name])
    model_payload, reason = _hit_at_1_model_artifact_status(store, retriever_name, mode=mode)
    if model_payload is None:
        raise FileNotFoundError(
            f"Unable to prepare current hit@1 model for {language}/{retriever_name}: {reason}"
        )
    return model_payload


def _cached_calibrated_ordinal_cache_status(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    *,
    candidate_sizes: list[int],
    base_model: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "calibrated_ordinal",
) -> tuple[dict[str, float] | None, dict[str, float] | None, str | None]:
    metrics, selection_metrics, reason = _cached_prediction_cache_status(
        store,
        predictor_name,
        language,
        retriever_name,
        mode=mode,
        required_metadata_version=CALIBRATED_ORDINAL_METADATA_VERSION,
        required_feature_dependency_version=PREDICTOR_FEATURE_DEPENDENCY_VERSION,
        required_query_signal_schema_version=QUERY_SIGNAL_SCHEMA_VERSION,
        evaluation_metric_key="dev_metrics",
        selection_metric_key="dev_metrics",
    )
    if metrics is None:
        return None, None, reason
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None, None, "cached calibrated ordinal outputs are missing metadata"
    cached_candidate_sizes = [int(value) for value in metadata.get("candidate_sizes", [])]
    if cached_candidate_sizes != [int(value) for value in candidate_sizes]:
        return None, None, (
            f"cached calibrated ordinal outputs use candidate sizes {cached_candidate_sizes}, "
            f"but {[int(value) for value in candidate_sizes]} are now required"
        )
    cached_base_model = str(metadata.get("base_model", ""))
    if cached_base_model != str(base_model):
        return None, None, (
            f"cached calibrated ordinal outputs use base model {cached_base_model or '<missing>'}, "
            f"but {base_model} is now required"
        )
    return metrics, selection_metrics, None


def _cached_qpp_cache_status(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    score_source: str,
    mapping: str,
    minimum_size: int,
    maximum_size: int,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "qpp",
    metadata_version: int = QPP_METADATA_VERSION,
    model_metadata_version: int = QPP_MODEL_METADATA_VERSION,
) -> tuple[dict[str, float] | None, dict[str, float] | None, str | None]:
    metrics, selection_metrics, reason = _cached_prediction_cache_status(
        store,
        predictor_name,
        language,
        retriever_name,
        mode=mode,
        required_metadata_version=metadata_version,
        required_feature_dependency_version=PREDICTOR_FEATURE_DEPENDENCY_VERSION,
        required_query_signal_schema_version=QUERY_SIGNAL_SCHEMA_VERSION,
        evaluation_metric_key="dev_metrics",
        selection_metric_key="dev_metrics",
    )
    if metrics is None:
        return None, None, reason
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None, None, f"cached {predictor_name} outputs are missing metadata"
    cached_score_source = metadata.get("score_source")
    if cached_score_source != score_source:
        return None, None, (
            f"cached {predictor_name} outputs use score source {cached_score_source}, "
            f"but {score_source} is now required"
        )
    cached_mapping = metadata.get("mapping")
    if cached_mapping != mapping:
        return None, None, (
            f"cached {predictor_name} outputs use mapping {cached_mapping}, "
            f"but {mapping} is now required"
        )
    cached_minimum = metadata.get("minimum_size")
    cached_maximum = metadata.get("maximum_size")
    if cached_minimum is None or int(cached_minimum) != int(minimum_size):
        return None, None, (
            f"cached {predictor_name} outputs use minimum size {cached_minimum}, "
            f"but {int(minimum_size)} is now required"
        )
    if cached_maximum is None or int(cached_maximum) != int(maximum_size):
        return None, None, (
            f"cached {predictor_name} outputs use maximum size {cached_maximum}, "
            f"but {int(maximum_size)} is now required"
        )
    manifest, manifest_reason = _qpp_model_artifact_status(
        store,
        retriever_name,
        mode=mode,
        predictor_name=predictor_name,
        metadata_version=metadata_version,
        model_metadata_version=model_metadata_version,
    )
    if manifest is None:
        return None, None, manifest_reason
    if manifest.get("score_source") != score_source:
        return None, None, (
            f"cached {predictor_name} model manifest uses score source {manifest.get('score_source')}, "
            f"but {score_source} is now required"
        )
    if manifest.get("mapping") != mapping:
        return None, None, (
            f"cached {predictor_name} model manifest uses mapping {manifest.get('mapping')}, "
            f"but {mapping} is now required"
        )
    manifest_minimum = manifest.get("minimum_size")
    manifest_maximum = manifest.get("maximum_size")
    if manifest_minimum is None or int(manifest_minimum) != int(minimum_size):
        return None, None, (
            f"cached {predictor_name} model manifest uses minimum size {manifest_minimum}, "
            f"but {int(minimum_size)} is now required"
        )
    if manifest_maximum is None or int(manifest_maximum) != int(maximum_size):
        return None, None, (
            f"cached {predictor_name} model manifest uses maximum size {manifest_maximum}, "
            f"but {int(maximum_size)} is now required"
        )
    return metrics, selection_metrics, None


def _hit_at_1_metrics_sort_key(metrics: dict[str, float] | None) -> tuple[float, float, float, float]:
    if metrics is None:
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    false_positives = float(metrics.get("false_positives", math.nan))
    precision = float(metrics.get("precision", math.nan))
    true_positives = float(metrics.get("true_positives", math.nan))
    recall = float(metrics.get("recall", math.nan))
    positive_predictions = float(metrics.get("positive_predictions", 0.0))
    false_positive_key = -false_positives if np.isfinite(false_positives) else float("-inf")
    precision_key = precision if np.isfinite(precision) else (1.0 if positive_predictions == 0 else float("-inf"))
    true_positive_key = true_positives if np.isfinite(true_positives) else float("-inf")
    recall_key = recall if np.isfinite(recall) else float("-inf")
    return false_positive_key, precision_key, true_positive_key, recall_key


def _select_best_hit_at_1_variants(
    retrievers: list[str],
    metrics_by_retriever: dict[str, dict[str, float] | None],
) -> list[str]:
    winners: list[str] = []
    grouped: dict[str, list[str]] = {}
    for retriever_name in retrievers:
        grouped.setdefault(base_retriever_name(retriever_name), []).append(retriever_name)
    for variants in grouped.values():
        best_name = None
        best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
        for retriever_name in variants:
            key = _hit_at_1_metrics_sort_key(metrics_by_retriever.get(retriever_name))
            if key > best_key:
                best_name = retriever_name
                best_key = key
        if best_name is not None:
            winners.append(best_name)
    return winners


def _in_top_max_depth_target(frame: pd.DataFrame, target_depth: int | None = None) -> pd.Series:
    if target_depth is None:
        return ~frame["retrieval_failure_within_max_depth"].astype(bool)
    return frame["gold_rank"] <= target_depth


def _in_top_max_depth_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["in_top_max_depth"].astype(bool)
    predicted = frame["predicted_in_top_max_depth"].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
    positive_predictions = tp + fp
    total = tp + fp + tn + fn
    precision = float(tp / positive_predictions) if positive_predictions else math.nan
    recall = float(tp / (tp + fn)) if tp + fn else math.nan
    f1 = _fbeta_score(precision, recall, beta=1.0)
    f2 = _fbeta_score(precision, recall, beta=2.0)
    return {
        "f2": f2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else math.nan,
        "positive_predictions": float(positive_predictions),
        "predicted_positive_rate": float(positive_predictions / total) if total else math.nan,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "true_negatives": float(tn),
        "false_negatives": float(fn),
    }


def _in_top_max_depth_threshold_policy(policy_name: str = IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME) -> dict[str, Any]:
    return {"name": policy_name}


def _in_top_max_depth_model_artifact_status(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "in_top_max_depth",
    threshold_policy_name: str = IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
) -> tuple[dict[str, Any] | None, str | None]:
    model_path = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "model.joblib"
    if not model_path.exists():
        return None, f"cached in-top-max-depth model is missing at {model_path}"
    model_payload = joblib.load(model_path)
    if not isinstance(model_payload, dict):
        return None, f"cached in-top-max-depth model at {model_path} does not contain a valid payload"
    cached_metadata_version = model_payload.get("predictor_metadata_version")
    if cached_metadata_version is None or int(cached_metadata_version) != IN_TOP_MAX_DEPTH_METADATA_VERSION:
        return None, (
            f"cached in-top-max-depth model uses predictor metadata version {cached_metadata_version}, "
            f"but version {IN_TOP_MAX_DEPTH_METADATA_VERSION} is required"
        )
    threshold_policy = model_payload.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        return None, f"cached in-top-max-depth model at {model_path} is missing threshold policy metadata"
    cached_policy_name = str(threshold_policy.get("name", ""))
    if cached_policy_name != threshold_policy_name:
        return None, (
            f"cached in-top-max-depth model was tuned with threshold policy {cached_policy_name or '<missing>'}, "
            f"but {threshold_policy_name} is now required"
        )
    cached_dependency_version = model_payload.get("feature_dependency_version")
    if cached_dependency_version is None or int(cached_dependency_version) != IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION:
        return None, (
            f"cached in-top-max-depth model uses feature dependency version {cached_dependency_version}, "
            f"but version {IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION} is required"
        )
    cached_query_signal_schema = model_payload.get("query_signal_schema_version")
    if cached_query_signal_schema is None or int(cached_query_signal_schema) != QUERY_SIGNAL_SCHEMA_VERSION:
        return None, (
            f"cached in-top-max-depth model uses query-signal schema version {cached_query_signal_schema}, "
            f"but version {QUERY_SIGNAL_SCHEMA_VERSION} is required"
        )
    for key in ("feature_depths", "threshold", "pipeline"):
        if key not in model_payload:
            return None, f"cached in-top-max-depth model at {model_path} is missing {key}"
    return model_payload, None


def _cached_in_top_max_depth_cache_status(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "in_top_max_depth",
    threshold_policy_name: str = IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
) -> tuple[dict[str, float] | None, str | None]:
    metadata = _cached_prediction_metadata(store, predictor_name, language, retriever_name, mode=mode)
    if metadata is None:
        return None, None
    if int(metadata.get("metadata_version", 0)) != IN_TOP_MAX_DEPTH_METADATA_VERSION:
        return None, (
            f"cached in-top-max-depth outputs use metadata version {int(metadata.get('metadata_version', 0))}, "
            f"but version {IN_TOP_MAX_DEPTH_METADATA_VERSION} is required"
        )
    compatibility = metadata.get("input_compatibility")
    if not isinstance(compatibility, dict):
        return None, "cached in-top-max-depth outputs do not record the current feature compatibility metadata"
    cached_dependency_version = compatibility.get("feature_dependency_version")
    if cached_dependency_version is None or int(cached_dependency_version) != IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION:
        return None, (
            f"cached in-top-max-depth outputs use feature dependency version {cached_dependency_version}, "
            f"but version {IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION} is required"
        )
    cached_query_signal_schema = compatibility.get("query_signal_schema_version")
    if cached_query_signal_schema is None or int(cached_query_signal_schema) != QUERY_SIGNAL_SCHEMA_VERSION:
        return None, (
            f"cached in-top-max-depth outputs use query-signal schema version {cached_query_signal_schema}, "
            f"but version {QUERY_SIGNAL_SCHEMA_VERSION} is required"
        )
    threshold_policy = metadata.get("threshold_policy")
    if not isinstance(threshold_policy, dict):
        return None, "cached in-top-max-depth outputs are missing threshold policy metadata"
    cached_policy_name = str(threshold_policy.get("name", ""))
    if cached_policy_name != threshold_policy_name:
        return None, (
            f"cached in-top-max-depth outputs were tuned with threshold policy {cached_policy_name or '<missing>'}, "
            f"but {threshold_policy_name} is now required"
        )
    _, model_reason = _in_top_max_depth_model_artifact_status(
        store,
        retriever_name,
        mode=mode,
        predictor_name=predictor_name,
        threshold_policy_name=threshold_policy_name,
    )
    if model_reason is not None:
        return None, model_reason
    if not isinstance(metadata.get("dev_metrics"), dict):
        return None, "cached in-top-max-depth outputs are missing dev metrics"
    return {str(key): float(value) for key, value in metadata["dev_metrics"].items()}, None


def _cached_in_top_max_depth_metrics(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "in_top_max_depth",
    threshold_policy_name: str = IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
) -> dict[str, float] | None:
    metrics, _ = _cached_in_top_max_depth_cache_status(
        store,
        language,
        retriever_name,
        mode=mode,
        predictor_name=predictor_name,
        threshold_policy_name=threshold_policy_name,
    )
    return metrics


def _in_top_max_depth_metrics_sort_key(metrics: dict[str, float] | None) -> tuple[float, float, float, float]:
    if metrics is None:
        return (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    false_positives = float(metrics.get("false_positives", math.nan))
    precision = float(metrics.get("precision", math.nan))
    true_positives = float(metrics.get("true_positives", math.nan))
    recall = float(metrics.get("recall", math.nan))
    positive_predictions = float(metrics.get("positive_predictions", 0.0))
    false_positive_key = -false_positives if np.isfinite(false_positives) else float("-inf")
    precision_key = precision if np.isfinite(precision) else (1.0 if positive_predictions == 0 else float("-inf"))
    true_positive_key = true_positives if np.isfinite(true_positives) else float("-inf")
    recall_key = recall if np.isfinite(recall) else float("-inf")
    return false_positive_key, precision_key, true_positive_key, recall_key


def _select_best_in_top_max_depth_variants(
    retrievers: list[str],
    metrics_by_retriever: dict[str, dict[str, float] | None],
) -> list[str]:
    winners: list[str] = []
    grouped: dict[str, list[str]] = {}
    for retriever_name in retrievers:
        grouped.setdefault(base_retriever_name(retriever_name), []).append(retriever_name)
    for variants in grouped.values():
        best_name = None
        best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
        for retriever_name in variants:
            key = _in_top_max_depth_metrics_sort_key(metrics_by_retriever.get(retriever_name))
            if key > best_key:
                best_name = retriever_name
                best_key = key
        if best_name is not None:
            winners.append(best_name)
    return winners


@dataclass(slots=True)
class PredictorContext:
    train_frames: dict[int, pd.DataFrame]
    dev_frames: dict[int, pd.DataFrame]
    test_frames: dict[int, pd.DataFrame]


class PredictorFeatureViews:
    def __init__(
        self,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
    ) -> None:
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self._cache: dict[tuple[str, str, str, tuple[str, ...]], dict[int, pd.DataFrame]] = {}
        self.excluded_features: frozenset[str] = frozenset()

    def _apply_exclusions(self, frames: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
        if not self.excluded_features:
            return frames
        return {
            depth: frame.drop(
                columns=[column for column in frame.columns if column.split("::")[-1] in self.excluded_features]
            )
            for depth, frame in frames.items()
        }

    def _requested_candidate_sizes(self, candidate_sizes: list[int] | tuple[int, ...] | None) -> list[int]:
        if hasattr(self.features, "_normalize_candidate_sizes"):
            return list(self.features._normalize_candidate_sizes(candidate_sizes))
        configured = getattr(getattr(self.features, "config", None), "candidate_sizes", None)
        if candidate_sizes is None and configured is not None:
            return [int(depth) for depth in configured]
        return [int(depth) for depth in (candidate_sizes or [])]

    def _build_base_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        if candidate_sizes:
            try:
                return self.features.build_features(split_key, retriever_name, candidate_sizes=candidate_sizes)
            except TypeError:
                pass
        return self.features.build_features(split_key, retriever_name)

    def build(
        self,
        split_key: str,
        target_retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None = None,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        target_retriever_name = self.retrieval.resolve_retriever_name(target_retriever_name)
        if mode == PREDICTOR_MODE_LOCAL:
            return self._apply_exclusions(
                self._build_base_features(split_key, target_retriever_name, candidate_sizes=candidate_sizes)
            )
        sources = self._canonical_sources(target_retriever_name, source_retrievers)
        requested_candidate_sizes = self._requested_candidate_sizes(candidate_sizes)
        cache_key = (split_key, target_retriever_name, mode, tuple(sources), tuple(requested_candidate_sizes))
        if cache_key in self._cache:
            return self._apply_exclusions(self._cache[cache_key])
        cached = self._load_cached_view(split_key, target_retriever_name, mode, sources, requested_candidate_sizes)
        if cached is not None:
            self._cache[cache_key] = cached
            return self._apply_exclusions(cached)
        self.console.log(
            f"Assembling {mode} predictor features for {split_key} / {target_retriever_name} "
            f"from {len(sources)} retriever views"
        )
        target_frames = self._build_base_features(split_key, target_retriever_name, candidate_sizes=requested_candidate_sizes)
        source_frames = {
            source_name: self._build_base_features(split_key, source_name, candidate_sizes=requested_candidate_sizes)
            for source_name in sources
            if source_name != target_retriever_name
        }
        frames: dict[int, pd.DataFrame] = {}
        for depth, target_frame in target_frames.items():
            frame = target_frame.copy()
            for source_name, by_depth in source_frames.items():
                source_frame = by_depth[depth]
                feature_columns = _feature_columns(source_frame)
                if not feature_columns:
                    continue
                merged = source_frame[PREDICTOR_VIEW_JOIN_COLUMNS + feature_columns].copy()
                merged.columns = PREDICTOR_VIEW_JOIN_COLUMNS + [f"{source_name}::{column}" for column in feature_columns]
                frame = frame.merge(merged, on=PREDICTOR_VIEW_JOIN_COLUMNS, how="left", sort=False)
            frames[depth] = frame
        if not requested_candidate_sizes:
            requested_candidate_sizes = sorted(int(depth) for depth in frames)
        self._write_cached_view(split_key, target_retriever_name, mode, sources, frames, requested_candidate_sizes)
        self._cache[cache_key] = frames
        return self._apply_exclusions(frames)

    def _canonical_sources(self, target_retriever_name: str, source_retrievers: list[str] | None) -> list[str]:
        requested = {target_retriever_name}
        if source_retrievers:
            for retriever_name in source_retrievers:
                requested.add(self.retrieval.resolve_retriever_name(retriever_name))
        ordered = []
        for retriever_name in self.retrieval.enabled_retrievers():
            if retriever_name in requested:
                ordered.append(retriever_name)
        for retriever_name in requested:
            if retriever_name not in ordered:
                ordered.append(retriever_name)
        return ordered

    def _view_root(self, split_key: str, target_retriever_name: str, mode: str) -> Path:
        return self.store.root_dir / "features" / "views" / mode / split_key / target_retriever_name

    def _view_manifest_path(self, split_key: str, target_retriever_name: str, mode: str) -> Path:
        return self._view_root(split_key, target_retriever_name, mode) / "manifest.json"

    def _view_frame_path(self, split_key: str, target_retriever_name: str, mode: str, depth: int) -> Path:
        return self._view_root(split_key, target_retriever_name, mode) / f"depth_{depth:04d}.parquet"

    def _load_cached_view(
        self,
        split_key: str,
        target_retriever_name: str,
        mode: str,
        sources: list[str],
        candidate_sizes: list[int],
    ) -> dict[int, pd.DataFrame] | None:
        if not candidate_sizes:
            return None
        manifest_path = self._view_manifest_path(split_key, target_retriever_name, mode)
        if not manifest_path.exists():
            return None
        manifest = self.store.read_json(manifest_path)
        if int(manifest.get("metadata_version", 0)) != PREDICTOR_VIEW_METADATA_VERSION:
            return None
        if int(manifest.get("feature_dependency_version", 0)) != PREDICTOR_FEATURE_DEPENDENCY_VERSION:
            return None
        if manifest.get("source_retrievers") != sources:
            return None
        if list(manifest.get("candidate_sizes", [])) != [int(depth) for depth in candidate_sizes]:
            return None
        frames = {}
        for depth in candidate_sizes:
            path = self._view_frame_path(split_key, target_retriever_name, mode, depth)
            if not path.exists():
                return None
            frames[int(depth)] = self.store.read_parquet(path)
        return frames

    def _write_cached_view(
        self,
        split_key: str,
        target_retriever_name: str,
        mode: str,
        sources: list[str],
        frames: dict[int, pd.DataFrame],
        candidate_sizes: list[int],
    ) -> None:
        manifest_path = self._view_manifest_path(split_key, target_retriever_name, mode)
        self.store.write_json(
            manifest_path,
            {
                "metadata_version": PREDICTOR_VIEW_METADATA_VERSION,
                "feature_dependency_version": PREDICTOR_FEATURE_DEPENDENCY_VERSION,
                "target_retriever": target_retriever_name,
                "mode": mode,
                "source_retrievers": sources,
                "candidate_sizes": [int(depth) for depth in candidate_sizes],
            },
        )
        for depth, frame in frames.items():
            self.store.write_parquet(self._view_frame_path(split_key, target_retriever_name, mode, depth), frame)


class CalibratedOrdinalPredictorRunner:
    predictor_name = "calibrated_ordinal"
    decoding_name = CALIBRATED_ORDINAL_DECODING_NAME

    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.views = PredictorFeatureViews(store, console, features, retrieval)

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary = {}
        selected = self._ordered_retrievers(retrievers)
        modes = _normalize_predictor_modes(self.config)
        local_selection_metrics: dict[str, dict[str, float] | None] = {}
        original_scopes = self._configured_document_scopes()
        try:
            for document_scope, scope_retrievers in self._scope_phases(selected):
                self.console.log(
                    f"Running ordinal predictor scope phase {document_scope} "
                    f"for {len(scope_retrievers)} retrievers"
                )
                self._set_enabled_document_scopes([document_scope])
                for retriever_name in scope_retrievers:
                    metrics, selection_metrics = self._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
                    summary[retriever_name] = metrics
                    local_selection_metrics[retriever_name] = selection_metrics
            self._set_enabled_document_scopes(original_scopes)
            if PREDICTOR_MODE_ALL_MEANINGFUL in modes:
                winners = _select_best_variants(selected, local_selection_metrics)
                if winners:
                    winner_text = ", ".join(
                        f"{base_retriever_name(name)} -> {name}"
                        for name in winners
                    )
                    self.console.log(f"Selected local winners for ordinal {PREDICTOR_MODE_ALL_MEANINGFUL}: {winner_text}")
                for retriever_name in winners:
                    metrics, _ = self._run_retriever(
                        language,
                        retriever_name,
                        PREDICTOR_MODE_ALL_MEANINGFUL,
                        source_retrievers=winners,
                    )
                    summary[_display_retriever_name(retriever_name, PREDICTOR_MODE_ALL_MEANINGFUL)] = metrics
        finally:
            self._set_enabled_document_scopes(original_scopes)
        return summary

    def _scope_phases(self, retrievers: list[str]) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for retriever_name in retrievers:
            grouped.setdefault(retriever_document_scope(retriever_name), []).append(retriever_name)
        ordered_scopes: list[str] = []
        for document_scope in self._configured_document_scopes():
            if document_scope in grouped and document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        for retriever_name in retrievers:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        return [(document_scope, grouped[document_scope]) for document_scope in ordered_scopes]

    def _set_enabled_document_scopes(self, scopes: list[str]) -> None:
        scopes = list(scopes)
        current_scopes = self._configured_document_scopes()
        if current_scopes == scopes:
            return
        self.config.enabled_document_scopes = scopes
        if hasattr(self.features, "_all_rankings_top100_cache"):
            self.features._all_rankings_top100_cache.clear()

    def _configured_document_scopes(self) -> list[str]:
        return list(getattr(self.config, "enabled_document_scopes", ["title_abstract"]))

    def _ordered_retrievers(self, retrievers: list[str] | None) -> list[str]:
        if retrievers:
            return [self.retrieval.resolve_retriever_name(name) for name in retrievers]
        enabled = [self.retrieval.resolve_retriever_name(name) for name in self.retrieval.enabled_retrievers()]
        original_order = {name: index for index, name in enumerate(enabled)}
        scope_order: list[str] = []
        for retriever_name in enabled:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in scope_order:
                scope_order.append(document_scope)
        scope_index = {document_scope: index for index, document_scope in enumerate(scope_order)}
        return sorted(
            enabled,
            key=lambda name: (
                scope_index[retriever_document_scope(name)],
                self._retriever_group(name),
                self._semantic_priority(name),
                original_order[name],
            ),
        )

    @staticmethod
    def _retriever_group(retriever_name: str) -> int:
        base_name = base_retriever_name(retriever_name)
        if base_name == "bm25":
            return 0
        if base_name in LEXICAL_RETRIEVERS:
            return 2
        return 1

    @staticmethod
    def _semantic_priority(retriever_name: str) -> int:
        priority_by_base_name = {
            "qwen3_embedding_0_6b": 0,
            "bge_m3": 1,
            "bge_large_en_v1_5": 2,
        }
        return priority_by_base_name.get(base_retriever_name(retriever_name), 100)

    def _candidate_sizes(self) -> list[int]:
        return _coarsen_candidate_sizes(getattr(self.config, "candidate_sizes", []))

    def _context(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None = None,
    ) -> PredictorContext:
        requested_candidate_sizes = self._candidate_sizes()
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        return PredictorContext(
            train_frames=self.views.build(
                train_key,
                retriever_name,
                mode,
                source_retrievers=source_retrievers,
                candidate_sizes=requested_candidate_sizes,
            ),
            dev_frames=self.views.build(
                dev_key,
                retriever_name,
                mode,
                source_retrievers=source_retrievers,
                candidate_sizes=requested_candidate_sizes,
            ),
            test_frames={},
        )

    def _run_retriever(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        display_name = _display_retriever_name(retriever_name, mode)
        self.console.section(f"Calibrated ordinal predictor for {display_name}")
        candidate_sizes = self._candidate_sizes()
        cached_metrics, cached_selection_metrics, cache_reason = self._cache_status(
            language, retriever_name, mode, candidate_sizes
        )
        if cached_metrics is not None:
            self.console.log(f"Using cached calibrated ordinal outputs for {display_name}")
            return cached_metrics, cached_selection_metrics
        if cache_reason:
            self.console.log(f"Rebuilding calibrated ordinal outputs for {display_name}: {cache_reason}")
        self.console.log(f"Preparing calibrated ordinal feature context for {display_name}")
        context = self._context(language, retriever_name, mode, source_retrievers=source_retrievers)
        self.console.log(f"Training calibrated ordinal solvability model for {display_name}")
        solvability_model, solvability_importances, solvability_calibration = self._fit_solvability_model(
            context.train_frames[max(candidate_sizes)],
            display_name,
        )
        self.console.log(f"Training calibrated ordinal cumulative models for {display_name}")
        stage_models = self._fit_cumulative_models(context.train_frames, display_name)
        reference = self._reference_output_frame(context.dev_frames, candidate_sizes)
        solvability_probabilities = _positive_class_probabilities(solvability_model, context.dev_frames[max(candidate_sizes)])
        stage_probabilities = _monotone_cumulative_probabilities(
            self._stage_probability_arrays(context.dev_frames, stage_models)
        )
        solvability_threshold, solvability_metrics = self._select_solvability_threshold(
            reference,
            solvability_probabilities,
            stage_probabilities,
            candidate_sizes,
        )
        self.console.log(
            f"Selected solvability threshold {solvability_threshold:.6f} for {display_name} "
            f"(end-to-end candidate_recall={solvability_metrics['candidate_recall']:.4f}, "
            f"solvability recall={solvability_metrics['recall']:.4f}, precision={solvability_metrics['precision']:.4f})"
        )
        dev_frame, dev_class_metrics = self._decode_ordinal_predictions(
            reference,
            solvability_probabilities,
            stage_probabilities,
            solvability_threshold,
            candidate_sizes,
        )
        self.console.log(
            f"Decoded calibrated ordinal predictions for {display_name} via {self.decoding_name} "
            "over per-class probabilities derived from the monotone cumulative stage probabilities "
            "(no per-depth threshold search)"
        )
        self.console.log(f"Generating calibrated ordinal signal summaries for {display_name}")
        self._attach_signal_summaries(
            dev_frame,
            context.dev_frames,
            context.train_frames,
            stage_models,
            solvability_importances,
        )
        self.console.log(f"Persisting calibrated ordinal models for {display_name}")
        self._persist_models(
            language,
            retriever_name,
            mode,
            candidate_sizes,
            solvability_threshold,
            solvability_model,
            solvability_importances,
            solvability_calibration,
            stage_models,
            source_retrievers=source_retrievers,
        )
        self.console.log(f"Writing calibrated ordinal outputs for {display_name}")
        self._write_outputs(
            language,
            retriever_name,
            display_name,
            dev_frame,
            mode,
            candidate_sizes,
            solvability_threshold,
            source_retrievers=source_retrievers,
            solvability_metrics=solvability_metrics,
            class_metrics=dev_class_metrics,
            solvability_calibration=solvability_calibration,
            stage_models=stage_models,
        )
        dev_metrics = _selection_metrics(dev_frame)
        return dev_metrics, dev_metrics

    def _cache_status(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        candidate_sizes: list[int],
    ) -> tuple[dict[str, float] | None, dict[str, float] | None, str | None]:
        return _cached_calibrated_ordinal_cache_status(
            self.store,
            language,
            retriever_name,
            candidate_sizes=candidate_sizes,
            base_model=self.config.predictors.ordinal.base_model,
            predictor_name=self.predictor_name,
            mode=mode,
        )

    def _fit_solvability_model(
        self,
        training_frame: pd.DataFrame,
        display_name: str,
    ) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
        feature_columns = _feature_columns(training_frame)
        target = training_frame["smallest_sufficient_size"].notna().astype(int)
        return _fit_calibrated_classifier(
            console=self.console,
            train_frame=training_frame,
            feature_columns=feature_columns,
            target=target,
            base_model_name=self.config.predictors.ordinal.base_model,
            seed=self.config.runtime.random_seed,
            purpose=f"calibrated ordinal solvability for {display_name}",
            method="isotonic",
        )

    def _fit_cumulative_models(
        self,
        training_frames: dict[int, pd.DataFrame],
        display_name: str,
    ) -> dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]]:
        stage_models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]] = {}
        for depth in sorted(training_frames):
            frame = training_frames[int(depth)]
            solvable = frame.loc[frame["smallest_sufficient_size"].notna()].copy()
            feature_columns = _feature_columns(solvable)
            target = solvable["gold_in_top_k"].astype(int)
            stage_models[int(depth)] = _fit_calibrated_classifier(
                console=self.console,
                train_frame=solvable,
                feature_columns=feature_columns,
                target=target,
                base_model_name=self.config.predictors.ordinal.base_model,
                seed=self.config.runtime.random_seed,
                purpose=f"calibrated ordinal cumulative depth {int(depth)} for {display_name}",
                method="isotonic",
            )
        return stage_models

    @staticmethod
    def _reference_output_frame(frames: dict[int, pd.DataFrame], candidate_sizes: list[int]) -> pd.DataFrame:
        ordered_depths = sorted(frames)
        reference = frames[ordered_depths[0]][["query_id", "document_id", "gold_rank", "smallest_sufficient_size"]].copy()
        reference.rename(columns={"smallest_sufficient_size": "raw_smallest_sufficient_size"}, inplace=True)
        reference["smallest_sufficient_size"] = _map_smallest_sufficient_sizes_to_grid(
            reference["raw_smallest_sufficient_size"],
            candidate_sizes,
        )
        return reference

    @staticmethod
    def _stage_probability_arrays(
        frames: dict[int, pd.DataFrame],
        models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]],
    ) -> dict[int, np.ndarray]:
        probabilities: dict[int, np.ndarray] = {}
        for depth in sorted(frames):
            model, _, _ = models[int(depth)]
            probabilities[int(depth)] = np.asarray(
                _positive_class_probabilities(model, frames[int(depth)]),
                dtype=np.float64,
            )
        return probabilities

    @classmethod
    def _select_solvability_threshold(
        cls,
        reference: pd.DataFrame,
        solvability_probabilities: np.ndarray,
        stage_probabilities: dict[int, np.ndarray],
        candidate_sizes: list[int],
    ) -> tuple[float, dict[str, float]]:
        values = np.asarray(solvability_probabilities, dtype=np.float64)
        truth = reference["smallest_sufficient_size"].notna().to_numpy(dtype=bool)
        gold_rank = reference["gold_rank"].to_numpy(dtype=np.float64)
        ordered_sizes = sorted(int(size) for size in candidate_sizes)
        finite_values = sorted({float(value) for value in values[np.isfinite(values)]}, reverse=True)
        if not finite_values:
            fallback = _binary_classification_summary(truth, np.zeros_like(truth, dtype=bool))
            fallback["candidate_recall"] = 0.0
            return 0.5, fallback
        recalls: list[float] = []
        for threshold in finite_values:
            output = cls._predict_from_probabilities(
                reference, values, stage_probabilities, threshold
            )
            recalls.append(
                _vectorized_candidate_recall(
                    gold_rank,
                    output["predicted_failure"].to_numpy(dtype=bool),
                    output["selected_size"].to_numpy(dtype=np.float64),
                )
            )
        best_recall = max(recalls)
        tied_indices = [index for index, value in enumerate(recalls) if value == best_recall]
        best_index = tied_indices[0]
        if len(tied_indices) > 1:
            best_key: tuple[float, float, float] | None = None
            for index in tied_indices:
                threshold = finite_values[index]
                output = cls._predict_from_probabilities(
                    reference, values, stage_probabilities, threshold
                )
                class_metrics = ordinal_predictor_metric_summary(output, ordered_sizes)
                mae = float(class_metrics["ordinal_mae"])
                key = (
                    _safe_sort_float(float(class_metrics["quadratic_weighted_cohen_kappa"])),
                    -mae if np.isfinite(mae) else float("-inf"),
                    float(threshold),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
        threshold = finite_values[best_index]
        predicted = values >= float(threshold)
        metrics = _binary_classification_summary(truth, predicted)
        try:
            metrics["mcc"] = float(matthews_corrcoef(truth, predicted))
        except ValueError:
            metrics["mcc"] = math.nan
        metrics["candidate_recall"] = float(best_recall)
        return float(threshold), metrics

    @classmethod
    def _predict_from_probabilities(
        cls,
        reference: pd.DataFrame,
        solvability_probabilities: np.ndarray,
        stage_probabilities: dict[int, np.ndarray],
        solvability_threshold: float,
        *,
        include_traces: bool = False,
    ) -> pd.DataFrame:
        ordered_depths = sorted(stage_probabilities)
        cumulative = np.column_stack(
            [np.asarray(stage_probabilities[int(depth)], dtype=np.float64) for depth in ordered_depths]
        )
        class_probabilities = np.diff(cumulative, axis=1, prepend=0.0)
        chosen_index = np.argmax(class_probabilities, axis=1)
        row_indices = np.arange(len(reference))
        chosen_sizes = np.asarray([ordered_depths[int(index)] for index in chosen_index], dtype=np.float64)
        chosen_confidences = class_probabilities[row_indices, chosen_index]
        solvability = np.asarray(solvability_probabilities, dtype=np.float64)
        predicted_solvable = solvability >= float(solvability_threshold)
        output = reference.copy()
        output["selected_size"] = np.where(predicted_solvable, chosen_sizes, np.nan)
        output["confidence"] = np.where(predicted_solvable, chosen_confidences, 1.0 - solvability)
        output["solvability_probability"] = solvability
        output["selected_stage_probability"] = cumulative[row_indices, chosen_index]
        output["predicted_failure"] = ~predicted_solvable
        output["visited_classifiers"] = np.where(predicted_solvable, chosen_index + 1, len(ordered_depths))
        if include_traces:
            output["stage_probabilities"] = [
                {str(depth): float(cumulative[row_index, column_index]) for column_index, depth in enumerate(ordered_depths)}
                for row_index in range(len(reference))
            ]
        return output

    def _decode_ordinal_predictions(
        self,
        reference: pd.DataFrame,
        solvability_probabilities: np.ndarray,
        stage_probabilities: dict[int, np.ndarray],
        solvability_threshold: float,
        candidate_sizes: list[int],
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        ordered_sizes = [int(depth) for depth in sorted(candidate_sizes)]
        output = self._predict_from_probabilities(
            reference,
            solvability_probabilities,
            stage_probabilities,
            solvability_threshold,
            include_traces=True,
        )
        class_metrics = ordinal_predictor_metric_summary(output, ordered_sizes)
        return output, class_metrics

    def _attach_signal_summaries(
        self,
        output: pd.DataFrame,
        evaluation_frames: dict[int, pd.DataFrame],
        training_frames: dict[int, pd.DataFrame],
        stage_models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]],
        solvability_importances: pd.DataFrame,
    ) -> None:
        summaries: list[str] = []
        max_depth = max(training_frames)
        for index, row in output.iterrows():
            if bool(row["predicted_failure"]):
                summaries.append(
                    _local_signal_summary(
                        evaluation_frames[max_depth].iloc[index],
                        training_frames[max_depth],
                        solvability_importances,
                    )
                )
                continue
            depth = int(row["selected_size"])
            _, importances, _ = stage_models[depth]
            summaries.append(
                _local_signal_summary(
                    evaluation_frames[depth].iloc[index],
                    training_frames[depth].loc[training_frames[depth]["smallest_sufficient_size"].notna()].copy(),
                    importances,
                )
            )
        output["strong_feature_signals"] = summaries

    def _persist_models(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        candidate_sizes: list[int],
        solvability_threshold: float,
        solvability_model: Any,
        solvability_importances: pd.DataFrame,
        solvability_calibration: dict[str, Any],
        stage_models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]],
        *,
        source_retrievers: list[str] | None = None,
    ) -> None:
        model_root = _prediction_model_root(self.store, self.predictor_name, retriever_name, mode=mode)
        joblib.dump(solvability_model, model_root / "solvability_model.joblib")
        solvability_importances.to_parquet(model_root / "solvability_importances.parquet", index=False)
        stage_calibration: dict[str, dict[str, Any]] = {}
        for depth, (model, importances, calibration_info) in stage_models.items():
            joblib.dump(model, model_root / f"depth_{int(depth):04d}.joblib")
            importances.to_parquet(model_root / f"depth_{int(depth):04d}_importances.parquet", index=False)
            stage_calibration[str(int(depth))] = dict(calibration_info)
        self.store.write_json(
            model_root / "manifest.json",
            {
                "model_metadata_version": CALIBRATED_ORDINAL_MODEL_METADATA_VERSION,
                "predictor_metadata_version": CALIBRATED_ORDINAL_METADATA_VERSION,
                "predictor_feature_mode": mode,
                "target_retriever": retriever_name,
                "source_retrievers": list(source_retrievers or [retriever_name]),
                "base_model": self.config.predictors.ordinal.base_model,
                "candidate_sizes": [int(depth) for depth in candidate_sizes],
                "decoding": self.decoding_name,
                "solvability_threshold": float(solvability_threshold),
                "training_split": "train",
                "evaluation_split": "dev",
                "language": language,
                "input_compatibility": {
                    "feature_dependency_version": PREDICTOR_FEATURE_DEPENDENCY_VERSION,
                    "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
                },
                "calibration": {
                    "solvability": dict(solvability_calibration),
                    "stages": stage_calibration,
                },
            },
        )

    def _write_outputs(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        output: pd.DataFrame,
        mode: str,
        candidate_sizes: list[int],
        solvability_threshold: float,
        *,
        source_retrievers: list[str] | None = None,
        solvability_metrics: dict[str, float],
        class_metrics: dict[str, float],
        solvability_calibration: dict[str, Any],
        stage_models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]],
    ) -> None:
        paths = _prediction_output_paths(self.store, self.predictor_name, language, retriever_name, mode=mode)
        paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        _safe_prediction_frame(output).to_parquet(paths["prediction"], index=False)
        selection_metrics = _selection_metrics(output, max_ranking_depth=self.config.max_ranking_depth)
        stage_calibration = {str(int(depth)): dict(calibration_info) for depth, (_, _, calibration_info) in stage_models.items()}
        metadata = {
            "metadata_version": CALIBRATED_ORDINAL_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": list(source_retrievers or [retriever_name]),
            "base_model": self.config.predictors.ordinal.base_model,
            "candidate_sizes": [int(depth) for depth in candidate_sizes],
            "decoding": self.decoding_name,
            "solvability_threshold": float(solvability_threshold),
            "training_split": "train",
            "evaluation_split": "dev",
            "input_compatibility": {
                "feature_dependency_version": PREDICTOR_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
            },
            "calibration": {
                "solvability": dict(solvability_calibration),
                "stages": stage_calibration,
            },
            "solvability_metrics": dict(solvability_metrics),
            "dev_metrics": selection_metrics,
            "dev_class_metrics": dict(class_metrics),
        }
        self.store.write_json(paths["metadata"], metadata)
        paths["report"].write_text(
            self._prediction_report(display_name, metadata, output),
            encoding="utf-8",
        )

    @staticmethod
    def _prediction_report(
        retriever_name: str,
        metadata: dict[str, Any],
        output: pd.DataFrame,
    ) -> str:
        selection = metadata["dev_metrics"]
        class_metrics = metadata["dev_class_metrics"]
        solvability = metadata["solvability_metrics"]
        lines = [
            f"# calibrated ordinal predictor report for {retriever_name}",
            "",
            "## Design",
            "",
            "- Separate solvability head tuned for high recall at the maximum candidate size.",
            "- Cumulative calibrated stage probabilities projected to a monotone sequence across candidate sizes.",
            "- Coarsened candidate-size grid to avoid unstable distinctions between adjacent large sizes.",
            "- Per-class probabilities derived from successive differences of the monotone cumulative "
            "probabilities (Frank & Hall, 2001, \"A Simple Approach to Ordinal Classification\"), decoded by "
            "taking the argmax class per query rather than a searched per-depth confidence threshold.",
            "",
            "## Metrics",
            "",
            f"- Evaluation split: {metadata['evaluation_split']}",
            f"- Base model: {metadata['base_model']}",
            f"- Candidate sizes: {metadata['candidate_sizes']}",
            f"- Decoding: {metadata['decoding']}",
            f"- Solvability threshold: {metadata['solvability_threshold']:.6f}",
            f"- Candidate recall: {selection['candidate_recall']:.4f}",
            f"- Mean selected size: {selection['mean_selected_size']:.2f}",
            f"- Median selected size: {selection['median_selected_size']:.2f}",
            f"- Failure rate: {selection['failure_rate']:.4f}",
            f"- Exact class accuracy: {class_metrics['exact_class_accuracy']:.4f}",
            f"- Within-one-class accuracy: {class_metrics['within_one_class_accuracy']:.4f}",
            f"- Ordinal MAE: {class_metrics['ordinal_mae']:.4f}",
            f"- Quadratic weighted Cohen's kappa: {class_metrics['quadratic_weighted_cohen_kappa']:.4f}",
            f"- Solvability precision: {solvability['precision']:.4f}",
            f"- Solvability recall: {solvability['recall']:.4f}",
            f"- Solvability MCC: {solvability['mcc']:.4f}",
            "",
            "## Most confident predictions",
            "",
        ]
        for row in output.sort_values("confidence", ascending=False).head(20).itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, solvability={row.solvability_probability:.4f}, "
                f"selected_size={row.selected_size}, failure={row.predicted_failure}, signals={row.strong_feature_signals}"
            )
        lines.extend(["", "## Least confident predictions", ""])
        for row in output.sort_values("confidence", ascending=True).head(20).itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, solvability={row.solvability_probability:.4f}, "
                f"selected_size={row.selected_size}, failure={row.predicted_failure}, signals={row.strong_feature_signals}"
            )
        return "\n".join(lines)


def _calibrated_ordinal_model_manifest_path(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "calibrated_ordinal",
) -> Path:
    return _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "manifest.json"


def _calibrated_ordinal_model_signature(
    store: ArtifactStore,
    retriever_name: str,
    manifest: dict[str, Any],
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "calibrated_ordinal",
) -> str:
    model_dir = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    solvability_model_path = model_dir / "solvability_model.joblib"
    if not solvability_model_path.exists():
        raise FileNotFoundError(
            f"Missing calibrated ordinal solvability model for {retriever_name}: {solvability_model_path}"
        )
    digest.update(_sha256_path(solvability_model_path).encode("ascii"))
    for depth in [int(value) for value in manifest.get("candidate_sizes", [])]:
        model_path = model_dir / f"depth_{depth:04d}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing calibrated ordinal model for {retriever_name} at depth {depth}: {model_path}"
            )
        digest.update(f"depth:{depth}".encode("ascii"))
        digest.update(_sha256_path(model_path).encode("ascii"))
    return digest.hexdigest()


def _calibrated_ordinal_cached_test_predictions(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    manifest: dict[str, Any],
    current_model_signature: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "calibrated_ordinal",
) -> pd.DataFrame | None:
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    if not paths["prediction"].exists() or not paths["metadata"].exists():
        return None
    metadata = store.read_json(paths["metadata"])
    if int(metadata.get("metadata_version", 0)) != CALIBRATED_ORDINAL_SPLIT_METADATA_VERSION:
        return None
    if metadata.get("evaluation_split") != "test":
        return None
    if int(metadata.get("predictor_metadata_version", 0)) != CALIBRATED_ORDINAL_METADATA_VERSION:
        return None
    if metadata.get("input_compatibility") != manifest.get("input_compatibility"):
        return None
    if metadata.get("source_retrievers") != manifest.get("source_retrievers"):
        return None
    if list(metadata.get("candidate_sizes", [])) != list(manifest.get("candidate_sizes", [])):
        return None
    if metadata.get("decoding") != CALIBRATED_ORDINAL_DECODING_NAME:
        return None
    cached_solvability_threshold = metadata.get("solvability_threshold")
    manifest_solvability_threshold = manifest.get("solvability_threshold")
    if cached_solvability_threshold is None or manifest_solvability_threshold is None:
        return None
    if not math.isclose(
        float(cached_solvability_threshold),
        float(manifest_solvability_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return None
    if metadata.get("model_signature") != current_model_signature:
        return None
    return store.read_parquet(paths["prediction"])


def load_or_create_calibrated_ordinal_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "calibrated_ordinal",
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    manifest_path = _calibrated_ordinal_model_manifest_path(
        store, retriever_name, mode=mode, predictor_name=predictor_name
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing calibrated ordinal model manifest for {language}/{retriever_name} "
            f"(predictor_name={predictor_name!r}): {manifest_path}. "
            f"Run the calibrated ordinal predictor first to create train-only model artifacts."
        )
    manifest = store.read_json(manifest_path)
    if int(manifest.get("model_metadata_version", 0)) != CALIBRATED_ORDINAL_MODEL_METADATA_VERSION:
        raise ValueError(
            f"Calibrated ordinal model manifest for {language}/{retriever_name} uses version "
            f"{int(manifest.get('model_metadata_version', 0))}, "
            f"but version {CALIBRATED_ORDINAL_MODEL_METADATA_VERSION} is required."
        )
    model_signature = _calibrated_ordinal_model_signature(
        store, retriever_name, manifest, mode=mode, predictor_name=predictor_name
    )
    cached = _calibrated_ordinal_cached_test_predictions(
        store,
        language,
        retriever_name,
        manifest,
        model_signature,
        mode=mode,
        predictor_name=predictor_name,
    )
    if cached is not None:
        console.log(f"Using cached calibrated ordinal test predictions for {retriever_name} ({predictor_name})")
        return cached
    candidate_sizes = [int(depth) for depth in manifest.get("candidate_sizes", [])]
    if not candidate_sizes:
        raise ValueError(
            f"Calibrated ordinal model manifest for {language}/{retriever_name} does not define any candidate sizes"
        )
    source_retrievers = [str(name) for name in manifest.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring calibrated ordinal test split for {retriever_name} ({predictor_name})")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frames = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=candidate_sizes,
    )
    model_dir = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    solvability_model = joblib.load(model_dir / "solvability_model.joblib")
    stage_models: dict[int, tuple[Any, pd.DataFrame, dict[str, Any]]] = {}
    for depth in candidate_sizes:
        model_path = model_dir / f"depth_{depth:04d}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing calibrated ordinal model for {language}/{retriever_name} at depth {depth}: {model_path}"
            )
        stage_models[int(depth)] = (
            joblib.load(model_path),
            pd.DataFrame(),
            dict(manifest.get("calibration", {}).get("stages", {}).get(str(int(depth)), {})),
        )
    reference = CalibratedOrdinalPredictorRunner._reference_output_frame(test_frames, candidate_sizes)
    solvability_probabilities = _positive_class_probabilities(solvability_model, test_frames[max(candidate_sizes)])
    stage_probabilities = _monotone_cumulative_probabilities(
        CalibratedOrdinalPredictorRunner._stage_probability_arrays(test_frames, stage_models)
    )
    solvability_threshold = manifest.get("solvability_threshold")
    if solvability_threshold is None:
        raise ValueError(
            f"Calibrated ordinal model manifest for {language}/{retriever_name} is missing solvability_threshold"
        )
    output = CalibratedOrdinalPredictorRunner._predict_from_probabilities(
        reference,
        solvability_probabilities,
        stage_probabilities,
        float(solvability_threshold),
        include_traces=True,
    )
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], _safe_prediction_frame(output))
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": CALIBRATED_ORDINAL_SPLIT_METADATA_VERSION,
            "predictor_metadata_version": CALIBRATED_ORDINAL_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "candidate_sizes": candidate_sizes,
            "decoding": CALIBRATED_ORDINAL_DECODING_NAME,
            "solvability_threshold": float(solvability_threshold),
            "training_split": manifest.get("training_split", "train"),
            "evaluation_split": "test",
            "input_compatibility": manifest.get("input_compatibility", {}),
            "model_signature": model_signature,
        },
    )
    return output


def _in_top_k_model_path(
    store: ArtifactStore,
    retriever_name: str,
    target_depth: int,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name_prefix: str = "in_top_k",
) -> Path:
    return (
        _prediction_model_dir(store, f"{predictor_name_prefix}{int(target_depth)}", retriever_name, mode=mode)
        / "model.joblib"
    )


def load_or_create_in_top_k_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    target_depth: int,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name_prefix: str = "in_top_k",
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    predictor_name = f"{predictor_name_prefix}{int(target_depth)}"
    model_path = _in_top_k_model_path(
        store, retriever_name, target_depth, mode=mode, predictor_name_prefix=predictor_name_prefix
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {predictor_name} model for {language}/{retriever_name}: {model_path}. "
            "Run scripts/run_in_top_max_depth_sweep_predictor.py first."
        )
    model_payload = joblib.load(model_path)
    if int(model_payload.get("predictor_metadata_version", 0)) != IN_TOP_MAX_DEPTH_METADATA_VERSION:
        raise ValueError(
            f"in-top-k={target_depth} model for {language}/{retriever_name} uses version "
            f"{int(model_payload.get('predictor_metadata_version', 0))}, "
            f"but version {IN_TOP_MAX_DEPTH_METADATA_VERSION} is required."
        )
    model_signature = _sha256_path(model_path)
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    if paths["prediction"].exists() and paths["metadata"].exists():
        metadata = store.read_json(paths["metadata"])
        if (
            int(metadata.get("metadata_version", 0)) == IN_TOP_K_SWEEP_SPLIT_METADATA_VERSION
            and metadata.get("evaluation_split") == "test"
            and metadata.get("model_signature") == model_signature
        ):
            console.log(f"Using cached in-top-k={target_depth} test predictions for {retriever_name}")
            return store.read_parquet(paths["prediction"])
    feature_depths = [int(depth) for depth in model_payload["feature_depths"]]
    source_retrievers = [str(name) for name in model_payload.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring in-top-k={target_depth} test split for {retriever_name}")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frames_by_depth = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=feature_depths,
    )
    merged_frame = InTopMaxDepthPredictorRunner._merge_multi_depth_frames(test_frames_by_depth)
    pipeline = model_payload["pipeline"]
    threshold = float(model_payload["threshold"])
    probabilities = _positive_class_probabilities(pipeline, merged_frame)
    output = merged_frame[["query_id", "document_id", "gold_rank"]].copy()
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output["predicted_in_top_k"] = output["confidence"] >= threshold
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], _safe_prediction_frame(output))
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": IN_TOP_K_SWEEP_SPLIT_METADATA_VERSION,
            "predictor_metadata_version": IN_TOP_MAX_DEPTH_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "target_depth": int(target_depth),
            "feature_depths": feature_depths,
            "threshold": threshold,
            "training_split": "train",
            "evaluation_split": "test",
            "model_signature": model_signature,
        },
    )
    return output


def _max_depth_model_path(
    store: ArtifactStore,
    predictor_name: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> Path:
    return _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "model.joblib"


def load_or_create_max_depth_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = IN_TOP_MAX_DEPTH_PREDICTOR_NAME,
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    model_path = _max_depth_model_path(store, predictor_name, retriever_name, mode=mode)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {predictor_name} model for {language}/{retriever_name}: {model_path}. "
            "Run the matching in-top-max-depth training script first."
        )
    model_payload = joblib.load(model_path)
    if int(model_payload.get("predictor_metadata_version", 0)) != IN_TOP_MAX_DEPTH_METADATA_VERSION:
        raise ValueError(
            f"{predictor_name} model for {language}/{retriever_name} uses version "
            f"{int(model_payload.get('predictor_metadata_version', 0))}, "
            f"but version {IN_TOP_MAX_DEPTH_METADATA_VERSION} is required."
        )
    model_signature = _sha256_path(model_path)
    paths = _prediction_split_output_paths(store, predictor_name, language, retriever_name, "test", mode=mode)
    if paths["prediction"].exists() and paths["metadata"].exists():
        metadata = store.read_json(paths["metadata"])
        if (
            int(metadata.get("metadata_version", 0)) == IN_TOP_K_SWEEP_SPLIT_METADATA_VERSION
            and metadata.get("evaluation_split") == "test"
            and metadata.get("model_signature") == model_signature
        ):
            console.log(f"Using cached {predictor_name} test predictions for {retriever_name}")
            return store.read_parquet(paths["prediction"])
    feature_depths = [int(depth) for depth in model_payload["feature_depths"]]
    source_retrievers = [str(name) for name in model_payload.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring {predictor_name} test split for {retriever_name}")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frames_by_depth = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=feature_depths,
    )
    merged_frame = InTopMaxDepthPredictorRunner._merge_multi_depth_frames(test_frames_by_depth)
    pipeline = model_payload["pipeline"]
    threshold = float(model_payload["threshold"])
    probabilities = _positive_class_probabilities(pipeline, merged_frame)
    output = merged_frame[["query_id", "document_id", "gold_rank"]].copy()
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output["predicted_in_top_max_depth"] = output["confidence"] >= threshold
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], _safe_prediction_frame(output))
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": IN_TOP_K_SWEEP_SPLIT_METADATA_VERSION,
            "predictor_metadata_version": IN_TOP_MAX_DEPTH_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "feature_depths": feature_depths,
            "threshold": threshold,
            "training_split": "train",
            "evaluation_split": "test",
            "model_signature": model_signature,
        },
    )
    return output


def _hit_at_1_cached_test_predictions(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    model_payload: dict[str, Any],
    current_model_signature: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame | None:
    paths = _prediction_split_output_paths(store, "hit_at_1", language, retriever_name, "test", mode=mode)
    if not paths["prediction"].exists() or not paths["metadata"].exists():
        return None
    metadata = store.read_json(paths["metadata"])
    if int(metadata.get("metadata_version", 0)) != HIT_AT_1_METADATA_VERSION:
        return None
    if metadata.get("evaluation_split") != "test":
        return None
    if metadata.get("model_signature") != current_model_signature:
        return None
    feature_depth = metadata.get("feature_depth")
    payload_feature_depth = model_payload.get("feature_depth")
    if feature_depth is None or payload_feature_depth is None or int(feature_depth) != int(payload_feature_depth):
        return None
    cached_threshold = metadata.get("threshold")
    payload_threshold = model_payload.get("threshold")
    if cached_threshold is None or payload_threshold is None:
        return None
    if not math.isclose(float(cached_threshold), float(payload_threshold), rel_tol=0.0, abs_tol=1e-12):
        return None
    threshold_policy = metadata.get("threshold_policy")
    payload_threshold_policy = model_payload.get("threshold_policy")
    if not isinstance(threshold_policy, dict) or not isinstance(payload_threshold_policy, dict):
        return None
    if threshold_policy != payload_threshold_policy:
        return None
    input_compatibility = metadata.get("input_compatibility")
    expected_compatibility = {
        "feature_dependency_version": int(model_payload.get("feature_dependency_version", HIT_AT_1_FEATURE_DEPENDENCY_VERSION)),
        "query_signal_schema_version": int(model_payload.get("query_signal_schema_version", QUERY_SIGNAL_SCHEMA_VERSION)),
    }
    if input_compatibility != expected_compatibility:
        return None
    if metadata.get("calibration") != model_payload.get("calibration", {"method": "none"}):
        return None
    return store.read_parquet(paths["prediction"])


def load_or_create_hit_at_1_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    model_path = _prediction_model_dir(store, "hit_at_1", retriever_name, mode=mode) / "model.joblib"
    model_payload = _ensure_current_hit_at_1_model(
        store,
        console,
        features,
        retrieval,
        language,
        retriever_name,
        mode=mode,
    )
    current_model_signature = _sha256_path(model_path)
    cached = _hit_at_1_cached_test_predictions(
        store,
        language,
        retriever_name,
        model_payload,
        current_model_signature,
        mode=mode,
    )
    if cached is not None:
        console.log(f"Using cached hit@1 test predictions for {retriever_name}")
        return cached
    feature_depth = int(model_payload["feature_depth"])
    threshold = float(model_payload["threshold"])
    model = model_payload["pipeline"]
    source_retrievers = [str(name) for name in model_payload.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring hit@1 test split for {retriever_name}")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frame = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=[feature_depth],
    )[feature_depth]
    probabilities = _positive_class_probabilities(model, test_frame)
    output = test_frame[
        ["query_id", "document_id", "language", "split_name", "retriever_name", "gold_rank"]
    ].copy()
    output["hit_at_1"] = _hit_at_1_target(test_frame).astype(bool)
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output["predicted_hit_at_1"] = output["confidence"] >= threshold
    paths = _prediction_split_output_paths(store, "hit_at_1", language, retriever_name, "test", mode=mode)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], output)
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": HIT_AT_1_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "evaluation_split": "test",
            "feature_depth": feature_depth,
            "threshold": threshold,
            "threshold_policy": dict(model_payload.get("threshold_policy", _hit_at_1_threshold_policy())),
            "input_compatibility": {
                "feature_dependency_version": int(model_payload.get("feature_dependency_version", HIT_AT_1_FEATURE_DEPENDENCY_VERSION)),
                "query_signal_schema_version": int(model_payload.get("query_signal_schema_version", QUERY_SIGNAL_SCHEMA_VERSION)),
            },
            "calibration": model_payload.get("calibration", {"method": "none"}),
            "model_signature": current_model_signature,
        },
    )
    return output


def _qpp_model_manifest_path(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "qpp",
) -> Path:
    return _prediction_model_dir(store, predictor_name, retriever_name, mode=mode) / "manifest.json"


def _qpp_model_artifact_status(
    store: ArtifactStore,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
    predictor_name: str = "qpp",
    metadata_version: int = QPP_METADATA_VERSION,
    model_metadata_version: int = QPP_MODEL_METADATA_VERSION,
) -> tuple[dict[str, Any] | None, str | None]:
    manifest_path = _qpp_model_manifest_path(store, retriever_name, mode=mode, predictor_name=predictor_name)
    if not manifest_path.exists():
        return None, f"cached {predictor_name} outputs are missing model manifest at {manifest_path}"
    manifest = store.read_json(manifest_path)
    if int(manifest.get("model_metadata_version", 0)) != model_metadata_version:
        return None, (
            f"cached {predictor_name} model manifest uses version {int(manifest.get('model_metadata_version', 0))}, "
            f"but version {model_metadata_version} is required"
        )
    if int(manifest.get("predictor_metadata_version", 0)) != metadata_version:
        return None, (
            f"cached {predictor_name} model manifest uses predictor metadata version "
            f"{int(manifest.get('predictor_metadata_version', 0))}, "
            f"but version {metadata_version} is required"
        )
    model_dir = _prediction_model_dir(store, predictor_name, retriever_name, mode=mode)
    failure_model_path = model_dir / "failure_model.joblib"
    if not failure_model_path.exists():
        return None, f"cached {predictor_name} outputs are missing failure model at {failure_model_path}"
    score_source = manifest.get("score_source")
    if score_source == "regression_full":
        score_model_path = model_dir / "score_model.joblib"
        if not score_model_path.exists():
            return None, f"cached {predictor_name} outputs are missing score model at {score_model_path}"
    return manifest, None


def _qpp_model_signature(
    store: ArtifactStore,
    retriever_name: str,
    manifest: dict[str, Any],
    mode: str = PREDICTOR_MODE_LOCAL,
) -> str:
    model_dir = _prediction_model_dir(store, "qpp", retriever_name, mode=mode)
    digest = hashlib.sha256()
    digest.update(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    failure_model_path = model_dir / "failure_model.joblib"
    if not failure_model_path.exists():
        raise FileNotFoundError(
            f"Missing QPP failure model for {retriever_name}: {failure_model_path}"
        )
    digest.update(_sha256_path(failure_model_path).encode("utf-8"))
    if manifest.get("score_source") == "regression_full":
        score_model_path = model_dir / "score_model.joblib"
        if not score_model_path.exists():
            raise FileNotFoundError(
                f"Missing QPP score model for {retriever_name}: {score_model_path}"
            )
        digest.update(_sha256_path(score_model_path).encode("utf-8"))
    return digest.hexdigest()


def _qpp_cached_test_predictions(
    store: ArtifactStore,
    language: str,
    retriever_name: str,
    manifest: dict[str, Any],
    current_model_signature: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame | None:
    paths = _prediction_split_output_paths(store, "qpp", language, retriever_name, "test", mode=mode)
    if not paths["prediction"].exists() or not paths["metadata"].exists():
        return None
    metadata = store.read_json(paths["metadata"])
    if int(metadata.get("metadata_version", 0)) != QPP_SPLIT_METADATA_VERSION:
        return None
    if metadata.get("evaluation_split") != "test":
        return None
    if int(metadata.get("predictor_metadata_version", 0)) != QPP_METADATA_VERSION:
        return None
    if metadata.get("input_compatibility") != manifest.get("input_compatibility"):
        return None
    if metadata.get("source_retrievers") != manifest.get("source_retrievers"):
        return None
    if metadata.get("model_signature") != current_model_signature:
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
    return store.read_parquet(paths["prediction"])


def load_or_create_qpp_test_predictions(
    store: ArtifactStore,
    console: Console,
    features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    mode: str = PREDICTOR_MODE_LOCAL,
) -> pd.DataFrame:
    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    manifest, reason = _qpp_model_artifact_status(store, retriever_name, mode=mode)
    if manifest is None:
        raise FileNotFoundError(
            f"Missing QPP model artifacts for {language}/{retriever_name}: {reason}. "
            f"Run the QPP predictor first to create train-only model artifacts."
        )
    model_signature = _qpp_model_signature(store, retriever_name, manifest, mode=mode)
    cached = _qpp_cached_test_predictions(
        store,
        language,
        retriever_name,
        manifest,
        model_signature,
        mode=mode,
    )
    if cached is not None:
        console.log(f"Using cached QPP test predictions for {retriever_name}")
        return cached
    feature_depth = manifest.get("feature_depth")
    if feature_depth is None:
        raise ValueError(f"QPP model manifest for {language}/{retriever_name} is missing feature_depth")
    feature_depth = int(feature_depth)
    source_retrievers = [str(name) for name in manifest.get("source_retrievers") or [retriever_name]]
    console.log(f"Scoring QPP test split for {retriever_name}")
    views = PredictorFeatureViews(store, console, features, retrieval)
    test_frame = views.build(
        f"{language}_test",
        retriever_name,
        mode,
        source_retrievers=source_retrievers,
        candidate_sizes=[feature_depth],
    )[feature_depth]
    model_dir = _prediction_model_dir(store, "qpp", retriever_name, mode=mode)
    failure_model = joblib.load(model_dir / "failure_model.joblib")
    score_source = str(manifest["score_source"])
    if score_source == "regression_full":
        score_pipeline = joblib.load(model_dir / "score_model.joblib")
        predicted_scores = score_pipeline.predict(test_frame)
    else:
        predicted_scores = test_frame[score_source].to_numpy(dtype=np.float64)
    hit_probabilities = _positive_class_probabilities(failure_model, test_frame)
    train_min = float(manifest["train_min"])
    train_max = float(manifest["train_max"])
    normalized_scores = np.asarray(
        [_normalize_qpp_score(value, train_min, train_max) for value in predicted_scores],
        dtype=np.float64,
    )
    mapping = str(manifest["mapping"])
    minimum_size = int(manifest["minimum_size"])
    maximum_size = int(manifest["maximum_size"])
    mapped_sizes = [
        _map_qpp_score(value, mapping, minimum_size, maximum_size)
        for value in normalized_scores
    ]
    failure_threshold = float(manifest.get("failure_threshold", 0.5))
    output = QppPredictorRunner._prediction_frame(
        test_frame,
        hit_probabilities,
        normalized_scores,
        mapped_sizes,
        failure_threshold,
    )
    paths = _prediction_split_output_paths(store, "qpp", language, retriever_name, "test", mode=mode)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    store.write_parquet(paths["prediction"], _safe_prediction_frame(output))
    store.write_json(
        paths["metadata"],
        {
            "metadata_version": QPP_SPLIT_METADATA_VERSION,
            "predictor_metadata_version": QPP_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": source_retrievers,
            "feature_depth": feature_depth,
            "score_source": score_source,
            "mapping": mapping,
            "minimum_size": minimum_size,
            "maximum_size": maximum_size,
            "train_min": train_min,
            "train_max": train_max,
            "failure_threshold": failure_threshold,
            "failure_threshold_policy": manifest.get("failure_threshold_policy", _qpp_failure_threshold_policy()),
            "calibration": manifest.get("calibration", {"method": "none"}),
            "training_split": manifest.get("training_split", "train"),
            "evaluation_split": "test",
            "input_compatibility": manifest.get("input_compatibility", {}),
            "model_signature": model_signature,
        },
    )
    return output


class HitAtOnePredictorRunner:
    predictor_name = "hit_at_1"
    compute_signal_summaries = True

    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.views = PredictorFeatureViews(store, console, features, retrieval)

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary = {}
        selected = self._ordered_retrievers(retrievers)
        modes = _normalize_predictor_modes(self.config)
        local_metrics: dict[str, dict[str, float] | None] = {}
        original_scopes = self._configured_document_scopes()
        try:
            for document_scope, scope_retrievers in self._scope_phases(selected):
                self.console.log(
                    f"Running hit@1 predictor scope phase {document_scope} "
                    f"for {len(scope_retrievers)} retrievers"
                )
                self._set_enabled_document_scopes([document_scope])
                for retriever_name in scope_retrievers:
                    metrics = self._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
                    summary[retriever_name] = metrics
                    local_metrics[retriever_name] = metrics
            self._set_enabled_document_scopes(original_scopes)
            if PREDICTOR_MODE_ALL_MEANINGFUL in modes:
                winners = _select_best_hit_at_1_variants(selected, local_metrics)
                if winners:
                    winner_text = ", ".join(f"{base_retriever_name(name)} -> {name}" for name in winners)
                    self.console.log(f"Selected local winners for hit@1 {PREDICTOR_MODE_ALL_MEANINGFUL}: {winner_text}")
                for retriever_name in winners:
                    metrics = self._run_retriever(
                        language,
                        retriever_name,
                        PREDICTOR_MODE_ALL_MEANINGFUL,
                        source_retrievers=winners,
                    )
                    summary[_display_retriever_name(retriever_name, PREDICTOR_MODE_ALL_MEANINGFUL)] = metrics
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
        self.console.section(f"Hit@1 predictor for {display_name}")
        cached_metrics, cache_reason = _cached_hit_at_1_cache_status(
            self.store,
            language,
            retriever_name,
            mode=mode,
            predictor_name=self.predictor_name,
        )
        if cached_metrics is not None:
            self.console.log(f"Using cached hit@1 outputs for {display_name}")
            return cached_metrics
        if cache_reason:
            self.console.log(f"Rebuilding hit@1 outputs for {display_name}: {cache_reason}")
        feature_depth = self._feature_depth()
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        self.console.log(f"Preparing hit@1 feature context for {display_name} at depth {feature_depth}")
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
        self.console.log(f"Writing hit@1 outputs for {display_name}")
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

    def _fit_and_predict(
        self,
        display_name: str,
        train_frame: pd.DataFrame,
        dev_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float], float, pd.DataFrame, Any, dict[str, Any]]:
        feature_columns = _feature_columns(train_frame)
        target = _hit_at_1_target(train_frame).astype(int)
        self.console.log(f"Training hit@1 random forest for {display_name}")
        importance_model = _fit_pipeline(
            _make_classifier("random_forest", self.config.runtime.random_seed),
            feature_columns,
            train_frame,
            target,
        )
        probability_model, calibration_info = self._fit_probability_model(train_frame, feature_columns, target, importance_model)
        importances = _feature_importances(importance_model, feature_columns)
        self.console.log(f"Scoring hit@1 dev split for {display_name}")
        probabilities = _positive_class_probabilities(probability_model, dev_frame)
        threshold, threshold_metrics = self._select_threshold(
            probabilities,
            _hit_at_1_target(dev_frame),
        )
        self.console.log(
            f"Selected hit@1 threshold {threshold:.6f} for {display_name} "
            f"(f0.5={threshold_metrics['f0_5']}, precision={threshold_metrics['precision']}, "
            f"recall={threshold_metrics['recall']}, "
            f"false_positives={int(threshold_metrics['false_positives'])})"
        )
        output = self._prediction_frame(dev_frame, probabilities, threshold, train_frame, importances)
        return output, _hit_at_1_metrics(output), float(threshold), importances, probability_model, calibration_info

    def _prediction_frame(
        self,
        evaluation_frame: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float,
        training_frame: pd.DataFrame,
        importances: pd.DataFrame,
    ) -> pd.DataFrame:
        output = evaluation_frame[
            ["query_id", "document_id", "language", "split_name", "retriever_name", "gold_rank"]
        ].copy()
        output["hit_at_1"] = _hit_at_1_target(evaluation_frame).astype(bool)
        output["confidence"] = np.asarray(probabilities, dtype=np.float64)
        output["predicted_hit_at_1"] = output["confidence"] >= float(threshold)
        output["strong_feature_signals"] = [
            _local_signal_summary(evaluation_frame.iloc[index], training_frame, importances)
            if self.compute_signal_summaries
            else ""
            for index in range(len(evaluation_frame))
        ]
        return output

    def _write_outputs(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        output: pd.DataFrame,
        metrics: dict[str, float],
        threshold: float,
        importances: pd.DataFrame,
        model: Any,
        feature_depth: int,
        mode: str = PREDICTOR_MODE_LOCAL,
        source_retrievers: list[str] | None = None,
        calibration_info: dict[str, Any] | None = None,
    ) -> None:
        prediction_paths = _prediction_output_paths(self.store, self.predictor_name, language, retriever_name, mode=mode)
        prediction_paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
        prediction_paths["report"].parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(prediction_paths["prediction"], index=False)
        metadata = {
            "metadata_version": HIT_AT_1_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": list(source_retrievers or [retriever_name]),
            "evaluation_split": "dev",
            "feature_depth": feature_depth,
            "threshold": threshold,
            "threshold_policy": _hit_at_1_threshold_policy(),
            "input_compatibility": {
                "feature_dependency_version": HIT_AT_1_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
            },
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
                "predictor_metadata_version": HIT_AT_1_METADATA_VERSION,
                "predictor_feature_mode": mode,
                "target_retriever": retriever_name,
                "source_retrievers": list(source_retrievers or [retriever_name]),
                "pipeline": model,
                "threshold": threshold,
                "feature_depth": feature_depth,
                "threshold_policy": _hit_at_1_threshold_policy(),
                "feature_dependency_version": HIT_AT_1_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
                "calibration": calibration_info or {"method": "none"},
            },
            model_root / "model.joblib",
        )
        importances.to_parquet(model_root / "feature_importances.parquet", index=False)

    @staticmethod
    def _select_threshold(
        probabilities: np.ndarray,
        target: pd.Series,
    ) -> tuple[float, dict[str, float]]:
        values = np.asarray(probabilities, dtype=np.float64)
        truth = target.astype(bool).to_numpy(dtype=bool)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 1.0, {
                "f0_5": math.nan,
                "f1": math.nan,
                "precision": math.nan,
                "recall": 0.0,
                "false_positives": 0.0,
                "true_positives": 0.0,
            }
        candidate_thresholds = sorted({float(value) for value in finite_values}, reverse=True)
        candidates: list[tuple[tuple[float, float, float, float, float], float, dict[str, float]]] = []
        for threshold in candidate_thresholds:
            predicted = values >= threshold
            tp = int(np.sum(predicted & truth))
            fp = int(np.sum(predicted & ~truth))
            fn = int(np.sum(~predicted & truth))
            positive_predictions = tp + fp
            if positive_predictions == 0:
                continue
            precision = float(tp / positive_predictions)
            recall = float(tp / (tp + fn)) if tp + fn else 0.0
            f0_5 = _fbeta_score(precision, recall, beta=0.5)
            f1 = _fbeta_score(precision, recall, beta=1.0)
            metrics = {
                "f0_5": f0_5,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "false_positives": float(fp),
                "true_positives": float(tp),
                "positive_predictions": float(positive_predictions),
            }
            candidates.append(
                ((f0_5, precision, recall, -float(fp), -float(threshold)), float(threshold), metrics)
            )
        if candidates:
            _, best_threshold, best_metrics = max(candidates, key=lambda item: item[0])
            return best_threshold, best_metrics
        return 1.0, {
            "f0_5": math.nan,
            "f1": math.nan,
            "precision": math.nan,
            "recall": 0.0,
            "false_positives": 0.0,
            "true_positives": 0.0,
        }

    @staticmethod
    def _prediction_report(
        retriever_name: str,
        metadata: dict[str, Any],
        output: pd.DataFrame,
    ) -> str:
        metrics = metadata["dev_metrics"]
        lines = [
            f"# hit@1 predictor report for {retriever_name}",
            "",
            "## Metrics",
            "",
            f"- Evaluation split: {metadata['evaluation_split']}",
            f"- Feature depth: {metadata['feature_depth']}",
            f"- Threshold: {metadata['threshold']:.6f}",
            f"- Threshold policy: {metadata['threshold_policy']['name']}",
            f"- Calibration: {metadata['calibration']['method']}",
            f"- F0.5: {metrics['f0_5']}",
            f"- F1: {metrics['f1']}",
            f"- Precision: {metrics['precision']}",
            f"- Recall: {metrics['recall']}",
            f"- False positives: {int(metrics['false_positives'])}",
            f"- True positives: {int(metrics['true_positives'])}",
            "",
            "## Most confident predicted hits",
            "",
        ]
        predicted_hits = output.loc[output["predicted_hit_at_1"]].sort_values("confidence", ascending=False).head(20)
        for row in predicted_hits.itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, gold_rank={row.gold_rank}, "
                f"hit_at_1={row.hit_at_1}, signals={row.strong_feature_signals}"
            )
        lines.extend(["", "## Most confident non-hits", ""])
        predicted_non_hits = output.loc[~output["predicted_hit_at_1"]].sort_values("confidence", ascending=False).head(20)
        for row in predicted_non_hits.itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, gold_rank={row.gold_rank}, "
                f"hit_at_1={row.hit_at_1}, signals={row.strong_feature_signals}"
            )
        return "\n".join(lines)

    def _feature_depth(self) -> int:
        return int(min(self.config.candidate_sizes))

    def _scope_phases(self, retrievers: list[str]) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for retriever_name in retrievers:
            grouped.setdefault(retriever_document_scope(retriever_name), []).append(retriever_name)
        ordered_scopes: list[str] = []
        for document_scope in self._configured_document_scopes():
            if document_scope in grouped and document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        for retriever_name in retrievers:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        return [(document_scope, grouped[document_scope]) for document_scope in ordered_scopes]

    def _set_enabled_document_scopes(self, scopes: list[str]) -> None:
        scopes = list(scopes)
        current_scopes = self._configured_document_scopes()
        if current_scopes == scopes:
            return
        self.config.enabled_document_scopes = scopes
        if hasattr(self.features, "_all_rankings_top100_cache"):
            self.features._all_rankings_top100_cache.clear()

    def _configured_document_scopes(self) -> list[str]:
        return list(getattr(self.config, "enabled_document_scopes", ["title_abstract"]))

    def _fit_probability_model(
        self,
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        target: pd.Series,
        fallback_model: Pipeline,
    ) -> tuple[Any, dict[str, Any]]:
        positive_count = int(target.astype(int).sum())
        negative_count = int(len(target) - positive_count)
        if positive_count == 0 or negative_count == 0:
            self.console.log(
                "Skipping hit@1 probability calibration because the training split contains only one class"
            )
            return fallback_model, {"method": "none", "reason": "single_class_training_data"}
        cv_folds = min(3, positive_count, negative_count)
        if cv_folds < 2:
            self.console.log(
                "Skipping hit@1 probability calibration because the class counts are too small for cross-validation"
            )
            return fallback_model, {"method": "none", "reason": "insufficient_class_counts_for_calibration"}
        self.console.log(f"Calibrating hit@1 probabilities with sigmoid calibration over {cv_folds} folds")
        calibrated_model = CalibratedClassifierCV(
            estimator=_pipeline_template(
                _make_classifier("random_forest", self.config.runtime.random_seed),
                feature_columns,
                train_frame,
            ),
            method="sigmoid",
            cv=cv_folds,
        )
        calibrated_model.fit(train_frame, target)
        return calibrated_model, {"method": "sigmoid", "cv_folds": int(cv_folds)}

    def _ordered_retrievers(self, retrievers: list[str] | None) -> list[str]:
        if retrievers:
            return [self.retrieval.resolve_retriever_name(name) for name in retrievers]
        enabled = [self.retrieval.resolve_retriever_name(name) for name in self.retrieval.enabled_retrievers()]
        original_order = {name: index for index, name in enumerate(enabled)}
        scope_order: list[str] = []
        for retriever_name in enabled:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in scope_order:
                scope_order.append(document_scope)
        scope_index = {document_scope: index for index, document_scope in enumerate(scope_order)}
        return sorted(
            enabled,
            key=lambda name: (
                scope_index[retriever_document_scope(name)],
                self._retriever_group(name),
                self._semantic_priority(name),
                original_order[name],
            ),
        )

    @staticmethod
    def _retriever_group(retriever_name: str) -> int:
        base_name = base_retriever_name(retriever_name)
        if base_name == "bm25":
            return 0
        if base_name in LEXICAL_RETRIEVERS:
            return 2
        return 1

    @staticmethod
    def _semantic_priority(retriever_name: str) -> int:
        priority_by_base_name = {
            "qwen3_embedding_0_6b": 0,
            "bge_m3": 1,
            "bge_large_en_v1_5": 2,
        }
        return priority_by_base_name.get(base_retriever_name(retriever_name), 100)


class InTopMaxDepthPredictorRunner:
    compute_signal_summaries = True

    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
        predictor_name: str = "in_top_max_depth",
        threshold_beta: float = 1.0,
        threshold_policy_name: str = IN_TOP_MAX_DEPTH_THRESHOLD_POLICY_NAME,
        threshold_objective: str = "mcc",
        target_depth: int | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.views = PredictorFeatureViews(store, console, features, retrieval)
        self.predictor_name = predictor_name
        self.threshold_beta = threshold_beta
        self.threshold_policy_name = threshold_policy_name
        self.threshold_objective = threshold_objective
        self.target_depth = target_depth

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary = {}
        selected = self._ordered_retrievers(retrievers)
        modes = _normalize_predictor_modes(self.config)
        local_metrics: dict[str, dict[str, float] | None] = {}
        original_scopes = self._configured_document_scopes()
        try:
            for document_scope, scope_retrievers in self._scope_phases(selected):
                self.console.log(
                    f"Running in-top-max-depth predictor scope phase {document_scope} "
                    f"for {len(scope_retrievers)} retrievers"
                )
                self._set_enabled_document_scopes([document_scope])
                for retriever_name in scope_retrievers:
                    metrics = self._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
                    summary[retriever_name] = metrics
                    local_metrics[retriever_name] = metrics
            self._set_enabled_document_scopes(original_scopes)
            if PREDICTOR_MODE_ALL_MEANINGFUL in modes:
                winners = _select_best_in_top_max_depth_variants(selected, local_metrics)
                if winners:
                    winner_text = ", ".join(f"{base_retriever_name(name)} -> {name}" for name in winners)
                    self.console.log(
                        f"Selected local winners for in-top-max-depth {PREDICTOR_MODE_ALL_MEANINGFUL}: {winner_text}"
                    )
                for retriever_name in winners:
                    metrics = self._run_retriever(
                        language,
                        retriever_name,
                        PREDICTOR_MODE_ALL_MEANINGFUL,
                        source_retrievers=winners,
                    )
                    summary[_display_retriever_name(retriever_name, PREDICTOR_MODE_ALL_MEANINGFUL)] = metrics
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
        self.console.section(f"{self._depth_label()} predictor for {display_name}")
        cached_metrics, cache_reason = _cached_in_top_max_depth_cache_status(
            self.store,
            language,
            retriever_name,
            mode=mode,
            predictor_name=self.predictor_name,
            threshold_policy_name=self.threshold_policy_name,
        )
        if cached_metrics is not None:
            self.console.log(f"Using cached in-top-max-depth outputs for {display_name}")
            return cached_metrics
        if cache_reason:
            self.console.log(f"Rebuilding in-top-max-depth outputs for {display_name}: {cache_reason}")
        feature_depths = self._feature_depths()
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        self.console.log(
            f"Preparing in-top-max-depth feature context for {display_name} across depths {feature_depths}"
        )
        train_frames_by_depth = self.views.build(
            train_key,
            retriever_name,
            mode,
            source_retrievers=source_retrievers,
            candidate_sizes=feature_depths,
        )
        dev_frames_by_depth = self.views.build(
            dev_key,
            retriever_name,
            mode,
            source_retrievers=source_retrievers,
            candidate_sizes=feature_depths,
        )
        train_frame = self._merge_multi_depth_frames(train_frames_by_depth)
        dev_frame = self._merge_multi_depth_frames(dev_frames_by_depth)
        output, metrics, threshold, importances, model, calibration_info, threshold_selection = self._fit_and_predict(
            display_name,
            train_frame,
            dev_frame,
        )
        self.console.log(f"Writing in-top-max-depth outputs for {display_name}")
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
            mode=mode,
            source_retrievers=source_retrievers,
            calibration_info=calibration_info,
            threshold_selection=threshold_selection,
        )
        return metrics

    @staticmethod
    def _merge_multi_depth_frames(frames_by_depth: dict[int, pd.DataFrame]) -> pd.DataFrame:
        ordered_depths = sorted(frames_by_depth)
        identifier_columns = [
            "query_id",
            "document_id",
            "language",
            "split_name",
            "retriever_name",
            "gold_rank",
            "smallest_sufficient_size",
            "retrieval_failure_within_max_depth",
        ]
        base_frame = frames_by_depth[ordered_depths[0]]
        merged = base_frame[[column for column in identifier_columns if column in base_frame.columns]].copy()
        for depth in ordered_depths:
            frame = frames_by_depth[depth]
            feature_columns = _feature_columns(frame)
            depth_frame = frame[["query_id", "document_id"] + feature_columns].copy()
            depth_frame.columns = ["query_id", "document_id"] + [
                f"depth_{depth}::{column}" for column in feature_columns
            ]
            merged = merged.merge(depth_frame, on=["query_id", "document_id"], how="left", sort=False)
        return merged

    def _fit_and_predict(
        self,
        display_name: str,
        train_frame: pd.DataFrame,
        dev_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, float], float, pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
        feature_columns = _feature_columns(train_frame)
        target = _in_top_max_depth_target(train_frame, self.target_depth).astype(int)
        self.console.log(
            f"Selecting in-top-max-depth threshold for {display_name} via "
            f"{IN_TOP_MAX_DEPTH_CV_FOLDS}-fold cross-validation on train"
        )
        out_of_fold_probabilities, cv_folds_used = self._cross_validated_probabilities(
            train_frame,
            feature_columns,
            target,
            display_name,
        )
        threshold, threshold_metrics = self._select_optimal_threshold(
            out_of_fold_probabilities,
            target,
            self.threshold_objective,
            self.threshold_beta,
        )
        if self.threshold_objective == "mcc":
            objective_label = "MCC"
        else:
            objective_label = f"F{self.threshold_beta:g}"
        solvable_examples = int(threshold_metrics["true_positives"] + threshold_metrics["false_negatives"])
        threshold_selection = {
            "policy": self.threshold_policy_name,
            "objective": self.threshold_objective,
            "beta": self.threshold_beta,
            "cv_folds": cv_folds_used,
            "cv_solvable_train_examples": solvable_examples,
            "cv_observed_false_negatives": int(threshold_metrics["false_negatives"]),
            "cv_precision": threshold_metrics["precision"],
            "cv_recall": threshold_metrics["recall"],
            "cv_score": threshold_metrics["score"],
        }
        self.console.log(
            f"Selected in-top-max-depth threshold {threshold:.6f} for {display_name} maximizing "
            f"{objective_label} across cross-validation (solvable_train_examples={solvable_examples}, "
            f"precision={threshold_metrics['precision']}, recall={threshold_metrics['recall']}, "
            f"score={threshold_metrics['score']})"
        )
        self.console.log(
            f"Training final in-top-max-depth random forest for {display_name} on the full train split "
            f"with class_weight={IN_TOP_MAX_DEPTH_CLASS_WEIGHT!r}"
        )
        probability_model, importances, calibration_info = _fit_calibrated_classifier(
            console=self.console,
            train_frame=train_frame,
            feature_columns=feature_columns,
            target=target,
            base_model_name="random_forest",
            seed=self.config.runtime.random_seed,
            purpose=f"in-top-max-depth predictor for {display_name}",
            class_weight=IN_TOP_MAX_DEPTH_CLASS_WEIGHT,
        )
        self.console.log(f"Scoring in-top-max-depth dev split for {display_name}")
        probabilities = _positive_class_probabilities(probability_model, dev_frame)
        output = self._prediction_frame(dev_frame, probabilities, threshold, train_frame, importances)
        return (
            output,
            _in_top_max_depth_metrics(output),
            float(threshold),
            importances,
            probability_model,
            calibration_info,
            threshold_selection,
        )

    def _cross_validated_probabilities(
        self,
        train_frame: pd.DataFrame,
        feature_columns: list[str],
        target: pd.Series,
        display_name: str,
    ) -> tuple[np.ndarray, int]:
        target_array = target.to_numpy()
        unique_classes, class_counts = np.unique(target_array, return_counts=True)
        if len(unique_classes) < 2 or int(class_counts.min()) < 2:
            self.console.log(
                f"Skipping cross-validation for {display_name} threshold selection because the training split "
                "does not contain enough examples of both classes; falling back to in-sample probabilities"
            )
            fallback_model = _fit_pipeline(
                _make_classifier(
                    "random_forest",
                    self.config.runtime.random_seed,
                    class_weight=IN_TOP_MAX_DEPTH_CLASS_WEIGHT,
                ),
                feature_columns,
                train_frame,
                target,
            )
            return _positive_class_probabilities(fallback_model, train_frame), 0
        n_splits = min(IN_TOP_MAX_DEPTH_CV_FOLDS, int(class_counts.min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.runtime.random_seed)
        out_of_fold_probabilities = np.full(len(train_frame), np.nan, dtype=np.float64)
        for fold_index, (fold_train_positions, fold_holdout_positions) in enumerate(
            splitter.split(train_frame, target_array), start=1
        ):
            fold_train_frame = train_frame.iloc[fold_train_positions]
            fold_train_target = target.iloc[fold_train_positions]
            fold_holdout_frame = train_frame.iloc[fold_holdout_positions]
            fold_model, _, _ = _fit_calibrated_classifier(
                console=self.console,
                train_frame=fold_train_frame,
                feature_columns=feature_columns,
                target=fold_train_target,
                base_model_name="random_forest",
                seed=self.config.runtime.random_seed,
                purpose=f"in-top-max-depth cross-validation fold {fold_index}/{n_splits} for {display_name}",
                class_weight=IN_TOP_MAX_DEPTH_CLASS_WEIGHT,
            )
            out_of_fold_probabilities[fold_holdout_positions] = _positive_class_probabilities(
                fold_model, fold_holdout_frame
            )
        return out_of_fold_probabilities, n_splits

    def _prediction_frame(
        self,
        evaluation_frame: pd.DataFrame,
        probabilities: np.ndarray,
        threshold: float,
        training_frame: pd.DataFrame,
        importances: pd.DataFrame,
    ) -> pd.DataFrame:
        output = evaluation_frame[
            ["query_id", "document_id", "language", "split_name", "retriever_name", "gold_rank"]
        ].copy()
        output["in_top_max_depth"] = _in_top_max_depth_target(evaluation_frame, self.target_depth).astype(bool)
        output["confidence"] = np.asarray(probabilities, dtype=np.float64)
        output["predicted_in_top_max_depth"] = output["confidence"] >= float(threshold)
        output["strong_feature_signals"] = [
            _local_signal_summary(evaluation_frame.iloc[index], training_frame, importances)
            if self.compute_signal_summaries
            else ""
            for index in range(len(evaluation_frame))
        ]
        return output

    def _write_outputs(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        output: pd.DataFrame,
        metrics: dict[str, float],
        threshold: float,
        importances: pd.DataFrame,
        model: Any,
        feature_depths: list[int],
        mode: str = PREDICTOR_MODE_LOCAL,
        source_retrievers: list[str] | None = None,
        calibration_info: dict[str, Any] | None = None,
        threshold_selection: dict[str, Any] | None = None,
    ) -> None:
        prediction_paths = _prediction_output_paths(self.store, self.predictor_name, language, retriever_name, mode=mode)
        prediction_paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
        prediction_paths["report"].parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(prediction_paths["prediction"], index=False)
        metadata = {
            "metadata_version": IN_TOP_MAX_DEPTH_METADATA_VERSION,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": list(source_retrievers or [retriever_name]),
            "evaluation_split": "dev",
            "feature_depths": [int(depth) for depth in feature_depths],
            "max_ranking_depth": int(self.config.max_ranking_depth),
            "target_depth": self._resolved_target_depth(),
            "target_depth_is_explicit": self.target_depth is not None,
            "threshold": threshold,
            "threshold_policy": _in_top_max_depth_threshold_policy(self.threshold_policy_name),
            "threshold_selection": threshold_selection or {},
            "input_compatibility": {
                "feature_dependency_version": IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
            },
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
                "predictor_metadata_version": IN_TOP_MAX_DEPTH_METADATA_VERSION,
                "predictor_feature_mode": mode,
                "target_retriever": retriever_name,
                "source_retrievers": list(source_retrievers or [retriever_name]),
                "pipeline": model,
                "threshold": threshold,
                "target_depth": self._resolved_target_depth(),
                "feature_depths": [int(depth) for depth in feature_depths],
                "threshold_policy": _in_top_max_depth_threshold_policy(self.threshold_policy_name),
                "feature_dependency_version": IN_TOP_MAX_DEPTH_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
                "calibration": calibration_info or {"method": "none"},
            },
            model_root / "model.joblib",
        )
        importances.to_parquet(model_root / "feature_importances.parquet", index=False)

    @staticmethod
    def _select_optimal_threshold(
        probabilities: np.ndarray,
        target: pd.Series,
        objective: str,
        beta: float,
    ) -> tuple[float, dict[str, float]]:
        if objective not in ("mcc", "fbeta"):
            raise ValueError(f"Unsupported threshold objective: {objective!r}")
        values = np.asarray(probabilities, dtype=np.float64)
        truth = target.astype(bool).to_numpy(dtype=bool)
        finite_mask = np.isfinite(values)
        values = values[finite_mask]
        truth = truth[finite_mask]
        if values.size == 0:
            return 1.0, {
                "precision": math.nan,
                "recall": math.nan,
                "score": math.nan,
                "true_positives": 0.0,
                "false_positives": 0.0,
                "true_negatives": 0.0,
                "false_negatives": 0.0,
                "positive_predictions": 0.0,
            }
        candidate_thresholds = sorted({0.0, 1.0, *[float(value) for value in values]})
        candidates = []
        for threshold in candidate_thresholds:
            predicted = values >= threshold
            summary = _binary_classification_summary(truth, predicted)
            if objective == "mcc":
                try:
                    score = float(matthews_corrcoef(truth, predicted))
                except ValueError:
                    score = math.nan
            else:
                score = _fbeta_score(summary["precision"], summary["recall"], beta=beta)
            metrics = {**summary, "score": score}
            key = (_safe_sort_float(score), _safe_sort_float(summary["recall"]), -float(threshold))
            candidates.append((key, float(threshold), metrics))
        _, best_threshold, best_metrics = max(candidates, key=lambda item: item[0])
        return best_threshold, best_metrics

    @staticmethod
    def _prediction_report(
        retriever_name: str,
        metadata: dict[str, Any],
        output: pd.DataFrame,
    ) -> str:
        metrics = metadata["dev_metrics"]
        threshold_selection = metadata.get("threshold_selection") or {}
        beta = threshold_selection.get("beta", 1.0)
        objective = threshold_selection.get("objective", "fbeta")
        if objective == "mcc":
            objective_label = "MCC"
        else:
            objective_label = f"F{beta:g}"
        target_depth = metadata.get("target_depth", metadata.get("max_ranking_depth"))
        report_title = (
            f"in-top-k={target_depth}" if metadata.get("target_depth_is_explicit") else "in-top-max-depth"
        )
        lines = [
            f"# {report_title} predictor report for {retriever_name}",
            "",
            "## Threshold selection",
            "",
            f"- Threshold policy: {metadata['threshold_policy']['name']}",
            f"- Threshold: {metadata['threshold']:.6f}",
            f"- Selection method: {IN_TOP_MAX_DEPTH_CV_FOLDS}-fold cross-validation on train only; "
            f"kept the threshold maximizing {objective_label} (class_weight=balanced random forest)",
            f"- CV folds used: {threshold_selection.get('cv_folds', 'n/a')}",
            f"- Solvable train examples covered by CV: {threshold_selection.get('cv_solvable_train_examples', 'n/a')}",
            f"- Observed false negatives in CV: {threshold_selection.get('cv_observed_false_negatives', 'n/a')}",
            f"- CV precision: {threshold_selection.get('cv_precision', 'n/a')}",
            f"- CV recall: {threshold_selection.get('cv_recall', 'n/a')}",
            f"- CV score ({objective_label}): {threshold_selection.get('cv_score', 'n/a')}",
            "",
            "## Dev evaluation (fixed threshold, no tuning on dev)",
            "",
            f"- Evaluation split: {metadata['evaluation_split']}",
            f"- Feature depths: {metadata['feature_depths']}",
            f"- Max ranking depth: {metadata['max_ranking_depth']}",
            f"- Target depth (k): {target_depth}",
            f"- Calibration: {metadata['calibration']['method']}",
            f"- F2: {metrics['f2']}",
            f"- F1: {metrics['f1']}",
            f"- Precision: {metrics['precision']}",
            f"- Recall: {metrics['recall']}",
            f"- False positives: {int(metrics['false_positives'])}",
            f"- True positives: {int(metrics['true_positives'])}",
            "",
            "## Most confident predicted in-top-max-depth queries",
            "",
        ]
        predicted_positive = output.loc[output["predicted_in_top_max_depth"]].sort_values(
            "confidence", ascending=False
        ).head(20)
        for row in predicted_positive.itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, gold_rank={row.gold_rank}, "
                f"in_top_max_depth={row.in_top_max_depth}, signals={row.strong_feature_signals}"
            )
        lines.extend(["", "## Most confident predicted-failure queries", ""])
        predicted_negative = output.loc[~output["predicted_in_top_max_depth"]].sort_values(
            "confidence", ascending=False
        ).head(20)
        for row in predicted_negative.itertuples(index=False):
            lines.append(
                f"- Query {row.query_id}: confidence={row.confidence:.4f}, gold_rank={row.gold_rank}, "
                f"in_top_max_depth={row.in_top_max_depth}, signals={row.strong_feature_signals}"
            )
        return "\n".join(lines)

    def _feature_depths(self) -> list[int]:
        return sorted({int(size) for size in self.config.candidate_sizes})

    def _resolved_target_depth(self) -> int:
        if self.target_depth is None:
            return int(self.config.max_ranking_depth)
        return int(self.target_depth)

    def _depth_label(self) -> str:
        if self.target_depth is None:
            return "In-top-max-depth"
        return f"In-top-k={self.target_depth}"

    def _scope_phases(self, retrievers: list[str]) -> list[tuple[str, list[str]]]:
        grouped: dict[str, list[str]] = {}
        for retriever_name in retrievers:
            grouped.setdefault(retriever_document_scope(retriever_name), []).append(retriever_name)
        ordered_scopes: list[str] = []
        for document_scope in self._configured_document_scopes():
            if document_scope in grouped and document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        for retriever_name in retrievers:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in ordered_scopes:
                ordered_scopes.append(document_scope)
        return [(document_scope, grouped[document_scope]) for document_scope in ordered_scopes]

    def _set_enabled_document_scopes(self, scopes: list[str]) -> None:
        scopes = list(scopes)
        current_scopes = self._configured_document_scopes()
        if current_scopes == scopes:
            return
        self.config.enabled_document_scopes = scopes
        if hasattr(self.features, "_all_rankings_top100_cache"):
            self.features._all_rankings_top100_cache.clear()

    def _configured_document_scopes(self) -> list[str]:
        return list(getattr(self.config, "enabled_document_scopes", ["title_abstract"]))

    def _ordered_retrievers(self, retrievers: list[str] | None) -> list[str]:
        if retrievers:
            return [self.retrieval.resolve_retriever_name(name) for name in retrievers]
        enabled = [self.retrieval.resolve_retriever_name(name) for name in self.retrieval.enabled_retrievers()]
        original_order = {name: index for index, name in enumerate(enabled)}
        scope_order: list[str] = []
        for retriever_name in enabled:
            document_scope = retriever_document_scope(retriever_name)
            if document_scope not in scope_order:
                scope_order.append(document_scope)
        scope_index = {document_scope: index for index, document_scope in enumerate(scope_order)}
        return sorted(
            enabled,
            key=lambda name: (
                scope_index[retriever_document_scope(name)],
                self._retriever_group(name),
                self._semantic_priority(name),
                original_order[name],
            ),
        )

    @staticmethod
    def _retriever_group(retriever_name: str) -> int:
        base_name = base_retriever_name(retriever_name)
        if base_name == "bm25":
            return 0
        if base_name in LEXICAL_RETRIEVERS:
            return 2
        return 1

    @staticmethod
    def _semantic_priority(retriever_name: str) -> int:
        priority_by_base_name = {
            "qwen3_embedding_0_6b": 0,
            "bge_m3": 1,
            "bge_large_en_v1_5": 2,
        }
        return priority_by_base_name.get(base_retriever_name(retriever_name), 100)


class QppPredictorRunner:
    predictor_name = "qpp"
    predictor_label = "QPP"
    compute_signal_summaries = True
    metadata_version = QPP_METADATA_VERSION
    model_metadata_version = QPP_MODEL_METADATA_VERSION

    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        features: FeatureBuilder,
        retrieval: RetrievalManager,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.views = PredictorFeatureViews(store, console, features, retrieval)

    @property
    def qpp_config(self) -> Any:
        return self.config.predictors.qpp

    @property
    def mapping_label(self) -> str:
        return self.qpp_config.mapping

    def run(self, language: str, retrievers: list[str] | None = None) -> dict[str, dict[str, float]]:
        summary = {}
        selected = [self.retrieval.resolve_retriever_name(name) for name in (retrievers or self.retrieval.enabled_retrievers())]
        modes = _normalize_predictor_modes(self.config)
        local_selection_metrics: dict[str, dict[str, float] | None] = {}
        for retriever_name in selected:
            metrics, selection_metrics = self._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
            summary[retriever_name] = metrics
            local_selection_metrics[retriever_name] = selection_metrics
        if PREDICTOR_MODE_ALL_MEANINGFUL in modes:
            winners = _select_best_variants(selected, local_selection_metrics)
            if winners:
                winner_text = ", ".join(
                    f"{base_retriever_name(name)} -> {name}"
                    for name in winners
                )
                self.console.log(f"Selected local winners for {self.predictor_label} {PREDICTOR_MODE_ALL_MEANINGFUL}: {winner_text}")
            for retriever_name in winners:
                metrics, _ = self._run_retriever(
                    language,
                    retriever_name,
                    PREDICTOR_MODE_ALL_MEANINGFUL,
                    source_retrievers=winners,
                )
                summary[_display_retriever_name(retriever_name, PREDICTOR_MODE_ALL_MEANINGFUL)] = metrics
        return summary

    def _run_retriever(
        self,
        language: str,
        retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None = None,
        failure_feature_columns: list[str] | None = None,
        score_feature_columns: list[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, float] | None]:
        display_name = _display_retriever_name(retriever_name, mode)
        self.console.section(f"{self.predictor_label} predictor for {display_name}")
        qpp_config = self.qpp_config
        cached_metrics, cached_selection_metrics, cache_reason = _cached_qpp_cache_status(
            self.store,
            language,
            retriever_name,
            qpp_config.score_source,
            self.mapping_label,
            qpp_config.minimum_size,
            qpp_config.maximum_size,
            mode=mode,
            predictor_name=self.predictor_name,
            metadata_version=self.metadata_version,
            model_metadata_version=self.model_metadata_version,
        )
        if cached_metrics is not None:
            self.console.log(f"Using cached {self.predictor_label} outputs for {display_name}")
            return cached_metrics, cached_selection_metrics
        if cache_reason:
            self.console.log(f"Rebuilding {self.predictor_label} outputs for {display_name}: {cache_reason}")
        train_key = f"{language}_train"
        dev_key = f"{language}_dev"
        self.console.log(f"Preparing {self.predictor_label} feature context for {display_name}")
        train_frame = self._split_frame(train_key, retriever_name, mode, source_retrievers)
        dev_frame = self._split_frame(dev_key, retriever_name, mode, source_retrievers)
        output, dev_metrics = self._fit_and_predict(
            language,
            retriever_name,
            display_name,
            train_frame,
            dev_frame,
            mode,
            source_retrievers=source_retrievers,
            failure_feature_columns=failure_feature_columns,
            score_feature_columns=score_feature_columns,
        )
        return _selection_metrics(output), dev_metrics

    def _split_frame(
        self,
        split_key: str,
        retriever_name: str,
        mode: str,
        source_retrievers: list[str] | None,
    ) -> pd.DataFrame:
        return self.views.build(split_key, retriever_name, mode, source_retrievers=source_retrievers)[self.config.max_ranking_depth]

    def _fit_and_predict(
        self,
        language: str,
        retriever_name: str,
        display_name: str,
        train_frame: pd.DataFrame,
        dev_frame: pd.DataFrame,
        mode: str,
        source_retrievers: list[str] | None = None,
        failure_feature_columns: list[str] | None = None,
        score_feature_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        output, threshold, threshold_metrics, calibration_info, _failure_importances, _score_importances = self._fit_and_predict_frame(
            train_frame,
            dev_frame,
            display_name,
            language=language,
            retriever_name=retriever_name,
            mode=mode,
            persist_model=True,
            source_retrievers=source_retrievers,
            evaluation_split="dev",
            failure_feature_columns=failure_feature_columns,
            score_feature_columns=score_feature_columns,
        )
        dev_metrics = _selection_metrics(output)
        prediction_paths = _prediction_output_paths(self.store, self.predictor_name, language, retriever_name, mode=mode)
        prediction_paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
        prediction_paths["report"].parent.mkdir(parents=True, exist_ok=True)
        self.console.log(f"Writing {self.predictor_label} outputs for {display_name}")
        output.to_parquet(prediction_paths["prediction"], index=False)
        metadata = {
            "metadata_version": self.metadata_version,
            "predictor_feature_mode": mode,
            "target_retriever": retriever_name,
            "source_retrievers": list(source_retrievers or [retriever_name]),
            "input_compatibility": {
                "feature_dependency_version": PREDICTOR_FEATURE_DEPENDENCY_VERSION,
                "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
            },
            "training_split": "train",
            "evaluation_split": "dev",
            "dev_metrics": dev_metrics,
            "feature_depth": int(self.config.max_ranking_depth),
            "score_source": self.qpp_config.score_source,
            "mapping": self.mapping_label,
            "minimum_size": int(self.qpp_config.minimum_size),
            "maximum_size": int(self.qpp_config.maximum_size),
            "failure_threshold": threshold,
            "failure_threshold_policy": _qpp_failure_threshold_policy(),
            "failure_threshold_metrics": threshold_metrics,
            "calibration": calibration_info,
        }
        self.store.write_json(prediction_paths["metadata"], metadata)
        prediction_paths["report"].write_text(
            _selection_prediction_report(
                self.predictor_name,
                display_name,
                metadata,
                output.sort_values("confidence", ascending=False).head(20),
                output.sort_values("confidence", ascending=True).head(20),
            ),
            encoding="utf-8",
        )
        return output, dev_metrics

    def _fit_and_predict_frame(
        self,
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        display_name: str,
        language: str | None = None,
        retriever_name: str | None = None,
        mode: str = PREDICTOR_MODE_LOCAL,
        persist_model: bool = False,
        source_retrievers: list[str] | None = None,
        evaluation_split: str = "dev",
        failure_feature_columns: list[str] | None = None,
        score_feature_columns: list[str] | None = None,
        precomputed_score_context: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, float, dict[str, float], dict[str, Any], pd.DataFrame, pd.DataFrame]:
        feature_columns = _feature_columns(training_frame)
        failure_columns = failure_feature_columns or feature_columns
        failure_target = 1 - training_frame["retrieval_failure_within_max_depth"].astype(int)
        self.console.log(f"Training {self.predictor_label} failure model for {display_name}")
        failure_model, failure_importances, calibration_info = _fit_calibrated_classifier(
            console=self.console,
            train_frame=training_frame,
            feature_columns=failure_columns,
            target=failure_target,
            base_model_name=self.qpp_config.failure_model,
            seed=self.config.runtime.random_seed,
            purpose=f"{self.predictor_label} success probabilities for {display_name}",
        )
        score_context = precomputed_score_context or self._fit_score_context(
            training_frame, evaluation_frame, display_name, score_feature_columns=score_feature_columns
        )
        score_source = self.qpp_config.score_source
        score_pipeline = score_context["score_pipeline"]
        score_importances = score_context["score_importances"]
        train_min = score_context["train_min"]
        train_max = score_context["train_max"]
        normalized_scores = score_context["normalized_scores"]
        selected_sizes = score_context["selected_sizes"]
        self.console.log(f"Predicting {self.predictor_label} {evaluation_split} split for {display_name}")
        hit_probabilities = _positive_class_probabilities(failure_model, evaluation_frame)
        if evaluation_split == "dev":
            failure_threshold, threshold_metrics = self._select_failure_threshold(
                hit_probabilities,
                evaluation_frame,
                selected_sizes,
            )
            self.console.log(
                f"Selected {self.predictor_label} failure threshold {failure_threshold:.6f} for {display_name} "
                f"(failure_f1={threshold_metrics['f1_failure']:.4f}, "
                f"failure_precision={threshold_metrics['precision']:.4f}, "
                f"failure_recall={threshold_metrics['recall']:.4f}, "
                f"balanced_accuracy={threshold_metrics['balanced_accuracy']:.4f})"
            )
        else:
            failure_threshold = 0.5
            threshold_metrics = {
                "f1_failure": math.nan,
                "f2_failure": math.nan,
                "balanced_accuracy": math.nan,
                **_binary_classification_summary(np.zeros(len(evaluation_frame), dtype=bool), np.zeros(len(evaluation_frame), dtype=bool)),
                **_default_metrics(),
            }
        output = self._prediction_frame(
            evaluation_frame,
            hit_probabilities,
            normalized_scores,
            selected_sizes,
            failure_threshold,
        )
        output["strong_feature_signals"] = [
            _local_signal_summary(
                evaluation_frame.iloc[index],
                training_frame,
                failure_importances if bool(output.iloc[index]["predicted_failure"]) else score_importances,
            )
            if self.compute_signal_summaries
            else ""
            for index in range(len(evaluation_frame))
        ]
        if persist_model and retriever_name is not None:
            if language is None:
                raise ValueError(f"language is required to persist {self.predictor_label} predictor artifacts")
            model_root = _prediction_model_root(self.store, self.predictor_name, retriever_name, mode=mode)
            joblib.dump(failure_model, model_root / "failure_model.joblib")
            failure_importances.to_parquet(model_root / "failure_importances.parquet", index=False)
            if score_pipeline is not None:
                joblib.dump(score_pipeline, model_root / "score_model.joblib")
            score_importances.to_parquet(model_root / "score_importances.parquet", index=False)
            feature_depth = int(training_frame["candidate_size"].iloc[0]) if "candidate_size" in training_frame.columns and not training_frame.empty else int(self.config.max_ranking_depth)
            self.store.write_json(
                model_root / "manifest.json",
                {
                    "model_metadata_version": self.model_metadata_version,
                    "predictor_metadata_version": self.metadata_version,
                    "predictor_feature_mode": mode,
                    "target_retriever": retriever_name,
                    "source_retrievers": list(source_retrievers or [retriever_name]),
                    "score_source": score_source,
                    "mapping": self.mapping_label,
                    "minimum_size": int(self.qpp_config.minimum_size),
                    "maximum_size": int(self.qpp_config.maximum_size),
                    "feature_depth": feature_depth,
                    "train_min": train_min,
                    "train_max": train_max,
                    "failure_threshold": failure_threshold,
                    "failure_threshold_policy": _qpp_failure_threshold_policy(),
                    "failure_threshold_metrics": threshold_metrics,
                    "training_split": "train",
                    "evaluation_split": evaluation_split,
                    "language": language,
                    "calibration": calibration_info,
                    "input_compatibility": {
                        "feature_dependency_version": PREDICTOR_FEATURE_DEPENDENCY_VERSION,
                        "query_signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
                    },
                },
            )
        return output, float(failure_threshold), threshold_metrics, calibration_info, failure_importances, score_importances

    @staticmethod
    def _prediction_frame(
        evaluation_frame: pd.DataFrame,
        success_probabilities: np.ndarray,
        normalized_scores: np.ndarray,
        mapped_sizes: list[int] | np.ndarray,
        failure_threshold: float,
    ) -> pd.DataFrame:
        output = evaluation_frame[["query_id", "document_id", "gold_rank", "smallest_sufficient_size"]].copy()
        output["confidence"] = np.asarray(success_probabilities, dtype=np.float64)
        output["predicted_failure"] = output["confidence"] < float(failure_threshold)
        output["normalized_qpp_score"] = np.asarray(normalized_scores, dtype=np.float64)
        output["selected_size"] = [
            math.nan if failure else float(size)
            for failure, size in zip(output["predicted_failure"].tolist(), mapped_sizes, strict=True)
        ]
        return output

    @staticmethod
    def _select_failure_threshold(
        success_probabilities: np.ndarray,
        evaluation_frame: pd.DataFrame,
        mapped_sizes: list[int] | np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        values = np.asarray(success_probabilities, dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.5, {
                "f1_failure": math.nan,
                "f2_failure": math.nan,
                "balanced_accuracy": math.nan,
                **_binary_classification_summary(
                    np.zeros(len(evaluation_frame), dtype=bool),
                    np.zeros(len(evaluation_frame), dtype=bool),
                ),
                **_default_metrics(),
            }
        thresholds = sorted({0.0, 1.0, *[float(value) for value in finite_values]})
        actual_failure = evaluation_frame["retrieval_failure_within_max_depth"].astype(bool).to_numpy(dtype=bool)
        candidates: list[tuple[tuple[float, float, float, float, float], float, dict[str, float]]] = []
        for threshold in thresholds:
            output = QppPredictorRunner._prediction_frame(
                evaluation_frame,
                values,
                np.zeros(len(values), dtype=np.float64),
                mapped_sizes,
                threshold,
            )
            summary = _binary_classification_summary(
                actual_failure,
                output["predicted_failure"].to_numpy(dtype=bool),
            )
            precision = float(summary["precision"])
            recall = float(summary["recall"])
            f1_failure = _fbeta_score(precision, recall, beta=1.0)
            f2_failure = _fbeta_score(precision, recall, beta=2.0)
            tn = float(summary["true_negatives"])
            fp = float(summary["false_positives"])
            specificity = float(tn / (tn + fp)) if (tn + fp) else math.nan
            balanced_accuracy = float((recall + specificity) / 2.0) if np.isfinite(recall) and np.isfinite(specificity) else math.nan
            selection_metrics = _selection_metrics(output)
            mean_selected_size = float(selection_metrics["mean_selected_size"])
            effective_mean_selected_size = 0.0 if math.isnan(mean_selected_size) else mean_selected_size
            metrics = {
                **summary,
                "f1_failure": f1_failure,
                "f2_failure": f2_failure,
                "balanced_accuracy": balanced_accuracy,
                **selection_metrics,
            }
            key = (
                _safe_sort_float(f1_failure),
                _safe_sort_float(balanced_accuracy),
                _safe_sort_float(selection_metrics["candidate_recall"]),
                -effective_mean_selected_size,
                -float(threshold),
            )
            candidates.append((key, float(threshold), metrics))
        _, best_threshold, best_metrics = max(candidates, key=lambda item: item[0])
        return best_threshold, best_metrics

    @staticmethod
    def _normalize_score(value: float, minimum: float, maximum: float) -> float:
        return _normalize_qpp_score(value, minimum, maximum)

    def _map_score(self, normalized_score: float) -> int:
        qpp = self.qpp_config
        return _map_qpp_score(
            normalized_score,
            qpp.mapping,
            int(qpp.minimum_size),
            int(qpp.maximum_size),
        )

    def _map_scores(self, normalized_scores: np.ndarray) -> list[int]:
        return [self._map_score(float(value)) for value in normalized_scores]

    def _fit_mapping(self, training_frame: pd.DataFrame, training_normalized_scores: np.ndarray) -> None:
        return None

    def _fit_score_context(
        self,
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        display_name: str,
        score_feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        feature_columns = _feature_columns(training_frame)
        score_source = self.qpp_config.score_source
        if score_source == "regression_full":
            score_columns = score_feature_columns or feature_columns
            self.console.log(f"Training {self.predictor_label} regression model for {display_name}")
            score_pipeline = _fit_pipeline(
                _make_regressor(self.qpp_config.regressor, self.config.runtime.random_seed),
                score_columns,
                training_frame,
                training_frame["rr_at_max_depth"],
            )
            score_importances = _feature_importances(score_pipeline, score_columns)
            training_scores = score_pipeline.predict(training_frame)
            predicted_scores = score_pipeline.predict(evaluation_frame)
        else:
            score_pipeline = None
            score_importances = pd.DataFrame({"feature": [score_source], "importance": [1.0]})
            training_scores = training_frame[score_source].to_numpy(dtype=np.float64)
            predicted_scores = evaluation_frame[score_source].to_numpy(dtype=np.float64)
        train_min = float(np.nanmin(training_scores))
        train_max = float(np.nanmax(training_scores))
        training_normalized_scores = np.array(
            [self._normalize_score(value, train_min, train_max) for value in training_scores], dtype=np.float64
        )
        self._fit_mapping(training_frame, training_normalized_scores)
        normalized_scores = np.array(
            [self._normalize_score(value, train_min, train_max) for value in predicted_scores], dtype=np.float64
        )
        selected_sizes = self._map_scores(normalized_scores)
        return {
            "score_pipeline": score_pipeline,
            "score_importances": score_importances,
            "train_min": train_min,
            "train_max": train_max,
            "normalized_scores": normalized_scores,
            "selected_sizes": selected_sizes,
        }


class TunedQppPredictorRunner(QppPredictorRunner):
    predictor_name = "tuned_qpp"
    predictor_label = "tuned QPP"
    metadata_version = TUNED_QPP_METADATA_VERSION
    model_metadata_version = TUNED_QPP_MODEL_METADATA_VERSION

    @property
    def qpp_config(self) -> Any:
        return self.config.predictors.tuned_qpp

    @property
    def mapping_label(self) -> str:
        return "isotonic"

    def _fit_mapping(self, training_frame: pd.DataFrame, training_normalized_scores: np.ndarray) -> None:
        non_failure_mask = ~training_frame["retrieval_failure_within_max_depth"].astype(bool).to_numpy()
        lower = int(self.qpp_config.minimum_size)
        upper = int(self.qpp_config.maximum_size)
        x = training_normalized_scores[non_failure_mask]
        y = np.clip(
            training_frame.loc[non_failure_mask, "gold_rank"].astype(float).to_numpy(),
            lower,
            upper,
        )
        calibrator = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=lower, y_max=upper)
        calibrator.fit(x, y)
        self._isotonic_calibrator = calibrator

    def _map_score(self, normalized_score: float) -> int:
        qpp = self.qpp_config
        lower = int(qpp.minimum_size)
        upper = int(qpp.maximum_size)
        predicted = float(self._isotonic_calibrator.predict(np.asarray([normalized_score]))[0])
        return int(min(max(math.ceil(predicted - 1e-12), lower), upper))

    def _map_scores(self, normalized_scores: np.ndarray) -> list[int]:
        qpp = self.qpp_config
        lower = int(qpp.minimum_size)
        upper = int(qpp.maximum_size)
        predicted = self._isotonic_calibrator.predict(np.asarray(normalized_scores, dtype=np.float64))
        return [int(min(max(math.ceil(value - 1e-12), lower), upper)) for value in predicted]
