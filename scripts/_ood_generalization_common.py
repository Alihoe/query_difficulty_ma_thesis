# scripts/_ood_generalization_common.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from query_difficulty.analysis import reciprocal_rank_at_k
from query_difficulty.config import _resolve_config_path
from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
    _LightweightFeaturePlan,
)
from query_difficulty.modeling import (
    PREDICTOR_MODE_LOCAL,
    CalibratedOrdinalPredictorRunner,
    InTopMaxDepthPredictorRunner,
    PredictorFeatureViews,
    _hit_at_1_target,
    _sha256_path,
    _monotone_cumulative_probabilities,
    _normalize_qpp_score,
    _positive_class_probabilities,
)
from query_difficulty.multistep import (
    ConditionMetrics,
    IN_TOP_K_SUPPORTED_DEPTHS,
    MultiStepPaths,
    SELECTOR_LABELS,
    _combined_selector_depth,
    _condition_name,
    _condition_rule_description,
    _policy_comparison_combinations,
    build_top_k_reranker,
    default_jina_compute_score_reranker_spec,
)
from query_difficulty.runtime import build_runtime

CLEF_CONFIG_PATH = "config/default.yaml"
FEATURE_MANIFEST_PATH = "config/lightweight_hit_at_1_features.json"
TARGET_RETRIEVER = "qwen3_embedding_0_6b"
TOP_K = 30
METRIC_K = 5
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"

HIT1_PREDICTOR_NAME = "lightweight/hit_at_1"
ORDINAL_PREDICTOR_NAME = "lightweight/calibrated_ordinal"
MAX_DEPTH_PREDICTOR_NAME = "lightweight/in_top_max_depth"
IN_TOP_K_PREDICTOR_NAME_TEMPLATE = "lightweight/in_top_k{depth}"
IN_TOP_K_DEPTHS = tuple(depth for depth in IN_TOP_K_SUPPORTED_DEPTHS if depth <= TOP_K)
QPP_PREDICTOR_NAME = "lightweight/tuned_qpp"


def build_lightweight_feature_builder(runtime: Any) -> LightweightHitAtOnePredictorFeatureBuilder:
    return LightweightHitAtOnePredictorFeatureBuilder(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features.bundle,
        runtime.retrieval,
        runtime.clustering,
        runtime.text_services,
        manifest_path=FEATURE_MANIFEST_PATH,
        consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
    )


def build_clef_reference_pool(clef_runtime: Any, qualifying_features: frozenset[str]) -> Any:
    clef_features = build_lightweight_feature_builder(clef_runtime)
    resolved = clef_features.retrieval.resolve_retriever_name(TARGET_RETRIEVER)
    clef_features._plan_cache[resolved] = _LightweightFeaturePlan(qualifying_features, is_lexical=False)
    pool = clef_features._semantic_reference_pool("en_test", TARGET_RETRIEVER)
    return pool, clef_features


def build_family_feature_builder(runtime: Any, clef_runtime: Any, qualifying_features: frozenset[str]) -> LightweightHitAtOnePredictorFeatureBuilder:
    reference_pool, clef_features = build_clef_reference_pool(clef_runtime, qualifying_features)
    resolved_clef_retriever = clef_features.retrieval.resolve_retriever_name(TARGET_RETRIEVER)
    threshold_bundle = clef_features._semantic_feature_thresholds(resolved_clef_retriever)
    features_builder = build_lightweight_feature_builder(runtime)
    resolved_retriever = features_builder.retrieval.resolve_retriever_name(TARGET_RETRIEVER)
    features_builder._semantic_threshold_bundle_cache[resolved_retriever] = threshold_bundle
    features_builder._plan_cache[resolved_retriever] = _LightweightFeaturePlan(qualifying_features, is_lexical=False)
    features_builder._semantic_reference_pool_cache[("en_test", resolved_retriever)] = reference_pool
    return features_builder


def score_frozen_hit1_model(
    console: Any,
    features_builder: LightweightHitAtOnePredictorFeatureBuilder,
    retrieval: Any,
    store_root: Path,
    language: str,
) -> tuple[pd.DataFrame, frozenset[str]]:
    model_path = store_root / "models" / HIT1_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib"
    model_payload = joblib.load(model_path)
    qualifying_features = frozenset(model_payload["qualifying_features"])
    console.log(f"[hit1] qualifying_features: {sorted(qualifying_features)}")
    feature_depth = int(model_payload["feature_depth"])
    threshold = float(model_payload["threshold"])
    model = model_payload["pipeline"]
    source_retrievers = [str(name) for name in model_payload.get("source_retrievers") or [TARGET_RETRIEVER]]
    views = PredictorFeatureViews(features_builder.store, console, features_builder, retrieval)
    test_frame = views.build(
        f"{language}_test", TARGET_RETRIEVER, PREDICTOR_MODE_LOCAL,
        source_retrievers=source_retrievers, candidate_sizes=[feature_depth],
    )[feature_depth]
    probabilities = _positive_class_probabilities(model, test_frame)
    output = test_frame[["query_id", "document_id", "gold_rank"]].copy()
    output["hit_at_1"] = _hit_at_1_target(test_frame).astype(bool)
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output["predicted_hit_at_1"] = output["confidence"] >= threshold
    return output, qualifying_features


def score_frozen_calibrated_ordinal_model(
    console: Any,
    features_builder: LightweightHitAtOnePredictorFeatureBuilder,
    store_root: Path,
    language: str,
) -> tuple[pd.DataFrame, frozenset[str]]:
    model_dir = store_root / "models" / ORDINAL_PREDICTOR_NAME / TARGET_RETRIEVER
    manifest = features_builder.store.read_json(model_dir / "manifest.json")
    candidate_sizes = [int(depth) for depth in manifest["candidate_sizes"]]
    solvability_model = joblib.load(model_dir / "solvability_model.joblib")
    stage_models = {depth: joblib.load(model_dir / f"depth_{depth:04d}.joblib") for depth in candidate_sizes}
    qualifying_features = frozenset(solvability_model.feature_names_in_.tolist()) - {
        "candidate_size", "document_id", "gold_in_top_k", "gold_rank", "language", "query_id",
        "retrieval_failure_within_max_depth", "retriever_name", "rr_at_max_depth",
        "smallest_sufficient_size", "split_name",
    }
    console.log(f"[ordinal] qualifying_features: {sorted(qualifying_features)}")
    test_frames = features_builder.build_features(f"{language}_test", TARGET_RETRIEVER, candidate_sizes=candidate_sizes)
    reference = CalibratedOrdinalPredictorRunner._reference_output_frame(test_frames, candidate_sizes)
    solvability_probabilities = _positive_class_probabilities(solvability_model, test_frames[max(candidate_sizes)])
    stage_probabilities = _monotone_cumulative_probabilities(
        {depth: np.asarray(_positive_class_probabilities(stage_models[depth], test_frames[depth]), dtype=np.float64) for depth in candidate_sizes}
    )
    output = CalibratedOrdinalPredictorRunner._predict_from_probabilities(
        reference, solvability_probabilities, stage_probabilities, float(manifest["solvability_threshold"]), include_traces=False
    )
    return output[["query_id", "document_id", "selected_size", "predicted_failure"]], qualifying_features


def score_frozen_depth_gate_model(
    console: Any,
    features_builder: LightweightHitAtOnePredictorFeatureBuilder,
    store_root: Path,
    language: str,
    predictor_name: str,
    predicted_column_name: str,
    label: str,
) -> tuple[pd.DataFrame, frozenset[str]]:
    model_path = store_root / "models" / predictor_name / TARGET_RETRIEVER / "model.joblib"
    model_payload = joblib.load(model_path)
    feature_depths = [int(depth) for depth in model_payload["feature_depths"]]
    pipeline = model_payload["pipeline"]
    threshold = float(model_payload["threshold"])
    qualifying_features = frozenset(
        column.split("::", 1)[1] if "::" in column else column
        for column in pipeline.feature_names_in_.tolist()
    ) - {
        "candidate_size", "document_id", "gold_in_top_k", "gold_rank", "language", "query_id",
        "retrieval_failure_within_max_depth", "retriever_name", "rr_at_max_depth",
        "smallest_sufficient_size", "split_name",
    }
    console.log(f"[{label}] qualifying_features: {sorted(qualifying_features)} at depths {feature_depths}")
    frames_by_depth = features_builder.build_features(f"{language}_test", TARGET_RETRIEVER, candidate_sizes=feature_depths)
    merged = InTopMaxDepthPredictorRunner._merge_multi_depth_frames(frames_by_depth)
    probabilities = _positive_class_probabilities(pipeline, merged)
    output = merged[["query_id", "document_id", "gold_rank"]].copy()
    output["confidence"] = np.asarray(probabilities, dtype=np.float64)
    output[predicted_column_name] = output["confidence"] >= threshold
    return output, qualifying_features


def score_frozen_tuned_qpp_model(
    console: Any,
    features_builder: LightweightHitAtOnePredictorFeatureBuilder,
    clef_runtime: Any,
    store_root: Path,
    language: str,
) -> tuple[pd.DataFrame, frozenset[str]]:
    model_dir = store_root / "models" / QPP_PREDICTOR_NAME / TARGET_RETRIEVER
    manifest = features_builder.store.read_json(model_dir / "manifest.json")
    failure_model = joblib.load(model_dir / "failure_model.joblib")
    score_model = joblib.load(model_dir / "score_model.joblib")
    failure_qualifying = _column_transformer_columns(failure_model)
    score_qualifying = _column_transformer_columns(score_model)
    qualifying_features = failure_qualifying | score_qualifying
    console.log(f"[qpp] failure qualifying_features: {sorted(failure_qualifying)}")
    console.log(f"[qpp] score qualifying_features: {sorted(score_qualifying)}")

    feature_depth = int(manifest["feature_depth"])
    minimum_size = int(manifest["minimum_size"])
    maximum_size = int(manifest["maximum_size"])
    train_min = float(manifest["train_min"])
    train_max = float(manifest["train_max"])
    failure_threshold = float(manifest["failure_threshold"])

    source_retrievers = [str(name) for name in manifest.get("source_retrievers") or [TARGET_RETRIEVER]]
    console.log(
        "[qpp] Refitting the isotonic score-to-size calibrator on CLEF en_train exactly as the in-domain "
        "lightweight tuned QPP loader does; no target-dataset data is used"
    )
    clef_views = PredictorFeatureViews(clef_runtime.store, console, clef_runtime.features, clef_runtime.retrieval)
    clef_calibration_frame = clef_views.build(
        f"{language}_train", TARGET_RETRIEVER, PREDICTOR_MODE_LOCAL,
        source_retrievers=source_retrievers, candidate_sizes=[feature_depth],
    )[feature_depth]
    console.log(f"[qpp] CLEF en_train calibration frame: {len(clef_calibration_frame)} queries")
    calibration_scores = score_model.predict(clef_calibration_frame)
    calibration_normalized_scores = np.asarray(
        [_normalize_qpp_score(value, train_min, train_max) for value in calibration_scores], dtype=np.float64
    )
    non_failure_mask = ~clef_calibration_frame["retrieval_failure_within_max_depth"].astype(bool).to_numpy()
    calibrator = IsotonicRegression(increasing=False, out_of_bounds="clip", y_min=minimum_size, y_max=maximum_size)
    calibrator.fit(
        calibration_normalized_scores[non_failure_mask],
        np.clip(clef_calibration_frame.loc[non_failure_mask, "gold_rank"].astype(float).to_numpy(), minimum_size, maximum_size),
    )

    test_frame = features_builder.build_features(f"{language}_test", TARGET_RETRIEVER, candidate_sizes=[feature_depth])[feature_depth]
    test_scores = score_model.predict(test_frame)
    test_normalized_scores = np.asarray(
        [_normalize_qpp_score(value, train_min, train_max) for value in test_scores], dtype=np.float64
    )
    predicted_sizes = calibrator.predict(test_normalized_scores)
    selected_sizes = [int(min(max(math.ceil(value - 1e-12), minimum_size), maximum_size)) for value in predicted_sizes]
    hit_probabilities = _positive_class_probabilities(failure_model, test_frame)
    output = test_frame[["query_id", "document_id", "gold_rank"]].copy()
    output["confidence"] = np.asarray(hit_probabilities, dtype=np.float64)
    output["predicted_failure"] = output["confidence"] < failure_threshold
    output["selected_size"] = [
        math.nan if failure else float(size)
        for failure, size in zip(output["predicted_failure"].tolist(), selected_sizes, strict=True)
    ]
    return output[["query_id", "document_id", "selected_size", "predicted_failure"]], qualifying_features


def _column_transformer_columns(pipeline: Any) -> frozenset[str]:
    estimator = pipeline
    if hasattr(estimator, "calibrated_classifiers_"):
        estimator = estimator.calibrated_classifiers_[0].estimator
    transformer = estimator.named_steps["transformer"]
    columns: set[str] = set()
    for _name, _trans, cols in transformer.transformers:
        columns.update(cols)
    return frozenset(columns)


def _rerank_prefix(base_docs: list[int], score_group: pd.DataFrame, top_k: int, rerank_depth: int) -> list[int]:
    depth = max(0, min(int(rerank_depth), top_k, len(base_docs)))
    if depth == 0:
        return base_docs
    reranked_head = (
        score_group.loc[score_group["rank"] <= depth]
        .sort_values(["reranker_score", "rank"], ascending=[False, True])["document_id"]
        .astype(int)
        .tolist()
    )
    tail_docs = base_docs[depth:]
    return reranked_head + tail_docs


def run_ood_generalization(config_path: str, dataset_label: str, gold_sets_path: str | None = None) -> dict[str, Any]:
    print(f"=== OOD generalization eval (hit1 only): {dataset_label} ===")
    runtime = build_runtime(config_path)
    clef_runtime = build_runtime(CLEF_CONFIG_PATH)
    model_root = clef_runtime.store.root_dir
    print(f"Loading frozen CLEF selector models from {model_root / 'models'}")
    console = runtime.console
    store = runtime.store

    model_path = model_root / "models" / HIT1_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib"
    qualifying_features = frozenset(joblib.load(model_path)["qualifying_features"])
    print(f"Frozen model's qualifying_features: {sorted(qualifying_features)}")
    features_builder = build_family_feature_builder(runtime, clef_runtime, qualifying_features)

    print(f"Ensuring bm25 and {TARGET_RETRIEVER} rankings for {dataset_label} en_test")
    runtime.retrieval.ensure_rankings("en_test", "bm25")
    base_ranking = runtime.retrieval.ensure_rankings("en_test", TARGET_RETRIEVER)
    print(f"Rankings ready for {dataset_label}: {base_ranking['query_id'].nunique()} queries")

    print(f"Running qwen3 + Jina reranker (top_k={TOP_K}) for supplementary pipeline context metrics")
    reranker_spec = default_jina_compute_score_reranker_spec(RERANKER_MODEL_NAME, quantize=True)
    paths = MultiStepPaths(store, "ood_generalization", "en", TARGET_RETRIEVER, reranker_spec, TOP_K)
    reranker_runner = build_top_k_reranker(console, store, runtime.retrieval, paths, reranker_spec, TOP_K)
    reranked_scores = reranker_runner.score_test_split(TARGET_RETRIEVER, "en")
    print(f"Reranked {reranked_scores['query_id'].nunique()} queries' top-{TOP_K} candidates")

    print("Scoring the frozen lightweight qwen3 Hit@1 classifier (no retraining)")
    predictions, _ = score_frozen_hit1_model(console, features_builder, runtime.retrieval, model_root, "en")
    print(f"Scored {len(predictions)} test queries")

    gold_sets: dict[int, list[int]] | None = None
    if gold_sets_path is not None:
        gold_sets = {int(key): value for key, value in store.read_json(_resolve_config_path(gold_sets_path)).items()}
        print(
            f"Loaded full gold sets for {len(gold_sets)} claims from {gold_sets_path}; "
            "recomputing actual Hit@1 as set membership against qwen3's own top-1 (pre-rerank)"
        )
        top1_by_query = (
            base_ranking.loc[base_ranking["rank"] == 1].set_index("query_id")["document_id"].astype(int).to_dict()
        )
        predictions = predictions.copy()
        predictions["hit_at_1"] = [
            int(top1_by_query.get(int(query_id), -1)) in gold_sets.get(int(query_id), [])
            for query_id in predictions["query_id"]
        ]

    predicted = predictions["predicted_hit_at_1"].astype(bool)
    actual = predictions["hit_at_1"].astype(bool)
    true_positives = int((predicted & actual).sum())
    false_positives = int((predicted & ~actual).sum())
    true_negatives = int((~predicted & ~actual).sum())
    false_negatives = int((~predicted & actual).sum())
    accuracy = float((predicted == actual).mean())
    print(
        f"Predicted-vs-actual Hit@1 confusion for {dataset_label}: "
        f"TP={true_positives} FP={false_positives} TN={true_negatives} FN={false_negatives} accuracy={accuracy:.4f}"
    )

    rerank_by_query: dict[int, list[int]] = {}
    for query_id, group in reranked_scores.groupby("query_id"):
        ordered = group.sort_values("reranker_score", ascending=False)
        rerank_by_query[int(query_id)] = [int(doc_id) for doc_id in ordered["document_id"].tolist()]

    query_ids = predictions["query_id"].astype(int).tolist()
    if gold_sets is not None:
        rr_at_5_values = [
            max(
                (reciprocal_rank_at_k(rerank_by_query.get(query_id, []), gold_id, 5) for gold_id in gold_sets.get(query_id, [])),
                default=0.0,
            )
            for query_id in query_ids
        ]
    else:
        gold_by_query = predictions.set_index("query_id")["document_id"].astype(int).to_dict()
        rr_at_5_values = [
            reciprocal_rank_at_k(rerank_by_query.get(query_id, []), gold_by_query[query_id], 5) for query_id in query_ids
        ]
    mrr_at_5 = float(np.mean(rr_at_5_values)) if rr_at_5_values else math.nan
    print(f"Supplementary context: full best-setting pipeline (qwen3 -> jina top_{TOP_K}) MRR@5 on {dataset_label} = {mrr_at_5:.4f}")

    summary = {
        "dataset": dataset_label,
        "num_queries": int(len(predictions)),
        "confusion": {"tp": true_positives, "fp": false_positives, "tn": true_negatives, "fn": false_negatives},
        "accuracy": accuracy,
        "pipeline_mrr_at_5": mrr_at_5,
    }
    results_predictions_path = store.path("ood_generalization", "predictions.parquet")
    results_summary_path = store.path("ood_generalization", "summary.json")
    store.write_parquet(results_predictions_path, predictions)
    store.write_json(results_summary_path, summary)
    print(f"Wrote predictions to {results_predictions_path}")
    print(f"Wrote summary to {results_summary_path}")
    print(summary)
    return summary


HIT1_COMPARISON_NAMESPACE = "ood_hit1_comparison"
HIT1_COMPARISON_SCHEMA_VERSION = 1
HIT1_COMPARISON_TOP_K_VALUES = tuple(depth for depth in IN_TOP_K_SUPPORTED_DEPTHS if depth <= TOP_K)
RERANKER_SCORE_NAMESPACE = "ood_full_comparison"


def hit1_reranker_spec() -> Any:
    return default_jina_compute_score_reranker_spec(RERANKER_MODEL_NAME, quantize=True)


def hit1_comparison_inputs(artifact_dir: Path, clef_artifact_dir: Path, gold_sets_path: str | None) -> dict[str, Any]:
    spec = hit1_reranker_spec()
    score_path = (
        artifact_dir / "multistep" / RERANKER_SCORE_NAMESPACE / "en" / TARGET_RETRIEVER / spec.method / spec.key
        / f"top_{TOP_K}" / spec.cache_dir_name / f"test_top{TOP_K}_scores.parquet"
    )
    model_path = clef_artifact_dir / "models" / HIT1_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib"
    return {
        "schema_version": HIT1_COMPARISON_SCHEMA_VERSION,
        "top_k_values": list(HIT1_COMPARISON_TOP_K_VALUES),
        "metric_k": METRIC_K,
        "hit1_model_signature": _sha256_path(model_path),
        "reranker_cache_dir": spec.cache_dir_name,
        "reranker_score_signature": _sha256_path(score_path) if score_path.exists() else None,
        "gold_sets_path": gold_sets_path,
        "gold_sets_signature": _sha256_path(_resolve_config_path(gold_sets_path)) if gold_sets_path else None,
    }


def hit1_comparison_summary_path(artifact_dir: Path) -> Path:
    return artifact_dir / HIT1_COMPARISON_NAMESPACE / "summary.json"


def run_hit1_comparison(
    config_path: str,
    dataset_label: str,
    clef_runtime: Any,
    gold_sets_path: str | None = None,
) -> dict[str, Any]:
    print(f"=== OOD Hit@1 multistage comparison: {dataset_label} ===")
    runtime = build_runtime(config_path)
    model_root = clef_runtime.store.root_dir
    console = runtime.console
    store = runtime.store

    print(f"Ensuring {TARGET_RETRIEVER} rankings for {dataset_label} en_test (cached rankings are reused)")
    runtime.retrieval.ensure_rankings("en_test", "bm25")
    base_ranking = runtime.retrieval.ensure_rankings("en_test", TARGET_RETRIEVER)
    print(f"Rankings ready: {base_ranking['query_id'].nunique()} queries")

    print(f"Loading jina reranker scores for the top {TOP_K} (cached scores are reused, missing ones are computed)")
    reranker_spec = hit1_reranker_spec()
    paths = MultiStepPaths(store, RERANKER_SCORE_NAMESPACE, "en", TARGET_RETRIEVER, reranker_spec, TOP_K)
    reranker_runner = build_top_k_reranker(console, store, runtime.retrieval, paths, reranker_spec, TOP_K)
    reranked_scores = reranker_runner.score_test_split(TARGET_RETRIEVER, "en")
    print(f"Reranker scores ready for {reranked_scores['query_id'].nunique()} queries")

    print("Scoring the frozen CLEF lightweight Hit@1 predictor (lightweight feature tables are reused when cached)")
    hit1_model_path = model_root / "models" / HIT1_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib"
    qualifying_features = frozenset(joblib.load(hit1_model_path)["qualifying_features"])
    features_builder = build_family_feature_builder(runtime, clef_runtime, qualifying_features)
    hit1_output, _ = score_frozen_hit1_model(console, features_builder, runtime.retrieval, model_root, "en")
    hit1_output = hit1_output.drop_duplicates("query_id").set_index("query_id")
    print(f"Scored {len(hit1_output)} queries")

    if gold_sets_path is not None:
        gold_sets = {int(key): [int(value) for value in values] for key, values in store.read_json(_resolve_config_path(gold_sets_path)).items()}
        print(f"Loaded full gold sets for {len(gold_sets)} claims from {gold_sets_path} (set-membership evaluation)")
    else:
        gold_sets = {int(query_id): [int(row.document_id)] for query_id, row in hit1_output.iterrows()}

    ranking_groups = {
        int(query_id): group.sort_values("rank")["document_id"].astype(int).tolist()
        for query_id, group in base_ranking.loc[base_ranking["rank"] <= max(TOP_K, METRIC_K)].groupby("query_id")
    }
    score_groups = {int(query_id): group for query_id, group in reranked_scores.groupby("query_id")}

    skipped = {"no_gold": 0, "no_ranking": 0, "no_reranker_scores": 0}
    outcome_rows: list[dict[str, Any]] = []
    for query_id, row in hit1_output.iterrows():
        query_id = int(query_id)
        gold_ids = gold_sets.get(query_id) or []
        if not gold_ids:
            skipped["no_gold"] += 1
            continue
        documents = ranking_groups.get(query_id)
        if documents is None:
            skipped["no_ranking"] += 1
            continue
        score_group = score_groups.get(query_id)
        if score_group is None:
            skipped["no_reranker_scores"] += 1
            continue

        def best_rr(ranked: list[int]) -> float:
            return max(reciprocal_rank_at_k(ranked, gold_id, METRIC_K) for gold_id in gold_ids)

        rerank = not bool(row["predicted_hit_at_1"])
        record: dict[str, Any] = {
            "query_id": query_id,
            "actual_hit_at_1": bool(documents and documents[0] in gold_ids),
            "predicted_hit_at_1": bool(row["predicted_hit_at_1"]),
            "hit1_confidence": float(row["confidence"]),
            "rr_no_rerank": best_rr(documents),
        }
        for top_k in HIT1_COMPARISON_TOP_K_VALUES:
            reranked = _rerank_prefix(documents, score_group, TOP_K, top_k)
            record[f"rr_rerank_all_top{top_k}"] = best_rr(reranked)
            record[f"rr_hit1_top{top_k}"] = best_rr(reranked if rerank else documents)
            record[f"depth_hit1_top{top_k}"] = top_k if rerank else 0
        outcome_rows.append(record)
    outcomes = pd.DataFrame(outcome_rows)
    print(f"Evaluated {len(outcomes)} queries; skipped: {skipped}")

    predicted = outcomes["predicted_hit_at_1"].astype(bool)
    actual = outcomes["actual_hit_at_1"].astype(bool)
    query_count = int(len(outcomes))
    results: dict[str, Any] = {}
    for top_k in HIT1_COMPARISON_TOP_K_VALUES:
        results[str(top_k)] = {
            "no_rerank": ConditionMetrics(
                mrr_at_5=float(outcomes["rr_no_rerank"].mean()), reranked_queries=0, reranked_pairs=0
            ).to_dict(),
            "rerank_all": ConditionMetrics(
                mrr_at_5=float(outcomes[f"rr_rerank_all_top{top_k}"].mean()),
                reranked_queries=query_count,
                reranked_pairs=query_count * top_k,
            ).to_dict(),
            "hit1": ConditionMetrics(
                mrr_at_5=float(outcomes[f"rr_hit1_top{top_k}"].mean()),
                reranked_queries=int((outcomes[f"depth_hit1_top{top_k}"] > 0).sum()),
                reranked_pairs=int(outcomes[f"depth_hit1_top{top_k}"].sum()),
            ).to_dict(),
        }
    summary = {
        "dataset": dataset_label,
        "language": "en",
        "retriever_name": TARGET_RETRIEVER,
        "reranker": RERANKER_MODEL_NAME,
        "evaluated_queries": query_count,
        "skipped_queries": skipped,
        "hit1_confusion": {
            "tp": int((predicted & actual).sum()),
            "fp": int((predicted & ~actual).sum()),
            "tn": int((~predicted & ~actual).sum()),
            "fn": int((~predicted & actual).sum()),
        },
        "results_by_top_k": results,
        "inputs": hit1_comparison_inputs(store.root_dir, model_root, gold_sets_path),
    }
    output_dir = store.root_dir / HIT1_COMPARISON_NAMESPACE
    output_dir.mkdir(parents=True, exist_ok=True)
    store.write_parquet(output_dir / "outcomes.parquet", outcomes)
    store.write_json(hit1_comparison_summary_path(store.root_dir), summary)
    print(f"Wrote {output_dir / 'outcomes.parquet'} and {hit1_comparison_summary_path(store.root_dir)}")
    return summary


def run_full_selector_comparison(config_path: str, dataset_label: str, gold_sets_path: str | None = None) -> dict[str, Any]:
    print(f"=== Full multistage selector comparison (OOD): {dataset_label} ===")
    runtime = build_runtime(config_path)
    clef_runtime = build_runtime(CLEF_CONFIG_PATH)
    model_root = clef_runtime.store.root_dir
    print(f"Loading frozen CLEF selector models from {model_root / 'models'}")
    console = runtime.console
    store = runtime.store

    print(f"Ensuring bm25 and {TARGET_RETRIEVER} rankings for {dataset_label} en_test")
    runtime.retrieval.ensure_rankings("en_test", "bm25")
    base_ranking = runtime.retrieval.ensure_rankings("en_test", TARGET_RETRIEVER)
    print(f"Rankings ready for {dataset_label}: {base_ranking['query_id'].nunique()} queries")

    print(f"Running qwen3 + Jina reranker (top_k={TOP_K})")
    reranker_spec = default_jina_compute_score_reranker_spec(RERANKER_MODEL_NAME, quantize=True)
    paths = MultiStepPaths(store, "ood_full_comparison", "en", TARGET_RETRIEVER, reranker_spec, TOP_K)
    reranker_runner = build_top_k_reranker(console, store, runtime.retrieval, paths, reranker_spec, TOP_K)
    reranked_scores = reranker_runner.score_test_split(TARGET_RETRIEVER, "en")
    print(f"Reranked {reranked_scores['query_id'].nunique()} queries' top-{TOP_K} candidates")

    print("Scoring hit1 (lightweight)")
    hit1_features = build_family_feature_builder(
        runtime, clef_runtime, frozenset(joblib.load(model_root / "models" / HIT1_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib")["qualifying_features"])
    )
    hit1_output, _ = score_frozen_hit1_model(console, hit1_features, runtime.retrieval, model_root, "en")

    print("Scoring ordinal (calibrated_ordinal, lightweight)")
    ordinal_model_dir = model_root / "models" / ORDINAL_PREDICTOR_NAME / TARGET_RETRIEVER
    ordinal_model_paths = [ordinal_model_dir / "solvability_model.joblib", *sorted(ordinal_model_dir.glob("depth_*.joblib"))]
    ordinal_qualifying = frozenset().union(
        *(_column_transformer_columns(joblib.load(path)) for path in ordinal_model_paths)
    )
    print(f"Ordinal feature plan: {len(ordinal_qualifying)} features from {len(ordinal_model_paths)} heads")
    ordinal_features = build_family_feature_builder(runtime, clef_runtime, ordinal_qualifying)
    ordinal_output, _ = score_frozen_calibrated_ordinal_model(console, ordinal_features, model_root, "en")

    print("Scoring max_depth (in_top_max_depth, lightweight)")
    max_depth_payload = joblib.load(model_root / "models" / MAX_DEPTH_PREDICTOR_NAME / TARGET_RETRIEVER / "model.joblib")
    max_depth_qualifying = frozenset(
        column.split("::", 1)[1] if "::" in column else column
        for column in max_depth_payload["pipeline"].feature_names_in_.tolist()
    ) - {
        "candidate_size", "document_id", "gold_in_top_k", "gold_rank", "language", "query_id",
        "retrieval_failure_within_max_depth", "retriever_name", "rr_at_max_depth",
        "smallest_sufficient_size", "split_name",
    }
    max_depth_features = build_family_feature_builder(runtime, clef_runtime, max_depth_qualifying)
    max_depth_output, _ = score_frozen_depth_gate_model(
        console, max_depth_features, model_root, "en", MAX_DEPTH_PREDICTOR_NAME, "predicted_in_top_max_depth", "max_depth"
    )

    in_top_k_outputs: dict[int, pd.DataFrame] = {}
    for depth in IN_TOP_K_DEPTHS:
        predictor_name = IN_TOP_K_PREDICTOR_NAME_TEMPLATE.format(depth=depth)
        print(f"Scoring in_top_k (in_top_k{depth}, lightweight) from {predictor_name}")
        in_top_k_payload = joblib.load(model_root / "models" / predictor_name / TARGET_RETRIEVER / "model.joblib")
        in_top_k_qualifying = frozenset(
            column.split("::", 1)[1] if "::" in column else column
            for column in in_top_k_payload["pipeline"].feature_names_in_.tolist()
        ) - {
            "candidate_size", "document_id", "gold_in_top_k", "gold_rank", "language", "query_id",
            "retrieval_failure_within_max_depth", "retriever_name", "rr_at_max_depth",
            "smallest_sufficient_size", "split_name",
        }
        in_top_k_features = build_family_feature_builder(runtime, clef_runtime, in_top_k_qualifying)
        in_top_k_outputs[depth], _ = score_frozen_depth_gate_model(
            console, in_top_k_features, model_root, "en", predictor_name, "predicted_in_top_k", f"in_top_k{depth}"
        )
    in_top_k_output = in_top_k_outputs[TOP_K]
    in_top_k_depth_frame: pd.DataFrame | None = None
    for depth in IN_TOP_K_DEPTHS:
        frame = in_top_k_outputs[depth][["query_id", "predicted_in_top_k"]].drop_duplicates("query_id").rename(
            columns={"predicted_in_top_k": f"predicted_in_top_{depth}"}
        )
        in_top_k_depth_frame = frame if in_top_k_depth_frame is None else in_top_k_depth_frame.merge(frame, on="query_id", how="inner")
    predicted_by_depth = np.column_stack(
        [in_top_k_depth_frame[f"predicted_in_top_{depth}"].astype(bool).to_numpy() for depth in IN_TOP_K_DEPTHS]
    )
    in_top_k_depth_frame["in_top_k_depth_selected_size"] = np.where(
        predicted_by_depth.any(axis=1),
        np.asarray(IN_TOP_K_DEPTHS, dtype=np.float64)[np.argmax(predicted_by_depth, axis=1)],
        np.nan,
    )
    in_top_k_depth_frame = in_top_k_depth_frame[["query_id", "in_top_k_depth_selected_size"]]
    distribution = in_top_k_depth_frame["in_top_k_depth_selected_size"].value_counts(dropna=False).sort_index()
    print(
        "In-top-k depth distribution: "
        + ", ".join(f"{'none' if pd.isna(size) else int(size)}={int(count)}" for size, count in distribution.items())
    )

    print("Scoring qpp (tuned_qpp, lightweight)")
    qpp_qualifying = _column_transformer_columns(
        joblib.load(model_root / "models" / QPP_PREDICTOR_NAME / TARGET_RETRIEVER / "failure_model.joblib")
    ) | _column_transformer_columns(
        joblib.load(model_root / "models" / QPP_PREDICTOR_NAME / TARGET_RETRIEVER / "score_model.joblib")
    )
    qpp_features = build_family_feature_builder(runtime, clef_runtime, qpp_qualifying)
    qpp_output, _ = score_frozen_tuned_qpp_model(console, qpp_features, clef_runtime, model_root, "en")

    print("Merging selector predictions")
    hit1_frame = hit1_output[["query_id", "predicted_hit_at_1"]].drop_duplicates("query_id")
    ordinal_frame = ordinal_output[["query_id", "selected_size", "predicted_failure"]].rename(
        columns={"selected_size": "ordinal_selected_size", "predicted_failure": "ordinal_predicted_failure"}
    ).drop_duplicates("query_id")
    qpp_frame = qpp_output[["query_id", "selected_size", "predicted_failure"]].rename(
        columns={"selected_size": "qpp_selected_size", "predicted_failure": "qpp_predicted_failure"}
    ).drop_duplicates("query_id")
    in_top_k_frame = in_top_k_output[["query_id", "predicted_in_top_k"]].drop_duplicates("query_id")
    max_depth_frame = max_depth_output[["query_id", "predicted_in_top_max_depth"]].drop_duplicates("query_id")
    merged = (
        hit1_frame.merge(ordinal_frame, on="query_id", how="inner")
        .merge(qpp_frame, on="query_id", how="inner")
        .merge(in_top_k_frame, on="query_id", how="inner")
        .merge(in_top_k_depth_frame, on="query_id", how="inner")
        .merge(max_depth_frame, on="query_id", how="inner")
    )
    merged["hit1_gate_pass"] = ~merged["predicted_hit_at_1"].astype(bool)
    merged["ordinal_gate_pass"] = ~merged["ordinal_predicted_failure"].astype(bool)
    merged["qpp_gate_pass"] = ~merged["qpp_predicted_failure"].astype(bool)
    merged["in_top_k_gate_pass"] = merged["predicted_in_top_k"].astype(bool)
    merged["in_top_k_depth_gate_pass"] = merged["in_top_k_depth_selected_size"].notna()
    merged["max_depth_gate_pass"] = merged["predicted_in_top_max_depth"].astype(bool)
    print(f"Merged selector frame: {len(merged)} queries with predictions from all 5 families")

    gold_sets: dict[int, list[int]] | None = None
    if gold_sets_path is not None:
        gold_sets = {int(key): value for key, value in store.read_json(_resolve_config_path(gold_sets_path)).items()}
        print(f"Loaded full gold sets for {len(gold_sets)} claims (set-membership evaluation)")
        gold_by_query = gold_sets
    else:
        gold_by_query = {
            int(row.query_id): [int(row.document_id)] for row in hit1_output[["query_id", "document_id"]].itertuples(index=False)
        }

    selector_by_query = {
        int(row.query_id): {
            "gate_pass": {
                "hit1": bool(row.hit1_gate_pass),
                "max_depth": bool(row.max_depth_gate_pass),
                "ordinal": bool(row.ordinal_gate_pass),
                "qpp": bool(row.qpp_gate_pass),
                "in_top_k": bool(row.in_top_k_gate_pass),
                "in_top_k_depth": bool(row.in_top_k_depth_gate_pass),
            },
            "depth": {
                "hit1": None,
                "max_depth": None,
                "ordinal": None if pd.isna(row.ordinal_selected_size) else int(float(row.ordinal_selected_size)),
                "qpp": None if pd.isna(row.qpp_selected_size) else int(float(row.qpp_selected_size)),
                "in_top_k": None,
                "in_top_k_depth": (
                    None if pd.isna(row.in_top_k_depth_selected_size) else int(float(row.in_top_k_depth_selected_size))
                ),
            },
        }
        for row in merged.itertuples(index=False)
    }

    score_groups = {int(query_id): group.copy() for query_id, group in reranked_scores.groupby("query_id")}
    evaluation_depth = max(TOP_K, METRIC_K)
    ranking_top_frame = base_ranking.loc[
        base_ranking["rank"] <= evaluation_depth, ["query_id", "rank", "document_id"]
    ].sort_values(["query_id", "rank"])
    ranking_groups = {int(query_id): group for query_id, group in ranking_top_frame.groupby("query_id")}

    combinations = _policy_comparison_combinations()
    outcome_rows: list[dict[str, Any]] = []
    for query_id, selector in selector_by_query.items():
        gold_ids = gold_by_query.get(query_id)
        if not gold_ids:
            continue
        ranking_group = ranking_groups.get(query_id)
        score_group = score_groups.get(query_id)
        if ranking_group is None or score_group is None:
            continue
        no_rerank_docs = ranking_group.sort_values("rank")["document_id"].astype(int).tolist()
        rerank_all_docs = _rerank_prefix(no_rerank_docs, score_group, TOP_K, TOP_K)

        def best_rr(documents: list[int]) -> float:
            return max((reciprocal_rank_at_k(documents, gold_id, METRIC_K) for gold_id in gold_ids), default=0.0)

        row_record: dict[str, Any] = {
            "query_id": query_id,
            "rr_no_rerank": best_rr(no_rerank_docs),
            "rr_rerank_all": best_rr(rerank_all_docs),
        }
        for combo in combinations:
            name = _condition_name(combo)
            selected, depth = _combined_selector_depth(combo, selector["gate_pass"], selector["depth"], TOP_K)
            documents = _rerank_prefix(no_rerank_docs, score_group, TOP_K, depth) if selected else no_rerank_docs
            row_record[f"rr_{name}"] = best_rr(documents)
            row_record[f"depth_{name}"] = depth if selected else 0
        outcome_rows.append(row_record)

    outcomes = pd.DataFrame(outcome_rows)
    print(f"Evaluated {len(outcomes)} queries across all conditions")

    rerank_all_key = f"rerank_all_top{TOP_K}"
    policy_metrics: dict[str, Any] = {
        "no_rerank": ConditionMetrics(
            mrr_at_5=float(outcomes["rr_no_rerank"].mean()) if len(outcomes) else 0.0, reranked_queries=0, reranked_pairs=0
        ).to_dict(),
        rerank_all_key: ConditionMetrics(
            mrr_at_5=float(outcomes["rr_rerank_all"].mean()) if len(outcomes) else 0.0,
            reranked_queries=int(len(outcomes)),
            reranked_pairs=int(len(outcomes)) * TOP_K,
        ).to_dict(),
    }
    policies: dict[str, Any] = {}
    for combo in combinations:
        name = _condition_name(combo)
        depth_column = f"depth_{name}"
        policy_metrics[name] = ConditionMetrics(
            mrr_at_5=float(outcomes[f"rr_{name}"].mean()) if len(outcomes) else 0.0,
            reranked_queries=int((outcomes[depth_column] > 0).sum()) if len(outcomes) else 0,
            reranked_pairs=int(outcomes[depth_column].sum()) if len(outcomes) else 0,
        ).to_dict()
        policies[name] = {"selectors": list(combo), "rule": _condition_rule_description(combo)}

    summary = {
        "dataset": dataset_label,
        "language": "en",
        "retriever_name": TARGET_RETRIEVER,
        "top_k": TOP_K,
        "metric_k": METRIC_K,
        "evaluated_queries": int(len(outcomes)),
        "selector_labels": dict(SELECTOR_LABELS),
        "policies": policies,
        "test": policy_metrics,
    }
    results_outcomes_path = store.path("ood_full_comparison", "outcomes.parquet")
    results_summary_path = store.path("ood_full_comparison", "summary.json")
    store.write_parquet(results_outcomes_path, outcomes)
    store.write_json(results_summary_path, summary)
    print(f"Wrote outcomes to {results_outcomes_path}")
    print(f"Wrote summary to {results_summary_path}")
    print(f"No rerank MRR@5: {policy_metrics['no_rerank']['mrr_at_5']:.4f}")
    print(f"Rerank all top_{TOP_K} MRR@5: {policy_metrics[rerank_all_key]['mrr_at_5']:.4f}")
    for combo in combinations:
        name = _condition_name(combo)
        metrics = policy_metrics[name]
        print(f"{name}: MRR@5={metrics['mrr_at_5']:.4f}, reranked_queries={metrics['reranked_queries']}/{len(outcomes)}")
    return summary
