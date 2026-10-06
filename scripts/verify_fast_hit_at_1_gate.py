from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import typer

from _hit_at_1_gate_cost_common import (
    DEFAULT_LANGUAGE,
    DEFAULT_METRIC_K,
    DEFAULT_PREDICTOR_NAME,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVER,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_TOP_K,
    FEATURE_CACHE_TABLES,
    cached_top_k_scores,
    default_request_size,
    effectiveness,
    load_deployed_gate,
    load_or_create_subset,
    multistep_run_dir,
    report_dir,
    reranker_spec,
    split_workload,
    write_json,
)
from query_difficulty.dataset import tokenize
from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
    _select_qualifying,
)
from query_difficulty.hit_at_1_gate_fast import (
    FEATURE_TRAINING_RR,
    PERTURBATION_COUNT,
    bm25_top_1_positions,
    build_pairs,
    environment_metadata,
    load_jina_reranker,
    perturbed_query_embeddings,
    run_gate,
    score_pairs,
    training_query_rr,
)
from query_difficulty.runtime import build_runtime


def _compare_features(features: dict[str, np.ndarray], cache: pd.DataFrame, query_ids: np.ndarray, columns: list[str]) -> dict[str, Any]:
    indexed = cache.set_index("query_id").loc[query_ids]
    result = {}
    for name in columns:
        new = np.asarray(features[name], dtype=np.float64)
        old = indexed[name].to_numpy(dtype=np.float64)
        both_nan = np.isnan(new) & np.isnan(old)
        equal = (new == old) | both_nan
        finite = np.isfinite(new) & np.isfinite(old)
        difference = np.abs(new[finite] - old[finite])
        result[name] = {
            "queries": int(len(new)),
            "exact_share": float(equal.mean()),
            "mismatches": int((~equal).sum()),
            "max_abs_difference": float(difference.max()) if difference.size else 0.0,
            "nan_mismatches": int((np.isnan(new) ^ np.isnan(old)).sum()),
        }
    return result


def _original_path(builder, split_key: str, retriever: str, query_ids: np.ndarray, depth: int) -> tuple[pd.DataFrame, float]:
    queries = builder._limited_queries(split_key)
    queries = queries.set_index("query_id", drop=False).loc[query_ids].reset_index(drop=True)
    ranking_frame = builder.retrieval.ensure_rankings(split_key, retriever)
    rankings = builder._rankings_by_query(split_key, retriever, ranking_frame=ranking_frame)
    query_signals = builder._query_signal_lookup(split_key)
    all_rankings_top100 = builder._bm25_and_self_rankings_top100(split_key, retriever)
    semantic_state = builder._semantic_state(split_key, retriever)
    semantic_thresholds = builder._semantic_feature_thresholds(retriever)
    plan = builder._plan(retriever)
    rows = []
    started = time.perf_counter()
    for query in queries.itertuples(index=False):
        query_id = int(query.query_id)
        signal_row = query_signals.loc[query_id]
        ranking = rankings[query_id]
        gold_rank = builder._gold_rank(ranking, int(query.document_id))
        smallest_size = builder._smallest_sufficient_size(gold_rank)
        shared = {
            "query_id": query_id,
            "document_id": int(query.document_id),
            "language": str(query.language),
            "split_name": split_key,
            "retriever_name": retriever,
            "gold_rank": gold_rank if gold_rank is not None else math.nan,
            "smallest_sufficient_size": smallest_size if smallest_size is not None else math.nan,
            "retrieval_failure_within_max_depth": float(gold_rank is None),
            "rr_at_max_depth": 0.0 if gold_rank is None else 1.0 / gold_rank,
        }
        lexical_pre = builder._lexical_pre_features(query, signal_row, retriever)
        semantic_context = builder._semantic_query_context(
            query, ranking, retriever, semantic_state, all_rankings_top100, threshold_bundle=semantic_thresholds
        )
        semantic_context["candidate_depths"] = (int(depth),)
        semantic_pre_static = builder._semantic_query_static_state(query, retriever, semantic_state)["pre_features"]
        agreement = builder._retriever_agreement_features(retriever, semantic_context, all_rankings_top100) if plan.needs_agreement else {}
        row = dict(shared)
        row["candidate_size"] = int(depth)
        row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
        row |= lexical_pre
        row |= semantic_pre_static
        if plan.needs_distance_weighted_rr:
            row["distance_weighted_training_query_rr_at_k"] = builder._distance_weighted_training_rr(
                query, retriever, int(depth), semantic_context["query_embedding"]
            )
        row |= builder._semantic_score_features(semantic_context, int(depth), retriever)
        if plan.needs_geometry:
            row |= builder._semantic_geometry_features(semantic_context, int(depth), retriever)
        if plan.needs_cluster:
            row |= builder._semantic_cluster_features(semantic_context, int(depth), retriever)
        if plan.needs_agreement:
            row |= agreement
        if plan.needs_perturbation:
            row |= builder._perturbation_features(query_id, int(depth), semantic_state, semantic_context)
        rows.append(_select_qualifying(row, plan))
        builder._discard_query_feature_caches(split_key, retriever, query_id)
    elapsed = time.perf_counter() - started
    return pd.DataFrame(rows), elapsed


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Language code."),
    retriever: str = typer.Option(DEFAULT_RETRIEVER, help="First-stage dense retriever."),
    predictor_name: str = typer.Option(DEFAULT_PREDICTOR_NAME, help="Deployed predictor namespace under artifacts/models."),
    feature_cache_table: Optional[str] = typer.Option(None, help="Feature table under artifacts/features holding the original features (default derived from the predictor)."),
    subset_size: int = typer.Option(DEFAULT_SUBSET_SIZE, help="Size of the stratified benchmark subset that is created here and reused by the benchmark."),
    seed: Optional[int] = typer.Option(None, help="Subset sampling seed (default: runtime.random_seed from the config)."),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Reranking depth."),
    metric_k: int = typer.Option(DEFAULT_METRIC_K, help="MRR cutoff."),
    reranker_model: str = typer.Option(DEFAULT_RERANKER_MODEL, help="Jina reranker model."),
    reranker_max_length: int = typer.Option(DEFAULT_RERANKER_MAX_LENGTH, help="Reranker maximum sequence length."),
    reranker_batch_size: int = typer.Option(DEFAULT_RERANKER_BATCH_SIZE, help="Reranker batch size."),
    request_size: Optional[int] = typer.Option(None, help="Pairs per compute_score request (default: thesis setting)."),
    jina_check_requests: int = typer.Option(1, help="Number of leading thesis-order requests rescored to check Jina reproduction (0 disables)."),
    original_path_timing: bool = typer.Option(True, help="Time the original per-query feature path on the benchmark subset."),
    check_dev: bool = typer.Option(True, help="Also verify the dev split features (exercises self-exclusion in the reference pool)."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    log = console.log
    console.section("Verification of the optimised lightweight Hit@1 gate")
    retriever = runtime.retrieval.resolve_retriever_name(retriever)
    spec = reranker_spec(reranker_model, reranker_max_length, reranker_batch_size)
    request_size = int(request_size or default_request_size(reranker_batch_size))
    seed = int(runtime.config.runtime.random_seed if seed is None else seed)
    output_dir = report_dir(runtime, language, retriever, spec, top_k, predictor_name)
    run_dir = multistep_run_dir(runtime, language, retriever, spec, top_k)
    cache_table = feature_cache_table or FEATURE_CACHE_TABLES.get(predictor_name)
    test_key = f"{language}_test"
    report: dict[str, Any] = {
        "predictor_name": predictor_name,
        "retriever": retriever,
        "environment": environment_metadata("cpu"),
    }

    console.section("1. Deployed gate state")
    gate = load_deployed_gate(runtime, language, retriever, predictor_name, test_key, log)
    model = gate.model
    report["model"] = {
        "path": str(model.model_path),
        "signature": gate.model_signature,
        "features": model.feature_columns,
        "threshold": model.threshold,
        "feature_depth": model.feature_depth,
        "reference_split_keys": gate.reference_split_keys,
    }

    console.section("2. Fast gate on the full test split vs. original cached features")
    workload = split_workload(runtime, test_key, retriever, top_k)
    log(f"Running the fast gate over {len(workload.queries)} {test_key} queries")
    result = run_gate(gate.state, workload.inputs)
    log("Fast gate component times (s): " + ", ".join(f"{k}={v:.3f}" for k, v in result.timings.items()))
    report["full_test_fast_gate_seconds"] = result.timings
    if cache_table:
        cache_path = runtime.store.root_dir / "features" / cache_table / test_key / retriever / f"depth_{model.feature_depth:04d}.parquet"
        if cache_path.exists():
            comparison = _compare_features(result.features, pd.read_parquet(cache_path), workload.inputs.query_ids, model.feature_columns)
            report["test_feature_agreement"] = comparison
            for name, values in comparison.items():
                log(f"  {name}: exact {values['exact_share']:.6f}, mismatches {values['mismatches']}, max |diff| {values['max_abs_difference']:.3e}")
        else:
            log(f"  No original feature cache at {cache_path}; feature comparison skipped")

    predictions_path = runtime.store.root_dir / "predictions" / predictor_name / language / retriever / "test_predictions.parquet"
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path).set_index("query_id").loc[workload.inputs.query_ids]
        probability_difference = np.abs(result.probabilities - predictions["confidence"].to_numpy(dtype=np.float64))
        skip_agreement = result.skip == predictions["predicted_hit_at_1"].to_numpy(dtype=bool)
        report["test_prediction_agreement"] = {
            "max_abs_probability_difference": float(probability_difference.max()),
            "skip_decision_agreement": float(skip_agreement.mean()),
            "skip_decision_mismatches": int((~skip_agreement).sum()),
        }
        log(f"  probabilities max |diff| {probability_difference.max():.3e}; skip decisions agree on {skip_agreement.mean():.6f}")
    else:
        log(f"  No cached test predictions at {predictions_path}; probability comparison skipped")

    outcomes_path = run_dir / "test_query_outcomes.parquet"
    if predictor_name == DEFAULT_PREDICTOR_NAME and outcomes_path.exists():
        outcomes = pd.read_parquet(outcomes_path).set_index("query_id").loc[workload.inputs.query_ids]
        rerank_agreement = (~result.skip) == outcomes["hit1_gate_pass"].to_numpy(dtype=bool)
        report["multistep_gate_agreement"] = {
            "rerank_decision_agreement": float(rerank_agreement.mean()),
            "mismatches": int((~rerank_agreement).sum()),
        }
        log(f"  rerank decisions agree with the thesis multistep run on {rerank_agreement.mean():.6f}")

    console.section("3. BM25 top-1 and query tokenisation")
    bm25_name = runtime.retrieval.paired_retriever("bm25", retriever)
    if gate.state.bm25 is not None:
        token_lists = [tokenize(text) for text in workload.inputs.query_texts]
        cached_tokens = [list(tokens) for tokens in workload.queries["query_tokens"]]
        token_agreement = float(np.mean([a == b for a, b in zip(token_lists, cached_tokens, strict=True)]))
        bm25_positions = bm25_top_1_positions(gate.state.bm25, token_lists)
        bm25_ranking = runtime.retrieval.ensure_rankings(test_key, bm25_name)
        cached_top_1 = bm25_ranking.loc[bm25_ranking["rank"] == 1].set_index("query_id")["document_id"].loc[workload.inputs.query_ids].to_numpy()
        top_1_agreement = float(np.mean(gate.state.document_ids[bm25_positions] == cached_top_1))
        report["bm25"] = {"online_tokenisation_agreement": token_agreement, "top_1_agreement_with_cached_ranking": top_1_agreement}
        log(f"  online tokenisation equals cached query tokens for {token_agreement:.6f}; BM25 top-1 equals cached {bm25_name} ranking for {top_1_agreement:.6f}")

    console.section("4. Batched vs per-query nearest-neighbour searches")
    if gate.state.document_index is not None and "perturbation_ranking_stability" in model.qualifying_features:
        perturbed = perturbed_query_embeddings(workload.inputs.query_ids, workload.inputs.query_embeddings, gate.state.random_seed)
        _, batched = gate.state.document_index.search(perturbed, model.feature_depth)
        identical = np.empty(len(workload.inputs.query_ids), dtype=bool)
        for row in range(len(identical)):
            block = perturbed[row * PERTURBATION_COUNT : (row + 1) * PERTURBATION_COUNT]
            _, single = gate.state.document_index.search(np.ascontiguousarray(block), model.feature_depth)
            identical[row] = np.array_equal(single, batched[row * PERTURBATION_COUNT : (row + 1) * PERTURBATION_COUNT])
        report["perturbation_batched_search_identical_share"] = float(identical.mean())
        log(f"  perturbed rankings: batched search identical to per-query search for {identical.mean():.6f} of the queries")
    if FEATURE_TRAINING_RR in model.qualifying_features:
        blas = training_query_rr(gate.state.reference_pool, workload.inputs.query_embeddings, workload.inputs.query_keys, exact_distances=False)
        exact = result.features[FEATURE_TRAINING_RR]
        alternative = dict(result.features)
        alternative[FEATURE_TRAINING_RR] = blas
        frame = pd.DataFrame({name: alternative[name] for name in model.feature_columns})
        blas_skip = np.asarray(model.pipeline.predict_proba(frame), dtype=np.float64)[:, model.positive_index] >= model.threshold
        report["training_rr_blas_alternative"] = {
            "max_abs_difference": float(np.nanmax(np.abs(blas - exact))),
            "changed_feature_values": int(np.sum(blas != exact)),
            "changed_skip_decisions": int(np.sum(blas_skip != result.skip)),
        }
        log(
            f"  training-query RR with FAISS BLAS distances instead of the original per-query kernel: max |diff| "
            f"{report['training_rr_blas_alternative']['max_abs_difference']:.3e}, changed skip decisions "
            f"{report['training_rr_blas_alternative']['changed_skip_decisions']} (not used; exact kernel kept)"
        )

    console.section("5. Effectiveness from cached rerank-all scores")
    cached_scores = cached_top_k_scores(run_dir, workload.inputs.query_ids, workload.top_document_ids, top_k)
    metrics = effectiveness(workload.top_document_ids, cached_scores, workload.gold_document_ids, ~result.skip, top_k, metric_k)
    report["full_test_effectiveness"] = {key: value for key, value in metrics.items() if not key.startswith("rr_")}
    log(
        f"  MRR@{metric_k}: no rerank {metrics['mrr_no_rerank']:.6f}, rerank all {metrics['mrr_rerank_all']:.6f}, "
        f"adaptive {metrics['mrr_adaptive']:.6f}; reranked {metrics['reranked_queries_adaptive']}/{metrics['queries']} queries "
        f"({metrics['reranked_pairs_adaptive']:,} of {metrics['reranked_pairs_rerank_all']:,} calls)"
    )
    summary_path = run_dir / "summary.json"
    if predictor_name == DEFAULT_PREDICTOR_NAME and summary_path.exists():
        summary = runtime.store.read_json(summary_path)["test"]
        rerank_all = summary[f"rerank_all_top{top_k}"]
        hit1 = summary["hit1"]
        report["thesis_summary_agreement"] = {
            "rerank_all_mrr_difference": abs(rerank_all["mrr_at_5"] - metrics["mrr_rerank_all"]),
            "adaptive_mrr_difference": abs(hit1["mrr_at_5"] - metrics["mrr_adaptive"]),
            "no_rerank_mrr_difference": abs(summary["no_rerank"]["mrr_at_5"] - metrics["mrr_no_rerank"]),
            "reranked_queries_thesis": int(hit1["reranked_queries"]),
            "reranked_pairs_thesis": int(hit1["reranked_pairs"]),
            "reranked_pairs_recomputed": int(metrics["reranked_pairs_adaptive"]),
        }
        log(
            f"  thesis summary: rerank all {rerank_all['mrr_at_5']:.6f}, Hit@1 {hit1['mrr_at_5']:.6f} with "
            f"{hit1['reranked_pairs']:,} calls; differences {report['thesis_summary_agreement']}"
        )

    console.section("6. Stratified benchmark subset")
    subset = load_or_create_subset(
        output_dir / "subset_query_ids.json", workload.inputs.query_ids, result.skip, subset_size, seed, gate.model_signature, log
    )
    subset_ids = np.asarray(subset["query_ids"], dtype=np.int64)
    report["subset"] = {key: value for key, value in subset.items() if key != "query_ids"}

    if check_dev and cache_table:
        console.section("7. Dev split features (self-excluding reference pool)")
        dev_key = f"{language}_dev"
        dev_cache = runtime.store.root_dir / "features" / cache_table / dev_key / retriever / f"depth_{model.feature_depth:04d}.parquet"
        if dev_cache.exists():
            dev_gate = load_deployed_gate(runtime, language, retriever, predictor_name, dev_key, log)
            dev_workload = split_workload(runtime, dev_key, retriever, top_k)
            dev_result = run_gate(dev_gate.state, dev_workload.inputs)
            comparison = _compare_features(dev_result.features, pd.read_parquet(dev_cache), dev_workload.inputs.query_ids, model.feature_columns)
            report["dev_feature_agreement"] = comparison
            for name, values in comparison.items():
                log(f"  {name}: exact {values['exact_share']:.6f}, mismatches {values['mismatches']}, max |diff| {values['max_abs_difference']:.3e}")
        else:
            log(f"  No dev feature cache at {dev_cache}; skipped")

    if original_path_timing and predictor_name == DEFAULT_PREDICTOR_NAME:
        console.section("8. Original per-query feature path vs. optimised gate on the benchmark subset")
        builder = LightweightHitAtOnePredictorFeatureBuilder(
            runtime.config,
            runtime.store,
            console,
            runtime.features.bundle,
            runtime.retrieval,
            runtime.clustering,
            runtime.text_services,
            consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
        )
        builder._active_language = language
        if builder.qualifying_features(retriever) != model.qualifying_features:
            raise RuntimeError("The original builder's feature plan differs from the deployed model's features")
        log("Untimed warm pass of the original path over the subset (loads lexical statistics and caches)")
        _original_path(builder, test_key, retriever, subset_ids, model.feature_depth)
        log("Timed pass of the original path over the subset")
        original_frame, original_seconds = _original_path(builder, test_key, retriever, subset_ids, model.feature_depth)
        prediction_started = time.perf_counter()
        original_probabilities = np.asarray(model.pipeline.predict_proba(original_frame), dtype=np.float64)[:, model.positive_index]
        original_prediction_seconds = time.perf_counter() - prediction_started
        subset_workload = split_workload(runtime, test_key, retriever, top_k, query_ids=subset_ids)
        run_gate(gate.state, subset_workload.inputs)
        fast = run_gate(gate.state, subset_workload.inputs)
        agreement = _compare_features(fast.features, original_frame, subset_ids, model.feature_columns)
        report["original_vs_optimised_subset"] = {
            "queries": int(len(subset_ids)),
            "original_feature_seconds": original_seconds,
            "original_batched_prediction_seconds": original_prediction_seconds,
            "original_total_seconds_excluding_online_bm25": original_seconds + original_prediction_seconds,
            "original_note": "BM25 top-1 was read from a precomputed BM25 ranking, so its online cost is not included in the original path",
            "optimised_component_seconds": fast.timings,
            "optimised_total_seconds_including_online_bm25": fast.timings["gate_total"],
            "speedup": (original_seconds + original_prediction_seconds) / fast.timings["gate_total"],
            "feature_agreement_original_path_vs_optimised": agreement,
            "probability_max_abs_difference": float(np.abs(original_probabilities - fast.probabilities).max()),
            "skip_decision_agreement": float(np.mean((original_probabilities >= model.threshold) == fast.skip)),
        }
        log(
            f"  original path: {1000.0 * (original_seconds + original_prediction_seconds) / len(subset_ids):.2f} ms/query "
            f"(BM25 precomputed); optimised gate: {1000.0 * fast.timings['gate_total'] / len(subset_ids):.2f} ms/query "
            f"(BM25 computed online); speedup x{report['original_vs_optimised_subset']['speedup']:.1f}"
        )
        for name, values in agreement.items():
            log(f"  original-path replica vs optimised {name}: exact {values['exact_share']:.6f}")

    if jina_check_requests > 0:
        console.section("9. Jina reproduction of thesis scores and inference_mode equivalence")
        reranker = load_jina_reranker(spec.model_name, spec.trust_remote_code, spec.use_flash_attn, spec.quantize_dynamic_int8, "cpu")
        query_text_by_id = workload.queries.set_index("query_id")["text"].astype(str).to_dict()
        document_text_by_id = {
            int(document_id): str(text)
            for document_id, text in zip(runtime.retrieval.document_ids.tolist(), runtime.retrieval.document_scope_bundle(retriever)["texts"], strict=True)
        }
        order = np.argsort(workload.inputs.query_ids, kind="mergesort")
        pairs = build_pairs(workload.inputs.query_ids[order], workload.top_document_ids[order, :top_k], query_text_by_id, document_text_by_id)
        cached_flat = cached_scores[order].reshape(-1)
        count = min(len(pairs), int(jina_check_requests) * request_size)
        log(f"Rescoring the first {count} thesis-order pairs ({jina_check_requests} request(s) of {request_size}) with and without inference_mode")
        grad_scores = []
        for start in range(0, count, request_size):
            grad_scores.append(np.asarray(reranker.compute_score(pairs[start : min(start + request_size, count)], batch_size=reranker_batch_size, max_length=reranker_max_length), dtype=np.float32).reshape(-1))
        grad_scores = np.concatenate(grad_scores)
        inference_scores = score_pairs(reranker, pairs[:count], reranker_batch_size, reranker_max_length, request_size)
        report["jina_check"] = {
            "pairs": int(count),
            "torch_num_threads": int(torch.get_num_threads()),
            "inference_mode_identical_to_grad_enabled": bool(np.array_equal(grad_scores, inference_scores)),
            "max_abs_difference_inference_mode_vs_grad_enabled": float(np.abs(grad_scores - inference_scores).max()),
            "max_abs_difference_vs_cached_thesis_scores": float(np.abs(inference_scores.astype(np.float64) - cached_flat[:count]).max()),
            "identical_to_cached_thesis_scores_share": float(np.mean(inference_scores.astype(np.float64) == cached_flat[:count])),
        }
        log(f"  {report['jina_check']}")

    write_json(output_dir / "verification.json", report)
    console.section("Verification summary")
    log(f"Saved {output_dir / 'verification.json'}")


if __name__ == "__main__":
    typer.run(main)
