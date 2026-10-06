from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import typer
from rank_bm25 import BM25Okapi

from _hit_at_1_gate_cost_common import (
    DEFAULT_LANGUAGE,
    DEFAULT_PREDICTOR_NAME,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVER,
    DEFAULT_TOP_K,
    close_competitor_threshold,
    load_deployed_gate,
    model_path,
    ranking_arrays,
    reference_split_keys,
    report_dir,
    reranker_spec,
    split_queries,
    split_workload,
    write_available_tables,
    write_json,
)
from query_difficulty.dataset import join_fields, tokenize
from query_difficulty.hit_at_1_gate_fast import (
    CLOSE_COMPETITOR_RANKING_DEPTH,
    FEATURE_BM25_TOP_1_AGREEMENT,
    FEATURE_CLOSE_COMPETITOR,
    FEATURE_TRAINING_RR,
    build_bm25_term_index,
    build_training_reference_pool,
    environment_metadata,
    estimate_close_competitor_threshold,
    gold_reciprocal_ranks,
    load_gate_model,
    training_query_rr,
)
from query_difficulty.runtime import build_runtime


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Language code."),
    retriever: str = typer.Option(DEFAULT_RETRIEVER, help="First-stage dense retriever."),
    predictor_name: str = typer.Option(DEFAULT_PREDICTOR_NAME, help="Deployed predictor namespace under artifacts/models."),
    repetitions: int = typer.Option(1, help="Number of complete from-scratch setup runs."),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Reranking depth (only used to locate the report directory)."),
    reranker_model: str = typer.Option(DEFAULT_RERANKER_MODEL, help="Jina reranker model (report directory)."),
    reranker_max_length: int = typer.Option(DEFAULT_RERANKER_MAX_LENGTH, help="Reranker maximum length (report directory)."),
    reranker_batch_size: int = typer.Option(DEFAULT_RERANKER_BATCH_SIZE, help="Reranker batch size (report directory)."),
    output_dir: Optional[str] = typer.Option(None, help="Override the report directory."),
    write_tables: bool = typer.Option(True, help="Write the LaTeX tables to latex/tables/efficiency."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    log = console.log
    console.section("Predictor-specific setup cost of the lightweight Hit@1 gate")
    retrieval = runtime.retrieval
    retriever = retrieval.resolve_retriever_name(retriever)
    spec = reranker_spec(reranker_model, reranker_max_length, reranker_batch_size)
    target_dir = report_dir(runtime, language, retriever, spec, top_k, predictor_name) if output_dir is None else Path(output_dir).resolve()
    test_key = f"{language}_test"
    pool_keys = reference_split_keys(test_key)
    semantic_spec = retrieval.semantic_spec(retriever)
    path = model_path(runtime, predictor_name, retriever)
    deployed_model = load_gate_model(path)
    required = deployed_model.qualifying_features
    feature_depth = deployed_model.feature_depth
    log(f"Deployed predictor {path} requires {sorted(required)}; reference pool splits for {test_key}: {pool_keys}")

    console.section("Shared prerequisites (excluded from predictor-specific setup)")
    shared: dict[str, float] = {}
    started = time.perf_counter()
    retrieval._load_encoder(semantic_spec)
    shared["qwen_encoder_loading"] = time.perf_counter() - started
    started = time.perf_counter()
    retrieval.ensure_semantic_embeddings(retriever)
    document_index = retrieval.ensure_semantic_index_object(retriever)
    shared["document_embeddings_and_index_loading"] = time.perf_counter() - started
    log(f"Shared prerequisites (not counted): {shared}")
    pool_queries = {key: split_queries(runtime, key) for key in pool_keys}
    publications = retrieval.publications[["title", "abstract"]].fillna("")
    training_query_count = int(sum(len(frame) for frame in pool_queries.values()))

    components: dict[str, list[float]] = {}
    totals: list[float] = []
    checks: dict[str, Any] = {}

    def record(name: str, seconds: float) -> None:
        components.setdefault(name, []).append(float(seconds))
        log(f"  {name}: {seconds:.2f}s")

    for repetition in range(int(repetitions)):
        console.section(f"Setup run {repetition + 1}/{repetitions}")
        total = 0.0
        fresh_embeddings: dict[str, np.ndarray] = {}
        fresh_rankings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if FEATURE_TRAINING_RR in required or FEATURE_CLOSE_COMPETITOR in required:
            started = time.perf_counter()
            for key, frame in pool_queries.items():
                if key != f"{language}_train" and FEATURE_TRAINING_RR not in required:
                    continue
                texts = [(semantic_spec.query_prefix + text).strip() for text in frame["text"].tolist()]
                fresh_embeddings[key] = retrieval._encode_texts_in_memory(semantic_spec, texts, f"{key} queries for setup timing")
            elapsed = time.perf_counter() - started
            record("training_query_encoding", elapsed)
            total += elapsed

            started = time.perf_counter()
            for key, embeddings in fresh_embeddings.items():
                depth = feature_depth
                if key == f"{language}_train" and FEATURE_CLOSE_COMPETITOR in required:
                    depth = max(depth, CLOSE_COMPETITOR_RANKING_DEPTH)
                scores, indices = document_index.search(np.ascontiguousarray(embeddings, dtype=np.float32), depth)
                fresh_rankings[key] = (np.asarray(retrieval.document_ids)[indices], scores.astype(np.float64))
            fresh_rr = {
                key: gold_reciprocal_ranks(fresh_rankings[key][0], pool_queries[key]["document_id"].to_numpy(dtype=np.int64), feature_depth)
                for key in fresh_rankings
            }
            elapsed = time.perf_counter() - started
            record("training_query_retrieval", elapsed)
            total += elapsed

        if FEATURE_TRAINING_RR in required:
            started = time.perf_counter()
            pool = build_training_reference_pool(
                [fresh_embeddings[key] for key in pool_keys],
                [[(key, int(query_id)) for query_id in pool_queries[key]["query_id"]] for key in pool_keys],
                [fresh_rr[key] for key in pool_keys],
            )
            elapsed = time.perf_counter() - started
            record("reference_index_construction", elapsed)
            total += elapsed

        if FEATURE_BM25_TOP_1_AGREEMENT in required:
            started = time.perf_counter()
            document_tokens = [tokenize(join_fields(title, abstract)) for title, abstract in zip(publications["title"], publications["abstract"], strict=True)]
            bm25_model = BM25Okapi(document_tokens)
            bm25_index = build_bm25_term_index(bm25_model, runtime.config.max_ranking_depth)
            elapsed = time.perf_counter() - started
            record("bm25_index_construction", elapsed)
            total += elapsed

        if FEATURE_CLOSE_COMPETITOR in required:
            started = time.perf_counter()
            train_key = f"{language}_train"
            threshold = estimate_close_competitor_threshold(pool_queries[train_key], fresh_rankings[train_key][0], fresh_rankings[train_key][1])
            elapsed = time.perf_counter() - started
            record("close_competitor_threshold_estimation", elapsed)
            total += elapsed

        started = time.perf_counter()
        load_gate_model(path)
        elapsed = time.perf_counter() - started
        record("predictor_loading", elapsed)
        total += elapsed
        totals.append(total)
        log(f"Total predictor-specific setup: {total:.2f}s")

        if repetition == 0:
            console.section("Reproduction checks of the from-scratch setup against the deployed artifacts (untimed)")
            for key, embeddings in fresh_embeddings.items():
                cached = np.asarray(retrieval.ensure_query_embeddings(key, retriever), dtype=np.float32)
                checks[f"{key}_embedding_max_abs_difference"] = float(np.abs(embeddings - cached).max())
                cached_ranked, _ = ranking_arrays(retrieval.ensure_rankings(key, retriever), pool_queries[key]["query_id"].to_numpy(), feature_depth)
                cached_rr = gold_reciprocal_ranks(cached_ranked, pool_queries[key]["document_id"].to_numpy(dtype=np.int64), feature_depth)
                checks[f"{key}_rr_agreement"] = float(np.mean(fresh_rr[key] == cached_rr))
            if FEATURE_CLOSE_COMPETITOR in required:
                checks["close_competitor_threshold_fresh"] = threshold
                checks["close_competitor_threshold_deployed"] = close_competitor_threshold(runtime, retriever)
            if FEATURE_BM25_TOP_1_AGREEMENT in required:
                deployed_bm25 = retrieval.ensure_bm25_model(retrieval.paired_retriever("bm25", retriever))
                checks["bm25_document_tokens_equal"] = bool(
                    all(list(a) == list(b) for a, b in zip(document_tokens, retrieval.document_scope_bundle(retriever)["tokens"], strict=True))
                )
                checks["bm25_idf_equal"] = bool(deployed_bm25.idf == bm25_model.idf)
                checks["bm25_doc_len_equal"] = bool(list(deployed_bm25.doc_len) == list(bm25_model.doc_len))
                checks["bm25_avgdl_equal"] = bool(deployed_bm25.avgdl == bm25_model.avgdl)
            if FEATURE_TRAINING_RR in required:
                gate = load_deployed_gate(runtime, language, retriever, predictor_name, test_key, log)
                workload = split_workload(runtime, test_key, retriever, feature_depth)
                deployed_values = training_query_rr(gate.state.reference_pool, workload.inputs.query_embeddings, workload.inputs.query_keys)
                fresh_values = training_query_rr(pool, workload.inputs.query_embeddings, workload.inputs.query_keys)
                checks["test_training_rr_fresh_vs_deployed_max_abs_difference"] = float(np.nanmax(np.abs(fresh_values - deployed_values)))
                checks["test_training_rr_fresh_vs_deployed_exact_share"] = float(np.mean(fresh_values == deployed_values))
            for name, value in checks.items():
                log(f"  {name}: {value}")

    payload = {
        "predictor_name": predictor_name,
        "retriever": retriever,
        "model_path": str(path),
        "required_features": sorted(required),
        "reference_split_keys": pool_keys,
        "training_query_count": training_query_count if FEATURE_TRAINING_RR in required else None,
        "repetitions": int(repetitions),
        "components": {name: {"seconds": values, "mean": float(np.mean(values))} for name, values in components.items()},
        "total_seconds": totals,
        "total_mean_seconds": float(np.mean(totals)),
        "excluded_shared_prerequisites_seconds": shared,
        "included": [
            "training_query_encoding: Qwen3 encoding of every reference-pool query with the deployed encoder settings",
            "training_query_retrieval: exact FAISS retrieval of the reference-pool queries and their RR@k labels (train top-100 for the threshold)",
            "reference_index_construction: exact L2 FAISS index over the reference-pool embeddings",
            "bm25_index_construction: corpus tokenisation, BM25Okapi statistics, and the BM25 term-contribution index",
            "close_competitor_threshold_estimation: estimation of the semantic close-competitor threshold from the train rankings",
            "predictor_loading: loading the fitted calibrated Random Forest from disk",
        ],
        "excluded": [
            "Qwen3 encoder loading, document embeddings, and the document FAISS index (shared with first-stage retrieval)",
            "Training of the predictor, feature selection, hyperparameter search, and other research-development costs",
        ],
        "reproduction_checks": checks,
        "environment": environment_metadata("cpu"),
    }
    write_json(target_dir / "setup_cost.json", payload)
    log(f"Saved {target_dir / 'setup_cost.json'}")
    if write_tables:
        write_available_tables(runtime, target_dir, language, retriever, predictor_name, log)


if __name__ == "__main__":
    typer.run(main)
