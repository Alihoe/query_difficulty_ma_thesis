from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.hit_at_1_gate_fast import (
    COMPONENT_BM25,
    COMPONENT_GATE_TOTAL,
    COMPONENT_ONLINE_TOTAL,
    COMPONENT_PERTURBATION,
    COMPONENT_REMAINING_AND_PREDICTION,
    COMPONENT_RERANK,
    COMPONENT_TRAINING_RR,
    FEATURE_BM25_TOP_1_AGREEMENT,
    FEATURE_TRAINING_RR,
    GateInputs,
    GateModel,
    GateState,
    build_bm25_term_index,
    build_training_reference_pool,
    document_position_lookup,
    gold_reciprocal_ranks,
    load_gate_model,
    ranking_arrays,
    reranked_reciprocal_ranks,
)
from query_difficulty.multistep import _COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER, JinaComputeScoreRerankerSpec

DEFAULT_LANGUAGE = "en"
DEFAULT_RETRIEVER = "qwen3_embedding_0_6b"
DEFAULT_PREDICTOR_NAME = "lightweight/hit_at_1"
DEFAULT_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
DEFAULT_RERANKER_MAX_LENGTH = 1024
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_TOP_K = 30
DEFAULT_METRIC_K = 5
DEFAULT_SUBSET_SIZE = 500
DEFAULT_WARMUP_REPETITIONS = 1
DEFAULT_REPETITIONS = 5
SELECTOR_NAMESPACE = "comparison_lightweight_selectors"
REPORT_TOPIC = "hit_at_1_gate_cost"
TABLE_CATEGORY = "efficiency"

FEATURE_CACHE_TABLES = {
    "lightweight/hit_at_1": "descriptive_lightweight_hit_at_1_bm25_and_self",
    "lightweight/hit_at_1_no_training_rr": "descriptive_lightweight_hit_at_1_no_training_rr",
}

RETRIEVER_DISPLAY_NAMES = {"qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B", "bm25": "BM25"}


def default_request_size(batch_size: int) -> int:
    return max(1, int(batch_size) * _COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER)


def reranker_spec(model_name: str, max_length: int, batch_size: int) -> JinaComputeScoreRerankerSpec:
    return JinaComputeScoreRerankerSpec(
        model_name=model_name,
        max_length=int(max_length),
        batch_size=int(batch_size),
        trust_remote_code=True,
        use_flash_attn=False,
        candidate_fields="title_abstract",
        quantize_dynamic_int8=True,
    )


def predictor_slug(predictor_name: str) -> str:
    return predictor_name.replace("/", "__")


def multistep_run_dir(runtime, language: str, retriever: str, spec: JinaComputeScoreRerankerSpec, top_k: int) -> Path:
    return (
        runtime.store.root_dir
        / "multistep"
        / SELECTOR_NAMESPACE
        / language
        / retriever
        / spec.method
        / spec.key
        / f"top_{int(top_k)}"
        / spec.cache_dir_name
    )


def report_dir(runtime, language: str, retriever: str, spec: JinaComputeScoreRerankerSpec, top_k: int, predictor_name: str) -> Path:
    return (
        runtime.store.root_dir
        / "reports"
        / REPORT_TOPIC
        / language
        / retriever
        / spec.method
        / spec.key
        / f"top_{int(top_k)}"
        / spec.cache_dir_name
        / predictor_slug(predictor_name)
    )


def table_paths(runtime, language: str, retriever: str, predictor_name: str) -> dict[str, Path]:
    suffix = "" if predictor_name == DEFAULT_PREDICTOR_NAME else f"_{predictor_slug(predictor_name).replace('lightweight__', '')}"
    root = runtime.config.latex_table_dir / TABLE_CATEGORY
    return {
        "online": root / f"table_hit_at_1_gate_online_cost_{retriever}_jina{suffix}_{language}.tex",
        "setup": root / f"table_hit_at_1_gate_setup_cost_{retriever}_jina{suffix}_{language}.tex",
        "robustness": root / f"table_hit_at_1_gate_cost_robustness_{retriever}_jina{suffix}_{language}.tex",
    }


def model_path(runtime, predictor_name: str, retriever: str) -> Path:
    return runtime.store.root_dir / "models" / predictor_name / retriever / "model.joblib"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    tmp_path.replace(path)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value)}")


def reference_split_keys(split_key: str) -> list[str]:
    language, split_name = split_key.split("_", 1)
    if split_name == "train":
        return [split_key]
    return [f"{language}_train", f"{language}_dev"]


def split_queries(runtime, split_key: str) -> pd.DataFrame:
    if runtime.config.runtime.limit_queries is not None:
        raise ValueError("The cost benchmark requires runtime.limit_queries to be unset so that all queries are used")
    return runtime.retrieval.queries[split_key].reset_index(drop=True)


def close_competitor_threshold(runtime, retriever: str) -> float:
    path = runtime.store.path("features", "thresholds", "semantic_close_competitor", f"{retriever}.json")
    if not path.exists():
        raise FileNotFoundError(f"Expected the deployed close-competitor threshold at {path}")
    return float(runtime.store.read_json(path)["threshold"])


@dataclass(slots=True)
class DeployedGate:
    state: GateState
    model: GateModel
    reference_split_keys: list[str]
    model_signature: str


def load_deployed_gate(runtime, language: str, retriever: str, predictor_name: str, evaluated_split_key: str, log) -> DeployedGate:
    retrieval = runtime.retrieval
    retriever = retrieval.resolve_retriever_name(retriever)
    path = model_path(runtime, predictor_name, retriever)
    log(f"Loading deployed gate model {path}")
    model = load_gate_model(path)
    log(
        f"Gate model: feature_depth={model.feature_depth}, threshold={model.threshold:.6f}, "
        f"features={model.feature_columns}"
    )
    log("Loading shared Qwen document embeddings and FAISS document index (shared with first-stage retrieval)")
    _, document_embeddings = retrieval.ensure_semantic_embeddings(retriever)
    document_index = retrieval.ensure_semantic_index_object(retriever)
    document_ids = np.asarray(retrieval.document_ids, dtype=np.int64)
    reference_pool = None
    pool_keys = reference_split_keys(evaluated_split_key)
    if FEATURE_TRAINING_RR in model.qualifying_features:
        embeddings, keys, reciprocal_ranks = [], [], []
        for split_key in pool_keys:
            queries = split_queries(runtime, split_key)
            query_ids = queries["query_id"].to_numpy(dtype=np.int64)
            ranked, _ = ranking_arrays(retrieval.ensure_rankings(split_key, retriever), query_ids, model.feature_depth)
            embeddings.append(np.asarray(retrieval.ensure_query_embeddings(split_key, retriever), dtype=np.float32))
            keys.append([(split_key, int(query_id)) for query_id in query_ids])
            reciprocal_ranks.append(gold_reciprocal_ranks(ranked, queries["document_id"].to_numpy(dtype=np.int64), model.feature_depth))
            log(f"  reference pool part {split_key}: {len(query_ids)} queries")
        reference_pool = build_training_reference_pool(embeddings, keys, reciprocal_ranks)
        log(f"Training-query reference pool for {evaluated_split_key}: {len(reference_pool.keys)} queries from {pool_keys}")
    bm25 = None
    if FEATURE_BM25_TOP_1_AGREEMENT in model.qualifying_features:
        bm25_name = retrieval.paired_retriever("bm25", retriever)
        log(f"Loading deployed BM25 model for {bm25_name} and building its term-contribution index")
        bm25 = build_bm25_term_index(retrieval.ensure_bm25_model(bm25_name), runtime.config.max_ranking_depth)
    state = GateState(
        model=model,
        document_ids=document_ids,
        document_position_lookup=document_position_lookup(document_ids),
        document_embeddings=np.asarray(document_embeddings, dtype=np.float32),
        document_index=document_index,
        reference_pool=reference_pool,
        bm25=bm25,
        close_competitor_threshold=close_competitor_threshold(runtime, retriever),
        random_seed=int(runtime.config.runtime.random_seed),
    )
    return DeployedGate(state=state, model=model, reference_split_keys=pool_keys, model_signature=sha256_file(path))


@dataclass(slots=True)
class SplitWorkload:
    split_key: str
    queries: pd.DataFrame
    inputs: GateInputs
    top_document_ids: np.ndarray
    top_scores: np.ndarray
    gold_document_ids: np.ndarray


def split_workload(runtime, split_key: str, retriever: str, depth: int, query_ids: np.ndarray | None = None) -> SplitWorkload:
    retrieval = runtime.retrieval
    queries = split_queries(runtime, split_key)
    embeddings = np.asarray(retrieval.ensure_query_embeddings(split_key, retriever), dtype=np.float32)
    if len(embeddings) != len(queries):
        raise ValueError(f"Query embeddings for {split_key} do not align with the query frame")
    if query_ids is not None:
        positions = pd.Index(queries["query_id"].to_numpy()).get_indexer(np.asarray(query_ids))
        if (positions < 0).any():
            raise KeyError("Some subset query ids are not part of the split")
        queries = queries.iloc[positions].reset_index(drop=True)
        embeddings = embeddings[positions]
    ids = queries["query_id"].to_numpy(dtype=np.int64)
    top_document_ids, top_scores = ranking_arrays(retrieval.ensure_rankings(split_key, retriever), ids, depth)
    inputs = GateInputs(
        query_ids=ids,
        query_texts=queries["text"].fillna("").astype(str).tolist(),
        query_keys=[(split_key, int(query_id)) for query_id in ids],
        query_embeddings=np.ascontiguousarray(embeddings),
        retrieval_document_ids=top_document_ids,
        retrieval_scores=top_scores,
    )
    return SplitWorkload(
        split_key=split_key,
        queries=queries,
        inputs=inputs,
        top_document_ids=top_document_ids,
        top_scores=top_scores,
        gold_document_ids=queries["document_id"].to_numpy(dtype=np.int64),
    )


def stratified_subset(query_ids: np.ndarray, skip: np.ndarray, size: int, seed: int) -> dict[str, Any]:
    query_ids = np.asarray(query_ids, dtype=np.int64)
    skip = np.asarray(skip, dtype=bool)
    total = len(query_ids)
    if size >= total:
        selected = query_ids.copy()
    else:
        skip_count = int(round(size * float(skip.sum()) / total))
        generator = np.random.default_rng(int(seed))
        chosen_skip = generator.choice(query_ids[skip], size=skip_count, replace=False)
        chosen_rerank = generator.choice(query_ids[~skip], size=int(size) - skip_count, replace=False)
        chosen = set(int(value) for value in np.concatenate([chosen_skip, chosen_rerank]))
        selected = np.array([int(value) for value in query_ids if int(value) in chosen], dtype=np.int64)
    selected_skip = skip[np.isin(query_ids, selected)]
    return {
        "method": "proportional stratified random sampling without replacement, strata = gate skip / rerank decision",
        "seed": int(seed),
        "requested_size": int(size),
        "size": int(len(selected)),
        "population_size": int(total),
        "population_skip_rate": float(skip.mean()),
        "subset_skip_rate": float(selected_skip.mean()) if len(selected_skip) else math.nan,
        "subset_skipped_queries": int(selected_skip.sum()),
        "subset_reranked_queries": int((~selected_skip).sum()),
        "query_ids": [int(value) for value in selected],
    }


def load_or_create_subset(
    path: Path,
    query_ids: np.ndarray,
    skip: np.ndarray,
    size: int,
    seed: int,
    model_signature: str,
    log,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if path.exists():
        existing = read_json(path)
        if (
            existing.get("seed") == int(seed)
            and existing.get("requested_size") == int(size)
            and existing.get("model_signature") == model_signature
        ):
            log(f"Reusing benchmark subset {path} ({existing['size']} queries)")
            return existing
        log(f"Existing subset at {path} does not match seed/size/model; recreating it")
    subset = stratified_subset(query_ids, skip, size, seed)
    subset["model_signature"] = model_signature
    subset |= extra or {}
    write_json(path, subset)
    log(
        f"Created stratified benchmark subset {path}: {subset['size']} queries, skip rate "
        f"{subset['subset_skip_rate']:.4f} (population {subset['population_skip_rate']:.4f})"
    )
    return subset


def cached_top_k_scores(run_dir: Path, query_ids: np.ndarray, top_document_ids: np.ndarray, top_k: int) -> np.ndarray:
    path = run_dir / f"test_top{int(top_k)}_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected cached rerank-all scores at {path}")
    frame = pd.read_parquet(path).sort_values(["query_id", "rank"], kind="mergesort")
    counts = frame.groupby("query_id", sort=False).size()
    if (counts != int(top_k)).any():
        raise ValueError(f"Cached scores at {path} do not contain exactly {top_k} candidates per query")
    ordered_query_ids = frame["query_id"].to_numpy()[:: int(top_k)]
    positions = pd.Index(ordered_query_ids).get_indexer(np.asarray(query_ids))
    if (positions < 0).any():
        raise KeyError("Some queries are missing from the cached reranker scores")
    documents = frame["document_id"].to_numpy(dtype=np.int64).reshape(-1, int(top_k))[positions]
    if not np.array_equal(documents, top_document_ids[:, : int(top_k)]):
        raise ValueError("Cached reranker candidates differ from the current first-stage ranking")
    return frame["reranker_score"].to_numpy(dtype=np.float64).reshape(-1, int(top_k))[positions]


def effectiveness(top_document_ids: np.ndarray, scores: np.ndarray, gold: np.ndarray, rerank: np.ndarray, top_k: int, metric_k: int) -> dict[str, Any]:
    rr_no_rerank = gold_reciprocal_ranks(top_document_ids, gold, metric_k)
    rr_rerank_all = reranked_reciprocal_ranks(top_document_ids[:, :top_k], scores, gold, metric_k)
    rr_adaptive = np.where(rerank, rr_rerank_all, rr_no_rerank)
    return {
        "queries": int(len(gold)),
        "mrr_no_rerank": float(rr_no_rerank.mean()),
        "mrr_rerank_all": float(rr_rerank_all.mean()),
        "mrr_adaptive": float(rr_adaptive.mean()),
        "reranked_queries_rerank_all": int(len(gold)),
        "reranked_pairs_rerank_all": int(len(gold)) * int(top_k),
        "reranked_queries_adaptive": int(rerank.sum()),
        "reranked_pairs_adaptive": int(rerank.sum()) * int(top_k),
        "reranked_share_adaptive": float(rerank.mean()),
        "rr_no_rerank": rr_no_rerank,
        "rr_rerank_all": rr_rerank_all,
        "rr_adaptive": rr_adaptive,
    }


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _pm(mean: float, sd: float, digits: int) -> str:
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$"


VARIABILITY_LABEL = (
    "run-to-run execution variability: standard deviation over repeated executions of the same fixed workload; "
    "not an uncertainty estimate over the full test set"
)
BREAK_EVEN_LABEL = (
    "hardware- and setup-specific: one-time deployment setup time divided by the difference of the measured mean "
    "online runtimes per query (rerank all minus adaptive) on the benchmark machine"
)
SHARED_FIRST_STAGE_EXCLUDED = [
    "Qwen3 document encoding and FAISS document index construction",
    "Qwen3 encoder loading, query encoding, and first-stage retrieval (query embedding and ranking)",
    "loading of document embeddings and the document index",
]
DEVELOPMENT_COSTS_EXCLUDED = [
    "predictor training and calibration",
    "feature selection (importance freeze, correlation pruning, RFE)",
    "hyperparameter and threshold search",
    "exploratory training of alternative predictors",
]


def pair_token_lengths(tokenizer, workload, query_text_by_id: dict[int, str], document_text_by_id: dict[int, str], top_k: int, max_length: int, log) -> np.ndarray:
    from query_difficulty.hit_at_1_gate_fast import build_pairs

    pairs = build_pairs(workload.inputs.query_ids, workload.top_document_ids[:, :top_k], query_text_by_id, document_text_by_id)
    lengths = np.empty(len(pairs), dtype=np.int64)
    chunk = 4096
    log(f"Tokenising {len(pairs):,} reranker inputs (max_length={max_length}) for the representativeness check")
    for start in range(0, len(pairs), chunk):
        part = pairs[start : start + chunk]
        encoded = tokenizer([pair[0] for pair in part], [pair[1] for pair in part], truncation=True, max_length=max_length)
        lengths[start : start + len(part)] = [len(ids) for ids in encoded["input_ids"]]
        if (start // chunk) % 10 == 0:
            log(f"  tokenised {min(start + chunk, len(pairs)):,}/{len(pairs):,} pairs")
    return lengths.reshape(-1, int(top_k))


def _length_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    summary = {"count": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}
    for quantile in (0.05, 0.25, 0.5, 0.75, 0.95):
        summary[f"q{int(round(quantile * 100)):02d}"] = float(np.quantile(values, quantile))
    return summary


def length_representativeness(per_pair: np.ndarray, sample_mask: np.ndarray, rerank_mask: np.ndarray) -> dict[str, Any]:
    from scipy.stats import ks_2samp

    per_query = per_pair.sum(axis=1)
    report: dict[str, Any] = {
        "unit": "Jina tokens per query-document input after truncation",
        "test": "two-sample Kolmogorov-Smirnov test (Massey 1951), subset vs. complete test set",
    }
    for name, population, sample in (
        ("all_queries", np.ones(len(per_query), dtype=bool), sample_mask),
        ("reranked_queries", rerank_mask, sample_mask & rerank_mask),
    ):
        population_pairs = per_pair[population].reshape(-1)
        sample_pairs = per_pair[sample].reshape(-1)
        query_ks = ks_2samp(per_query[sample], per_query[population])
        pair_ks = ks_2samp(sample_pairs, population_pairs)
        report[name] = {
            "full_test_per_pair": _length_summary(population_pairs),
            "subset_per_pair": _length_summary(sample_pairs),
            "full_test_per_query_total": _length_summary(per_query[population]),
            "subset_per_query_total": _length_summary(per_query[sample]),
            "ks_per_query_total": {"statistic": float(query_ks.statistic), "p_value": float(query_ks.pvalue)},
            "ks_per_pair": {"statistic": float(pair_ks.statistic), "p_value": float(pair_ks.pvalue)},
        }
    return report


def run_environment_snapshot() -> dict[str, Any]:
    import os
    import time as time_module

    import faiss
    import torch

    return {
        "timestamp": time_module.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": os.getpid(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "faiss_omp_max_threads": int(faiss.omp_get_max_threads()),
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def break_even(setup: dict[str, Any] | None, aggregated: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"label": BREAK_EVEN_LABEL, "queries": None}
    if setup is None or aggregated is None:
        result["status"] = "pending"
        return result
    setup_seconds = float(np.mean(setup["total_seconds"]))
    full = float(aggregated["strategies"]["full"]["online_per_query_mean"])
    adaptive = float(aggregated["strategies"]["adaptive"]["online_per_query_mean"])
    difference = full - adaptive
    result.update(
        {
            "setup_seconds": setup_seconds,
            "full_online_seconds_per_query_mean": full,
            "adaptive_online_seconds_per_query_mean": adaptive,
            "online_saving_seconds_per_query": difference,
        }
    )
    if difference > 0:
        result["queries"] = int(math.ceil(setup_seconds / difference))
        result["status"] = "reached"
    else:
        result["status"] = "not reached (adaptive is not faster per query)"
    return result


def write_online_table(
    path: Path,
    aggregated: dict[str, Any],
    retriever: str,
    language: str,
    reference: dict[str, Any] | None = None,
) -> None:
    full = aggregated["strategies"]["full"]
    adaptive = aggregated["strategies"]["adaptive"]
    saving = aggregated["online_saving"]
    n_queries = aggregated["workload"]["queries"]
    repetitions = aggregated["workload"]["measured_repetitions"]
    warmups = aggregated["workload"]["warmup_repetitions"]
    full_test = aggregated.get("full_test_effectiveness") or {}
    display = RETRIEVER_DISPLAY_NAMES.get(retriever, retriever)
    workload_text = (
        f"the complete {n_queries:,}-query English test set"
        if aggregated["workload"].get("is_full_test")
        else f"one fixed stratified subset of {n_queries:,} English test queries"
    )
    caption = (
        f"Online computational cost of full and adaptive reranking with {display} and Jina at reranking depth $k={aggregated['workload']['top_k']}$, "
        f"measured on {workload_text} (CPU, dynamic int8 quantization). Values are means $\\pm$ SD over {repetitions} repeated executions "
        f"of this same workload after {warmups} warm-up run{'s' if warmups != 1 else ''}, with alternating strategy order; the SD reflects "
        "run-to-run execution variability, not uncertainty over the full test set. Online cost starts after first-stage retrieval and "
        "includes the gate (feature computation and prediction) and reranking. Per-query times are normalized by all queries of the workload."
    )
    if full_test:
        caption += (
            f" On the complete test set, the gate reranks {100.0 * full_test['reranked_share_adaptive']:.1f}\\% of the queries "
            f"({full_test['reranked_pairs_adaptive']:,} instead of {full_test['reranked_pairs_rerank_all']:,} reranker calls; "
            f"MRR@5 {full_test['mrr_adaptive']:.4f} vs. {full_test['mrr_rerank_all']:.4f})."
        )
    if reference is not None:
        reference_full = reference["strategies"]["full"]
        reference_adaptive = reference["strategies"]["adaptive"]
        reference_runs = reference["workload"]["measured_repetitions"]
        caption += (
            f" On a larger stratified subset of {reference['workload']['queries']:,} test queries "
            f"({reference_runs} measured run{'s' if reference_runs != 1 else ''}), the online time was "
            f"{reference_full['online_per_query_mean']:.2f} s/query for rerank all and {reference_adaptive['online_per_query_mean']:.2f} s/query "
            f"for the adaptive gate (gate {1000.0 * reference_adaptive['gate_per_query_mean']:.2f} ms/query), an online saving of "
            f"{100.0 * reference['online_saving']['mean']:.1f}\\%."
        )
    label = path.stem.replace("table_", "tab:")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l|rrrr|rr}",
        "\\toprule",
        "Strategy & Gate (ms/query) & Rerank (s/query) & Online total (s/query) & Workload total (h) & Reranker calls & Online saving \\\\",
        "\\midrule",
        (
            f"Rerank all & - & {_pm(full['rerank_per_query_mean'], full['rerank_per_query_sd'], 3)} & "
            f"{_pm(full['online_per_query_mean'], full['online_per_query_sd'], 3)} & "
            f"{_pm(full['online_total_mean'] / 3600.0, full['online_total_sd'] / 3600.0, 2)} & "
            f"{full['reranker_calls']:,} & - \\\\"
        ),
        (
            f"Adaptive Hit@1 & {_pm(1000.0 * adaptive['gate_per_query_mean'], 1000.0 * adaptive['gate_per_query_sd'], 2)} & "
            f"{_pm(adaptive['rerank_per_query_mean'], adaptive['rerank_per_query_sd'], 3)} & "
            f"{_pm(adaptive['online_per_query_mean'], adaptive['online_per_query_sd'], 3)} & "
            f"{_pm(adaptive['online_total_mean'] / 3600.0, adaptive['online_total_sd'] / 3600.0, 2)} & "
            f"{adaptive['reranker_calls']:,} & {_pm(100.0 * saving['mean'], 100.0 * saving['sd'], 1)}\\% \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


SETUP_TABLE_ROWS = (
    ("training_query_encoding", "Training-query encoding"),
    ("training_query_retrieval", "Training-query retrieval (RR@5 labels)"),
    ("reference_index_construction", "Training-query reference index construction"),
    ("bm25_index_construction", "BM25 index construction (incl. corpus tokenisation)"),
    ("close_competitor_threshold_estimation", "Close-competitor threshold estimation"),
    ("predictor_loading", "Loading the trained predictor"),
)


def write_setup_table(path: Path, setup: dict[str, Any], aggregated: dict[str, Any] | None, retriever: str) -> None:
    components = setup["components"]
    display = RETRIEVER_DISPLAY_NAMES.get(retriever, retriever)
    repetitions = int(setup["repetitions"])
    digits = 1

    def cell(name: str) -> str:
        values = components[name]["seconds"]
        mean, sd = _mean_sd(values)
        if repetitions > 1:
            return _pm(mean, sd, digits)
        return f"{mean:.{digits}f}"

    encoding_count = setup.get("training_query_count")
    rows = []
    for key, label in SETUP_TABLE_ROWS:
        if key not in components:
            continue
        text = label
        if key == "training_query_encoding" and encoding_count:
            text = f"{label} ({encoding_count:,} queries)"
        rows.append(f"{text} & {cell(key)} \\\\")
    total_mean, total_sd = _mean_sd(setup["total_seconds"])
    total_cell = _pm(total_mean, total_sd, digits) if repetitions > 1 else f"{total_mean:.{digits}f}"
    estimate = break_even(setup, aggregated)
    if estimate["queries"] is not None:
        break_even_text = f"{estimate['queries']:,} queries"
    elif aggregated is None:
        break_even_text = "-"
    else:
        break_even_text = "not reached"
    split_names = {"train": "training", "dev": "validation", "test": "test"}
    pool_text = " and ".join(split_names.get(key.split("_", 1)[1], key.split("_", 1)[1]) for key in setup.get("reference_split_keys", []))
    caption = (
        f"One-time deployment setup cost of the already trained lightweight {display} Hit@1 gate (seconds, CPU). "
        f"The training-query reference pool consists of the {pool_text} queries, as used by the deployed predictor. "
        "Shared first-stage costs (document encoding and indexing, encoder loading, first-stage retrieval) are excluded, "
        "as are model-development costs (training, feature selection, hyperparameter search). "
        "The break-even workload is hardware- and setup-specific: the setup time divided by the difference of the measured "
        "mean online runtimes per query of rerank all and the adaptive gate on the benchmark machine."
    )
    if repetitions > 1:
        caption += f" Values are means and standard deviations over {repetitions} runs."
    label = path.stem.replace("table_", "tab:")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Deployment setup component & Time (s) \\\\",
        "\\midrule",
        *rows,
        "\\midrule",
        f"Total deployment setup & {total_cell} \\\\",
        f"Break-even workload (hardware-specific) & {break_even_text} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_robustness_table(path: Path, summary: dict[str, Any], retriever: str) -> None:
    display = RETRIEVER_DISPLAY_NAMES.get(retriever, retriever)
    statistics = summary["statistics"]
    seeds = ", ".join(str(seed) for seed in summary["seeds"])
    caption = (
        f"Robustness of the online saving across {summary['completed_subsets']} independently sampled stratified subsets of "
        f"{summary['subset_size']:,} English test queries each ({display}, Jina, $k={summary['top_k']}$, CPU, dynamic int8 quantization; "
        f"seeds {seeds}). Each subset approximately preserves the full-test skip proportion of the gate "
        f"({100.0 * summary['population_skip_rate']:.1f}\\%), and both strategies were executed once per subset after "
        f"{summary['warmup_repetitions']} warm-up run{'s' if summary['warmup_repetitions'] != 1 else ''}, with alternating strategy order. "
        "SD, minimum and maximum are taken across subsets and therefore combine workload composition and execution variability."
    )
    rows = []
    for key, label, scale, digits, unit in (
        ("full_online_seconds_per_query", "Rerank all, online time", 1.0, 3, "s/query"),
        ("adaptive_online_seconds_per_query", "Adaptive Hit@1, online time", 1.0, 3, "s/query"),
        ("adaptive_gate_seconds_per_query", "Adaptive Hit@1, gate time", 1000.0, 2, "ms/query"),
        ("adaptive_reranked_share", "Reranked queries", 100.0, 1, "\\%"),
        ("relative_online_saving", "Relative online saving", 100.0, 1, "\\%"),
    ):
        values = statistics[key]
        rows.append(
            f"{label} ({unit}) & {scale * values['mean']:.{digits}f} & {scale * values['sd']:.{digits}f} & "
            f"{scale * values['min']:.{digits}f} & {scale * values['max']:.{digits}f} \\\\"
        )
    label = path.stem.replace("table_", "tab:")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l|rrrr}",
        "\\toprule",
        "Quantity & Mean & SD & Min & Max \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nested_subset_dirs(output_dir: Path) -> list[Path]:
    return sorted(path for path in Path(output_dir).glob("subset_*") if (path / "aggregated.json").exists())


def cost_summary(setup: dict[str, Any] | None, aggregated: dict[str, Any] | None, setup_path: Path, aggregated_path: Path) -> dict[str, Any]:
    online: dict[str, Any] = {"status": "pending", "source": str(aggregated_path)}
    if aggregated is not None:
        online = {
            "source": str(aggregated_path),
            "definition": "online = gate (features and prediction) + reranking of the selected queries; full = reranking of all queries",
            "workload_queries": aggregated["workload"]["queries"],
            "measured_repetitions": aggregated["workload"]["measured_repetitions"],
            "variability": VARIABILITY_LABEL,
            "full_online_seconds_per_query": {
                "mean": aggregated["strategies"]["full"]["online_per_query_mean"],
                "sd": aggregated["strategies"]["full"]["online_per_query_sd"],
            },
            "adaptive_online_seconds_per_query": {
                "mean": aggregated["strategies"]["adaptive"]["online_per_query_mean"],
                "sd": aggregated["strategies"]["adaptive"]["online_per_query_sd"],
            },
            "adaptive_gate_seconds_per_query": {
                "mean": aggregated["strategies"]["adaptive"].get("gate_per_query_mean"),
                "sd": aggregated["strategies"]["adaptive"].get("gate_per_query_sd"),
            },
            "relative_online_saving": {"mean": aggregated["online_saving"]["mean"], "sd": aggregated["online_saving"]["sd"]},
        }
    deployment: dict[str, Any] = {"status": "pending", "source": str(setup_path)}
    if setup is not None:
        deployment = {
            "source": str(setup_path),
            "definition": "one-time setup needed to deploy the already trained predictor",
            "components_seconds_mean": {name: values["mean"] for name, values in setup["components"].items()},
            "total_seconds_mean": float(np.mean(setup["total_seconds"])),
            "repetitions": setup["repetitions"],
        }
    return {
        "shared_first_stage_costs": {"status": "excluded (shared by both strategies)", "items": SHARED_FIRST_STAGE_EXCLUDED},
        "deployment_setup_cost": deployment,
        "online_cost": online,
        "break_even": break_even(setup, aggregated),
        "model_development_costs": {"status": "excluded", "items": DEVELOPMENT_COSTS_EXCLUDED},
    }


def write_available_tables(
    runtime,
    output_dir: Path,
    language: str,
    retriever: str,
    predictor_name: str,
    log,
    parent_dir: Path | None = None,
) -> None:
    paths = table_paths(runtime, language, retriever, predictor_name)
    if parent_dir is None and nested_subset_dirs(output_dir):
        child = nested_subset_dirs(output_dir)[0]
        log(f"The thesis tables are written from the repeated nested-subset run in {child}; regenerating them from there")
        write_available_tables(runtime, child, language, retriever, predictor_name, log, parent_dir=output_dir)
        return
    aggregated_path = output_dir / "aggregated.json"
    setup_path = (parent_dir if parent_dir is not None else output_dir) / "setup_cost.json"
    reference_path = parent_dir / "aggregated.json" if parent_dir is not None else None
    aggregated = read_json(aggregated_path) if aggregated_path.exists() else None
    setup = read_json(setup_path) if setup_path.exists() else None
    reference = read_json(reference_path) if reference_path is not None and reference_path.exists() else None
    if parent_dir is not None and reference is None:
        log(f"No aggregated timings of the larger subset at {reference_path} yet; caption omits that comparison")
    if aggregated is not None:
        write_online_table(paths["online"], aggregated, retriever, language, reference=reference)
        log(f"Wrote online cost table {paths['online']}")
    else:
        log(f"No aggregated online timings at {aggregated_path} yet; online table not written")
    if setup is not None:
        write_setup_table(paths["setup"], setup, aggregated, retriever)
        log(f"Wrote setup cost table {paths['setup']}" + ("" if aggregated else " (break-even pending online timings)"))
    else:
        log(f"No setup timings at {setup_path} yet; setup table not written")
    write_cost_summary(output_dir, log, parent_dir=parent_dir)


def write_cost_summary(output_dir: Path, log, parent_dir: Path | None = None) -> dict[str, Any]:
    aggregated_path = Path(output_dir) / "aggregated.json"
    setup_path = (Path(parent_dir) if parent_dir is not None else Path(output_dir)) / "setup_cost.json"
    aggregated = read_json(aggregated_path) if aggregated_path.exists() else None
    setup = read_json(setup_path) if setup_path.exists() else None
    summary = cost_summary(setup, aggregated, setup_path, aggregated_path)
    write_json(Path(output_dir) / "cost_summary.json", summary)
    log(f"Wrote cost categories and break-even to {Path(output_dir) / 'cost_summary.json'}: break-even {summary['break_even'].get('queries')}")
    return summary


ONLINE_COMPONENTS = (
    COMPONENT_BM25,
    COMPONENT_TRAINING_RR,
    COMPONENT_PERTURBATION,
    COMPONENT_REMAINING_AND_PREDICTION,
    COMPONENT_GATE_TOTAL,
    COMPONENT_RERANK,
    COMPONENT_ONLINE_TOTAL,
)
