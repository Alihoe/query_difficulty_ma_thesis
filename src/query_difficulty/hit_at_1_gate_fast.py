from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import faiss
import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew

from query_difficulty.dataset import tokenize
from query_difficulty.features import RANK_BIASED_OVERLAP_P, _rank_biased_overlap_ceiling
from query_difficulty.retrieval import _top_k

PERTURBATION_COUNT = 30
PERTURBATION_NOISE_POWER_RATIO = 0.06
TRAINING_REFERENCE_NEIGHBOR_COUNT = 10
TRAINING_REFERENCE_MINIMUM_SEARCH_COUNT = 32

FEATURE_BM25_TOP_1_AGREEMENT = "bm25_top_1_agreement"
FEATURE_TRAINING_RR = "distance_weighted_training_query_rr_at_k"
FEATURE_PERTURBATION = "perturbation_ranking_stability"
FEATURE_MAX_PAIRWISE_COSINE = "maximum_top_k_pairwise_cosine_similarity"
FEATURE_TOP_1_COSINE = "top_1_cosine_similarity"
FEATURE_SKEWNESS = "top_k_cosine_skewness"
FEATURE_STANDARD_DEVIATION = "top_k_cosine_standard_deviation"
FEATURE_CLOSE_COMPETITOR = "close_competitor_fraction"

GATE_FEATURES = frozenset({
    FEATURE_BM25_TOP_1_AGREEMENT,
    FEATURE_TRAINING_RR,
    FEATURE_PERTURBATION,
    FEATURE_MAX_PAIRWISE_COSINE,
    FEATURE_TOP_1_COSINE,
    FEATURE_SKEWNESS,
    FEATURE_STANDARD_DEVIATION,
    FEATURE_CLOSE_COMPETITOR,
})

COMPONENT_BM25 = "bm25_top_1"
COMPONENT_TRAINING_RR = "training_query_rr"
COMPONENT_PERTURBATION = "perturbation_stability"
COMPONENT_REMAINING_AND_PREDICTION = "remaining_features_and_prediction"
COMPONENT_GATE_TOTAL = "gate_total"
COMPONENT_RERANK = "rerank"
COMPONENT_ONLINE_TOTAL = "online_total"
GATE_COMPONENTS = (COMPONENT_BM25, COMPONENT_TRAINING_RR, COMPONENT_PERTURBATION, COMPONENT_REMAINING_AND_PREDICTION)


@dataclass(slots=True)
class GateModel:
    pipeline: Any
    threshold: float
    feature_depth: int
    qualifying_features: frozenset[str]
    feature_columns: list[str]
    positive_index: int
    model_path: Path


def load_gate_model(model_path: Path) -> GateModel:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    qualifying = frozenset(str(name) for name in payload["qualifying_features"])
    unsupported = sorted(qualifying - GATE_FEATURES)
    if unsupported:
        raise ValueError(f"The fast gate does not implement the feature(s) {unsupported} required by {model_path}")
    classes = [int(value) for value in pipeline.classes_]
    if 1 not in classes:
        raise ValueError(f"The model at {model_path} has no positive class among {classes}")
    return GateModel(
        pipeline=pipeline,
        threshold=float(payload["threshold"]),
        feature_depth=int(payload["feature_depth"]),
        qualifying_features=qualifying,
        feature_columns=sorted(qualifying),
        positive_index=classes.index(1),
        model_path=Path(model_path),
    )


@dataclass(slots=True)
class Bm25TermIndex:
    vocabulary: dict[str, int]
    indptr: np.ndarray
    document_positions: np.ndarray
    contributions: np.ndarray
    document_count: int
    ranking_depth: int


def build_bm25_term_index(model: Any, ranking_depth: int) -> Bm25TermIndex:
    doc_len = np.array(model.doc_len)
    length_normalization = model.k1 * (1 - model.b + model.b * doc_len / model.avgdl)
    vocabulary = {term: position for position, term in enumerate(model.idf.keys())}
    term_positions: list[int] = []
    document_positions: list[int] = []
    frequencies: list[int] = []
    for document_position, frequencies_by_term in enumerate(model.doc_freqs):
        for term, frequency in frequencies_by_term.items():
            term_position = vocabulary.get(term)
            if term_position is None:
                continue
            term_positions.append(term_position)
            document_positions.append(document_position)
            frequencies.append(frequency)
    term_array = np.asarray(term_positions, dtype=np.int64)
    document_array = np.asarray(document_positions, dtype=np.int64)
    frequency_array = np.asarray(frequencies, dtype=np.int64)
    order = np.lexsort((document_array, term_array))
    term_array = term_array[order]
    document_array = document_array[order]
    frequency_array = frequency_array[order]
    idf_values = np.array([(model.idf.get(term) or 0) for term in vocabulary], dtype=np.float64)
    contributions = idf_values[term_array] * (
        frequency_array * (model.k1 + 1) / (frequency_array + length_normalization[document_array])
    )
    indptr = np.zeros(len(vocabulary) + 1, dtype=np.int64)
    np.add.at(indptr, term_array + 1, 1)
    indptr = np.cumsum(indptr)
    return Bm25TermIndex(
        vocabulary=vocabulary,
        indptr=indptr,
        document_positions=document_array,
        contributions=contributions,
        document_count=int(model.corpus_size),
        ranking_depth=int(min(int(ranking_depth), int(model.corpus_size))),
    )


def bm25_scores(index: Bm25TermIndex, tokens: list[str], buffer: np.ndarray | None = None) -> np.ndarray:
    scores = np.zeros(index.document_count, dtype=np.float64) if buffer is None else buffer
    if buffer is not None:
        scores.fill(0.0)
    for token in tokens:
        term_position = index.vocabulary.get(token)
        if term_position is None:
            continue
        start = index.indptr[term_position]
        end = index.indptr[term_position + 1]
        scores[index.document_positions[start:end]] += index.contributions[start:end]
    return scores


def bm25_top_1_positions(index: Bm25TermIndex, token_lists: list[list[str]]) -> np.ndarray:
    top_positions = np.empty(len(token_lists), dtype=np.int64)
    buffer = np.zeros(index.document_count, dtype=np.float64)
    for row, tokens in enumerate(token_lists):
        scores = bm25_scores(index, tokens, buffer).astype(np.float32)
        maximum = scores.max()
        tied = np.flatnonzero(scores == maximum)
        if tied.size == 1:
            top_positions[row] = int(tied[0])
        else:
            order, _ = _top_k(scores, index.ranking_depth)
            top_positions[row] = int(order[0])
    return top_positions


@dataclass(slots=True)
class TrainingReferencePool:
    index: faiss.Index
    reciprocal_ranks: np.ndarray
    keys: list[tuple[str, int]]
    positions: dict[tuple[str, int], int]


def gold_reciprocal_ranks(ranked_document_ids: np.ndarray, gold_document_ids: np.ndarray, depth: int) -> np.ndarray:
    head = np.asarray(ranked_document_ids)[:, : int(depth)]
    matches = head == np.asarray(gold_document_ids)[:, np.newaxis]
    found = matches.any(axis=1)
    ranks = matches.argmax(axis=1).astype(np.float64) + 1.0
    return np.where(found, 1.0 / ranks, 0.0).astype(np.float64)


def build_training_reference_pool(
    embeddings: list[np.ndarray],
    keys: list[list[tuple[str, int]]],
    reciprocal_ranks: list[np.ndarray],
) -> TrainingReferencePool:
    matrix = np.ascontiguousarray(np.vstack(embeddings).astype(np.float32, copy=False))
    flat_keys = [key for part in keys for key in part]
    index = faiss.IndexFlatL2(int(matrix.shape[1]))
    index.add(matrix)
    return TrainingReferencePool(
        index=index,
        reciprocal_ranks=np.concatenate(reciprocal_ranks).astype(np.float64),
        keys=flat_keys,
        positions={key: position for position, key in enumerate(flat_keys)},
    )


class _SequentialFaissDistances:
    def __init__(self, query_count: int) -> None:
        self.query_count = int(query_count)
        self.previous: int | None = None

    def __enter__(self) -> None:
        self.previous = int(faiss.cvar.distance_compute_blas_threshold)
        faiss.cvar.distance_compute_blas_threshold = self.query_count + 1

    def __exit__(self, *exc: Any) -> None:
        faiss.cvar.distance_compute_blas_threshold = int(self.previous)


def training_query_rr(
    pool: TrainingReferencePool,
    query_embeddings: np.ndarray,
    query_keys: list[tuple[str, int]],
    exact_distances: bool = True,
) -> np.ndarray:
    query_count = len(query_keys)
    result = np.full(query_count, math.nan, dtype=np.float64)
    pool_size = len(pool.keys)
    if pool_size == 0 or query_count == 0:
        return result
    self_positions = np.array([pool.positions.get(key, -1) for key in query_keys], dtype=np.int64)
    excluded = (self_positions >= 0).astype(np.int64)
    available = np.maximum(pool_size - excluded, 0)
    target_counts = np.minimum(TRAINING_REFERENCE_NEIGHBOR_COUNT, available)
    search_counts = np.minimum(pool_size, np.maximum(target_counts + excluded, TRAINING_REFERENCE_MINIMUM_SEARCH_COUNT))
    queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    for search_count in np.unique(search_counts):
        rows = np.flatnonzero(search_counts == search_count)
        if exact_distances:
            with _SequentialFaissDistances(len(rows)):
                distances, indices = pool.index.search(np.ascontiguousarray(queries[rows]), int(search_count))
        else:
            distances, indices = pool.index.search(np.ascontiguousarray(queries[rows]), int(search_count))
        valid = (indices >= 0) & (indices != self_positions[rows, np.newaxis])
        selected = valid & (np.cumsum(valid, axis=1) <= target_counts[rows, np.newaxis])
        weights_all = 1.0 / (1.0 + np.sqrt(np.maximum(distances.astype(np.float64), 0.0)))
        for local, row in enumerate(rows):
            mask = selected[local]
            if not mask.any():
                continue
            weights = weights_all[local][mask]
            total = float(weights.sum())
            if total <= 0.0 or not np.isfinite(total):
                continue
            normalized = (weights / total).astype(np.float64, copy=False)
            rr_values = pool.reciprocal_ranks[indices[local][mask]]
            result[row] = float(np.dot(rr_values, normalized))
    return result


def perturbed_query_embeddings(query_ids: np.ndarray, query_embeddings: np.ndarray, random_seed: int) -> np.ndarray:
    query_count, dimension = query_embeddings.shape
    perturbed = np.empty((query_count * PERTURBATION_COUNT, dimension), dtype=np.float32)
    for row in range(query_count):
        query_embedding = query_embeddings[row]
        power = float(np.mean(np.square(query_embedding)))
        noise_std = math.sqrt(power * PERTURBATION_NOISE_POWER_RATIO)
        generator = np.random.default_rng(int(random_seed) + int(query_ids[row]))
        perturbations = generator.normal(0.0, noise_std, size=(PERTURBATION_COUNT, len(query_embedding))).astype(np.float32)
        block = query_embedding[np.newaxis, :] + perturbations
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        perturbed[row * PERTURBATION_COUNT : (row + 1) * PERTURBATION_COUNT] = block / np.clip(norms, 1e-12, None)
    return perturbed


def normalized_rank_biased_overlap_batch(baseline: np.ndarray, candidates: np.ndarray, p: float = RANK_BIASED_OVERLAP_P) -> np.ndarray:
    depth = int(baseline.shape[-1])
    equal = baseline[:, np.newaxis, :, np.newaxis] == candidates[:, :, np.newaxis, :]
    cumulative = equal.cumsum(axis=2, dtype=np.int64).cumsum(axis=3, dtype=np.int64)
    agreement = np.zeros(candidates.shape[:2], dtype=np.float64)
    for index in range(depth):
        agreement += cumulative[:, :, index, index] / (index + 1) * (p ** index)
    rank_biased_overlap = (1 - p) * agreement
    return rank_biased_overlap / _rank_biased_overlap_ceiling(depth, p=p)


def perturbation_ranking_stability(
    query_ids: np.ndarray,
    query_embeddings: np.ndarray,
    baseline_document_ids: np.ndarray,
    document_index: faiss.Index,
    document_ids: np.ndarray,
    random_seed: int,
    depth: int,
) -> np.ndarray:
    query_count = len(query_ids)
    max_depth = int(min(baseline_document_ids.shape[1], len(document_ids), int(depth)))
    perturbed = perturbed_query_embeddings(query_ids, query_embeddings, random_seed)
    _, indices = document_index.search(perturbed, max_depth)
    perturbed_document_ids = np.asarray(document_ids)[indices].reshape(query_count, PERTURBATION_COUNT, max_depth)
    scores = normalized_rank_biased_overlap_batch(baseline_document_ids[:, :max_depth], perturbed_document_ids)
    return scores.mean(axis=1)


def maximum_pairwise_cosine(document_embeddings: np.ndarray, top_positions: np.ndarray) -> np.ndarray:
    embeddings = document_embeddings[top_positions]
    cosine = np.clip(np.matmul(embeddings, embeddings.transpose(0, 2, 1)), -1.0, 1.0)
    rows, cols = np.triu_indices(top_positions.shape[1], k=1)
    return cosine[:, rows, cols].max(axis=1).astype(np.float64)


def score_features(scores: np.ndarray, required: frozenset[str], close_competitor_threshold: float) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    features: dict[str, np.ndarray] = {}
    if FEATURE_TOP_1_COSINE in required:
        features[FEATURE_TOP_1_COSINE] = values[:, 0].copy()
    if FEATURE_CLOSE_COMPETITOR in required:
        if np.isfinite(close_competitor_threshold):
            features[FEATURE_CLOSE_COMPETITOR] = np.mean(values >= values[:, :1] - float(close_competitor_threshold), axis=1)
        else:
            features[FEATURE_CLOSE_COMPETITOR] = np.full(len(values), math.nan)
    if FEATURE_STANDARD_DEVIATION in required:
        features[FEATURE_STANDARD_DEVIATION] = np.std(values, axis=1)
    if FEATURE_SKEWNESS in required:
        if not np.isfinite(values).all():
            raise ValueError("Non-finite retrieval scores are not supported by the batched skewness computation")
        constant = np.isclose(values, values[:, :1]).all(axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = skew(values, axis=1, bias=False)
        features[FEATURE_SKEWNESS] = np.where(constant, 0.0, skewness).astype(np.float64)
    return features


CLOSE_COMPETITOR_SAMPLE_SIZE = 2048
CLOSE_COMPETITOR_RANKING_DEPTH = 100


def estimate_close_competitor_threshold(
    train_queries: pd.DataFrame,
    ranked_document_ids: np.ndarray,
    ranked_scores: np.ndarray,
) -> float:
    from query_difficulty.features import FeatureBuilder

    sample = FeatureBuilder._sample_query_frame(train_queries.reset_index(drop=True), CLOSE_COMPETITOR_SAMPLE_SIZE)
    positions = pd.Index(train_queries["query_id"].to_numpy()).get_indexer(sample["query_id"].to_numpy())
    positive: list[float] = []
    negative: list[float] = []
    for position, gold_id in zip(positions, sample["document_id"].to_numpy(dtype=np.int64), strict=True):
        similarities = np.asarray(ranked_scores[position], dtype=np.float64)[:CLOSE_COMPETITOR_RANKING_DEPTH]
        diffs = similarities[0] - similarities
        matches = np.flatnonzero(np.asarray(ranked_document_ids[position])[:CLOSE_COMPETITOR_RANKING_DEPTH] == gold_id)
        gold_rank = int(matches[0]) + 1 if matches.size else None
        if gold_rank is not None and gold_rank <= len(diffs):
            positive.append(float(diffs[gold_rank - 1]))
        mask = np.ones(len(diffs), dtype=bool)
        if gold_rank is not None and gold_rank <= len(diffs):
            mask[gold_rank - 1] = False
        negative.extend(float(value) for value in diffs[mask])
    candidates = FeatureBuilder._quantile_candidates(np.asarray(positive + negative, dtype=np.float64))
    if candidates.size == 0:
        return 0.0
    positive_array = np.asarray(positive, dtype=np.float64)
    negative_array = np.asarray(negative, dtype=np.float64)
    best = (float("-inf"), 0.0)
    for threshold in candidates:
        tpr = np.mean(positive_array <= threshold) if positive else 0.0
        fpr = np.mean(negative_array <= threshold) if negative else 0.0
        score = tpr - fpr
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


@dataclass(slots=True)
class GateState:
    model: GateModel
    document_ids: np.ndarray
    document_position_lookup: np.ndarray
    document_embeddings: np.ndarray | None
    document_index: faiss.Index | None
    reference_pool: TrainingReferencePool | None
    bm25: Bm25TermIndex | None
    close_competitor_threshold: float
    random_seed: int

    def positions_of(self, document_ids: np.ndarray) -> np.ndarray:
        positions = self.document_position_lookup[np.asarray(document_ids, dtype=np.int64)]
        if (positions < 0).any():
            raise KeyError("Encountered document ids outside the indexed collection")
        return positions


def document_position_lookup(document_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(document_ids, dtype=np.int64)
    lookup = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids), dtype=np.int64)
    return lookup


@dataclass(slots=True)
class GateInputs:
    query_ids: np.ndarray
    query_texts: list[str]
    query_keys: list[tuple[str, int]]
    query_embeddings: np.ndarray
    retrieval_document_ids: np.ndarray
    retrieval_scores: np.ndarray


@dataclass(slots=True)
class GateResult:
    features: dict[str, np.ndarray]
    probabilities: np.ndarray
    skip: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)


def run_gate(state: GateState, inputs: GateInputs, clock: Callable[[], float] = time.perf_counter) -> GateResult:
    required = state.model.qualifying_features
    depth = state.model.feature_depth
    top_document_ids = np.asarray(inputs.retrieval_document_ids)[:, :depth]
    top_scores = np.asarray(inputs.retrieval_scores)[:, :depth]
    features: dict[str, np.ndarray] = {}
    timings = {component: 0.0 for component in GATE_COMPONENTS}
    gate_started = clock()

    if FEATURE_BM25_TOP_1_AGREEMENT in required:
        started = clock()
        token_lists = [tokenize(text) for text in inputs.query_texts]
        bm25_positions = bm25_top_1_positions(state.bm25, token_lists)
        bm25_top_1_ids = state.document_ids[bm25_positions]
        features[FEATURE_BM25_TOP_1_AGREEMENT] = (top_document_ids[:, 0] == bm25_top_1_ids).astype(np.float64)
        timings[COMPONENT_BM25] = clock() - started

    if FEATURE_TRAINING_RR in required:
        started = clock()
        features[FEATURE_TRAINING_RR] = training_query_rr(state.reference_pool, inputs.query_embeddings, inputs.query_keys)
        timings[COMPONENT_TRAINING_RR] = clock() - started

    if FEATURE_PERTURBATION in required:
        started = clock()
        features[FEATURE_PERTURBATION] = perturbation_ranking_stability(
            inputs.query_ids,
            inputs.query_embeddings,
            top_document_ids,
            state.document_index,
            state.document_ids,
            state.random_seed,
            depth,
        )
        timings[COMPONENT_PERTURBATION] = clock() - started

    started = clock()
    features |= score_features(top_scores, required, state.close_competitor_threshold)
    if FEATURE_MAX_PAIRWISE_COSINE in required:
        features[FEATURE_MAX_PAIRWISE_COSINE] = maximum_pairwise_cosine(
            state.document_embeddings, state.positions_of(top_document_ids)
        )
    frame = pd.DataFrame({name: features[name] for name in state.model.feature_columns})
    probabilities = np.asarray(state.model.pipeline.predict_proba(frame), dtype=np.float64)[:, state.model.positive_index]
    skip = probabilities >= state.model.threshold
    timings[COMPONENT_REMAINING_AND_PREDICTION] = clock() - started
    timings[COMPONENT_GATE_TOTAL] = clock() - gate_started
    return GateResult(features=features, probabilities=probabilities, skip=skip, timings=timings)


def ranking_arrays(ranking_frame: pd.DataFrame, query_ids: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    head = ranking_frame.loc[ranking_frame["rank"] <= int(depth), ["query_id", "rank", "document_id", "score"]]
    head = head.sort_values(["query_id", "rank"], kind="mergesort")
    counts = head.groupby("query_id", sort=False).size()
    if (counts != int(depth)).any():
        raise ValueError(f"Every query needs exactly {depth} ranked documents")
    document_ids = head["document_id"].to_numpy(dtype=np.int64).reshape(-1, int(depth))
    scores = head["score"].to_numpy(dtype=np.float64).reshape(-1, int(depth))
    order = pd.Index(head["query_id"].to_numpy()[:: int(depth)]).get_indexer(np.asarray(query_ids))
    if (order < 0).any():
        raise KeyError("Some requested queries are missing from the ranking")
    return document_ids[order], scores[order]


def reranked_reciprocal_ranks(
    top_document_ids: np.ndarray,
    reranker_scores: np.ndarray,
    gold_document_ids: np.ndarray,
    metric_k: int,
) -> np.ndarray:
    ranks = np.broadcast_to(np.arange(1, top_document_ids.shape[1] + 1), top_document_ids.shape)
    order = np.lexsort((ranks, -np.asarray(reranker_scores, dtype=np.float64)), axis=1)
    reranked = np.take_along_axis(top_document_ids, order, axis=1)
    return gold_reciprocal_ranks(reranked, gold_document_ids, metric_k)


def load_jina_reranker(
    model_name: str,
    trust_remote_code: bool,
    use_flash_attn: bool,
    quantize_dynamic_int8: bool,
    device: str,
) -> Any:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        dtype="auto",
        trust_remote_code=trust_remote_code,
        use_flash_attn=use_flash_attn,
    )
    model.to(device)
    model.eval()
    if quantize_dynamic_int8:
        if device != "cpu":
            raise ValueError("Dynamic int8 quantization is only supported on CPU")
        model = model.float()
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


def build_pairs(
    query_ids: np.ndarray,
    top_document_ids: np.ndarray,
    query_text_by_id: dict[int, str],
    document_text_by_id: dict[int, str],
) -> list[list[str]]:
    return [
        [str(query_text_by_id[int(query_id)]), str(document_text_by_id[int(document_id)])]
        for query_id, row in zip(query_ids, top_document_ids, strict=True)
        for document_id in row
    ]


def score_pairs(
    model: Any,
    pairs: list[list[str]],
    batch_size: int,
    max_length: int,
    request_size: int,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), int(request_size)):
            batch_scores = model.compute_score(pairs[start : start + int(request_size)], batch_size=int(batch_size), max_length=int(max_length))
            outputs.append(np.asarray(batch_scores, dtype=np.float32).reshape(-1))
            if progress is not None:
                progress(min(start + int(request_size), len(pairs)), len(pairs))
    return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)


def synchronize_device(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cpu_model_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def environment_metadata(device: str) -> dict[str, Any]:
    import torch

    physical_cores = None
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        physical_cores = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": _cpu_model_name(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": physical_cores,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_quantized_engine": str(torch.backends.quantized.engine),
        "torch_mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "faiss_omp_max_threads": int(faiss.omp_get_max_threads()),
        "faiss_blas_threshold_default": int(faiss.cvar.distance_compute_blas_threshold),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "package_versions": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "scikit-learn",
                "faiss-cpu",
                "numpy",
                "scipy",
                "pandas",
                "rank-bm25",
                "joblib",
            )
        },
    }
