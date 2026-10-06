from __future__ import annotations

import math
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import faiss
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import cramervonmises, genpareto, skew
from tqdm.auto import tqdm

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.clustering import ClusterArtifacts, ClusterManager
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console
from query_difficulty.dataset import DatasetBundle, TOKEN_PATTERN
from query_difficulty.retrieval import (
    RetrievalManager,
    base_retriever_name,
    is_lexical_retriever,
    load_rankings_by_query,
    retriever_document_scope,
)
from query_difficulty.text import TextServices


def _entropy(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    positive = values[values > 0]
    if positive.size == 0:
        return math.nan
    probs = positive / positive.sum()
    return float(-(probs * np.log(probs)).sum())


def _normalized_entropy(proportions: np.ndarray) -> float:
    positive = proportions[proportions > 0]
    if positive.size < 2:
        return 0.0
    return float(-(positive * np.log(positive)).sum() / np.log(len(positive)))


def _safe_skew(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    if np.allclose(values, values[0]):
        return 0.0
    return float(skew(values, bias=False))


def _safe_slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    coeffs = np.polyfit(x, values.astype(np.float64), deg=1)
    return float(coeffs[0])


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    centered = values - np.nanmax(values)
    weights = np.exp(centered)
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


def _upper_triangle(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.array([], dtype=np.float32)
    row, col = np.triu_indices(values.shape[0], k=1)
    return values[row, col]


def _pairwise_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    pairs = _upper_triangle(values)
    if pairs.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return float(pairs.min()), float(pairs.max()), float(pairs.mean()), float(np.std(pairs))


def _dense_connected_components(adjacency: np.ndarray) -> tuple[int, np.ndarray]:
    size = int(adjacency.shape[0])
    if size == 0:
        return 0, np.asarray([], dtype=np.int32)
    labels = np.full(size, -1, dtype=np.int32)
    component_count = 0
    for start in range(size):
        if labels[start] != -1:
            continue
        labels[start] = component_count
        stack = [start]
        while stack:
            node = stack.pop()
            unlabeled = labels < 0
            neighbors = np.flatnonzero(adjacency[node] & unlabeled)
            if neighbors.size == 0:
                continue
            labels[neighbors] = component_count
            stack.extend(int(index) for index in neighbors)
        component_count += 1
    return component_count, labels


def _ratio(count: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return float(count) / float(total)


def _stream_top_document_ids_by_query(frame: pd.DataFrame, limit: int) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    current_query_id: int | None = None
    current_documents: list[int] | None = None
    current_count = 0
    query_ids = frame["query_id"].to_numpy(copy=False)
    document_ids = frame["document_id"].to_numpy(copy=False)
    for query_id, document_id in zip(query_ids, document_ids, strict=False):
        query_value = int(query_id)
        if current_query_id != query_value:
            current_query_id = query_value
            current_documents = []
            grouped[current_query_id] = current_documents
            current_count = 0
        if current_count < limit:
            current_documents.append(int(document_id))
            current_count += 1
    return grouped


RANK_BIASED_OVERLAP_P = 0.9


def _rank_biased_overlap(left: list[int], right: list[int], p: float = 0.9) -> float:
    if not left and not right:
        return 1.0
    agreement = 0.0
    seen_left: set[int] = set()
    seen_right: set[int] = set()
    depth = max(len(left), len(right))
    for index in range(depth):
        if index < len(left):
            seen_left.add(left[index])
        if index < len(right):
            seen_right.add(right[index])
        overlap = len(seen_left & seen_right)
        agreement += overlap / (index + 1) * (p ** index)
    return float((1 - p) * agreement)


def _rank_biased_overlap_ceiling(depth: int, p: float = RANK_BIASED_OVERLAP_P) -> float:
    if depth <= 0:
        return 1.0
    return float(1.0 - (p ** depth))


def _normalized_rank_biased_overlap(left: list[int], right: list[int], p: float = RANK_BIASED_OVERLAP_P) -> float:
    depth = max(len(left), len(right))
    ceiling = _rank_biased_overlap_ceiling(depth, p=p)
    if ceiling <= 0.0:
        return math.nan
    return float(_rank_biased_overlap(left, right, p=p) / ceiling)


class SurpriseCalibrator:
    def __init__(self) -> None:
        self.threshold: float | None = None
        self.shape: float | None = None
        self.scale: float | None = None

    def fit(self, scores: np.ndarray) -> bool:
        values = np.asarray(scores, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 20:
            return False
        values.sort()
        candidate_indices = range(max(5, values.size // 2), values.size - 10, max(1, values.size // 25))
        best_state: tuple[float, float, float, float] | None = None
        for index in candidate_indices:
            threshold = float(values[index])
            excess = values[index:] - threshold
            if np.allclose(excess, 0.0):
                continue
            try:
                shape, _, scale = genpareto.fit(excess, floc=0.0)
            except Exception:
                continue
            if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
                continue
            clipped_shape = float(min(shape, 1.0))
            cdf = lambda item: genpareto.cdf(item, clipped_shape, loc=0.0, scale=scale)
            try:
                statistic = float(cramervonmises(excess, cdf).statistic)
            except Exception:
                continue
            if best_state is None or statistic < best_state[0]:
                best_state = (statistic, threshold, clipped_shape, float(scale))
        if best_state is None:
            return False
        _, self.threshold, self.shape, self.scale = best_state
        return True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None or self.shape is None or self.scale is None:
            return np.full(len(scores), math.nan, dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
        shifted = np.maximum(values - self.threshold, 0.0)
        probs = genpareto.cdf(shifted, self.shape, loc=0.0, scale=self.scale)
        probs = np.clip(probs, 0.0, 1.0 - 1e-12)
        return -np.log1p(-probs)


@dataclass(slots=True)
class LexicalCollectionStats:
    vocabulary: dict[str, int]
    count_matrix: sparse.spmatrix
    binary_matrix: sparse.spmatrix
    document_lengths: np.ndarray
    document_frequency: np.ndarray
    collection_frequency: np.ndarray
    collection_probability: np.ndarray
    total_terms: int
    document_count: int
    tfidf_matrix: sparse.spmatrix
    tfidf_vocabulary: dict[str, int]


@dataclass(slots=True)
class SemanticReferencePool:
    current_split_key: str
    retriever_name: str
    reference_count: int
    index: faiss.Index | None
    query_keys: list[tuple[str, int]]
    query_index: dict[tuple[str, int], int]
    rr_by_depth: dict[int, np.ndarray]


QUERY_SIGNAL_SCHEMA_VERSION = 2
DESCRIPTIVE_FEATURE_SCHEMA_VERSION = 4
GOLD_AWARE_FEATURE_SCHEMA_VERSION = 2
PAPER_GOLD_AWARE_CANDIDATE_SIZES = (5, 10, 20, 30, 50, 60, 100, 120, 1000)
QUERY_SIGNAL_REQUIRED_COLUMNS = {
    "query_id",
    "signal_schema_version",
    "raw_token_count",
    "x_artifact_ratio",
    "x_mention_ratio",
    "stopword_ratio",
    "repeated_term_ratio",
    "numeral_ratio",
    "acronym_ratio",
    "proper_noun_ratio",
    "mean_dependency_link_span",
    "mean_query_term_polysemy",
    "sentiment_polarity",
    "sentiment_intensity",
    "entities",
}
FEATURE_FRAME_REQUIRED_COLUMNS = {
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
    "query_scope",
    "query_scope_non_stopwords",
    "proper_noun_ratio",
    "mean_dependency_link_span",
    "mean_query_term_polysemy",
    "sentiment_polarity",
    "sentiment_intensity",
}
CLUSTER_FEATURE_REQUIRED_COLUMNS = {
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
    "nearest_centroid_cosine_spherical",
    "nearest_second_nearest_centroid_margin_spherical",
    "nearest_centroid_cosine_hdbscan",
    "nearest_second_nearest_centroid_margin_hdbscan",
    "cluster_weighted_corpus_familiarity",
    "spherical_query_nearest_cluster_share_in_top_k",
    "spherical_retrieval_score_mass_of_query_nearest_cluster_in_top_k",
    "spherical_distinct_clusters_in_top_k",
    "spherical_cluster_entropy_in_top_k",
    "spherical_score_weighted_cluster_entropy_in_top_k",
    "spherical_largest_cluster_share_in_top_k",
    "spherical_dominant_cluster_enrichment_in_top_k",
    "hdbscan_query_nearest_cluster_share_in_top_k",
    "hdbscan_retrieval_score_mass_of_query_nearest_cluster_in_top_k",
    "hdbscan_distinct_clusters_in_top_k",
    "hdbscan_cluster_entropy_in_top_k",
    "hdbscan_score_weighted_cluster_entropy_in_top_k",
    "hdbscan_largest_cluster_share_in_top_k",
    "hdbscan_dominant_cluster_enrichment_in_top_k",
    "hdbscan_noise_fraction_in_top_k",
}
GOLD_AWARE_FEATURE_REQUIRED_COLUMNS = {
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
    "query_length_tokens",
    "gold_title_length_characters",
    "gold_title_length_tokens",
    "gold_abstract_length_characters",
    "gold_abstract_length_tokens",
    "gold_authors_length_characters",
    "gold_authors_length_tokens",
    "gold_venue_length_characters",
    "gold_venue_length_tokens",
    "gold_title_query_coverage",
    "gold_abstract_query_coverage",
    "gold_authors_query_coverage",
    "gold_venue_query_coverage",
    "gold_title_abstract_entity_overlap",
    "gold_query_term_coverage",
    "gold_unmatched_idf_mass_ratio",
    "x_artifact_ratio",
    "x_mention_ratio",
    "gold_document_retrieval_score",
    "top_1_gold_score_gap",
    "top_1_gold_score_ratio",
    "rank_k_gold_score_gap",
    "gold_score_top_k_z_score",
    "gold_document_surprise_score",
    "gold_document_surprise_threshold_indicator",
    "top_1_gold_query_term_coverage_gap",
    "top_1_gold_unmatched_idf_mass_gap",
    "gold_document_cosine_similarity",
    "top_1_cosine_similarity",
    "top_2_cosine_similarity",
    "top_3_cosine_similarity",
    "top_4_cosine_similarity",
    "top_5_cosine_similarity",
    "top_1_gold_cosine_gap",
    "top_1_top_10_mean_gap",
    "top_1_agreement_vs_bge_m3",
    "top_1_agreement_vs_multilingual_e5_base",
    "mean_top_k_gold_cosine_gap",
    "rank_k_gold_cosine_gap",
    "gold_cosine_top_k_z_score",
    "gold_cosine_above_top_k_mean",
    "gold_document_semantic_surprise_score",
    "gold_document_semantic_surprise_threshold_indicator",
    "gold_top_k_centroid_cosine",
    "mean_gold_top_k_document_cosine",
    "minimum_gold_top_k_document_cosine",
    "maximum_gold_top_k_document_cosine",
    "gold_top_k_cosine_standard_deviation",
    "gold_document_graph_degree",
    "gold_document_retrieval_coherency_degree",
    "gold_connected_to_largest_coherency_component",
    "spherical_gold_cluster_centroid_cosine",
    "spherical_gold_cluster_centroid_rank",
    "spherical_gold_cluster_present_in_top_k",
    "spherical_gold_cluster_share_in_top_k",
    "spherical_gold_cluster_score_mass_in_top_k",
    "spherical_gold_cluster_enrichment_in_top_k",
    "hdbscan_gold_cluster_centroid_cosine",
    "hdbscan_gold_cluster_centroid_rank",
    "hdbscan_gold_cluster_present_in_top_k",
    "hdbscan_gold_cluster_share_in_top_k",
    "hdbscan_gold_cluster_score_mass_in_top_k",
    "hdbscan_gold_cluster_enrichment_in_top_k",
    "hdbscan_gold_document_labelled_as_noise",
    "gold_document_perturbation_retention",
    "gold_document_cosine_perturbation",
}
UNIT_INTERVAL_QUERY_SIGNAL_COLUMNS = {
    "x_artifact_ratio",
    "x_mention_ratio",
    "stopword_ratio",
    "repeated_term_ratio",
    "numeral_ratio",
    "acronym_ratio",
    "proper_noun_ratio",
}
UNIT_INTERVAL_FEATURE_COLUMNS = {
    "retrieval_failure_within_max_depth",
    "gold_in_top_k",
    "proper_noun_ratio",
    "query_scope",
    "query_scope_non_stopwords",
}
QUERY_SCOPE_COLUMNS = {
    "query_scope",
    "query_scope_non_stopwords",
}


class FeatureBuilder:
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        bundle: DatasetBundle,
        retrieval: RetrievalManager,
        clusters: ClusterManager,
        text_services: TextServices,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.bundle = bundle
        self.retrieval = retrieval
        self.clusters = clusters
        self.text_services = text_services
        self.publications = bundle.publications
        self.document_lookup = self.publications.set_index("document_id", drop=False)
        self.document_position = {
            int(document_id): int(index)
            for index, document_id in enumerate(self.publications["document_id"].tolist())
        }
        self._lexical_stats_cache: dict[str, LexicalCollectionStats] = {}
        self._document_entities_cache: dict[str, dict[int, set[str]]] = {}
        self._document_entities_dirty: set[str] = set()
        self._feature_table_cache: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
        self._cluster_feature_table_cache: dict[tuple[str, str, tuple[int, ...]], dict[int, pd.DataFrame]] = {}
        self._gold_aware_feature_table_cache: dict[tuple[str, str, tuple[int, ...]], dict[int, pd.DataFrame]] = {}
        self._query_signal_cache: dict[str, pd.DataFrame] = {}
        self._query_signal_lookup_cache: dict[str, pd.DataFrame] = {}
        self._all_rankings_top100_cache: dict[tuple[str, str], dict[str, dict[int, list[int]]]] = {}
        self._semantic_state_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._ranking_lookup_cache: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}
        self._semantic_reference_pool_cache: dict[tuple[str, str], SemanticReferencePool] = {}
        self._semantic_reference_neighbor_cache: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
        self._semantic_threshold_bundle_cache: dict[str, dict[str, float]] = {}
        self._semantic_query_static_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._cluster_label_count_cache: dict[tuple[str, str], dict[int, int]] = {}
        self._surprise_score_cache: dict[tuple[str, str, int], np.ndarray] = {}
        self._surprise_calibrator_cache: dict[tuple[str, str, int], SurpriseCalibrator | None] = {}
        self._perturbation_feature_cache: dict[tuple[str, str, int], dict[int, float]] = {}
        self._threshold_cache: dict[tuple[str, str], float] = {}
        self._feature_checkpoint_batch_size = max(1, int(self.config.runtime.feature_checkpoint_batch_size))

    @property
    def lexical_stats(self) -> LexicalCollectionStats:
        return self._lexical_stats("bm25")

    @property
    def document_entities(self) -> dict[int, set[str]]:
        return self._document_entities_for_retriever("bm25")

    def _lexical_stats(self, retriever_name: str) -> LexicalCollectionStats:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        document_scope = retriever_document_scope(retriever_name)
        if document_scope not in self._lexical_stats_cache:
            self._lexical_stats_cache[document_scope] = self._lexical_collection_stats(retriever_name)
        return self._lexical_stats_cache[document_scope]

    def _document_entities_for_retriever(self, retriever_name: str) -> dict[int, set[str]]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        document_scope = retriever_document_scope(retriever_name)
        if document_scope not in self._document_entities_cache:
            self._document_entities_cache[document_scope] = self._document_entities(document_scope)
        return self._document_entities_cache[document_scope]

    def build_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        active_candidate_sizes = self._normalize_candidate_sizes(candidate_sizes)
        cache_key = (split_key, retriever_name, tuple(active_candidate_sizes))
        if cache_key in self._feature_table_cache:
            return self._feature_table_cache[cache_key]
        cached = self._load_cached_features(split_key, retriever_name, active_candidate_sizes)
        missing_candidate_sizes = [depth for depth in active_candidate_sizes if depth not in cached]
        if not missing_candidate_sizes:
            self.console.log(f"Using cached feature tables for {split_key} / {retriever_name}")
            self._feature_table_cache[cache_key] = cached
            return cached
        self.console.log(f"Generating feature tables for {split_key} / {retriever_name}")
        self._active_language = split_key.split("_", 1)[0]
        queries = self.bundle.queries[split_key]
        if self.config.runtime.limit_queries is not None:
            queries = queries.head(self.config.runtime.limit_queries).copy()
        query_ids = queries["query_id"].astype(int).tolist()
        checkpoint = self._prepare_feature_checkpoint(split_key, retriever_name, query_ids, missing_candidate_sizes)
        if checkpoint["processed_queries"] > 0:
            self.console.log(
                f"Resuming feature generation for {split_key} / {retriever_name} "
                f"from {checkpoint['processed_queries']} of {checkpoint['total_queries']} queries"
            )
        if checkpoint["processed_queries"] >= checkpoint["total_queries"]:
            frames = self._finalize_feature_checkpoint(
                split_key,
                retriever_name,
                checkpoint["batches"],
                checkpoint["total_queries"],
                missing_candidate_sizes,
            )
            self._persist_document_entities()
            combined = self._merge_feature_frames(active_candidate_sizes, cached, frames)
            self._feature_table_cache[cache_key] = combined
            return combined
        ranking_frame = self.retrieval.ensure_rankings(split_key, retriever_name)
        rankings = self._rankings_by_query(split_key, retriever_name, ranking_frame=ranking_frame)
        query_signals = self._query_signal_lookup(split_key)
        semantic = not is_lexical_retriever(retriever_name)
        all_rankings_top100 = {}
        semantic_state = None
        semantic_thresholds: dict[str, float] | None = None
        if semantic:
            depth_text = ", ".join(str(depth) for depth in missing_candidate_sizes)
            self.console.log(
                f"Preparing semantic prerequisites for {split_key} / {retriever_name}: "
                f"{len(self.retrieval.enabled_retrievers())} agreement ranking views, clusters, and embeddings"
            )
            prerequisites_start = perf_counter()
            all_rankings_top100 = self._all_rankings_top100(
                split_key, retriever_document_scope(retriever_name), target_retriever_name=retriever_name
            )
            semantic_state = self._semantic_state(split_key, retriever_name)
            semantic_thresholds = self._semantic_feature_thresholds(retriever_name)
            prerequisites_elapsed = perf_counter() - prerequisites_start
            self.console.log(
                f"Prepared semantic prerequisites for {split_key} / {retriever_name} "
                f"in {prerequisites_elapsed:.1f}s"
            )
            self.console.log(
                f"Starting semantic feature extraction for {split_key} / {retriever_name}: "
                f"{len(queries)} queries across candidate sizes [{depth_text}]"
            )
            if checkpoint["processed_queries"] == 0:
                self.console.log(
                    f"First progress update for {split_key} / {retriever_name} appears "
                    f"after the first query completes"
                )
        lexical_artifacts = self.retrieval.ensure_lexical_artifacts(retriever_name) if not semantic else None
        if not semantic:
            self._ensure_document_entities(retriever_name)
        pending_rows: dict[int, list[dict[str, Any]]] = {depth: [] for depth in missing_candidate_sizes}
        pending_query_ids: list[int] = []
        first_pending_query_index = checkpoint["processed_queries"]
        first_pending_query_logged = False
        progress = tqdm(
            total=len(queries),
            desc=f"{split_key} {retriever_name} features",
            unit="query",
            initial=checkpoint["processed_queries"],
            **self.console.progress_kwargs(len(queries)),
        )
        try:
            for query_index, query in enumerate(queries.itertuples(index=False)):
                if query_index < checkpoint["processed_queries"]:
                    continue
                query_id = int(query.query_id)
                if semantic and not first_pending_query_logged and query_index == first_pending_query_index:
                    self.console.log(
                        f"Computing first pending semantic query for {split_key} / {retriever_name} "
                        f"(query_id={query_id})"
                    )
                    first_pending_query_logged = True
                language = str(query.language)
                signal_row = query_signals.loc[query_id]
                ranking = rankings[query_id]
                gold_rank = self._gold_rank(ranking, int(query.document_id))
                smallest_size = self._smallest_sufficient_size(gold_rank)
                shared = {
                    "query_id": query_id,
                    "document_id": int(query.document_id),
                    "language": language,
                    "split_name": split_key,
                    "retriever_name": retriever_name,
                    "gold_rank": gold_rank if gold_rank is not None else math.nan,
                    "smallest_sufficient_size": smallest_size if smallest_size is not None else math.nan,
                    "retrieval_failure_within_max_depth": float(gold_rank is None),
                    "rr_at_max_depth": 0.0 if gold_rank is None else 1.0 / gold_rank,
                }
                lexical_pre = self._lexical_pre_features(query, signal_row, retriever_name)
                if not semantic:
                    full_scores = self.retrieval._score_lexical_query(lexical_artifacts, list(query.query_tokens))
                    lexical_context = self._lexical_query_context(query, ranking, full_scores, retriever_name)
                    for depth in missing_candidate_sizes:
                        row = dict(shared)
                        row["candidate_size"] = depth
                        row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
                        row |= lexical_pre
                        row |= self._lexical_post_features(query, lexical_context, depth, retriever_name)
                        pending_rows[depth].append(row)
                else:
                    semantic_context = self._semantic_query_context(
                        query,
                        ranking,
                        retriever_name,
                        semantic_state,
                        all_rankings_top100,
                        threshold_bundle=semantic_thresholds,
                    )
                    semantic_context["candidate_depths"] = tuple(int(depth) for depth in missing_candidate_sizes)
                    semantic_pre_static = self._semantic_query_static_state(query, retriever_name, semantic_state)["pre_features"]
                    agreement_features = self._retriever_agreement_features(retriever_name, semantic_context, all_rankings_top100)
                    for depth in missing_candidate_sizes:
                        row = dict(shared)
                        row["candidate_size"] = depth
                        row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
                        row |= lexical_pre
                        row |= semantic_pre_static
                        row["distance_weighted_training_query_rr_at_k"] = self._distance_weighted_training_rr(
                            query,
                            retriever_name,
                            depth,
                            semantic_context["query_embedding"],
                        )
                        row |= self._semantic_score_features(semantic_context, depth, retriever_name)
                        row |= self._semantic_geometry_features(semantic_context, depth, retriever_name)
                        row |= self._semantic_cluster_features(semantic_context, depth, retriever_name)
                        row |= agreement_features
                        row |= self._perturbation_features(query_id, depth, semantic_state, semantic_context)
                        pending_rows[depth].append(row)
                pending_query_ids.append(query_id)
                progress.update(1)
                if semantic and query_index == first_pending_query_index:
                    self.console.log(
                        f"Completed first pending semantic query for {split_key} / {retriever_name}; "
                        f"progress bar now reflects per-query throughput"
                    )
                self._discard_query_feature_caches(split_key, retriever_name, query_id)
                if len(pending_query_ids) >= self._feature_checkpoint_batch_size:
                    batch_info = self._store_feature_checkpoint_batch(
                        split_key,
                        retriever_name,
                        missing_candidate_sizes,
                        checkpoint["next_batch_id"],
                        pending_query_ids,
                        pending_rows,
                        checkpoint["processed_queries"] + len(pending_query_ids),
                        checkpoint["total_queries"],
                    )
                    checkpoint["batches"].append(batch_info)
                    checkpoint["processed_queries"] += batch_info["query_count"]
                    checkpoint["next_batch_id"] = int(batch_info["batch_id"]) + 1
                    pending_rows = {depth: [] for depth in missing_candidate_sizes}
                    pending_query_ids = []
        finally:
            progress.close()
        if pending_query_ids:
            batch_info = self._store_feature_checkpoint_batch(
                split_key,
                retriever_name,
                missing_candidate_sizes,
                checkpoint["next_batch_id"],
                pending_query_ids,
                pending_rows,
                checkpoint["processed_queries"] + len(pending_query_ids),
                checkpoint["total_queries"],
            )
            checkpoint["batches"].append(batch_info)
            checkpoint["processed_queries"] += batch_info["query_count"]
        frames = self._finalize_feature_checkpoint(
            split_key,
            retriever_name,
            checkpoint["batches"],
            checkpoint["total_queries"],
            missing_candidate_sizes,
        )
        self._persist_document_entities()
        combined = self._merge_feature_frames(active_candidate_sizes, cached, frames)
        self._feature_table_cache[cache_key] = combined
        return combined

    def build_cluster_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        active_candidate_sizes = self._normalize_candidate_sizes(candidate_sizes)
        cache_key = (split_key, retriever_name, tuple(active_candidate_sizes))
        if cache_key in self._cluster_feature_table_cache:
            return self._cluster_feature_table_cache[cache_key]

        cached = self._load_cached_cluster_features(split_key, retriever_name, active_candidate_sizes)
        missing_candidate_sizes = [depth for depth in active_candidate_sizes if depth not in cached]
        if not missing_candidate_sizes:
            self.console.log(f"Using cached cluster feature tables for {split_key} / {retriever_name}")
            self._cluster_feature_table_cache[cache_key] = cached
            return cached

        if is_lexical_retriever(retriever_name):
            raise ValueError(f"Cluster features are only defined for semantic retrievers, got '{retriever_name}'")

        self.console.log(f"Generating cluster feature tables for {split_key} / {retriever_name}")
        ranking_frame = self.retrieval.ensure_rankings(split_key, retriever_name)
        rankings = self._rankings_by_query(split_key, retriever_name, ranking_frame=ranking_frame)
        query_frame = self._limited_queries(split_key).reset_index(drop=True)
        _, document_embeddings = self.retrieval.ensure_semantic_embeddings(retriever_name)
        query_embeddings = self.retrieval.ensure_query_embeddings(split_key, retriever_name)
        spherical = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "spherical_kmeans")
        hdbscan = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "hdbscan")

        rows_by_depth: dict[int, list[dict[str, Any]]] = {depth: [] for depth in missing_candidate_sizes}
        progress = tqdm(
            total=len(query_frame),
            desc=f"{split_key} {retriever_name} cluster features",
            unit="query",
            **self.console.progress_kwargs(len(query_frame)),
        )
        try:
            for position, query in enumerate(query_frame.itertuples(index=False)):
                query_id = int(query.query_id)
                ranking = rankings[query_id]
                query_embedding = query_embeddings[position]
                similarities = ranking["score"].to_numpy(dtype=np.float64)
                top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
                context = {
                    "query_embedding": query_embedding,
                    "similarities": similarities,
                    "top_document_ids": top_document_ids,
                    "spherical": spherical,
                    "hdbscan": hdbscan,
                }
                gold_rank = self._gold_rank(ranking, int(query.document_id))
                smallest_size = self._smallest_sufficient_size(gold_rank)
                cluster_pre = self._cluster_pre_features(query_embedding, spherical, hdbscan)
                shared = {
                    "query_id": query_id,
                    "document_id": int(query.document_id),
                    "language": str(query.language),
                    "split_name": split_key,
                    "retriever_name": retriever_name,
                    "gold_rank": gold_rank if gold_rank is not None else math.nan,
                    "smallest_sufficient_size": smallest_size if smallest_size is not None else math.nan,
                    "retrieval_failure_within_max_depth": float(gold_rank is None),
                    "rr_at_max_depth": 0.0 if gold_rank is None else 1.0 / gold_rank,
                }
                for depth in missing_candidate_sizes:
                    row = dict(shared)
                    row["candidate_size"] = depth
                    row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
                    row |= cluster_pre
                    row |= self._semantic_cluster_features(context, depth, retriever_name)
                    rows_by_depth[depth].append(row)
                progress.update(1)
        finally:
            progress.close()

        generated = {
            depth: pd.DataFrame(rows).reset_index(drop=True).copy()
            for depth, rows in rows_by_depth.items()
        }
        self._write_cluster_feature_tables(split_key, retriever_name, generated)
        combined = self._merge_feature_frames(active_candidate_sizes, cached, generated)
        self._cluster_feature_table_cache[cache_key] = combined
        return combined

    def build_gold_aware_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        active_candidate_sizes = self._normalize_gold_aware_candidate_sizes(retriever_name, candidate_sizes)
        cache_key = (split_key, retriever_name, tuple(active_candidate_sizes))
        if cache_key in self._gold_aware_feature_table_cache:
            return self._gold_aware_feature_table_cache[cache_key]

        self._active_language = split_key.split("_", 1)[0]
        cached = self._load_cached_gold_aware_features(split_key, retriever_name, active_candidate_sizes)
        missing_candidate_sizes = [depth for depth in active_candidate_sizes if depth not in cached]
        if not missing_candidate_sizes:
            self.console.log(f"Using cached gold-aware feature tables for {split_key} / {retriever_name}")
            self._gold_aware_feature_table_cache[cache_key] = cached
            return cached

        query_frame = self._limited_queries(split_key).reset_index(drop=True)
        query_ids = query_frame["query_id"].astype(int).tolist()
        checkpoint = self._prepare_feature_checkpoint(
            split_key,
            retriever_name,
            query_ids,
            missing_candidate_sizes,
            checkpoint_namespace="gold_aware",
            feature_schema_version=GOLD_AWARE_FEATURE_SCHEMA_VERSION,
            checkpoint_label="gold-aware feature",
        )
        if checkpoint["processed_queries"] > 0:
            self.console.log(
                f"Resuming gold-aware feature generation for {split_key} / {retriever_name} "
                f"from {checkpoint['processed_queries']} of {checkpoint['total_queries']} queries"
            )
        if checkpoint["processed_queries"] >= checkpoint["total_queries"]:
            generated = self._finalize_feature_checkpoint(
                split_key,
                retriever_name,
                checkpoint["batches"],
                checkpoint["total_queries"],
                missing_candidate_sizes,
                checkpoint_namespace="gold_aware",
                checkpoint_label="gold-aware feature",
                frame_writer=self._write_gold_aware_feature_tables,
            )
            combined = self._merge_feature_frames(active_candidate_sizes, cached, generated)
            self._gold_aware_feature_table_cache[cache_key] = combined
            return combined

        self.console.log(f"Generating gold-aware feature tables for {split_key} / {retriever_name}")
        ranking_frame = self.retrieval.ensure_rankings(split_key, retriever_name)
        rankings = self._rankings_by_query(split_key, retriever_name, ranking_frame=ranking_frame)
        query_signals = self._query_signal_lookup(split_key)
        peer_top_1_by_retriever = self._gold_aware_peer_top_1_by_query(split_key, retriever_name)
        semantic = not is_lexical_retriever(retriever_name)
        lexical_artifacts = self.retrieval.ensure_lexical_artifacts(retriever_name) if not semantic else None
        semantic_state = None
        semantic_thresholds: dict[str, float] | None = None
        if semantic:
            semantic_state = self._semantic_state(split_key, retriever_name)
            semantic_thresholds = self._semantic_feature_thresholds(retriever_name)
        self._ensure_document_entities(retriever_name)

        pending_rows: dict[int, list[dict[str, Any]]] = {depth: [] for depth in missing_candidate_sizes}
        pending_query_ids: list[int] = []
        progress = tqdm(
            total=len(query_frame),
            desc=f"{split_key} {retriever_name} gold-aware features",
            unit="query",
            initial=checkpoint["processed_queries"],
            **self.console.progress_kwargs(len(query_frame)),
        )
        try:
            for position, query in enumerate(query_frame.itertuples(index=False)):
                if position < checkpoint["processed_queries"]:
                    continue
                query_id = int(query.query_id)
                gold_document_id = int(query.document_id)
                ranking = rankings[query_id]
                similarities = ranking["score"].to_numpy(dtype=np.float64)
                top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
                gold_rank = self._gold_rank(ranking, gold_document_id)
                smallest_size = self._smallest_sufficient_size(gold_rank)
                signal_row = query_signals.loc[query_id]
                gold_document = self.document_lookup.loc[gold_document_id]
                lexical_static = None
                semantic_context = None
                semantic_static = None
                lexical_context = None
                paper_features = self._gold_aware_paper_analysis_features(
                    query=query,
                    signal_row=signal_row,
                    retriever_name=retriever_name,
                    similarities=similarities,
                    current_top_1_document_id=int(top_document_ids[0]),
                    gold_document=gold_document,
                    peer_top_1_by_retriever=peer_top_1_by_retriever,
                    include_semantic_signals=semantic,
                )
                if semantic:
                    semantic_context = self._gold_aware_semantic_query_context(
                        query,
                        ranking,
                        retriever_name,
                        semantic_state,
                        threshold_bundle=semantic_thresholds,
                    )
                    semantic_context["candidate_depths"] = tuple(int(depth) for depth in missing_candidate_sizes)
                    semantic_static = self._gold_aware_semantic_static_features(semantic_context, gold_document_id)
                else:
                    full_scores = self.retrieval._score_lexical_query(lexical_artifacts, list(query.query_tokens))
                    lexical_context = self._lexical_query_context(query, ranking, full_scores, retriever_name)
                    lexical_static = self._gold_aware_lexical_static_features(
                        query,
                        lexical_context,
                        retriever_name,
                        gold_document_id,
                    )
                shared = {
                    "query_id": query_id,
                    "document_id": gold_document_id,
                    "language": str(query.language),
                    "split_name": split_key,
                    "retriever_name": retriever_name,
                    "gold_rank": gold_rank if gold_rank is not None else math.nan,
                    "smallest_sufficient_size": smallest_size if smallest_size is not None else math.nan,
                    "retrieval_failure_within_max_depth": float(gold_rank is None),
                    "rr_at_max_depth": 0.0 if gold_rank is None else 1.0 / gold_rank,
                }
                for depth in missing_candidate_sizes:
                    row = dict(shared)
                    row["candidate_size"] = depth
                    row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
                    row |= paper_features
                    if semantic:
                        row |= self._empty_gold_aware_lexical_ranking_features()
                        row |= semantic_static
                        row |= self._gold_aware_semantic_features(semantic_context, depth)
                        row |= self._gold_aware_semantic_geometry_features(semantic_context, depth)
                        row |= self._gold_aware_cluster_features(semantic_context, gold_document_id, depth, "spherical_")
                        row |= self._gold_aware_cluster_features(semantic_context, gold_document_id, depth, "hdbscan_")
                        row |= self._gold_aware_perturbation_features(query_id, gold_document_id, depth, semantic_context)
                    else:
                        row |= lexical_static
                        row |= self._gold_aware_lexical_depth_features(lexical_context, gold_document_id, depth)
                        row |= self._empty_gold_aware_semantic_features()
                        row |= self._empty_gold_aware_semantic_geometry_features()
                        row |= self._empty_gold_aware_cluster_features("spherical_")
                        row |= self._empty_gold_aware_cluster_features("hdbscan_")
                        row |= self._empty_gold_aware_perturbation_features()
                    pending_rows[depth].append(row)
                pending_query_ids.append(query_id)
                progress.update(1)
                self._discard_query_feature_caches(split_key, retriever_name, query_id)
                if len(pending_query_ids) >= self._feature_checkpoint_batch_size:
                    batch_info = self._store_feature_checkpoint_batch(
                        split_key,
                        retriever_name,
                        missing_candidate_sizes,
                        checkpoint["next_batch_id"],
                        pending_query_ids,
                        pending_rows,
                        checkpoint["processed_queries"] + len(pending_query_ids),
                        checkpoint["total_queries"],
                        checkpoint_namespace="gold_aware",
                        checkpoint_label="gold-aware feature",
                    )
                    checkpoint["batches"].append(batch_info)
                    checkpoint["processed_queries"] += batch_info["query_count"]
                    checkpoint["next_batch_id"] = int(batch_info["batch_id"]) + 1
                    pending_rows = {depth: [] for depth in missing_candidate_sizes}
                    pending_query_ids = []
        finally:
            progress.close()

        if pending_query_ids:
            batch_info = self._store_feature_checkpoint_batch(
                split_key,
                retriever_name,
                missing_candidate_sizes,
                checkpoint["next_batch_id"],
                pending_query_ids,
                pending_rows,
                checkpoint["processed_queries"] + len(pending_query_ids),
                checkpoint["total_queries"],
                checkpoint_namespace="gold_aware",
                checkpoint_label="gold-aware feature",
            )
            checkpoint["batches"].append(batch_info)
            checkpoint["processed_queries"] += batch_info["query_count"]
        generated = self._finalize_feature_checkpoint(
            split_key,
            retriever_name,
            checkpoint["batches"],
            checkpoint["total_queries"],
            missing_candidate_sizes,
            checkpoint_namespace="gold_aware",
            checkpoint_label="gold-aware feature",
            frame_writer=self._write_gold_aware_feature_tables,
        )
        combined = self._merge_feature_frames(active_candidate_sizes, cached, generated)
        self._gold_aware_feature_table_cache[cache_key] = combined
        return combined

    def _normalize_candidate_sizes(self, candidate_sizes: list[int] | tuple[int, ...] | None) -> list[int]:
        allowed = {int(depth) for depth in self.config.candidate_sizes}
        requested = [int(depth) for depth in (candidate_sizes or self.config.candidate_sizes)]
        invalid = [depth for depth in requested if depth not in allowed]
        if invalid:
            raise ValueError(f"Unsupported candidate sizes requested: {invalid}")
        ordered: list[int] = []
        for depth in self.config.candidate_sizes:
            if int(depth) in requested and int(depth) not in ordered:
                ordered.append(int(depth))
        return ordered

    def _normalize_gold_aware_candidate_sizes(
        self,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None,
    ) -> list[int]:
        if candidate_sizes is not None:
            return self._normalize_candidate_sizes(candidate_sizes)
        if is_lexical_retriever(retriever_name):
            allowed = {int(depth) for depth in self.config.candidate_sizes}
            defaults = [int(depth) for depth in PAPER_GOLD_AWARE_CANDIDATE_SIZES if int(depth) in allowed]
            if defaults:
                return defaults
        return self._normalize_candidate_sizes(None)

    @staticmethod
    def _merge_feature_frames(
        candidate_sizes: list[int],
        cached: dict[int, pd.DataFrame],
        generated: dict[int, pd.DataFrame],
    ) -> dict[int, pd.DataFrame]:
        frames = {}
        for depth in candidate_sizes:
            if depth in generated:
                frames[int(depth)] = generated[int(depth)]
            elif depth in cached:
                frames[int(depth)] = cached[int(depth)]
        return frames

    def _load_cached_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
    ) -> dict[int, pd.DataFrame]:
        frames = {}
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth in candidate_sizes:
            path = self.store.path("features", "descriptive", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            if path.exists():
                metadata_path = self._feature_table_metadata_path(split_key, retriever_name, int(depth))
                if not metadata_path.exists():
                    self.console.log(
                        f"Cached feature table for {split_key} / {retriever_name} at depth {int(depth)} "
                        "is missing feature-schema metadata. Recomputing this depth."
                    )
                    continue
                metadata = self.store.read_json(metadata_path)
                if int(metadata.get("feature_schema_version", 0)) != DESCRIPTIVE_FEATURE_SCHEMA_VERSION:
                    self.console.log(
                        f"Cached feature table for {split_key} / {retriever_name} at depth {int(depth)} uses "
                        f"feature schema version {int(metadata.get('feature_schema_version', 0))}, but "
                        f"version {DESCRIPTIVE_FEATURE_SCHEMA_VERSION} is required. Recomputing this depth."
                    )
                    continue
                frame = self.store.read_parquet(path)
                try:
                    self._validate_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
                except ValueError as error:
                    self.console.log(
                        f"Cached feature table for {split_key} / {retriever_name} at depth {int(depth)} failed validation: "
                        f"{error}. Recomputing this depth."
                    )
                    continue
                frames[int(depth)] = frame
        return frames

    def _load_cached_cluster_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
    ) -> dict[int, pd.DataFrame]:
        frames = {}
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth in candidate_sizes:
            path = self.store.path("features", "semantic_cluster", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            if not path.exists():
                continue
            frame = self.store.read_parquet(path)
            try:
                self._validate_cluster_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
            except ValueError as error:
                self.console.log(
                    f"Cached cluster feature table for {split_key} / {retriever_name} at depth {int(depth)} failed validation: "
                    f"{error}. Recomputing this depth."
                )
                continue
            frames[int(depth)] = frame
        return frames

    def _load_cached_gold_aware_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
    ) -> dict[int, pd.DataFrame]:
        frames = {}
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth in candidate_sizes:
            path = self.store.path("features", "gold_aware", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            metadata_path = self._gold_aware_feature_table_metadata_path(split_key, retriever_name, int(depth))
            if not path.exists() or not metadata_path.exists():
                continue
            metadata = self.store.read_json(metadata_path)
            if int(metadata.get("feature_schema_version", 0)) != GOLD_AWARE_FEATURE_SCHEMA_VERSION:
                self.console.log(
                    f"Cached gold-aware feature table for {split_key} / {retriever_name} at depth {int(depth)} uses "
                    f"feature schema version {int(metadata.get('feature_schema_version', 0))}, but "
                    f"version {GOLD_AWARE_FEATURE_SCHEMA_VERSION} is required. Recomputing this depth."
                )
                continue
            frame = self.store.read_parquet(path)
            try:
                self._validate_gold_aware_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
            except ValueError as error:
                self.console.log(
                    f"Cached gold-aware feature table for {split_key} / {retriever_name} at depth {int(depth)} failed validation: "
                    f"{error}. Recomputing this depth."
                )
                continue
            frames[int(depth)] = frame
        return frames

    @staticmethod
    def _candidate_size_signature(candidate_sizes: list[int]) -> str:
        return "depths_" + "_".join(f"{int(depth):04d}" for depth in candidate_sizes)

    def _feature_checkpoint_root(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        checkpoint_namespace: str | None = None,
    ) -> Path:
        root = self.store.root_dir / "features" / "checkpoints"
        if checkpoint_namespace:
            root = root / checkpoint_namespace
        return root / split_key / retriever_name / self._candidate_size_signature(candidate_sizes)

    def _feature_checkpoint_state_path(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        checkpoint_namespace: str | None = None,
    ) -> Path:
        return self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / "state.json"

    def _feature_checkpoint_query_ids_path(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        checkpoint_namespace: str | None = None,
    ) -> Path:
        return self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / "query_ids.json"

    def _feature_checkpoint_batch_metadata_path(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        batch_id: int,
        checkpoint_namespace: str | None = None,
    ) -> Path:
        return self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / "batches" / f"batch_{batch_id:05d}.json"

    def _feature_checkpoint_batch_frame_path(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        batch_id: int,
        checkpoint_namespace: str | None = None,
    ) -> Path:
        return self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / "batch_frames" / f"batch_{batch_id:05d}.parquet"

    def _feature_checkpoint_batch_path(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        depth: int,
        batch_id: int,
        checkpoint_namespace: str | None = None,
    ) -> Path:
        return self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / f"depth_{depth:04d}" / f"batch_{batch_id:05d}.parquet"

    def _clear_feature_checkpoint(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        checkpoint_namespace: str | None = None,
    ) -> None:
        root = self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        if root.exists():
            shutil.rmtree(root)

    def _prepare_feature_checkpoint(
        self,
        split_key: str,
        retriever_name: str,
        query_ids: list[int],
        candidate_sizes: list[int],
        *,
        checkpoint_namespace: str | None = None,
        feature_schema_version: int = DESCRIPTIVE_FEATURE_SCHEMA_VERSION,
        checkpoint_label: str = "feature",
        plan_identity: Any = None,
    ) -> dict[str, Any]:
        state_path = self._feature_checkpoint_state_path(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        query_ids_path = self._feature_checkpoint_query_ids_path(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        if state_path.exists() and query_ids_path.exists():
            try:
                state = self.store.read_json(state_path)
                saved_query_ids = [int(value) for value in self.store.read_json(query_ids_path)]
            except Exception:
                self._clear_feature_checkpoint(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
            else:
                if self._feature_checkpoint_matches(
                    state,
                    split_key,
                    retriever_name,
                    query_ids,
                    saved_query_ids,
                    candidate_sizes,
                    feature_schema_version=feature_schema_version,
                    plan_identity=plan_identity,
                ):
                    batches = self._feature_checkpoint_batches(
                        split_key,
                        retriever_name,
                        candidate_sizes,
                        checkpoint_namespace=checkpoint_namespace,
                    )
                    processed_queries = sum(int(batch["query_count"]) for batch in batches)
                    if processed_queries <= len(query_ids):
                        return {
                            "processed_queries": processed_queries,
                            "total_queries": len(query_ids),
                            "batches": batches,
                            "next_batch_id": len(batches) + 1,
                        }
                self.console.log(
                    f"Discarding stale {checkpoint_label} checkpoint for {split_key} / {retriever_name} "
                    f"for candidate sizes {candidate_sizes}"
                )
                self._clear_feature_checkpoint(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        state = {
            "split_key": split_key,
            "retriever_name": retriever_name,
            "candidate_sizes": [int(depth) for depth in candidate_sizes],
            "batch_size": self._feature_checkpoint_batch_size,
            "storage_format": "combined_parquet_v2",
            "feature_schema_version": int(feature_schema_version),
            "plan_identity": plan_identity,
            "total_queries": len(query_ids),
            "processed_queries": 0,
            "completed_batches": 0,
        }
        self.store.write_json(state_path, state)
        self.store.write_json(query_ids_path, [int(query_id) for query_id in query_ids])
        return {
            "processed_queries": 0,
            "total_queries": len(query_ids),
            "batches": [],
            "next_batch_id": 1,
        }

    def _feature_checkpoint_matches(
        self,
        state: dict[str, Any],
        split_key: str,
        retriever_name: str,
        query_ids: list[int],
        saved_query_ids: list[int],
        candidate_sizes: list[int],
        *,
        feature_schema_version: int = DESCRIPTIVE_FEATURE_SCHEMA_VERSION,
        plan_identity: Any = None,
    ) -> bool:
        return (
            state.get("split_key") == split_key
            and state.get("retriever_name") == retriever_name
            and list(state.get("candidate_sizes", [])) == [int(depth) for depth in candidate_sizes]
            and int(state.get("feature_schema_version", 0)) == int(feature_schema_version)
            and state.get("plan_identity") == plan_identity
            and saved_query_ids == [int(query_id) for query_id in query_ids]
        )

    def _feature_table_metadata_path(self, split_key: str, retriever_name: str, depth: int) -> Path:
        return self.store.root_dir / "features" / "descriptive" / split_key / retriever_name / f"depth_{depth:04d}_metadata.json"

    def _gold_aware_feature_table_metadata_path(self, split_key: str, retriever_name: str, depth: int) -> Path:
        return self.store.root_dir / "features" / "gold_aware" / split_key / retriever_name / f"depth_{depth:04d}_metadata.json"

    def _feature_checkpoint_batches(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        checkpoint_namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        metadata_dir = self._feature_checkpoint_root(split_key, retriever_name, candidate_sizes, checkpoint_namespace) / "batches"
        if not metadata_dir.exists():
            return []
        batches: list[dict[str, Any]] = []
        expected_batch_id = 1
        for path in sorted(metadata_dir.glob("batch_*.json")):
            try:
                metadata = self.store.read_json(path)
            except Exception:
                break
            batch_id = int(metadata.get("batch_id", expected_batch_id))
            if batch_id != expected_batch_id:
                break
            if not self._feature_checkpoint_batch_exists(
                split_key,
                retriever_name,
                candidate_sizes,
                batch_id,
                checkpoint_namespace=checkpoint_namespace,
            ):
                break
            metadata["batch_id"] = batch_id
            metadata["query_count"] = int(metadata["query_count"])
            batches.append(metadata)
            expected_batch_id += 1
        return batches

    def _feature_checkpoint_batch_exists(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        batch_id: int,
        checkpoint_namespace: str | None = None,
    ) -> bool:
        if self._feature_checkpoint_batch_frame_path(
            split_key,
            retriever_name,
            candidate_sizes,
            batch_id,
            checkpoint_namespace,
        ).exists():
            return True
        return all(
            self._feature_checkpoint_batch_path(
                split_key,
                retriever_name,
                candidate_sizes,
                depth,
                batch_id,
                checkpoint_namespace,
            ).exists()
            for depth in candidate_sizes
        )

    def _read_feature_checkpoint_batch_frame(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        batch_id: int,
        checkpoint_namespace: str | None = None,
    ) -> pd.DataFrame:
        batch_frame_path = self._feature_checkpoint_batch_frame_path(
            split_key,
            retriever_name,
            candidate_sizes,
            batch_id,
            checkpoint_namespace,
        )
        if batch_frame_path.exists():
            return self.store.read_parquet(batch_frame_path)
        parts = [
            self.store.read_parquet(
                self._feature_checkpoint_batch_path(
                    split_key,
                    retriever_name,
                    candidate_sizes,
                    depth,
                    batch_id,
                    checkpoint_namespace,
                )
            )
            for depth in candidate_sizes
        ]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def _store_feature_checkpoint_batch(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        batch_id: int,
        batch_query_ids: list[int],
        rows_by_depth: dict[int, list[dict[str, Any]]],
        processed_queries: int,
        total_queries: int,
        *,
        checkpoint_namespace: str | None = None,
        checkpoint_label: str = "feature",
    ) -> dict[str, Any]:
        ordered_rows: list[dict[str, Any]] = []
        for depth in candidate_sizes:
            ordered_rows.extend(rows_by_depth[depth])
        frame = pd.DataFrame(ordered_rows)
        self.store.write_parquet(
            self._feature_checkpoint_batch_frame_path(
                split_key,
                retriever_name,
                candidate_sizes,
                batch_id,
                checkpoint_namespace,
            ),
            frame,
        )
        metadata = {
            "batch_id": batch_id,
            "query_count": len(batch_query_ids),
            "first_query_id": int(batch_query_ids[0]),
            "last_query_id": int(batch_query_ids[-1]),
            "processed_queries": processed_queries,
            "storage_format": "combined_parquet_v2",
        }
        self.store.write_json(
            self._feature_checkpoint_batch_metadata_path(
                split_key,
                retriever_name,
                candidate_sizes,
                batch_id,
                checkpoint_namespace,
            ),
            metadata,
        )
        state_path = self._feature_checkpoint_state_path(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        state = self.store.read_json(state_path)
        state["processed_queries"] = processed_queries
        state["completed_batches"] = batch_id
        state["storage_format"] = "combined_parquet_v2"
        self.store.write_json(state_path, state)
        previous_processed = processed_queries - len(batch_query_ids)
        previous_percent = int((previous_processed * 100) / total_queries) if total_queries else 100
        current_percent = int((processed_queries * 100) / total_queries) if total_queries else 100
        if current_percent > previous_percent or processed_queries >= total_queries:
            self.console.log(
                f"Stored {checkpoint_label} checkpoint batch {batch_id:05d} for {split_key} / {retriever_name}: "
                f"{processed_queries}/{total_queries} queries ({current_percent}%)"
            )
        return metadata

    def _finalize_feature_checkpoint(
        self,
        split_key: str,
        retriever_name: str,
        batches: list[dict[str, Any]],
        total_queries: int,
        candidate_sizes: list[int],
        *,
        checkpoint_namespace: str | None = None,
        checkpoint_label: str = "feature",
        frame_writer: Callable[[str, str, dict[int, pd.DataFrame]], None] | None = None,
    ) -> dict[int, pd.DataFrame]:
        processed_queries = sum(int(batch["query_count"]) for batch in batches)
        if processed_queries != total_queries:
            raise ValueError(
                f"{checkpoint_label} checkpoint for {split_key} / {retriever_name} is incomplete: "
                f"{processed_queries} of {total_queries} queries"
            )
        if batches:
            self.console.log(
                f"Finalizing {checkpoint_label} tables for {split_key} / {retriever_name} from {len(batches)} checkpoint batches"
            )
        combined_parts = [
            self._read_feature_checkpoint_batch_frame(
                split_key,
                retriever_name,
                candidate_sizes,
                int(batch["batch_id"]),
                checkpoint_namespace=checkpoint_namespace,
            )
            for batch in batches
        ]
        combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
        frames = {
            depth: combined.loc[combined["candidate_size"] == depth].reset_index(drop=True).copy()
            for depth in candidate_sizes
        }
        writer = frame_writer or self._write_feature_tables
        writer(split_key, retriever_name, frames)
        self._clear_feature_checkpoint(split_key, retriever_name, candidate_sizes, checkpoint_namespace)
        return frames

    def _write_feature_tables(self, split_key: str, retriever_name: str, frames: dict[int, pd.DataFrame]) -> None:
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth, frame in frames.items():
            self._validate_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
            descriptive_path = self.store.path("features", "descriptive", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            efficient_path = self.store.path("features", "efficient", split_key, retriever_name, f"depth_{depth:04d}.npz")
            columns_path = self.store.path("features", "efficient", split_key, retriever_name, f"depth_{depth:04d}_columns.json")
            metadata_path = self._feature_table_metadata_path(split_key, retriever_name, int(depth))
            self.store.write_parquet(descriptive_path, frame)
            numeric_columns = [column for column in frame.columns if column not in {"language", "split_name", "retriever_name"}]
            array = frame[numeric_columns].to_numpy(dtype=np.float32, copy=True)
            self.store.write_npz(efficient_path, values=array)
            self.store.write_json(columns_path, numeric_columns)
            self.store.write_json(
                metadata_path,
                {
                    "feature_schema_version": DESCRIPTIVE_FEATURE_SCHEMA_VERSION,
                    "split_key": split_key,
                    "retriever_name": retriever_name,
                    "candidate_size": int(depth),
                    "numeric_columns": numeric_columns,
                },
            )
        depth_text = ", ".join(str(int(depth)) for depth in sorted(frames))
        self.console.log(
            f"Validated and wrote feature tables for {split_key} / {retriever_name} at candidate sizes [{depth_text}]"
        )

    def _write_cluster_feature_tables(self, split_key: str, retriever_name: str, frames: dict[int, pd.DataFrame]) -> None:
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth, frame in frames.items():
            self._validate_cluster_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
            descriptive_path = self.store.path("features", "semantic_cluster", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            efficient_path = self.store.path(
                "features",
                "semantic_cluster_efficient",
                split_key,
                retriever_name,
                f"depth_{depth:04d}.npz",
            )
            columns_path = self.store.path(
                "features",
                "semantic_cluster_efficient",
                split_key,
                retriever_name,
                f"depth_{depth:04d}_columns.json",
            )
            self.store.write_parquet(descriptive_path, frame)
            numeric_columns = [column for column in frame.columns if column not in {"language", "split_name", "retriever_name"}]
            array = frame[numeric_columns].to_numpy(dtype=np.float32, copy=True)
            self.store.write_npz(efficient_path, values=array)
            self.store.write_json(columns_path, numeric_columns)
        depth_text = ", ".join(str(int(depth)) for depth in sorted(frames))
        self.console.log(
            f"Validated and wrote cluster feature tables for {split_key} / {retriever_name} at candidate sizes [{depth_text}]"
        )

    def _write_gold_aware_feature_tables(self, split_key: str, retriever_name: str, frames: dict[int, pd.DataFrame]) -> None:
        expected_query_ids = self._limited_queries(split_key)["query_id"].astype(int).tolist()
        for depth, frame in frames.items():
            self._validate_gold_aware_feature_frame(split_key, retriever_name, int(depth), frame, expected_query_ids)
            descriptive_path = self.store.path("features", "gold_aware", split_key, retriever_name, f"depth_{depth:04d}.parquet")
            efficient_path = self.store.path("features", "gold_aware_efficient", split_key, retriever_name, f"depth_{depth:04d}.npz")
            columns_path = self.store.path("features", "gold_aware_efficient", split_key, retriever_name, f"depth_{depth:04d}_columns.json")
            metadata_path = self._gold_aware_feature_table_metadata_path(split_key, retriever_name, int(depth))
            self.store.write_parquet(descriptive_path, frame)
            numeric_columns = [column for column in frame.columns if column not in {"language", "split_name", "retriever_name"}]
            array = frame[numeric_columns].to_numpy(dtype=np.float32, copy=True)
            self.store.write_npz(efficient_path, values=array)
            self.store.write_json(columns_path, numeric_columns)
            self.store.write_json(
                metadata_path,
                {
                    "feature_schema_version": GOLD_AWARE_FEATURE_SCHEMA_VERSION,
                    "split_key": split_key,
                    "retriever_name": retriever_name,
                    "candidate_size": int(depth),
                    "numeric_columns": numeric_columns,
                },
            )
        depth_text = ", ".join(str(int(depth)) for depth in sorted(frames))
        self.console.log(
            f"Validated and wrote gold-aware feature tables for {split_key} / {retriever_name} at candidate sizes [{depth_text}]"
        )

    def _discard_query_feature_caches(self, split_key: str, retriever_name: str, query_id: int) -> None:
        resolved_name = self.retrieval.resolve_retriever_name(retriever_name)
        query_key = (split_key, resolved_name, int(query_id))
        self._semantic_reference_neighbor_cache.pop(query_key, None)
        self._semantic_query_static_cache.pop(query_key, None)
        self._perturbation_feature_cache.pop(query_key, None)
        self._surprise_score_cache.pop(("lexical", resolved_name, int(query_id)), None)
        self._surprise_score_cache.pop(("semantic", resolved_name, int(query_id)), None)
        if hasattr(self, "_surprise_calibrator_cache"):
            self._surprise_calibrator_cache.pop(("lexical", resolved_name, int(query_id)), None)
            self._surprise_calibrator_cache.pop(("semantic", resolved_name, int(query_id)), None)

    def _lexical_collection_stats(self, retriever_name: str) -> LexicalCollectionStats:
        artifacts = self.retrieval.ensure_lexical_artifacts(self.retrieval.paired_retriever("keyword_overlap", retriever_name))
        tfidf_artifacts = self.retrieval.ensure_lexical_artifacts(self.retrieval.paired_retriever("tfidf", retriever_name))
        count_matrix = artifacts.count_matrix.tocsr()
        binary_matrix = artifacts.binary_matrix.tocsr()
        document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel().astype(np.int32)
        collection_frequency = np.asarray(count_matrix.sum(axis=0)).ravel().astype(np.int64)
        total_terms = int(collection_frequency.sum())
        collection_probability = np.where(collection_frequency > 0, collection_frequency / total_terms, 0.0)
        return LexicalCollectionStats(
            vocabulary=artifacts.count_vectorizer.vocabulary_,
            count_matrix=count_matrix,
            binary_matrix=binary_matrix,
            document_lengths=np.asarray(count_matrix.sum(axis=1)).ravel().astype(np.int32),
            document_frequency=document_frequency,
            collection_frequency=collection_frequency,
            collection_probability=collection_probability,
            total_terms=total_terms,
            document_count=count_matrix.shape[0],
            tfidf_matrix=tfidf_artifacts.matrix.tocsr(),
            tfidf_vocabulary=tfidf_artifacts.vectorizer.vocabulary_,
        )

    def _document_entities(self, document_scope: str) -> dict[int, set[str]]:
        filename = "document_entities.parquet" if document_scope == "title_abstract" else f"document_entities__{document_scope}.parquet"
        path = self.store.path("corpus", filename)
        if path.exists():
            frame = self.store.read_parquet(path)
            return {int(row.document_id): set(row.entities) for row in frame.itertuples(index=False)}
        return {}

    def _document_entity_set(self, document_id: int, retriever_name: str = "bm25") -> set[str]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        document_scope = retriever_document_scope(retriever_name)
        entities = self._document_entities_for_retriever(retriever_name)
        if document_id in entities:
            return entities[document_id]
        fields = self.retrieval.document_scope_bundle(retriever_name)["texts"]
        values = self.text_services.entity_set(fields[self.document_position[int(document_id)]], "en")
        entities[int(document_id)] = values
        self._document_entities_dirty.add(document_scope)
        return values

    def _persist_document_entities(self) -> None:
        for document_scope in list(self._document_entities_dirty):
            entities = self._document_entities_cache.get(document_scope)
            if entities is None:
                continue
            filename = "document_entities.parquet" if document_scope == "title_abstract" else f"document_entities__{document_scope}.parquet"
            path = self.store.path("corpus", filename)
            records = [
                {"document_id": int(document_id), "entities": sorted(values)}
                for document_id, values in sorted(entities.items())
            ]
            self.store.write_parquet(path, pd.DataFrame(records))
            self._document_entities_dirty.discard(document_scope)

    def _ensure_document_entities(self, retriever_name: str) -> None:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        document_scope = retriever_document_scope(retriever_name)
        entities = self._document_entities_for_retriever(retriever_name)
        missing_ids = [int(document_id) for document_id in self.publications["document_id"].tolist() if int(document_id) not in entities]
        if not missing_ids:
            return
        self.console.log(f"Computing document entity cache for {len(missing_ids)} publications with {retriever_name}")
        batch_size = 128
        scope_texts = self.retrieval.document_scope_bundle(retriever_name)["texts"]
        progress = tqdm(
            total=len(missing_ids),
            desc=f"{retriever_name} document entities",
            unit="doc",
            **self.console.progress_kwargs(len(missing_ids)),
        )
        try:
            for start in range(0, len(missing_ids), batch_size):
                end = min(start + batch_size, len(missing_ids))
                batch_ids = missing_ids[start:end]
                texts = [scope_texts[self.document_position[int(document_id)]] for document_id in batch_ids]
                batch_entities = self.text_services.entity_sets(texts, "en", batch_size=batch_size)
                for document_id, values in zip(batch_ids, batch_entities, strict=True):
                    entities[int(document_id)] = values
                self._document_entities_dirty.add(document_scope)
                progress.update(len(batch_ids))
        finally:
            progress.close()
        self._persist_document_entities()

    def _query_signals(self, split_key: str) -> pd.DataFrame:
        if split_key in self._query_signal_cache:
            return self._query_signal_cache[split_key]
        queries = self._limited_queries(split_key)
        expected_query_ids = queries["query_id"].astype(int).tolist()
        use_persistent_cache = self.config.runtime.limit_queries is None
        path = self.store.path("queries", f"{split_key}_signals.parquet")
        if use_persistent_cache and path.exists():
            self.console.log(f"Validating cached query linguistic signals for {split_key}")
            frame = self.store.read_parquet(path)
            if self._query_signal_cache_matches(frame, expected_query_ids):
                try:
                    self._validate_query_signal_frame(split_key, frame, expected_query_ids)
                except ValueError as error:
                    self.console.log(
                        f"Cached query linguistic signals for {split_key} failed validation: {error}. Recomputing."
                    )
                else:
                    self.console.log(
                        f"Using validated query linguistic signals for {split_key} ({len(frame)} queries)"
                    )
                    self._query_signal_cache[split_key] = frame
                    return frame
            else:
                self.console.log(f"Cached query linguistic signals for {split_key} do not match the current query ids. Recomputing.")
        self.console.log(
            f"Computing query linguistic signals for {split_key} with spaCy POS/DEP/NER, WordNet polysemy, and sentiment"
        )
        records = []
        batch_size = 64
        progress = tqdm(
            total=len(queries),
            desc=f"{split_key} query signals",
            unit="query",
            **self.console.progress_kwargs(len(queries)),
        )
        try:
            for start in range(0, len(queries), batch_size):
                end = min(start + batch_size, len(queries))
                batch = queries.iloc[start:end].reset_index(drop=True)
                texts = batch["text"].fillna("").tolist()
                token_lists = batch["query_tokens"].tolist()
                language = str(batch["language"].iloc[0]) if not batch.empty else split_key.split("_", 1)[0]
                signals_batch = self.text_services.analyze_queries(texts, language, token_lists, batch_size=batch_size)
                for query, signals in zip(batch.itertuples(index=False), signals_batch, strict=True):
                    raw_tokens = TOKEN_PATTERN.findall(query.text or "")
                    repeated = sum(max(value - 1, 0) for value in Counter(query.query_tokens).values())
                    mention_count = sum(1 for token in raw_tokens if token.startswith("@"))
                    hashtag_count = sum(1 for token in raw_tokens if token.startswith("#"))
                    url_count = sum(1 for token in raw_tokens if token.lower().startswith(("http", "www")))
                    rt_count = sum(1 for token in raw_tokens if token == "RT")
                    numeral_count = sum(1 for token in raw_tokens if any(character.isdigit() for character in token))
                    acronym_count = sum(1 for token in raw_tokens if token.isupper() and len(token) > 1 and token.isalpha())
                    records.append(
                        {
                            "query_id": int(query.query_id),
                            "signal_schema_version": QUERY_SIGNAL_SCHEMA_VERSION,
                            "raw_token_count": len(raw_tokens),
                            "x_artifact_ratio": _ratio(mention_count + hashtag_count + url_count + rt_count, len(query.query_tokens)),
                            "x_mention_ratio": _ratio(mention_count, len(query.query_tokens)),
                            "stopword_ratio": signals.stopword_ratio,
                            "repeated_term_ratio": _ratio(repeated, len(query.query_tokens)),
                            "numeral_ratio": _ratio(numeral_count, len(query.query_tokens)),
                            "acronym_ratio": _ratio(acronym_count, len(query.query_tokens)),
                            "proper_noun_ratio": signals.proper_noun_ratio,
                            "mean_dependency_link_span": signals.dependency_link_span,
                            "mean_query_term_polysemy": signals.mean_polysemy,
                            "sentiment_polarity": signals.sentiment_polarity,
                            "sentiment_intensity": signals.sentiment_intensity,
                            "entities": sorted(signals.entities),
                        }
                    )
                progress.update(len(batch))
        finally:
            progress.close()
        frame = pd.DataFrame(records)
        self._validate_query_signal_frame(split_key, frame, expected_query_ids)
        if use_persistent_cache:
            self.store.write_parquet(path, frame)
        self.console.log(f"Computed and validated query linguistic signals for {split_key} ({len(frame)} queries)")
        self._query_signal_cache[split_key] = frame
        return frame

    @staticmethod
    def _query_signal_cache_matches(frame: pd.DataFrame, expected_query_ids: list[int]) -> bool:
        if "query_id" not in frame.columns:
            return False
        cached_query_ids = frame["query_id"].astype(int).tolist()
        return cached_query_ids == expected_query_ids

    def _validate_query_signal_frame(
        self,
        split_key: str,
        frame: pd.DataFrame,
        expected_query_ids: list[int],
    ) -> None:
        missing_columns = sorted(QUERY_SIGNAL_REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"missing columns {missing_columns}")
        if not self._query_signal_cache_matches(frame, expected_query_ids):
            raise ValueError("query ids do not match the current split")
        duplicate_count = int(frame["query_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"duplicate query ids detected ({duplicate_count} rows)")
        version_values = {
            int(value)
            for value in pd.to_numeric(frame["signal_schema_version"], errors="coerce").dropna().tolist()
        }
        if version_values != {QUERY_SIGNAL_SCHEMA_VERSION}:
            raise ValueError(
                f"signal schema version mismatch; expected {QUERY_SIGNAL_SCHEMA_VERSION}, found {sorted(version_values) or ['missing']}"
            )
        self._validate_numeric_column(frame, "raw_token_count", minimum=0.0)
        self._validate_numeric_column(frame, "mean_dependency_link_span", minimum=0.0)
        for column in sorted(UNIT_INTERVAL_QUERY_SIGNAL_COLUMNS):
            self._validate_numeric_column(frame, column, minimum=0.0, maximum=1.0)
        language = split_key.split("_", 1)[0]
        if language == "en":
            self._validate_numeric_column(frame, "mean_query_term_polysemy", minimum=1.0)
            self._validate_numeric_column(frame, "sentiment_polarity", minimum=-1.0, maximum=1.0)
            self._validate_numeric_column(frame, "sentiment_intensity", minimum=0.0, maximum=1.0)

    def _validate_feature_frame(
        self,
        split_key: str,
        retriever_name: str,
        depth: int,
        frame: pd.DataFrame,
        expected_query_ids: list[int] | None = None,
    ) -> None:
        missing_columns = sorted(FEATURE_FRAME_REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"missing columns {missing_columns}")
        expected_ids = expected_query_ids or self._limited_queries(split_key)["query_id"].astype(int).tolist()
        actual_ids = frame["query_id"].astype(int).tolist()
        if actual_ids != expected_ids:
            raise ValueError("query ids or query ordering do not match the current split")
        duplicate_count = int(frame["query_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"duplicate feature rows detected for query ids ({duplicate_count} duplicates)")
        self._validate_numeric_column(frame, "candidate_size", minimum=float(depth), maximum=float(depth))
        self._validate_numeric_column(frame, "rr_at_max_depth", minimum=0.0, maximum=1.0)
        for column in sorted(UNIT_INTERVAL_FEATURE_COLUMNS):
            allow_na = frame[column].isna().to_numpy() if column in QUERY_SCOPE_COLUMNS else None
            self._validate_numeric_column(frame, column, minimum=0.0, maximum=1.0, allow_na=allow_na)
        self._validate_numeric_column(frame, "mean_dependency_link_span", minimum=0.0)
        failure_mask = frame["retrieval_failure_within_max_depth"].astype(float).eq(1.0)
        self._validate_numeric_column(frame, "gold_rank", minimum=1.0, allow_na=failure_mask)
        self._validate_numeric_column(frame, "smallest_sufficient_size", minimum=1.0, allow_na=failure_mask)
        unexpected_rr_zero = (~failure_mask) & frame["rr_at_max_depth"].astype(float).le(0.0)
        if bool(unexpected_rr_zero.any()):
            raise ValueError(
                f"rr_at_max_depth contains non-positive values for retrieved gold documents ({int(unexpected_rr_zero.sum())} rows)"
            )
        language = split_key.split("_", 1)[0]
        if language == "en":
            self._validate_numeric_column(frame, "mean_query_term_polysemy", minimum=1.0)
            self._validate_numeric_column(frame, "sentiment_polarity", minimum=-1.0, maximum=1.0)
            self._validate_numeric_column(frame, "sentiment_intensity", minimum=0.0, maximum=1.0)
        minimum_depth = int(min(self.config.candidate_sizes))
        if depth == minimum_depth:
            expected_empty_columns = ["mean_score_decrease", "mean_cosine_decrease", "preceding_candidate_set_coherence_ratio", "query_adjusted_coherence_ratio"]
            for column in expected_empty_columns:
                if column in frame.columns and not frame[column].isna().all():
                    raise ValueError(f"{column} is expected to be fully empty at depth {depth}")
        if "top_1_rank_k_score_ratio" in frame.columns and "score_at_rank_k" in frame.columns:
            zero_boundary_mask = frame["score_at_rank_k"].astype(float).eq(0.0)
            self._validate_numeric_column(frame, "top_1_rank_k_score_ratio", minimum=1.0, allow_na=zero_boundary_mask)
        numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        for column in numeric_columns:
            series = frame[column]
            if series.dropna().empty:
                continue
            values = series.astype(float).to_numpy(dtype=np.float64, copy=False)
            if np.isinf(values[~np.isnan(values)]).any():
                raise ValueError(f"{column} contains infinite values")

    def _validate_cluster_feature_frame(
        self,
        split_key: str,
        retriever_name: str,
        depth: int,
        frame: pd.DataFrame,
        expected_query_ids: list[int] | None = None,
    ) -> None:
        missing_columns = sorted(CLUSTER_FEATURE_REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"missing columns {missing_columns}")
        expected_ids = expected_query_ids or self._limited_queries(split_key)["query_id"].astype(int).tolist()
        actual_ids = frame["query_id"].astype(int).tolist()
        if actual_ids != expected_ids:
            raise ValueError("query ids or query ordering do not match the current split")
        duplicate_count = int(frame["query_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"duplicate cluster feature rows detected for query ids ({duplicate_count} duplicates)")
        self._validate_numeric_column(frame, "candidate_size", minimum=float(depth), maximum=float(depth))
        self._validate_numeric_column(frame, "rr_at_max_depth", minimum=0.0, maximum=1.0)
        self._validate_numeric_column(frame, "retrieval_failure_within_max_depth", minimum=0.0, maximum=1.0)
        self._validate_numeric_column(frame, "gold_in_top_k", minimum=0.0, maximum=1.0)
        failure_mask = frame["retrieval_failure_within_max_depth"].astype(float).eq(1.0)
        self._validate_numeric_column(frame, "gold_rank", minimum=1.0, allow_na=failure_mask)
        self._validate_numeric_column(frame, "smallest_sufficient_size", minimum=1.0, allow_na=failure_mask)

    def _validate_gold_aware_feature_frame(
        self,
        split_key: str,
        retriever_name: str,
        depth: int,
        frame: pd.DataFrame,
        expected_query_ids: list[int] | None = None,
    ) -> None:
        missing_columns = sorted(GOLD_AWARE_FEATURE_REQUIRED_COLUMNS - set(frame.columns))
        if missing_columns:
            raise ValueError(f"missing columns {missing_columns}")
        expected_ids = expected_query_ids or self._limited_queries(split_key)["query_id"].astype(int).tolist()
        actual_ids = frame["query_id"].astype(int).tolist()
        if actual_ids != expected_ids:
            raise ValueError("query ids or query ordering do not match the current split")
        duplicate_count = int(frame["query_id"].duplicated().sum())
        if duplicate_count:
            raise ValueError(f"duplicate gold-aware feature rows detected for query ids ({duplicate_count} duplicates)")
        self._validate_numeric_column(frame, "candidate_size", minimum=float(depth), maximum=float(depth))
        self._validate_numeric_column(frame, "rr_at_max_depth", minimum=0.0, maximum=1.0)
        self._validate_numeric_column(frame, "retrieval_failure_within_max_depth", minimum=0.0, maximum=1.0)
        self._validate_numeric_column(frame, "gold_in_top_k", minimum=0.0, maximum=1.0)
        lexical_mask = np.ones(len(frame), dtype=bool) if is_lexical_retriever(retriever_name) else None
        self._validate_numeric_column(
            frame,
            "spherical_gold_cluster_present_in_top_k",
            minimum=0.0,
            maximum=1.0,
            allow_na=lexical_mask,
        )
        self._validate_numeric_column(
            frame,
            "hdbscan_gold_cluster_present_in_top_k",
            minimum=0.0,
            maximum=1.0,
            allow_na=lexical_mask,
        )
        self._validate_numeric_column(
            frame,
            "hdbscan_gold_document_labelled_as_noise",
            minimum=0.0,
            maximum=1.0,
            allow_na=lexical_mask,
        )
        failure_mask = frame["retrieval_failure_within_max_depth"].astype(float).eq(1.0)
        self._validate_numeric_column(frame, "gold_rank", minimum=1.0, allow_na=failure_mask)
        self._validate_numeric_column(frame, "smallest_sufficient_size", minimum=1.0, allow_na=failure_mask)

    @staticmethod
    def _validate_numeric_column(
        frame: pd.DataFrame,
        column: str,
        minimum: float | None = None,
        maximum: float | None = None,
        allow_na: pd.Series | np.ndarray | None = None,
    ) -> None:
        series = frame[column].astype(float)
        values = series.to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(values)
        not_na_mask = ~series.isna().to_numpy()
        if np.any(not_na_mask & ~finite_mask):
            raise ValueError(f"{column} contains non-finite values")
        if allow_na is None:
            allowed_na_mask = np.zeros(len(frame), dtype=bool)
        else:
            allowed_na_mask = np.asarray(allow_na, dtype=bool)
            if allowed_na_mask.shape[0] != len(frame):
                raise ValueError(f"allow_na mask for {column} has incorrect length")
        unexpected_na_mask = series.isna().to_numpy() & ~allowed_na_mask
        if np.any(unexpected_na_mask):
            raise ValueError(f"{column} contains unexpected NaN values ({int(np.sum(unexpected_na_mask))} rows)")
        if minimum is not None:
            below_mask = not_na_mask & (values < float(minimum))
            if np.any(below_mask):
                raise ValueError(f"{column} contains values below {minimum}")
        if maximum is not None:
            above_mask = not_na_mask & (values > float(maximum))
            if np.any(above_mask):
                raise ValueError(f"{column} contains values above {maximum}")

    def _query_signal_lookup(self, split_key: str) -> pd.DataFrame:
        if split_key in self._query_signal_lookup_cache:
            return self._query_signal_lookup_cache[split_key]
        lookup = self._query_signals(split_key).set_index("query_id", drop=False)
        self._query_signal_lookup_cache[split_key] = lookup
        return lookup

    def _rankings_by_query(
        self,
        split_key: str,
        retriever_name: str,
        ranking_frame: pd.DataFrame | None = None,
    ) -> dict[int, pd.DataFrame]:
        cache_key = (split_key, self.retrieval.resolve_retriever_name(retriever_name))
        if cache_key in self._ranking_lookup_cache:
            return self._ranking_lookup_cache[cache_key]
        frame = ranking_frame if ranking_frame is not None else self.retrieval.ensure_rankings(split_key, retriever_name)
        lookup = load_rankings_by_query(frame)
        self._ranking_lookup_cache[cache_key] = lookup
        return lookup

    @staticmethod
    def _sample_query_frame(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(frame) <= limit:
            return frame
        positions = np.linspace(0, len(frame) - 1, num=limit, dtype=np.int64)
        return frame.iloc[np.unique(positions)].reset_index(drop=True)

    @staticmethod
    def _quantile_candidates(values: np.ndarray, count: int = 256) -> np.ndarray:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.array([], dtype=np.float64)
        if finite.size <= count:
            return np.unique(finite)
        quantiles = np.linspace(0.0, 1.0, num=count)
        return np.unique(np.quantile(finite, quantiles))

    def _lexical_pre_features(self, query: Any, signal_row: pd.Series, retriever_name: str) -> dict[str, float]:
        stats = self._lexical_stats(retriever_name)
        tokens = list(query.query_tokens)
        indexed_occurrence_values = []
        indexed_unique_values = {}
        stopwords = self.text_services.stopwords(query.language)
        binary_columns = []
        non_stopword_columns = []
        indexed_non_stop_tokens = []
        for token in tokens:
            index = stats.vocabulary.get(token)
            if index is not None:
                idf = self._idf_value(stats, index)
                indexed_occurrence_values.append(idf)
                if token not in indexed_unique_values:
                    indexed_unique_values[token] = idf
                binary_columns.append(index)
                if token not in stopwords:
                    non_stopword_columns.append(index)
                    indexed_non_stop_tokens.append(token)
        scope = self._query_scope(stats, binary_columns)
        non_stopword_scope = self._query_scope(stats, non_stopword_columns)
        mean_ictf = self._mean_ictf(stats, tokens)
        simplified_clarity = self._simplified_clarity(stats, tokens, stopwords)
        idf_ratio = self._idf_ratio(stats, indexed_non_stop_tokens)
        oov = sum(1 for token in tokens if token not in stats.vocabulary)
        return {
            "query_length_characters": float(len(query.text or "")),
            "query_length_tokens": float(len(tokens)),
            "x_artifact_ratio": float(signal_row["x_artifact_ratio"]),
            "x_mention_ratio": float(signal_row["x_mention_ratio"]),
            "english_indicator": float(query.language == "en"),
            "german_indicator": float(query.language == "de"),
            "french_indicator": float(query.language == "fr"),
            "mean_query_term_idf": float(np.mean(indexed_occurrence_values)) if indexed_occurrence_values else math.nan,
            "maximum_query_term_idf": float(np.max(indexed_occurrence_values)) if indexed_occurrence_values else math.nan,
            "query_term_idf_standard_deviation": float(np.std(indexed_occurrence_values)) if indexed_occurrence_values else math.nan,
            "mean_query_term_ictf": mean_ictf,
            "query_scope": scope,
            "query_scope_non_stopwords": non_stopword_scope,
            "out_of_vocabulary_ratio": _ratio(oov, len(tokens)),
            "stopword_ratio": float(signal_row["stopword_ratio"]),
            "repeated_term_ratio": float(signal_row["repeated_term_ratio"]),
            "simplified_clarity_score": simplified_clarity,
            "idf_maximum_to_minimum_ratio": idf_ratio,
            "numeral_ratio": float(signal_row["numeral_ratio"]),
            "acronym_ratio": float(signal_row["acronym_ratio"]),
            "proper_noun_ratio": float(signal_row["proper_noun_ratio"]),
            "mean_query_term_polysemy": float(signal_row["mean_query_term_polysemy"]),
            "mean_dependency_link_span": float(signal_row["mean_dependency_link_span"]),
            "sentiment_polarity": float(signal_row["sentiment_polarity"]),
            "sentiment_intensity": float(signal_row["sentiment_intensity"]),
        }

    def _lexical_query_context(self, query: Any, ranking: pd.DataFrame, full_scores: np.ndarray, retriever_name: str) -> dict[str, Any]:
        top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
        score_values = ranking["score"].to_numpy(dtype=np.float64)
        title_tokens = []
        entity_sets = []
        for document_id in top_document_ids:
            document = self.document_lookup.loc[int(document_id)]
            title_tokens.append(document["document_tokens"])
            entity_sets.append(self._document_entity_set(int(document_id), retriever_name))
        return {
            "top_document_ids": top_document_ids,
            "scores": score_values,
            "full_scores": np.asarray(full_scores, dtype=np.float64),
            "documents": self.document_lookup.loc[top_document_ids].reset_index(drop=True),
            "entity_sets": entity_sets,
        }

    def _lexical_post_features(self, query: Any, context: dict[str, Any], depth: int, retriever_name: str) -> dict[str, float]:
        stats = self._lexical_stats(retriever_name)
        documents = context["documents"].head(depth).reset_index(drop=True)
        top_ids = documents["document_id"].to_numpy(dtype=np.int64)
        positions = [self.document_position[int(document_id)] for document_id in top_ids]
        scores = context["scores"][:depth]
        top1 = documents.iloc[0]
        query_token_set = set(query.query_tokens)
        indexed_query_tokens = [token for token in query.query_tokens if token in stats.vocabulary]
        top1_terms = set(top1["document_tokens"])
        idf_mass_total = sum(self._idf_value(stats, stats.vocabulary[token]) for token in indexed_query_tokens) if indexed_query_tokens else math.nan
        idf_mass_missing = sum(self._idf_value(stats, stats.vocabulary[token]) for token in indexed_query_tokens if token not in top1_terms) if indexed_query_tokens else math.nan
        tfidf_similarity = (stats.tfidf_matrix[positions] @ stats.tfidf_matrix[positions].T).toarray()
        tfidf_min, tfidf_max, tfidf_mean, _ = _pairwise_summary(tfidf_similarity)
        entity_matrix = self._entity_similarity_matrix(top_ids, retriever_name)
        entity_min, entity_max, entity_mean, _ = _pairwise_summary(entity_matrix)
        graph_threshold = self._lexical_graph_threshold(retriever_name)
        graph_density, component_share = self._graph_features(tfidf_similarity, graph_threshold)
        full_scores = context["full_scores"]
        positive_mass = full_scores[full_scores > 0].sum()
        surprise_scores = self._lexical_surprise_scores(query.query_id, retriever_name, context["scores"])
        surprise_threshold = self._lexical_surprise_threshold(retriever_name)
        collection_score = self._lexical_collection_score(retriever_name, list(query.query_tokens))
        previous_depth = self._previous_depth(depth)
        previous_mean = float(np.mean(context["scores"][:previous_depth])) if previous_depth else math.nan
        score_k = float(context["scores"][depth - 1])
        score_k_plus_1 = float(context["scores"][depth]) if len(context["scores"]) > depth else math.nan
        top_scores_entropy = _entropy(scores)
        return {
            "top_1_title_length_characters": float(len(top1["title"])),
            "top_1_title_length_tokens": float(len(top1["title_tokens"])),
            "top_1_title_query_coverage": _ratio(len(query_token_set & set(top1["title_tokens"])), len(query_token_set)),
            "top_1_abstract_length_characters": float(len(top1["abstract"])),
            "top_1_abstract_length_tokens": float(len(top1["abstract_tokens"])),
            "top_1_abstract_query_coverage": _ratio(len(query_token_set & set(top1["abstract_tokens"])), len(query_token_set)),
            "top_1_authors_length_characters": float(len(top1["authors"])),
            "top_1_authors_length_tokens": float(len(top1["authors_tokens"])),
            "top_1_authors_query_coverage": _ratio(len(query_token_set & set(top1["authors_tokens"])), len(query_token_set)),
            "top_1_venue_length_characters": float(len(top1["venue"])),
            "top_1_venue_length_tokens": float(len(top1["venue_tokens"])),
            "top_1_venue_query_coverage": _ratio(len(query_token_set & set(top1["venue_tokens"])), len(query_token_set)),
            "top_1_title_abstract_entity_overlap": self._entity_overlap(query.query_id, query.split_key, top1["document_id"], retriever_name),
            "top_1_query_term_coverage": _ratio(len(set(indexed_query_tokens) & top1_terms), len(set(indexed_query_tokens))) if indexed_query_tokens else math.nan,
            "top_1_unmatched_idf_mass_ratio": (idf_mass_missing / idf_mass_total) if indexed_query_tokens and idf_mass_total else math.nan,
            "minimum_top_k_pairwise_tfidf_similarity": tfidf_min,
            "maximum_top_k_pairwise_tfidf_similarity": tfidf_max,
            "mean_top_k_pairwise_tfidf_similarity": tfidf_mean,
            "minimum_top_k_pairwise_entity_similarity": entity_min,
            "maximum_top_k_pairwise_entity_similarity": entity_max,
            "mean_top_k_pairwise_entity_similarity": entity_mean,
            "clarity_score": self._clarity_score(retriever_name, top_ids, scores),
            "retrieval_coherency_graph_density": graph_density,
            "largest_retrieval_coherency_component_share": component_share,
            "top_1_score": float(scores[0]),
            "top_1_top_2_score_margin": float(scores[0] - scores[1]) if len(scores) > 1 else math.nan,
            "score_at_rank_k": score_k,
            "score_at_rank_k_plus_1": score_k_plus_1,
            "rank_k_rank_k_plus_1_score_margin": float(score_k - score_k_plus_1) if np.isfinite(score_k_plus_1) else math.nan,
            "top_1_rank_k_score_drop": float(scores[0] - score_k),
            "mean_top_k_score": float(np.mean(scores)),
            "median_top_k_score": float(np.median(scores)),
            "top_k_score_standard_deviation": float(np.std(scores)),
            "top_k_score_coefficient_of_variation": float(np.std(scores) / abs(np.mean(scores))) if np.mean(scores) != 0 else math.nan,
            "top_k_score_entropy": top_scores_entropy,
            "top_k_score_skewness": _safe_skew(scores),
            "top_k_score_slope": _safe_slope(scores),
            "boundary_score_slope": _safe_slope(context["scores"][max(0, depth - 10):depth]),
            "close_competitor_fraction": float(np.mean(scores >= 0.75 * scores[0])) if scores[0] != 0 else 1.0,
            "cumulative_top_k_score_mass": float(np.sum(scores[scores > 0]) / positive_mass) if positive_mass > 0 else math.nan,
            "boundary_surprise_score": float(surprise_scores[depth - 1]) if len(surprise_scores) >= depth else math.nan,
            "surprising_document_fraction": float(np.mean(surprise_scores[:depth] >= surprise_threshold)) if np.isfinite(surprise_threshold) else math.nan,
            "normalized_query_commitment": self._lexical_nqc(scores, collection_score),
            "weighted_information_gain": self._lexical_wig(scores, collection_score, len(indexed_query_tokens)),
            "top_1_rank_k_score_ratio": float(scores[0] / score_k) if score_k != 0 else math.nan,
            "mean_score_decrease": float(previous_mean - np.mean(scores)) if previous_depth else math.nan,
        }

    def _semantic_state(self, split_key: str, retriever_name: str) -> dict[str, Any]:
        cache_key = (split_key, retriever_name)
        if cache_key in self._semantic_state_cache:
            self.console.log(f"Using in-memory semantic state for {split_key} / {retriever_name}")
            return self._semantic_state_cache[cache_key]
        self.console.log(f"Preparing semantic state for {split_key} / {retriever_name}")
        state_start = perf_counter()
        self.console.log(f"Loading semantic document embeddings for {retriever_name}")
        spec, document_embeddings = self.retrieval.ensure_semantic_embeddings(retriever_name)
        self.console.log(f"Loading semantic query embeddings for {split_key} / {retriever_name}")
        query_embeddings = self.retrieval.ensure_query_embeddings(split_key, retriever_name)
        self.console.log(f"Loading semantic search index for {retriever_name}")
        index = self.retrieval.ensure_semantic_index_object(retriever_name)
        self.console.log(f"Loading spherical k-means clusters for {retriever_name}")
        spherical = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "spherical_kmeans")
        self.console.log(f"Loading HDBSCAN clusters for {retriever_name}")
        hdbscan = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "hdbscan")
        centroid = document_embeddings.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        query_frame = self._limited_queries(split_key).reset_index(drop=True)
        state = {
            "spec": spec,
            "document_embeddings": document_embeddings,
            "query_embeddings": query_embeddings,
            "index": index,
            "collection_centroid": centroid.astype(np.float32),
            "spherical": spherical,
            "hdbscan": hdbscan,
            "query_frame": query_frame,
            "query_positions": {int(query_id): position for position, query_id in enumerate(query_frame["query_id"].tolist())},
        }
        self._semantic_state_cache[cache_key] = state
        state_elapsed = perf_counter() - state_start
        self.console.log(f"Prepared semantic state for {split_key} / {retriever_name} in {state_elapsed:.1f}s")
        return state

    def _semantic_feature_thresholds(self, retriever_name: str) -> dict[str, float]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        if retriever_name in self._semantic_threshold_bundle_cache:
            self.console.log(f"Using in-memory semantic thresholds for {retriever_name}")
            return self._semantic_threshold_bundle_cache[retriever_name]
        self.console.log(f"Preparing semantic thresholds for {retriever_name}")
        started = perf_counter()
        bundle = {
            "semantic_close_competitor_threshold": self._semantic_close_competitor_threshold(retriever_name),
            "semantic_surprise_threshold": self._semantic_surprise_threshold(retriever_name),
            "semantic_graph_threshold": self._semantic_graph_threshold(retriever_name),
        }
        elapsed = perf_counter() - started
        self.console.log(f"Prepared semantic thresholds for {retriever_name} in {elapsed:.1f}s")
        self._semantic_threshold_bundle_cache[retriever_name] = bundle
        return bundle

    def _semantic_query_static_state(
        self,
        query: Any,
        retriever_name: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        split_key = str(query.split_key)
        query_id = int(query.query_id)
        if not hasattr(self, "_semantic_query_static_cache"):
            self._semantic_query_static_cache = {}
        cache_key = (split_key, retriever_name, query_id)
        if cache_key in self._semantic_query_static_cache:
            return self._semantic_query_static_cache[cache_key]

        query_position = state["query_positions"][query_id]
        query_embedding = state["query_embeddings"][query_position]
        collection_centroid = state["collection_centroid"]
        spherical = state["spherical"]
        hdbscan = state["hdbscan"]
        spherical_query_embedding = self._cluster_view_query_embedding(query_embedding, spherical)
        hdbscan_query_embedding = self._cluster_view_query_embedding(query_embedding, hdbscan)
        nearest_spherical, spherical_margin = self._nearest_centroid_features(spherical_query_embedding, spherical)
        nearest_hdbscan, hdbscan_margin = self._nearest_centroid_features(hdbscan_query_embedding, hdbscan)
        pre_features = {
            "query_collection_centroid_cosine": float(np.dot(query_embedding, collection_centroid)),
            "nearest_centroid_cosine_spherical": nearest_spherical,
            "nearest_second_nearest_centroid_margin_spherical": spherical_margin,
            "nearest_centroid_cosine_hdbscan": nearest_hdbscan,
            "nearest_second_nearest_centroid_margin_hdbscan": hdbscan_margin,
            "cluster_weighted_corpus_familiarity": self._cluster_weighted_corpus_familiarity(spherical_query_embedding, spherical),
        }
        static_state = {
            "query_embedding": query_embedding,
            "spherical_query_embedding": spherical_query_embedding,
            "hdbscan_query_embedding": hdbscan_query_embedding,
            "pre_features": pre_features,
        }
        self._semantic_query_static_cache[cache_key] = static_state
        return static_state

    def _semantic_query_context(
        self,
        query: Any,
        ranking: pd.DataFrame,
        retriever_name: str,
        state: dict[str, Any],
        all_rankings_top100: dict[str, dict[int, list[int]]],
        threshold_bundle: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        split_key = str(query.split_key)
        static_state = self._semantic_query_static_state(query, retriever_name, state)
        query_embedding = static_state["query_embedding"]
        top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
        document_positions = np.array([self.document_position[int(document_id)] for document_id in top_document_ids], dtype=np.int64)
        top_embeddings = state["document_embeddings"][document_positions]
        similarities = ranking["score"].to_numpy(dtype=np.float64)
        if threshold_bundle is None:
            threshold_bundle = self._semantic_feature_thresholds(retriever_name)
        return {
            "split_key": split_key,
            "query_embedding": query_embedding,
            "spherical_query_embedding": static_state["spherical_query_embedding"],
            "hdbscan_query_embedding": static_state["hdbscan_query_embedding"],
            "ranking": ranking,
            "top_document_ids": top_document_ids,
            "document_positions": document_positions,
            "top_embeddings": top_embeddings,
            "similarities": similarities,
            "similarities_mean": float(np.mean(similarities)),
            "similarities_std": float(np.std(similarities)),
            "collection_centroid": state["collection_centroid"],
            "query_collection_centroid_cosine": static_state["pre_features"]["query_collection_centroid_cosine"],
            "spherical": state["spherical"],
            "hdbscan": state["hdbscan"],
            "spherical_labels": self._labels_for_documents(top_document_ids, state["spherical"]),
            "hdbscan_labels": self._labels_for_documents(top_document_ids, state["hdbscan"]),
            "semantic_rankings_top100": {
                name: all_rankings_top100[name][int(query.query_id)]
                for name in self.retrieval.enabled_semantic_retrievers()
                if name in all_rankings_top100
            },
            "bm25_top100": all_rankings_top100[self.retrieval.paired_retriever("bm25", retriever_name)][int(query.query_id)],
            "semantic_close_competitor_threshold": threshold_bundle["semantic_close_competitor_threshold"],
            "semantic_surprise_threshold": threshold_bundle["semantic_surprise_threshold"],
            "semantic_graph_threshold": threshold_bundle["semantic_graph_threshold"],
            "semantic_surprise_scores": self._semantic_surprise_scores(int(query.query_id), retriever_name, similarities),
            "retriever_name": retriever_name,
            "query_id": int(query.query_id),
            "index": state["index"],
            "_geometry_cache": {},
        }

    def _semantic_pre_features(self, query: Any, retriever_name: str, state: dict[str, Any], depth: int) -> dict[str, float]:
        static_state = self._semantic_query_static_state(query, retriever_name, state)
        rows = dict(static_state["pre_features"])
        rows["distance_weighted_training_query_rr_at_k"] = self._distance_weighted_training_rr(
            query,
            retriever_name,
            depth,
            static_state["query_embedding"],
        )
        return rows

    def _cluster_pre_features(
        self,
        query_embedding: np.ndarray,
        spherical: ClusterArtifacts,
        hdbscan: ClusterArtifacts,
    ) -> dict[str, float]:
        spherical_query_embedding = self._cluster_view_query_embedding(query_embedding, spherical)
        hdbscan_query_embedding = self._cluster_view_query_embedding(query_embedding, hdbscan)
        nearest_spherical, spherical_margin = self._nearest_centroid_features(spherical_query_embedding, spherical)
        nearest_hdbscan, hdbscan_margin = self._nearest_centroid_features(hdbscan_query_embedding, hdbscan)
        return {
            "nearest_centroid_cosine_spherical": nearest_spherical,
            "nearest_second_nearest_centroid_margin_spherical": spherical_margin,
            "nearest_centroid_cosine_hdbscan": nearest_hdbscan,
            "nearest_second_nearest_centroid_margin_hdbscan": hdbscan_margin,
            "cluster_weighted_corpus_familiarity": self._cluster_weighted_corpus_familiarity(spherical_query_embedding, spherical),
        }

    def _semantic_score_features(self, context: dict[str, Any], depth: int, retriever_name: str) -> dict[str, float]:
        similarities = context["similarities"][:depth]
        full_similarities = context["similarities"]
        score_k = float(full_similarities[depth - 1])
        score_k_plus_1 = float(full_similarities[depth]) if len(full_similarities) > depth else math.nan
        close_threshold = float(context["semantic_close_competitor_threshold"])
        surprise_scores = context["semantic_surprise_scores"]
        surprise_threshold = float(context["semantic_surprise_threshold"])
        previous_depth = self._previous_depth(depth)
        previous_mean = float(np.mean(full_similarities[:previous_depth])) if previous_depth else math.nan
        full_mean = float(context["similarities_mean"])
        full_std = float(context["similarities_std"])
        collection_cosine = float(context["query_collection_centroid_cosine"])
        return {
            "top_1_cosine_similarity": float(similarities[0]),
            "top_1_top_2_cosine_margin": float(similarities[0] - similarities[1]) if len(similarities) > 1 else math.nan,
            "cosine_similarity_at_rank_k": score_k,
            "cosine_similarity_at_rank_k_plus_1": score_k_plus_1,
            "rank_k_rank_k_plus_1_cosine_margin": float(score_k - score_k_plus_1) if np.isfinite(score_k_plus_1) else math.nan,
            "top_1_rank_k_cosine_drop": float(similarities[0] - score_k),
            "mean_top_k_cosine_similarity": float(np.mean(similarities)),
            "median_top_k_cosine_similarity": float(np.median(similarities)),
            "top_k_cosine_standard_deviation": float(np.std(similarities)),
            "top_k_cosine_entropy": _entropy(similarities),
            "top_k_cosine_skewness": _safe_skew(similarities),
            "top_k_cosine_slope": _safe_slope(similarities),
            "boundary_cosine_slope": _safe_slope(full_similarities[max(0, depth - 10):depth]),
            "close_competitor_fraction": float(np.mean(similarities >= similarities[0] - close_threshold)) if np.isfinite(close_threshold) else math.nan,
            "above_mean_cosine_fraction": float(np.mean(similarities > np.mean(similarities))),
            "boundary_semantic_surprise_score": float(surprise_scores[depth - 1]) if len(surprise_scores) >= depth else math.nan,
            "surprising_document_fraction": float(np.mean(surprise_scores[:depth] >= surprise_threshold)) if np.isfinite(surprise_threshold) else math.nan,
            "mean_cosine_decrease": float(previous_mean - np.mean(similarities)) if previous_depth else math.nan,
            "boundary_cosine_z_score": float((score_k - full_mean) / full_std) if full_std > 0 else math.nan,
            "standardized_top_1_rank_k_cosine_drop": float((similarities[0] - score_k) / full_std) if full_std > 0 else math.nan,
            "dense_weighted_information_gain": self._dense_wig(similarities, collection_cosine, len(context["query_embedding"])),
            "dense_normalized_query_commitment": self._dense_nqc(similarities, collection_cosine),
        }

    def _semantic_geometry_features(self, context: dict[str, Any], depth: int, retriever_name: str) -> dict[str, float]:
        cache = context.setdefault("_geometry_cache", {})
        embeddings = context["top_embeddings"][:depth]
        query_embedding = context["query_embedding"]
        prefix_sums = cache.get("prefix_sums")
        if prefix_sums is None:
            prefix_sums = np.cumsum(context["top_embeddings"], axis=0, dtype=context["top_embeddings"].dtype)
            cache["prefix_sums"] = prefix_sums
        centroid = prefix_sums[depth - 1] / depth
        centroid_norm = np.linalg.norm(centroid)
        centroid = centroid / max(centroid_norm, 1e-12)
        query_centroid_cosine = float(np.dot(query_embedding, centroid))
        centroid_distances = 1.0 - np.clip(embeddings @ centroid, -1.0, 1.0)
        full_cosine_matrix = cache.get("full_cosine_matrix")
        if full_cosine_matrix is None:
            full_cosine_matrix = _cosine_similarity_matrix(context["top_embeddings"])
            cache["full_cosine_matrix"] = full_cosine_matrix
        cosine_matrix = full_cosine_matrix[:depth, :depth]
        pair_min, pair_max, pair_mean, pair_std = _pairwise_summary(cosine_matrix)
        previous_depth = self._previous_depth(depth)
        query_document_cosines = cache.get("query_document_cosines")
        if query_document_cosines is None:
            query_document_cosines = np.clip(context["top_embeddings"] @ query_embedding, -1.0, 1.0)
            cache["query_document_cosines"] = query_document_cosines
        coherence_ratio, adjusted_ratio = self._coherence_ratios(
            context,
            depth,
            full_cosine_matrix=full_cosine_matrix,
            query_document_cosines=query_document_cosines,
        )
        graph_density, average_neighbour_degree = self._mean_threshold_graph_features(cosine_matrix)
        semantic_graph_threshold = float(context["semantic_graph_threshold"])
        semantic_graph_density, semantic_component_share = self._graph_features(cosine_matrix, semantic_graph_threshold)
        autocorrelation = self._embedding_spatial_autocorrelation(cosine_matrix, context["similarities"][:depth])
        prefix_mins = cache.get("prefix_mins")
        prefix_maxs = cache.get("prefix_maxs")
        if prefix_mins is None or prefix_maxs is None:
            prefix_mins = np.minimum.accumulate(context["top_embeddings"], axis=0)
            prefix_maxs = np.maximum.accumulate(context["top_embeddings"], axis=0)
            cache["prefix_mins"] = prefix_mins
            cache["prefix_maxs"] = prefix_maxs
        side_lengths = np.maximum(prefix_maxs[depth - 1], query_embedding) - np.minimum(prefix_mins[depth - 1], query_embedding)
        side_lengths = np.clip(side_lengths, 1e-12, None)
        return {
            "query_top_k_centroid_cosine": query_centroid_cosine,
            "top_k_pairwise_coherence": pair_mean,
            "mean_top_k_centroid_distance": float(np.mean(centroid_distances)),
            "top_k_centroid_distance_standard_deviation": float(np.std(centroid_distances)),
            "maximum_top_k_centroid_distance": float(np.max(centroid_distances)),
            "minimum_top_k_pairwise_cosine_similarity": pair_min,
            "maximum_top_k_pairwise_cosine_similarity": pair_max,
            "top_k_pairwise_cosine_standard_deviation": pair_std,
            "preceding_candidate_set_coherence_ratio": coherence_ratio if previous_depth else math.nan,
            "query_adjusted_coherence_ratio": adjusted_ratio if previous_depth else math.nan,
            "average_neighbour_degree": average_neighbour_degree,
            "graph_density": graph_density,
            "embedding_spatial_autocorrelation": autocorrelation,
            "semantic_retrieval_coherency_graph_density": semantic_graph_density,
            "largest_semantic_retrieval_coherency_component_share": semantic_component_share,
            "log_query_document_hyper_volume": float(np.log(side_lengths).sum()),
        }

    def _semantic_cluster_features(self, context: dict[str, Any], depth: int, retriever_name: str) -> dict[str, float]:
        scores = context["similarities"][:depth]
        weights = _softmax((scores - np.mean(scores)) / np.std(scores)) if np.std(scores) > 0 else np.full(depth, 1.0 / depth)
        rows = {}
        for algorithm in ["spherical", "hdbscan"]:
            artifacts = context[algorithm]
            cluster_query_embedding = context.get(f"{algorithm}_query_embedding")
            if cluster_query_embedding is None:
                cluster_query_embedding = self._cluster_view_query_embedding(context["query_embedding"], artifacts)
                context[f"{algorithm}_query_embedding"] = cluster_query_embedding
            labels = context.get(f"{algorithm}_labels")
            if labels is None:
                labels = self._labels_for_documents(context["top_document_ids"], artifacts)
                context[f"{algorithm}_labels"] = labels
            depth_labels = labels[:depth]
            prefix = f"{algorithm}_"
            rows[prefix + "query_nearest_cluster_share_in_top_k"] = self._query_nearest_cluster_share(
                cluster_query_embedding,
                depth_labels,
                artifacts,
            )
            rows[prefix + "retrieval_score_mass_of_query_nearest_cluster_in_top_k"] = self._query_nearest_cluster_score_mass(
                cluster_query_embedding,
                depth_labels,
                artifacts,
                weights,
            )
            rows[prefix + "distinct_clusters_in_top_k"] = self._distinct_clusters(depth_labels, algorithm == "hdbscan")
            rows[prefix + "cluster_entropy_in_top_k"] = self._cluster_entropy(depth_labels, algorithm == "hdbscan")
            rows[prefix + "score_weighted_cluster_entropy_in_top_k"] = self._score_weighted_cluster_entropy(depth_labels, weights, algorithm == "hdbscan")
            rows[prefix + "largest_cluster_share_in_top_k"] = self._largest_cluster_share(depth_labels, algorithm == "hdbscan")
            rows[prefix + "dominant_cluster_enrichment_in_top_k"] = self._dominant_cluster_enrichment(depth_labels, artifacts, algorithm == "hdbscan")
            if algorithm == "hdbscan":
                rows[prefix + "noise_fraction_in_top_k"] = float(np.mean(np.asarray(depth_labels) == -1))
        return rows

    def _gold_aware_semantic_features(self, gold_document_cosine: float, similarities: np.ndarray, depth: int) -> dict[str, float]:
        top_k_similarities = similarities[:depth]
        top_1_similarity = float(top_k_similarities[0])
        rank_k_similarity = float(top_k_similarities[-1])
        top_k_mean = float(np.mean(top_k_similarities))
        top_k_std = float(np.std(top_k_similarities))
        if top_k_std > 0.0:
            z_score = float((gold_document_cosine - top_k_mean) / top_k_std)
        else:
            z_score = 0.0
        return {
            "gold_document_cosine_similarity": float(gold_document_cosine),
            "top_1_gold_cosine_gap": float(top_1_similarity - gold_document_cosine),
            "rank_k_gold_cosine_gap": float(rank_k_similarity - gold_document_cosine),
            "gold_cosine_top_k_z_score": z_score,
        }

    def _gold_aware_paper_analysis_features(
        self,
        query: Any,
        signal_row: pd.Series,
        retriever_name: str,
        similarities: np.ndarray,
        current_top_1_document_id: int,
        gold_document: pd.Series,
        peer_top_1_by_retriever: dict[str, dict[int, int]],
        include_semantic_signals: bool = True,
    ) -> dict[str, float]:
        query_token_set = set(query.query_tokens)
        title_tokens = set(gold_document["title_tokens"])
        abstract_tokens = set(gold_document["abstract_tokens"])
        if include_semantic_signals:
            top_10_similarities = similarities[: min(10, len(similarities))]
            top_1_top_10_mean_gap = (
                float(top_10_similarities[0] - np.mean(top_10_similarities))
                if top_10_similarities.size
                else math.nan
            )
            top_1_cosine = self._ranked_similarity_at(similarities, 1)
            top_2_cosine = self._ranked_similarity_at(similarities, 2)
            top_3_cosine = self._ranked_similarity_at(similarities, 3)
            top_4_cosine = self._ranked_similarity_at(similarities, 4)
            top_5_cosine = self._ranked_similarity_at(similarities, 5)
        else:
            top_1_top_10_mean_gap = math.nan
            top_1_cosine = math.nan
            top_2_cosine = math.nan
            top_3_cosine = math.nan
            top_4_cosine = math.nan
            top_5_cosine = math.nan
        return {
            "query_length_tokens": float(len(query.query_tokens)),
            "gold_title_length_characters": float(len(str(gold_document.get("title", "")).strip())),
            "gold_title_length_tokens": float(len(gold_document["title_tokens"])),
            "gold_abstract_length_characters": float(len(str(gold_document.get("abstract", "")).strip())),
            "gold_abstract_length_tokens": float(len(gold_document["abstract_tokens"])),
            "gold_authors_length_characters": float(len(str(gold_document.get("authors", "")).strip())),
            "gold_authors_length_tokens": float(len(gold_document["authors_tokens"])),
            "gold_venue_length_characters": float(len(str(gold_document.get("venue", "")).strip())),
            "gold_venue_length_tokens": float(len(gold_document["venue_tokens"])),
            "gold_title_query_coverage": _ratio(len(query_token_set & title_tokens), len(query_token_set)),
            "gold_abstract_query_coverage": _ratio(len(query_token_set & abstract_tokens), len(query_token_set)),
            "gold_authors_query_coverage": _ratio(
                len(query_token_set & set(gold_document["authors_tokens"])),
                len(query_token_set),
            ),
            "gold_venue_query_coverage": _ratio(
                len(query_token_set & set(gold_document["venue_tokens"])),
                len(query_token_set),
            ),
            "gold_title_abstract_entity_overlap": self._entity_overlap(
                int(query.query_id),
                str(query.split_key),
                int(gold_document["document_id"]),
                retriever_name,
            ),
            "x_artifact_ratio": float(signal_row["x_artifact_ratio"]),
            "x_mention_ratio": float(signal_row["x_mention_ratio"]),
            "top_1_cosine_similarity": top_1_cosine,
            "top_2_cosine_similarity": top_2_cosine,
            "top_3_cosine_similarity": top_3_cosine,
            "top_4_cosine_similarity": top_4_cosine,
            "top_5_cosine_similarity": top_5_cosine,
            "top_1_top_10_mean_gap": top_1_top_10_mean_gap,
            "top_1_agreement_vs_bge_m3": self._top_1_agreement_with_peer(
                current_top_1_document_id,
                int(query.query_id),
                retriever_name,
                "bge_m3",
                peer_top_1_by_retriever,
            ),
            "top_1_agreement_vs_multilingual_e5_base": self._top_1_agreement_with_peer(
                current_top_1_document_id,
                int(query.query_id),
                retriever_name,
                "multilingual_e5_base",
                peer_top_1_by_retriever,
            ),
        }

    def _gold_aware_lexical_static_features(
        self,
        query: Any,
        context: dict[str, Any],
        retriever_name: str,
        gold_document_id: int,
    ) -> dict[str, float]:
        stats = self._lexical_stats(retriever_name)
        gold_document = self.document_lookup.loc[int(gold_document_id)]
        top_1_document = context["documents"].iloc[0]
        indexed_query_tokens = [token for token in query.query_tokens if token in stats.vocabulary]
        indexed_query_token_set = set(indexed_query_tokens)
        gold_terms = set(gold_document["document_tokens"])
        top_1_terms = set(top_1_document["document_tokens"])
        if indexed_query_tokens:
            idf_mass_total = sum(self._idf_value(stats, stats.vocabulary[token]) for token in indexed_query_tokens)
            gold_missing_mass = sum(
                self._idf_value(stats, stats.vocabulary[token])
                for token in indexed_query_tokens
                if token not in gold_terms
            )
            top_1_missing_mass = sum(
                self._idf_value(stats, stats.vocabulary[token])
                for token in indexed_query_tokens
                if token not in top_1_terms
            )
            gold_query_coverage = _ratio(len(indexed_query_token_set & gold_terms), len(indexed_query_token_set))
            gold_unmatched_idf_mass_ratio = (
                float(gold_missing_mass / idf_mass_total)
                if idf_mass_total > 0.0
                else math.nan
            )
            top_1_query_coverage = _ratio(len(indexed_query_token_set & top_1_terms), len(indexed_query_token_set))
            top_1_unmatched_idf_mass_ratio = (
                float(top_1_missing_mass / idf_mass_total)
                if idf_mass_total > 0.0
                else math.nan
            )
        else:
            gold_query_coverage = math.nan
            gold_unmatched_idf_mass_ratio = math.nan
            top_1_query_coverage = math.nan
            top_1_unmatched_idf_mass_ratio = math.nan
        gold_score = float(context["full_scores"][self.document_position[int(gold_document_id)]])
        top_1_score = float(context["scores"][0])
        query_id = getattr(query, "query_id", None)
        calibrator = (
            self._lexical_surprise_calibrator(int(query_id), retriever_name, context["scores"])
            if query_id is not None
            else None
        )
        if calibrator is None:
            gold_surprise_score = self._gold_aware_surprise_score(
                context["scores"],
                gold_score,
                semantic=False,
            )
        else:
            gold_surprise_score = self._gold_aware_surprise_score(
                context["scores"],
                gold_score,
                semantic=False,
                calibrator=calibrator,
            )
        surprise_threshold = self._lexical_surprise_threshold(retriever_name)
        return {
            "gold_query_term_coverage": gold_query_coverage,
            "gold_unmatched_idf_mass_ratio": gold_unmatched_idf_mass_ratio,
            "gold_document_retrieval_score": gold_score,
            "top_1_gold_score_gap": float(top_1_score - gold_score),
            "top_1_gold_score_ratio": float(top_1_score / gold_score) if gold_score != 0.0 else math.nan,
            "gold_document_surprise_score": gold_surprise_score,
            "gold_document_surprise_threshold_indicator": (
                float(gold_surprise_score >= surprise_threshold)
                if np.isfinite(gold_surprise_score) and np.isfinite(surprise_threshold)
                else math.nan
            ),
            "top_1_gold_query_term_coverage_gap": (
                float(top_1_query_coverage - gold_query_coverage)
                if np.isfinite(top_1_query_coverage) and np.isfinite(gold_query_coverage)
                else math.nan
            ),
            "top_1_gold_unmatched_idf_mass_gap": (
                float(gold_unmatched_idf_mass_ratio - top_1_unmatched_idf_mass_ratio)
                if np.isfinite(gold_unmatched_idf_mass_ratio) and np.isfinite(top_1_unmatched_idf_mass_ratio)
                else math.nan
            ),
        }

    def _gold_aware_lexical_depth_features(
        self,
        context: dict[str, Any],
        gold_document_id: int,
        depth: int,
    ) -> dict[str, float]:
        scores = context["scores"][:depth]
        gold_score = float(context["full_scores"][self.document_position[int(gold_document_id)]])
        score_k = float(context["scores"][depth - 1])
        top_k_std = float(np.std(scores))
        return {
            "rank_k_gold_score_gap": float(score_k - gold_score),
            "gold_score_top_k_z_score": (
                float((gold_score - np.mean(scores)) / top_k_std)
                if top_k_std > 0.0
                else math.nan
            ),
        }

    @staticmethod
    def _empty_gold_aware_lexical_ranking_features() -> dict[str, float]:
        return {
            "gold_query_term_coverage": math.nan,
            "gold_unmatched_idf_mass_ratio": math.nan,
            "gold_document_retrieval_score": math.nan,
            "top_1_gold_score_gap": math.nan,
            "top_1_gold_score_ratio": math.nan,
            "rank_k_gold_score_gap": math.nan,
            "gold_score_top_k_z_score": math.nan,
            "gold_document_surprise_score": math.nan,
            "gold_document_surprise_threshold_indicator": math.nan,
            "top_1_gold_query_term_coverage_gap": math.nan,
            "top_1_gold_unmatched_idf_mass_gap": math.nan,
        }

    def _gold_aware_semantic_query_context(
        self,
        query: Any,
        ranking: pd.DataFrame,
        retriever_name: str,
        state: dict[str, Any],
        threshold_bundle: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        static_state = self._semantic_query_static_state(query, retriever_name, state)
        query_embedding = static_state["query_embedding"]
        top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
        document_positions = np.array([self.document_position[int(document_id)] for document_id in top_document_ids], dtype=np.int64)
        top_embeddings = state["document_embeddings"][document_positions]
        similarities = ranking["score"].to_numpy(dtype=np.float64)
        gold_document_embedding = state["document_embeddings"][self.document_position[int(query.document_id)]]
        if threshold_bundle is None:
            threshold_bundle = self._semantic_feature_thresholds(retriever_name)
        return {
            "split_key": str(query.split_key),
            "retriever_name": retriever_name,
            "query_id": int(query.query_id),
            "query_embedding": query_embedding,
            "gold_document_embedding": gold_document_embedding,
            "gold_document_cosine": float(np.clip(np.dot(query_embedding, gold_document_embedding), -1.0, 1.0)),
            "top_document_ids": top_document_ids,
            "top_embeddings": top_embeddings,
            "similarities": similarities,
            "spherical_query_embedding": static_state["spherical_query_embedding"],
            "hdbscan_query_embedding": static_state["hdbscan_query_embedding"],
            "spherical": state["spherical"],
            "hdbscan": state["hdbscan"],
            "spherical_labels": self._labels_for_documents(top_document_ids, state["spherical"]),
            "hdbscan_labels": self._labels_for_documents(top_document_ids, state["hdbscan"]),
            "semantic_graph_threshold": float(threshold_bundle["semantic_graph_threshold"]),
            "semantic_surprise_threshold": float(threshold_bundle["semantic_surprise_threshold"]),
            "index": state["index"],
            "_geometry_cache": {},
        }

    def _gold_aware_semantic_static_features(
        self,
        context: dict[str, Any],
        gold_document_id: int,
    ) -> dict[str, float]:
        gold_document_cosine = float(context["gold_document_cosine"])
        top_1_similarity = float(context["similarities"][0])
        calibrator = self._semantic_surprise_calibrator(
            int(context["query_id"]),
            str(context["retriever_name"]),
            context["similarities"],
        )
        if calibrator is None:
            gold_surprise_score = self._gold_aware_surprise_score(
                context["similarities"],
                gold_document_cosine,
                semantic=True,
            )
        else:
            gold_surprise_score = self._gold_aware_surprise_score(
                context["similarities"],
                gold_document_cosine,
                semantic=True,
                calibrator=calibrator,
            )
        surprise_threshold = float(context["semantic_surprise_threshold"])
        return {
            "gold_document_cosine_similarity": gold_document_cosine,
            "top_1_gold_cosine_gap": float(top_1_similarity - gold_document_cosine),
            "gold_document_semantic_surprise_score": gold_surprise_score,
            "gold_document_semantic_surprise_threshold_indicator": (
                float(gold_surprise_score >= surprise_threshold)
                if np.isfinite(gold_surprise_score) and np.isfinite(surprise_threshold)
                else math.nan
            ),
        }

    @staticmethod
    def _gold_aware_surprise_score(
        ranking_scores: np.ndarray,
        candidate_score: float,
        *,
        semantic: bool,
        calibrator: SurpriseCalibrator | None = None,
    ) -> float:
        if not np.isfinite(candidate_score):
            return math.nan
        fit_values = np.asarray(ranking_scores, dtype=np.float64)
        fit_values = fit_values[np.isfinite(fit_values)]
        if fit_values.size == 0:
            return math.nan
        if semantic:
            top_similarity = float(np.max(fit_values))
            fit_values = top_similarity - (1.0 - fit_values)
            candidate_value = top_similarity - (1.0 - float(candidate_score))
        else:
            candidate_value = float(candidate_score)
        active_calibrator = calibrator
        if active_calibrator is None:
            active_calibrator = SurpriseCalibrator()
        if active_calibrator.threshold is None and not active_calibrator.fit(fit_values):
            return math.nan
        transformed = active_calibrator.transform(np.asarray([candidate_value], dtype=np.float64))
        value = float(transformed[0]) if transformed.size else math.nan
        return value if np.isfinite(value) else math.nan

    @staticmethod
    def _empty_gold_aware_semantic_features() -> dict[str, float]:
        return {
            "gold_document_cosine_similarity": math.nan,
            "top_1_gold_cosine_gap": math.nan,
            "mean_top_k_gold_cosine_gap": math.nan,
            "rank_k_gold_cosine_gap": math.nan,
            "gold_cosine_top_k_z_score": math.nan,
            "gold_cosine_above_top_k_mean": math.nan,
            "gold_document_semantic_surprise_score": math.nan,
            "gold_document_semantic_surprise_threshold_indicator": math.nan,
        }

    def _gold_aware_semantic_features(
        self,
        context: dict[str, Any] | None = None,
        depth: int | None = None,
        *,
        gold_document_cosine: float | None = None,
        similarities: np.ndarray | None = None,
    ) -> dict[str, float]:
        if depth is None:
            raise TypeError("depth must be provided")
        if context is None:
            if gold_document_cosine is None or similarities is None:
                raise TypeError("gold_document_cosine and similarities must be provided when context is omitted")
            top_k_similarities = np.asarray(similarities, dtype=np.float64)[:depth]
            gold_document_cosine = float(gold_document_cosine)
        else:
            top_k_similarities = context["similarities"][:depth]
            gold_document_cosine = float(context["gold_document_cosine"])
        rank_k_similarity = float(top_k_similarities[-1])
        top_k_mean = float(np.mean(top_k_similarities))
        top_k_std = float(np.std(top_k_similarities))
        return {
            "mean_top_k_gold_cosine_gap": float(top_k_mean - gold_document_cosine),
            "rank_k_gold_cosine_gap": float(rank_k_similarity - gold_document_cosine),
            "gold_cosine_top_k_z_score": (
                float((gold_document_cosine - top_k_mean) / top_k_std)
                if top_k_std > 0.0
                else math.nan
            ),
            "gold_cosine_above_top_k_mean": float(gold_document_cosine > top_k_mean),
        }

    @staticmethod
    def _empty_gold_aware_semantic_geometry_features() -> dict[str, float]:
        return {
            "gold_top_k_centroid_cosine": math.nan,
            "mean_gold_top_k_document_cosine": math.nan,
            "minimum_gold_top_k_document_cosine": math.nan,
            "maximum_gold_top_k_document_cosine": math.nan,
            "gold_top_k_cosine_standard_deviation": math.nan,
            "gold_document_graph_degree": math.nan,
            "gold_document_retrieval_coherency_degree": math.nan,
            "gold_connected_to_largest_coherency_component": math.nan,
        }

    def _gold_aware_semantic_geometry_features(self, context: dict[str, Any], depth: int) -> dict[str, float]:
        cache = context["_geometry_cache"]
        prefix_sums = cache.get("prefix_sums")
        if prefix_sums is None:
            prefix_sums = np.cumsum(context["top_embeddings"], axis=0, dtype=context["top_embeddings"].dtype)
            cache["prefix_sums"] = prefix_sums
        centroid = prefix_sums[depth - 1] / depth
        centroid_norm = np.linalg.norm(centroid)
        centroid = centroid / max(centroid_norm, 1e-12)
        gold_embedding = context["gold_document_embedding"]
        top_embeddings = context["top_embeddings"][:depth]
        gold_top_k_cosines = np.clip(top_embeddings @ gold_embedding, -1.0, 1.0)
        full_cosine_matrix = cache.get("full_cosine_matrix")
        if full_cosine_matrix is None:
            full_cosine_matrix = _cosine_similarity_matrix(context["top_embeddings"])
            cache["full_cosine_matrix"] = full_cosine_matrix
        cosine_matrix = full_cosine_matrix[:depth, :depth]
        upper_triangle = _upper_triangle(cosine_matrix)
        mean_pairwise_similarity = float(np.mean(upper_triangle)) if upper_triangle.size else math.nan
        semantic_graph_threshold = float(context["semantic_graph_threshold"])
        full_adjacency = cache.get("full_semantic_graph_adjacency")
        if full_adjacency is None:
            full_adjacency = full_cosine_matrix > semantic_graph_threshold
            np.fill_diagonal(full_adjacency, False)
            cache["full_semantic_graph_adjacency"] = full_adjacency
        adjacency = full_adjacency[:depth, :depth]
        _, component_labels = _dense_connected_components(adjacency)
        component_share = np.bincount(component_labels) if component_labels.size else np.array([], dtype=np.int64)
        largest_component_label = int(np.argmax(component_share)) if component_share.size else None
        if largest_component_label is None:
            connected_to_largest = math.nan
        else:
            largest_component_mask = component_labels == largest_component_label
            connected_to_largest = float(np.any(gold_top_k_cosines[largest_component_mask] > semantic_graph_threshold))
        return {
            "gold_top_k_centroid_cosine": float(np.dot(gold_embedding, centroid)),
            "mean_gold_top_k_document_cosine": float(np.mean(gold_top_k_cosines)),
            "minimum_gold_top_k_document_cosine": float(np.min(gold_top_k_cosines)),
            "maximum_gold_top_k_document_cosine": float(np.max(gold_top_k_cosines)),
            "gold_top_k_cosine_standard_deviation": float(np.std(gold_top_k_cosines)),
            "gold_document_graph_degree": (
                float(np.mean(gold_top_k_cosines > mean_pairwise_similarity))
                if np.isfinite(mean_pairwise_similarity)
                else math.nan
            ),
            "gold_document_retrieval_coherency_degree": float(np.mean(gold_top_k_cosines > semantic_graph_threshold)),
            "gold_connected_to_largest_coherency_component": connected_to_largest,
        }

    @staticmethod
    def _gold_aware_cluster_weights(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return np.asarray([], dtype=np.float64)
        score_std = float(np.std(scores))
        if score_std <= 0.0:
            return np.full(len(scores), 1.0 / len(scores), dtype=np.float64)
        standardized_scores = (scores - np.mean(scores)) / score_std
        return _softmax(standardized_scores)

    @staticmethod
    def _empty_gold_aware_cluster_features(prefix: str) -> dict[str, float]:
        rows = {
            prefix + "gold_cluster_centroid_cosine": math.nan,
            prefix + "gold_cluster_centroid_rank": math.nan,
            prefix + "gold_cluster_present_in_top_k": math.nan,
            prefix + "gold_cluster_share_in_top_k": math.nan,
            prefix + "gold_cluster_score_mass_in_top_k": math.nan,
            prefix + "gold_cluster_enrichment_in_top_k": math.nan,
        }
        if prefix == "hdbscan_":
            rows["hdbscan_gold_document_labelled_as_noise"] = math.nan
        return rows

    def _gold_aware_cluster_features(
        self,
        context: dict[str, Any] | None = None,
        gold_document_id: int | None = None,
        depth: int | None = None,
        prefix: str | None = None,
        *,
        query_embedding: np.ndarray | None = None,
        labels: list[int] | np.ndarray | None = None,
        artifacts: ClusterArtifacts | None = None,
    ) -> dict[str, float]:
        if gold_document_id is None or depth is None or prefix is None:
            raise TypeError("gold_document_id, depth, and prefix must be provided")
        if context is None:
            if query_embedding is None or labels is None or artifacts is None:
                raise TypeError("query_embedding, labels, and artifacts must be provided when context is omitted")
            weights = np.full(int(depth), 1.0 / int(depth), dtype=np.float64)
        elif prefix == "spherical_":
            query_embedding = context["spherical_query_embedding"]
            labels = context["spherical_labels"]
            artifacts = context["spherical"]
            weights = self._gold_aware_cluster_weights_for_depth(context, int(depth))
        elif prefix == "hdbscan_":
            query_embedding = context["hdbscan_query_embedding"]
            labels = context["hdbscan_labels"]
            artifacts = context["hdbscan"]
            weights = self._gold_aware_cluster_weights_for_depth(context, int(depth))
        else:
            raise ValueError(f"Unsupported gold-aware cluster prefix '{prefix}'")
        gold_label = artifacts.label_lookup.get(int(gold_document_id))
        if gold_label is None:
            raise KeyError(f"Missing cluster label for document_id={gold_document_id}")

        labelled_as_noise = float(prefix == "hdbscan_" and int(gold_label) == -1)
        if labelled_as_noise:
            return {
                prefix + "gold_cluster_centroid_cosine": math.nan,
                prefix + "gold_cluster_centroid_rank": math.nan,
                prefix + "gold_cluster_present_in_top_k": 0.0,
                prefix + "gold_cluster_share_in_top_k": 0.0,
                prefix + "gold_cluster_score_mass_in_top_k": 0.0,
                prefix + "gold_cluster_enrichment_in_top_k": 0.0,
                "hdbscan_gold_document_labelled_as_noise": 1.0,
            }

        depth_labels = np.asarray(labels[:depth], dtype=np.int64)
        gold_cluster_share = float(np.mean(depth_labels == int(gold_label)))
        gold_cluster_score_mass = float(np.sum(weights[depth_labels == int(gold_label)])) if weights.size else 0.0
        collection_share = float(artifacts.label_share_lookup.get(int(gold_label), 0.0))
        enrichment = gold_cluster_share / collection_share if collection_share > 0.0 else 0.0
        rows = {
            prefix + "gold_cluster_centroid_cosine": self._cluster_centroid_similarity(query_embedding, artifacts, int(gold_label)),
            prefix + "gold_cluster_centroid_rank": self._cluster_centroid_rank(query_embedding, artifacts, int(gold_label)),
            prefix + "gold_cluster_present_in_top_k": float(np.any(depth_labels == int(gold_label))),
            prefix + "gold_cluster_share_in_top_k": gold_cluster_share,
            prefix + "gold_cluster_score_mass_in_top_k": gold_cluster_score_mass,
            prefix + "gold_cluster_enrichment_in_top_k": float(enrichment),
        }
        if prefix == "hdbscan_":
            rows["hdbscan_gold_document_labelled_as_noise"] = 0.0
        return rows

    def _gold_aware_cluster_weights_for_depth(self, context: dict[str, Any], depth: int) -> np.ndarray:
        cache = context.setdefault("_gold_aware_cluster_cache", {})
        weights_by_depth = cache.setdefault("weights_by_depth", {})
        weights = weights_by_depth.get(int(depth))
        if weights is None:
            weights = self._gold_aware_cluster_weights(context["similarities"][:depth])
            weights_by_depth[int(depth)] = weights
        return weights

    @staticmethod
    def _empty_gold_aware_perturbation_features() -> dict[str, float]:
        return {
            "gold_document_perturbation_retention": math.nan,
            "gold_document_cosine_perturbation": math.nan,
        }

    def _gold_aware_perturbation_features(
        self,
        query_id: int,
        gold_document_id: int,
        depth: int,
        context: dict[str, Any],
    ) -> dict[str, float]:
        cache = context.setdefault("_gold_aware_perturbation_cache", {})
        candidate_depths = self._perturbation_candidate_depths(context, required_depth=depth)
        run = self._perturbation_run(query_id, context, candidate_depths=candidate_depths)
        if "gold_document_cosine_perturbation" not in cache:
            gold_cosines = np.clip(run["perturbed_embeddings"] @ context["gold_document_embedding"], -1.0, 1.0)
            cache["gold_document_cosine_perturbation"] = float(
                np.mean(np.abs(gold_cosines - float(context["gold_document_cosine"])))
            )
        retention_by_depth = cache.setdefault("retention_by_depth", {})
        if int(depth) not in retention_by_depth:
            if int(depth) > int(run["max_depth"]):
                retention_by_depth[int(depth)] = math.nan
            else:
                retention_by_depth[int(depth)] = float(
                    np.mean(
                        [
                            int(gold_document_id) in perturbed_ids[: int(depth)]
                            for perturbed_ids in run["perturbed_rankings"]
                        ]
                    )
                )
        return {
            "gold_document_perturbation_retention": float(retention_by_depth[int(depth)]),
            "gold_document_cosine_perturbation": float(cache["gold_document_cosine_perturbation"]),
        }

    @staticmethod
    def _ranked_similarity_at(similarities: np.ndarray, rank: int) -> float:
        index = int(rank) - 1
        if index < 0 or index >= len(similarities):
            return math.nan
        return float(similarities[index])

    def _gold_aware_peer_top_1_by_query(self, split_key: str, retriever_name: str) -> dict[str, dict[int, int]]:
        peer_top_1_by_retriever: dict[str, dict[int, int]] = {}
        for peer_base_name in ("bge_m3", "multilingual_e5_base"):
            try:
                peer_name = self.retrieval.paired_retriever(peer_base_name, retriever_name)
            except KeyError:
                continue
            peer_documents = self._top_document_ids_by_query(split_key, peer_name, limit=1)
            peer_top_1_by_retriever[peer_base_name] = {
                int(query_id): int(document_ids[0])
                for query_id, document_ids in peer_documents.items()
                if document_ids
            }
        return peer_top_1_by_retriever

    def _top_document_ids_by_query(self, split_key: str, retriever_name: str, limit: int) -> dict[int, list[int]]:
        resolved = self.retrieval.resolve_retriever_name(retriever_name)
        cache_key = (split_key, resolved)
        cached = self.retrieval._ranking_cache.get(cache_key)
        if cached is not None:
            return _stream_top_document_ids_by_query(cached, limit=limit)

        ranking_path = self.store.path("rankings", split_key, f"{resolved}_top_{self.config.max_ranking_depth}.parquet")
        if ranking_path.exists():
            return self._stream_top_document_ids_from_parquet(ranking_path, limit=limit)

        ranking = self.retrieval.ensure_rankings(split_key, resolved)
        return _stream_top_document_ids_by_query(ranking, limit=limit)

    @staticmethod
    def _stream_top_document_ids_from_parquet(path: Path, limit: int) -> dict[int, list[int]]:
        import pyarrow.parquet as pq

        grouped: dict[int, list[int]] = {}
        current_query_id: int | None = None
        current_documents: list[int] | None = None
        current_count = 0

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=["query_id", "document_id"], batch_size=65536):
            query_ids = batch.column(0).to_numpy(zero_copy_only=False)
            document_ids = batch.column(1).to_numpy(zero_copy_only=False)
            for query_id, document_id in zip(query_ids, document_ids, strict=False):
                query_value = int(query_id)
                if current_query_id != query_value:
                    current_query_id = query_value
                    current_documents = []
                    grouped[current_query_id] = current_documents
                    current_count = 0
                if current_count < limit:
                    current_documents.append(int(document_id))
                    current_count += 1
        return grouped

    def _top_1_agreement_with_peer(
        self,
        current_top_1_document_id: int,
        query_id: int,
        retriever_name: str,
        peer_base_name: str,
        peer_top_1_by_retriever: dict[str, dict[int, int]],
    ) -> float:
        peer_rankings = peer_top_1_by_retriever.get(peer_base_name)
        if not peer_rankings:
            return math.nan
        peer_document_id = peer_rankings.get(int(query_id))
        if peer_document_id is None:
            return math.nan
        return float(int(current_top_1_document_id) == int(peer_document_id))

    def _cluster_view_query_embedding(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts) -> np.ndarray:
        projected = self.clusters.project_embeddings_for_cluster_view(query_embedding[np.newaxis, :], artifacts)
        return projected[0]

    def _retriever_agreement_features(
        self,
        retriever_name: str,
        context: dict[str, Any],
        all_rankings_top100: dict[str, dict[int, list[int]]],
    ) -> dict[str, float]:
        cached = context.get("_retriever_agreement_features")
        if cached is not None:
            return cached
        current = context["semantic_rankings_top100"][retriever_name]
        peers = [name for name in self.retrieval.enabled_semantic_retrievers() if name != retriever_name]
        top1_matches = [float(current[0] == context["semantic_rankings_top100"][name][0]) for name in peers if context["semantic_rankings_top100"].get(name)]
        top100_overlaps = [
            self._jaccard(current, context["semantic_rankings_top100"][name])
            for name in peers
            if context["semantic_rankings_top100"].get(name)
        ]
        bm25 = context["bm25_top100"]
        other_retrievers = [name for name in all_rankings_top100.keys() if name != retriever_name]
        peer_candidate_lists = {name: all_rankings_top100[name][context["query_id"]] for name in other_retrievers}
        peer_candidate_sets = {name: set(candidates) for name, candidates in peer_candidate_lists.items()}
        consensus_values = [
            float(np.mean([float(document_id in candidates) for candidates in peer_candidate_sets.values()]))
            if peer_candidate_sets else math.nan
            for document_id in current
        ]
        top1_consensus_support = [
            float(candidates[0] == current[0]) for candidates in peer_candidate_lists.values()
        ]
        rows = {
            "mean_semantic_top_1_agreement": float(np.mean(top1_matches)) if top1_matches else math.nan,
            "maximum_semantic_top_1_agreement": float(np.max(top1_matches)) if top1_matches else math.nan,
            "mean_semantic_top_100_agreement": float(np.mean(top100_overlaps)) if top100_overlaps else math.nan,
            "maximum_semantic_top_100_agreement": float(np.max(top100_overlaps)) if top100_overlaps else math.nan,
            "bm25_top_1_agreement": float(current[0] == bm25[0]),
            "bm25_top_100_agreement": self._jaccard(current, bm25),
            "top_1_consensus_frequency": float(np.mean(top1_consensus_support)) if top1_consensus_support else math.nan,
            "top_100_document_consensus": float(np.nanmean(consensus_values)) if consensus_values else math.nan,
        }
        context["_retriever_agreement_features"] = rows
        return rows

    def _perturbation_features(self, query_id: int, depth: int, state: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
        split_key = str(context["split_key"])
        retriever_name = self.retrieval.resolve_retriever_name(str(context["retriever_name"]))
        cache_key = (split_key, retriever_name, int(query_id))
        perturbation_scores = self._perturbation_feature_cache.get(cache_key)
        if perturbation_scores is None or int(depth) not in perturbation_scores:
            perturbation_scores = self._perturbation_scores_by_depth(query_id, context, required_depth=int(depth))
            self._perturbation_feature_cache[cache_key] = perturbation_scores
        return {"perturbation_ranking_stability": float(perturbation_scores.get(int(depth), math.nan))}

    def _perturbation_candidate_depths(
        self,
        context: dict[str, Any],
        required_depth: int | None = None,
    ) -> list[int]:
        configured = context.get("candidate_depths")
        if configured is None:
            config = getattr(self, "config", None)
            configured = getattr(config, "candidate_sizes", ()) if config is not None else ()
            candidate_depths = {int(value) for value in configured if int(value) > 0}
        else:
            candidate_depths = {int(value) for value in configured if int(value) > 0}
        if required_depth is not None and int(required_depth) > 0:
            candidate_depths.add(int(required_depth))
        return sorted(candidate_depths)

    def _perturbation_run(
        self,
        query_id: int,
        context: dict[str, Any],
        candidate_depths: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        if candidate_depths is None:
            candidate_depths = self._perturbation_candidate_depths(context)
        requested_max_depth = max((int(value) for value in candidate_depths if int(value) > 0), default=0)
        cached = context.get("_perturbation_run")
        if cached is not None and int(cached.get("max_depth", 0)) >= requested_max_depth:
            return cached
        query_embedding = context["query_embedding"]
        power = float(np.mean(np.square(query_embedding)))
        noise_std = math.sqrt(power * 0.06)
        generator = np.random.default_rng(self.config.runtime.random_seed + query_id)
        perturbations = generator.normal(0.0, noise_std, size=(30, len(query_embedding))).astype(np.float32)
        perturbed = query_embedding[np.newaxis, :] + perturbations
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / np.clip(norms, 1e-12, None)
        index = context["index"]
        baseline = context["top_document_ids"].tolist()
        max_depth = min(
            len(baseline),
            len(self.retrieval.document_ids),
            requested_max_depth,
        )
        if max_depth <= 0:
            run = {
                "perturbed_embeddings": perturbed,
                "perturbed_rankings": [],
                "max_depth": 0,
            }
            context["_perturbation_run"] = run
            return run
        _, indices = index.search(perturbed.astype(np.float32), max_depth)
        document_ids = self.retrieval.document_ids
        perturbed_rankings = [
            [int(document_ids[int(item)]) for item in row.tolist()]
            for row in indices
        ]
        run = {
            "perturbed_embeddings": perturbed,
            "perturbed_rankings": perturbed_rankings,
            "max_depth": int(max_depth),
        }
        context["_perturbation_run"] = run
        return run

    def _perturbation_scores_by_depth(
        self,
        query_id: int,
        context: dict[str, Any],
        required_depth: int | None = None,
    ) -> dict[int, float]:
        split_key = str(context["split_key"])
        retriever_name = self.retrieval.resolve_retriever_name(str(context["retriever_name"]))
        cache_key = (split_key, retriever_name, int(query_id))
        cached = self._perturbation_feature_cache.get(cache_key)
        if cached is not None and (required_depth is None or int(required_depth) in cached):
            return cached

        candidate_depths = self._perturbation_candidate_depths(context, required_depth=required_depth)
        run = self._perturbation_run(query_id, context, candidate_depths=candidate_depths)
        baseline = context["top_document_ids"].tolist()
        max_depth = int(run["max_depth"])
        if max_depth <= 0:
            return dict(cached or {})
        perturbed_rankings = run["perturbed_rankings"]
        baseline = context["top_document_ids"].tolist()
        scores_by_depth = dict(cached or {})
        for candidate_depth in candidate_depths:
            if not 0 < int(candidate_depth) <= max_depth:
                continue
            if candidate_depth in scores_by_depth:
                continue
            baseline_prefix = baseline[:candidate_depth]
            scores = [
                _normalized_rank_biased_overlap(baseline_prefix, perturbed_ids[:candidate_depth])
                for perturbed_ids in perturbed_rankings
            ]
            scores_by_depth[candidate_depth] = float(np.mean(scores))
        return scores_by_depth

    def _all_rankings_top100(
        self, split_key: str, document_scope: str, target_retriever_name: str | None = None
    ) -> dict[str, dict[int, list[int]]]:
        cache_key = (split_key, document_scope)
        if cache_key in self._all_rankings_top100_cache:
            self.console.log(f"Using in-memory top-100 agreement rankings for {split_key} / {document_scope}")
            return self._all_rankings_top100_cache[cache_key]
        enabled = [
            name for name in self.retrieval.enabled_retrievers()
            if retriever_document_scope(name) == document_scope
        ]
        self.console.log(
            f"Preparing top-100 agreement rankings for {split_key} / {document_scope} across {len(enabled)} retrievers"
        )
        cache = {}
        for retriever_name in enabled:
            frame = self.retrieval.ensure_rankings(split_key, retriever_name)
            cache[retriever_name] = _stream_top_document_ids_by_query(frame, limit=100)
            if retriever_name != target_retriever_name:
                self.retrieval.discard_cached_ranking(split_key, retriever_name)
        self._all_rankings_top100_cache[cache_key] = cache
        self.console.log(f"Prepared top-100 agreement rankings for {split_key} / {document_scope}")
        return cache

    def _gold_rank(self, ranking: pd.DataFrame, gold_id: int) -> int | None:
        matches = ranking.loc[ranking["document_id"] == gold_id, "rank"]
        if matches.empty:
            return None
        return int(matches.iloc[0])

    def _smallest_sufficient_size(self, gold_rank: int | None) -> int | None:
        if gold_rank is None:
            return None
        for depth in self.config.candidate_sizes:
            if gold_rank <= depth:
                return depth
        return None

    def _idf(self, index: int) -> float:
        return self._idf_value(self.lexical_stats, index)

    @staticmethod
    def _idf_value(stats: LexicalCollectionStats, index: int) -> float:
        document_frequency = int(stats.document_frequency[index])
        if document_frequency <= 0:
            return 0.0
        numerator = math.log2((stats.document_count + 0.5) / document_frequency)
        denominator = math.log2(stats.document_count + 1.0)
        return numerator / denominator

    @staticmethod
    def _query_scope(stats: LexicalCollectionStats, columns: list[int]) -> float:
        if not columns:
            return math.nan
        unique_columns = sorted(set(columns))
        matches = np.asarray(stats.binary_matrix[:, unique_columns].sum(axis=1)).ravel() > 0
        return float(matches.mean())

    @staticmethod
    def _mean_ictf(stats: LexicalCollectionStats, tokens: list[str]) -> float:
        values = []
        for token in tokens:
            index = stats.vocabulary.get(token)
            if index is None:
                continue
            collection_frequency = int(stats.collection_frequency[index])
            if collection_frequency <= 0:
                continue
            values.append(math.log2(stats.total_terms / collection_frequency))
        return float(np.mean(values)) if values else math.nan

    @staticmethod
    def _simplified_clarity(stats: LexicalCollectionStats, tokens: list[str], stopwords: set[str]) -> float:
        retained = [token for token in tokens if token not in stopwords and token in stats.vocabulary]
        if not retained:
            return math.nan
        counts = Counter(retained)
        total = sum(counts.values())
        value = 0.0
        for token, count in counts.items():
            probability_query = count / total
            index = stats.vocabulary[token]
            probability_collection = float(stats.collection_probability[index])
            value += probability_query * math.log(probability_query / probability_collection)
        return float(value)

    def _idf_ratio(self, stats: LexicalCollectionStats, tokens: list[str]) -> float:
        unique_tokens = sorted(set(token for token in tokens if token in stats.vocabulary))
        if not unique_tokens:
            return math.nan
        if len(unique_tokens) == 1:
            return 1.0
        values = [self._idf_value(stats, stats.vocabulary[token]) for token in unique_tokens]
        minimum = min(values)
        return float(max(values) / minimum) if minimum else math.nan

    def _entity_overlap(self, query_id: int, split_key: str, document_id: int, retriever_name: str) -> float:
        query_signals = self._query_signal_lookup(split_key)
        query_entities = set(query_signals.loc[query_id, "entities"])
        document_entities = self._document_entity_set(int(document_id), retriever_name)
        return _ratio(len(query_entities & document_entities), len(query_entities))

    def _entity_similarity_matrix(self, document_ids: np.ndarray, retriever_name: str) -> np.ndarray:
        size = len(document_ids)
        matrix = np.zeros((size, size), dtype=np.float32)
        entity_sets = [self._document_entity_set(int(document_id), retriever_name) for document_id in document_ids]
        for i in range(size):
            matrix[i, i] = 1.0
            for j in range(i + 1, size):
                left = entity_sets[i]
                right = entity_sets[j]
                if not left or not right:
                    similarity = 0.0
                else:
                    similarity = len(left & right) / math.sqrt(len(left) * len(right))
                matrix[i, j] = similarity
                matrix[j, i] = similarity
        return matrix

    def _graph_features(self, similarity_matrix: np.ndarray, threshold: float) -> tuple[float, float]:
        if similarity_matrix.shape[0] < 2:
            return math.nan, math.nan
        adjacency = similarity_matrix > threshold
        np.fill_diagonal(adjacency, False)
        edge_count = adjacency.sum() / 2
        density = float((2 * edge_count) / (similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1)))
        _, labels = _dense_connected_components(adjacency)
        _, counts = np.unique(labels, return_counts=True)
        component_share = float(counts.max() / similarity_matrix.shape[0]) if counts.size else 0.0
        return density, component_share

    def _mean_threshold_graph_features(self, similarity_matrix: np.ndarray) -> tuple[float, float]:
        if similarity_matrix.shape[0] < 2:
            return math.nan, math.nan
        mean_similarity = float(np.mean(_upper_triangle(similarity_matrix)))
        adjacency = (similarity_matrix > mean_similarity).astype(np.int8)
        np.fill_diagonal(adjacency, 0)
        degrees = adjacency.sum(axis=1)
        neighbour_degree = []
        for index, degree in enumerate(degrees):
            if degree == 0:
                neighbour_degree.append(0.0)
                continue
            neighbours = np.where(adjacency[index] > 0)[0]
            neighbour_degree.append(float(np.mean(degrees[neighbours])))
        density, _ = self._graph_features(similarity_matrix, mean_similarity)
        return density, float(np.mean(neighbour_degree))

    def _clarity_score(self, retriever_name: str, top_ids: np.ndarray, scores: np.ndarray) -> float:
        stats = self._lexical_stats(retriever_name)
        positive_scores = np.clip(scores, 0.0, None)
        if positive_scores.sum() <= 0:
            weights = np.full(len(scores), 1.0 / len(scores))
        else:
            weights = positive_scores / positive_scores.sum()
        indices = [self.document_position[int(document_id)] for document_id in top_ids]
        term_counts = stats.count_matrix[indices]
        weighted_counts = np.asarray(term_counts.transpose().dot(weights), dtype=np.float64).ravel()
        total_weighted_count = float(weighted_counts.sum())
        if total_weighted_count <= 0:
            return math.nan
        query_distribution = weighted_counts / total_weighted_count
        mask = query_distribution > 0
        return float(np.sum(query_distribution[mask] * np.log(query_distribution[mask] / stats.collection_probability[mask])))

    def _lexical_surprise_scores(self, query_id: int, retriever_name: str, scores: np.ndarray) -> np.ndarray:
        cache_key = ("lexical", retriever_name, int(query_id))
        if cache_key in self._surprise_score_cache:
            return self._surprise_score_cache[cache_key]
        path = self.store.path("features", "surprise", "lexical", retriever_name, f"{query_id}.npz")
        if path.exists():
            try:
                values = np.load(path)["values"]
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                path.unlink()
            else:
                self._lexical_surprise_calibrator(int(query_id), retriever_name, scores)
                self._surprise_score_cache[cache_key] = values
                return values
        calibrator = self._lexical_surprise_calibrator(int(query_id), retriever_name, scores)
        if calibrator is None:
            values = np.full(len(scores), math.nan, dtype=np.float64)
        else:
            values = calibrator.transform(scores)
        self.store.write_npz(path, values=values)
        self._surprise_score_cache[cache_key] = values
        return values

    def _semantic_surprise_scores(self, query_id: int, retriever_name: str, similarities: np.ndarray) -> np.ndarray:
        cache_key = ("semantic", retriever_name, int(query_id))
        if cache_key in self._surprise_score_cache:
            return self._surprise_score_cache[cache_key]
        path = self.store.path("features", "surprise", "semantic", retriever_name, f"{query_id}.npz")
        if path.exists():
            try:
                values = np.load(path)["values"]
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                path.unlink()
            else:
                self._semantic_surprise_calibrator(int(query_id), retriever_name, similarities)
                self._surprise_score_cache[cache_key] = values
                return values
        reversed_distances = self._semantic_surprise_fit_values(similarities)
        calibrator = self._semantic_surprise_calibrator(int(query_id), retriever_name, similarities)
        if calibrator is None:
            values = np.full(len(similarities), math.nan, dtype=np.float64)
        else:
            values = calibrator.transform(reversed_distances)
        self.store.write_npz(path, values=values)
        self._surprise_score_cache[cache_key] = values
        return values

    @staticmethod
    def _semantic_surprise_fit_values(similarities: np.ndarray) -> np.ndarray:
        values = np.asarray(similarities, dtype=np.float64)
        if values.size == 0:
            return values
        return float(np.max(values)) - (1.0 - values)

    def _surprise_calibrator(
        self,
        score_type: str,
        query_id: int,
        retriever_name: str,
        fit_values: np.ndarray,
    ) -> SurpriseCalibrator | None:
        if not hasattr(self, "_surprise_calibrator_cache"):
            self._surprise_calibrator_cache = {}
        cache_key = (score_type, retriever_name, int(query_id))
        if cache_key in self._surprise_calibrator_cache:
            return self._surprise_calibrator_cache[cache_key]
        calibrator = SurpriseCalibrator()
        if not calibrator.fit(np.asarray(fit_values, dtype=np.float64)):
            self._surprise_calibrator_cache[cache_key] = None
            return None
        self._surprise_calibrator_cache[cache_key] = calibrator
        return calibrator

    def _lexical_surprise_calibrator(
        self,
        query_id: int,
        retriever_name: str,
        scores: np.ndarray,
    ) -> SurpriseCalibrator | None:
        return self._surprise_calibrator("lexical", query_id, retriever_name, scores)

    def _semantic_surprise_calibrator(
        self,
        query_id: int,
        retriever_name: str,
        similarities: np.ndarray,
    ) -> SurpriseCalibrator | None:
        return self._surprise_calibrator(
            "semantic",
            query_id,
            retriever_name,
            self._semantic_surprise_fit_values(similarities),
        )

    def _lexical_surprise_threshold(self, retriever_name: str) -> float:
        cache_key = ("lexical_surprise_threshold", retriever_name)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        path = self.store.path("features", "surprise", "lexical", retriever_name, "threshold.json")
        if path.exists():
            threshold = float(self.store.read_json(path)["threshold"])
            self._threshold_cache[cache_key] = threshold
            return threshold
        threshold = self._surprise_threshold_for_retriever(retriever_name, semantic=False)
        self.store.write_json(path, {"threshold": threshold})
        self._threshold_cache[cache_key] = threshold
        return threshold

    def _semantic_surprise_threshold(self, retriever_name: str) -> float:
        cache_key = ("semantic_surprise_threshold", retriever_name)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        path = self.store.path("features", "surprise", "semantic", retriever_name, "threshold.json")
        if path.exists():
            threshold = float(self.store.read_json(path)["threshold"])
            self._threshold_cache[cache_key] = threshold
            return threshold
        self.console.log(f"Estimating semantic surprise threshold for {retriever_name}")
        threshold = self._surprise_threshold_for_retriever(retriever_name, semantic=True)
        self.store.write_json(path, {"threshold": threshold})
        self._threshold_cache[cache_key] = threshold
        return threshold

    def _surprise_threshold_for_retriever(self, retriever_name: str, semantic: bool) -> float:
        train_key = self._reference_train_key(retriever_name)
        frame = self.retrieval.ensure_rankings(train_key, retriever_name)
        query_frame = self._sample_query_frame(self._limited_queries(train_key), 2048)
        thresholds = np.linspace(0.0, 8.0, 41)
        scores = []
        grouped = self._rankings_by_query(train_key, retriever_name, ranking_frame=frame)
        for threshold in thresholds:
            f1_values = []
            for query in query_frame.itertuples(index=False):
                if int(query.query_id) not in grouped:
                    continue
                ranking = grouped[int(query.query_id)]
                values = self._semantic_surprise_scores(int(query.query_id), retriever_name, ranking["score"].to_numpy(dtype=np.float64)) if semantic else self._lexical_surprise_scores(int(query.query_id), retriever_name, ranking["score"].to_numpy(dtype=np.float64))
                positives = values >= threshold
                gold_rank = self._gold_rank(ranking, int(query.document_id))
                true_positive = int(gold_rank is not None and gold_rank <= len(values) and positives[gold_rank - 1])
                false_positive = int(positives.sum()) - true_positive
                false_negative = 1 - true_positive
                denominator = 2 * true_positive + false_positive + false_negative
                f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
            scores.append((float(np.mean(f1_values)), float(threshold)))
        scores.sort(reverse=True)
        return scores[0][1]

    def _lexical_graph_threshold(self, retriever_name: str) -> float:
        cache_key = ("lexical_graph_threshold", retriever_name)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        path = self.store.path("features", "thresholds", "lexical_graph", f"{retriever_name}.json")
        if path.exists():
            threshold = float(self.store.read_json(path)["threshold"])
            self._threshold_cache[cache_key] = threshold
            return threshold
        train_key = self._reference_train_key(retriever_name)
        frame = self.retrieval.ensure_rankings(train_key, retriever_name)
        stats = self._lexical_stats(retriever_name)
        values = []
        for _, group in frame.groupby("query_id", sort=False):
            top_ids = group.head(100)["document_id"].to_numpy(dtype=np.int64)
            positions = [self.document_position[int(document_id)] for document_id in top_ids]
            similarity = (stats.tfidf_matrix[positions] @ stats.tfidf_matrix[positions].T).toarray()
            pair_values = _upper_triangle(similarity)
            if pair_values.size:
                values.append(float(pair_values.mean()))
        threshold = float(np.mean(values)) if values else 0.0
        self.store.write_json(path, {"threshold": threshold})
        self._threshold_cache[cache_key] = threshold
        return threshold

    def _semantic_graph_threshold(self, retriever_name: str) -> float:
        cache_key = ("semantic_graph_threshold", retriever_name)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        path = self.store.path("features", "thresholds", "semantic_graph", f"{retriever_name}.json")
        if path.exists():
            threshold = float(self.store.read_json(path)["threshold"])
            self._threshold_cache[cache_key] = threshold
            return threshold
        self.console.log(f"Estimating semantic graph threshold for {retriever_name}")
        train_key = self._reference_train_key(retriever_name)
        frame = self.retrieval.ensure_rankings(train_key, retriever_name)
        state = self._semantic_state(train_key, retriever_name)
        grouped = self._rankings_by_query(train_key, retriever_name, ranking_frame=frame)
        query_frame = self._sample_query_frame(self._limited_queries(train_key), 1024)
        values = []
        for query in query_frame.itertuples(index=False):
            group = grouped.get(int(query.query_id))
            if group is None:
                continue
            top_ids = group.head(100)["document_id"].to_numpy(dtype=np.int64)
            positions = np.array([self.document_position[int(document_id)] for document_id in top_ids], dtype=np.int64)
            embeddings = state["document_embeddings"][positions]
            pair_values = _upper_triangle(_cosine_similarity_matrix(embeddings))
            if pair_values.size:
                values.append(float(pair_values.mean()))
        threshold = float(np.mean(values)) if values else 0.0
        self.store.write_json(path, {"threshold": threshold})
        self._threshold_cache[cache_key] = threshold
        return threshold

    def _semantic_close_competitor_threshold(self, retriever_name: str) -> float:
        cache_key = ("semantic_close_competitor_threshold", retriever_name)
        if cache_key in self._threshold_cache:
            return self._threshold_cache[cache_key]
        path = self.store.path("features", "thresholds", "semantic_close_competitor", f"{retriever_name}.json")
        if path.exists():
            threshold = float(self.store.read_json(path)["threshold"])
            self._threshold_cache[cache_key] = threshold
            return threshold
        self.console.log(f"Estimating semantic close-competitor threshold for {retriever_name}")
        train_key = self._reference_train_key(retriever_name)
        frame = self.retrieval.ensure_rankings(train_key, retriever_name)
        queries = self._sample_query_frame(self._limited_queries(train_key), 2048)
        positive = []
        negative = []
        grouped = self._rankings_by_query(train_key, retriever_name, ranking_frame=frame)
        for query in queries.itertuples(index=False):
            if int(query.query_id) not in grouped:
                continue
            ranking = grouped[int(query.query_id)]
            similarities = ranking["score"].to_numpy(dtype=np.float64)[:100]
            diffs = similarities[0] - similarities
            gold_rank = self._gold_rank(ranking, int(query.document_id))
            if gold_rank is not None and gold_rank <= len(diffs):
                positive.append(float(diffs[gold_rank - 1]))
            mask = np.ones(len(diffs), dtype=bool)
            if gold_rank is not None and gold_rank <= len(diffs):
                mask[gold_rank - 1] = False
            negative.extend(float(value) for value in diffs[mask])
        candidates = self._quantile_candidates(np.asarray(positive + negative, dtype=np.float64))
        if candidates.size == 0:
            self.store.write_json(path, {"threshold": 0.0})
            self._threshold_cache[cache_key] = 0.0
            return 0.0
        best = (float("-inf"), 0.0)
        for threshold in candidates:
            tpr = np.mean([value <= threshold for value in positive]) if positive else 0.0
            fpr = np.mean([value <= threshold for value in negative]) if negative else 0.0
            score = tpr - fpr
            if score > best[0]:
                best = (score, float(threshold))
        self.store.write_json(path, {"threshold": best[1]})
        self._threshold_cache[cache_key] = best[1]
        return best[1]

    def _lexical_collection_score(self, retriever_name: str, tokens: list[str]) -> float:
        stats = self._lexical_stats(retriever_name)
        if base_retriever_name(retriever_name) == "tfidf":
            artifacts = self.retrieval.ensure_lexical_artifacts(self.retrieval.paired_retriever("tfidf", retriever_name))
            query_vector = artifacts.vectorizer.transform([" ".join(tokens)])
            cache_name = f"_tfidf_collection_vector__{retriever_name}"
            if not hasattr(self, cache_name):
                counts = stats.collection_frequency.astype(np.float64)
                idf = artifacts.vectorizer.idf_
                weights = counts / max(float(counts.sum()), 1.0) * idf
                norm = np.linalg.norm(weights)
                setattr(self, cache_name, weights / max(norm, 1e-12))
            return float(query_vector.toarray().ravel() @ getattr(self, cache_name))
        bm25 = self.retrieval.ensure_lexical_artifacts(self.retrieval.paired_retriever("bm25", retriever_name)).model
        score = 0.0
        document_length = float(stats.total_terms)
        for token in tokens:
            if token not in bm25.idf:
                continue
            frequency = int(stats.collection_frequency[stats.vocabulary[token]])
            numerator = frequency * (bm25.k1 + 1.0)
            denominator = frequency + bm25.k1 * (1.0 - bm25.b + bm25.b * document_length / bm25.avgdl)
            score += bm25.idf[token] * (numerator / max(denominator, 1e-12))
        return float(score)

    @staticmethod
    def _lexical_nqc(scores: np.ndarray, collection_score: float) -> float:
        if collection_score == 0:
            return math.nan
        return float(np.sqrt(np.mean((scores - np.mean(scores)) ** 2)) / abs(collection_score))

    @staticmethod
    def _lexical_wig(scores: np.ndarray, collection_score: float, query_length: int) -> float:
        if query_length <= 0:
            return math.nan
        return float(np.mean(scores - collection_score) / math.sqrt(query_length))

    @staticmethod
    def _dense_wig(similarities: np.ndarray, collection_cosine: float, query_length: int) -> float:
        if query_length <= 0:
            return math.nan
        return float(np.mean(similarities - collection_cosine) / math.sqrt(query_length))

    @staticmethod
    def _dense_nqc(similarities: np.ndarray, collection_cosine: float) -> float:
        if collection_cosine == 0:
            return math.nan
        return float(np.std(similarities) / abs(collection_cosine))

    @staticmethod
    def _previous_depth(depth: int) -> int | None:
        ordering = [5, 10, 20, 30, 50, 60, 100, 120, 200, 1000]
        index = ordering.index(depth)
        return ordering[index - 1] if index > 0 else None

    def _reference_train_key(self, retriever_name: str) -> str:
        return f"{self._active_language}_train"

    def _nearest_centroid_features(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts) -> tuple[float, float]:
        centroids = artifacts.centroids
        if centroids.size == 0:
            return 0.0, 0.0
        metric = str(artifacts.summary.get("centroid_query_metric", "cosine"))
        if metric == "inverse_euclidean":
            distances = np.linalg.norm(centroids - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
            order = np.sort(similarities)[::-1]
        else:
            similarities = centroids @ query_embedding
            order = np.sort(similarities)[::-1]
        nearest = float(order[0])
        margin = float(order[0] - order[1]) if len(order) > 1 else nearest
        return nearest, margin

    def _cluster_centroid_rank(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts, target_label: int) -> float:
        if artifacts.centroids.size == 0 or target_label not in set(int(label) for label in artifacts.centroid_labels):
            return math.nan
        similarities = self._cluster_centroid_similarities(query_embedding, artifacts)
        order = np.argsort(similarities)[::-1]
        ranked_labels = [int(artifacts.centroid_labels[index]) for index in order]
        return float(ranked_labels.index(int(target_label)) + 1)

    def _cluster_centroid_similarity(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts, target_label: int) -> float:
        if artifacts.centroids.size == 0 or target_label not in set(int(label) for label in artifacts.centroid_labels):
            return math.nan
        similarities = self._cluster_centroid_similarities(query_embedding, artifacts)
        label_to_similarity = {
            int(label): float(similarities[index])
            for index, label in enumerate(artifacts.centroid_labels)
        }
        return float(label_to_similarity[int(target_label)])

    @staticmethod
    def _cluster_centroid_similarities(query_embedding: np.ndarray, artifacts: ClusterArtifacts) -> np.ndarray:
        metric = str(artifacts.summary.get("centroid_query_metric", "cosine"))
        if metric == "inverse_euclidean":
            return 1.0 / (1.0 + np.linalg.norm(artifacts.centroids - query_embedding, axis=1))
        return artifacts.centroids @ query_embedding

    def _semantic_reference_pool(self, split_key: str, retriever_name: str) -> SemanticReferencePool:
        cache_key = (split_key, self.retrieval.resolve_retriever_name(retriever_name))
        if cache_key in self._semantic_reference_pool_cache:
            return self._semantic_reference_pool_cache[cache_key]
        language, split_name = split_key.split("_", 1)
        if split_name == "train":
            reference_keys = [split_key]
        else:
            reference_keys = [f"{language}_train", f"{language}_dev"]
        self.console.log(f"Preparing semantic reference pool for {split_key} / {retriever_name}")
        embeddings_parts: list[np.ndarray] = []
        query_keys: list[tuple[str, int]] = []
        gold_ranks: list[float] = []
        for reference_key in reference_keys:
            if reference_key not in self.bundle.queries:
                continue
            frame = self._limited_queries(reference_key)
            ranking = self._rankings_by_query(reference_key, retriever_name)
            state = self._semantic_state(reference_key, retriever_name)
            for position, record in enumerate(frame.itertuples(index=False)):
                query_id = int(record.query_id)
                if query_id not in ranking:
                    continue
                embeddings_parts.append(state["query_embeddings"][position : position + 1])
                query_keys.append((reference_key, query_id))
                gold_rank = self._gold_rank(ranking[query_id], int(record.document_id))
                gold_ranks.append(float(gold_rank) if gold_rank is not None else math.inf)
        if embeddings_parts:
            embedding_matrix = np.vstack(embeddings_parts).astype(np.float32, copy=False)
        else:
            embedding_matrix = np.zeros((0, 0), dtype=np.float32)
        reference_index = None
        if embedding_matrix.size:
            reference_index = faiss.IndexFlatL2(int(embedding_matrix.shape[1]))
            reference_index.add(np.ascontiguousarray(embedding_matrix, dtype=np.float32))
        gold_rank_array = np.asarray(gold_ranks, dtype=np.float64)
        rr_by_depth = {
            int(depth): np.where(gold_rank_array <= float(depth), 1.0 / gold_rank_array, 0.0).astype(np.float64)
            for depth in self.config.candidate_sizes
        }
        pool = SemanticReferencePool(
            current_split_key=split_key,
            retriever_name=retriever_name,
            reference_count=len(query_keys),
            index=reference_index,
            query_keys=query_keys,
            query_index={key: index for index, key in enumerate(query_keys)},
            rr_by_depth=rr_by_depth,
        )
        self._semantic_reference_pool_cache[cache_key] = pool
        self.console.log(
            f"Prepared semantic reference pool for {split_key} / {retriever_name} "
            f"with {len(query_keys)} reference queries"
        )
        return pool

    def _semantic_reference_neighborhood(
        self,
        query: Any,
        retriever_name: str,
        query_embedding: np.ndarray,
    ) -> tuple[SemanticReferencePool, np.ndarray, np.ndarray]:
        split_key = str(query.split_key)
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        cache_key = (split_key, retriever_name, int(query.query_id))
        pool = self._semantic_reference_pool(split_key, retriever_name)
        if cache_key in self._semantic_reference_neighbor_cache:
            indices, weights = self._semantic_reference_neighbor_cache[cache_key]
            return pool, indices, weights
        if pool.reference_count == 0 or pool.index is None:
            empty_indices = np.array([], dtype=np.int32)
            empty_weights = np.array([], dtype=np.float64)
            self._semantic_reference_neighbor_cache[cache_key] = (empty_indices, empty_weights)
            return pool, empty_indices, empty_weights
        self_index = pool.query_index.get((split_key, int(query.query_id)))
        excluded = 1 if self_index is not None else 0
        available_count = max(len(pool.query_keys) - excluded, 0)
        if available_count == 0:
            empty_indices = np.array([], dtype=np.int32)
            empty_weights = np.array([], dtype=np.float64)
            self._semantic_reference_neighbor_cache[cache_key] = (empty_indices, empty_weights)
            return pool, empty_indices, empty_weights
        target_count = min(10, available_count)
        search_count = min(len(pool.query_keys), max(target_count + excluded, 32))
        distances, indices = pool.index.search(
            np.ascontiguousarray(query_embedding[np.newaxis, :], dtype=np.float32),
            search_count,
        )
        ordered_indices: list[int] = []
        ordered_weights: list[float] = []
        for squared_distance, candidate_index in zip(distances[0], indices[0], strict=False):
            candidate = int(candidate_index)
            if candidate < 0:
                continue
            if self_index is not None and candidate == self_index:
                continue
            distance = math.sqrt(max(float(squared_distance), 0.0))
            ordered_indices.append(candidate)
            ordered_weights.append(1.0 / (1.0 + distance))
            if len(ordered_indices) >= target_count:
                break
        if not ordered_indices:
            empty_indices = np.array([], dtype=np.int32)
            empty_weights = np.array([], dtype=np.float64)
            self._semantic_reference_neighbor_cache[cache_key] = (empty_indices, empty_weights)
            return pool, empty_indices, empty_weights
        order = np.asarray(ordered_indices, dtype=np.int32)
        weights = np.asarray(ordered_weights, dtype=np.float64)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            normalized = np.array([], dtype=np.float64)
            order = np.array([], dtype=np.int32)
        else:
            normalized = (weights / total).astype(np.float64, copy=False)
        self._semantic_reference_neighbor_cache[cache_key] = (order, normalized)
        return pool, order, normalized

    def _distance_weighted_training_rr(self, query: Any, retriever_name: str, depth: int, query_embedding: np.ndarray) -> float:
        pool, neighbor_indices, neighbor_weights = self._semantic_reference_neighborhood(query, retriever_name, query_embedding)
        if neighbor_indices.size == 0 or neighbor_weights.size == 0:
            return math.nan
        rr_values = pool.rr_by_depth[int(depth)][neighbor_indices]
        if rr_values.size == 0:
            return math.nan
        return float(np.dot(rr_values, neighbor_weights))

    def _cluster_weighted_corpus_familiarity(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts) -> float:
        if artifacts.centroids.size == 0:
            return 0.0
        if not hasattr(self, "_cluster_label_count_cache"):
            self._cluster_label_count_cache = {}
        cache_key = (artifacts.retriever_name, artifacts.algorithm)
        counts = self._cluster_label_count_cache.get(cache_key)
        if counts is None:
            counts = artifacts.assignments.loc[artifacts.assignments["label"] >= 0, "label"].value_counts().to_dict()
            self._cluster_label_count_cache[cache_key] = counts
        weighted = np.zeros_like(query_embedding, dtype=np.float64)
        for centroid, label in zip(artifacts.centroids, artifacts.centroid_labels, strict=True):
            distance_vector = centroid - query_embedding
            norm = np.linalg.norm(distance_vector)
            if norm == 0:
                continue
            weighted += (counts.get(label, 0) / max(norm ** 2, 1e-12)) * (distance_vector / norm)
        return float(np.linalg.norm(weighted) / max(len(artifacts.centroid_labels), 1))

    def _coherence_ratios(
        self,
        context: dict[str, Any],
        depth: int,
        full_cosine_matrix: np.ndarray | None = None,
        query_document_cosines: np.ndarray | None = None,
    ) -> tuple[float, float]:
        previous_depth = self._previous_depth(depth)
        if previous_depth is None:
            return math.nan, math.nan
        embeddings = context["top_embeddings"][:depth]
        previous = embeddings[:previous_depth]
        added = embeddings[previous_depth:depth]
        if len(previous) < 2 or len(added) < 2:
            return math.nan, math.nan
        if full_cosine_matrix is None:
            previous_matrix = _cosine_similarity_matrix(previous)
            added_matrix = _cosine_similarity_matrix(added)
        else:
            previous_matrix = full_cosine_matrix[:previous_depth, :previous_depth]
            added_matrix = full_cosine_matrix[previous_depth:depth, previous_depth:depth]
        previous_pairs = _upper_triangle(previous_matrix)
        added_pairs = _upper_triangle(added_matrix)
        if added_pairs.size == 0 or np.mean(added_pairs) == 0:
            return math.nan, math.nan
        ratio = float(np.mean(previous_pairs) / np.mean(added_pairs))
        if query_document_cosines is None:
            query_embedding = context["query_embedding"]

            def adjusted_mean(matrix: np.ndarray, weights: np.ndarray) -> float:
                values = []
                for i in range(matrix.shape[0]):
                    for j in range(i + 1, matrix.shape[1]):
                        values.append(matrix[i, j] * weights[i] * weights[j])
                return float(np.mean(values)) if values else 0.0

            previous_weights = np.clip(previous @ query_embedding, -1.0, 1.0)
            added_weights = np.clip(added @ query_embedding, -1.0, 1.0)
        else:
            previous_weights = query_document_cosines[:previous_depth]
            added_weights = query_document_cosines[previous_depth:depth]

            def adjusted_mean(matrix: np.ndarray, weights: np.ndarray) -> float:
                values = []
                for i in range(matrix.shape[0]):
                    for j in range(i + 1, matrix.shape[1]):
                        values.append(matrix[i, j] * weights[i] * weights[j])
                return float(np.mean(values)) if values else 0.0

        added_adjusted = adjusted_mean(added_matrix, added_weights)
        if added_adjusted == 0:
            return ratio, math.nan
        adjusted_ratio = adjusted_mean(previous_matrix, previous_weights) / added_adjusted
        return ratio, float(adjusted_ratio)

    def _embedding_spatial_autocorrelation(self, similarity_matrix: np.ndarray, similarities: np.ndarray) -> float:
        if len(similarities) < 2:
            return math.nan
        score_std = float(np.std(similarities))
        if score_std == 0:
            return math.nan
        standardized = (similarities - np.mean(similarities)) / score_std
        mean_similarity = float(np.mean(_upper_triangle(similarity_matrix)))
        adjacency = (similarity_matrix > mean_similarity).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)
        degrees = adjacency.sum(axis=1, keepdims=True)
        diffusion = np.divide(adjacency, degrees, out=np.zeros_like(adjacency), where=degrees > 0)
        diffused = diffusion @ standardized
        diffused_norm = float(np.linalg.norm(diffused))
        if diffused_norm == 0:
            return math.nan
        return float(np.dot(standardized, diffused) / (np.linalg.norm(standardized) * diffused_norm))

    @staticmethod
    def _log_query_document_hyper_volume(query_embedding: np.ndarray, embeddings: np.ndarray) -> float:
        stacked = np.vstack([query_embedding, embeddings])
        side_lengths = stacked.max(axis=0) - stacked.min(axis=0)
        side_lengths = np.clip(side_lengths, 1e-12, None)
        return float(np.log(side_lengths).sum())

    @staticmethod
    def _labels_for_documents(document_ids: np.ndarray, artifacts: ClusterArtifacts) -> list[int]:
        return [int(artifacts.label_lookup[int(document_id)]) for document_id in document_ids]

    @staticmethod
    def _distinct_clusters(labels: list[int], exclude_noise: bool) -> float:
        values = {label for label in labels if not (exclude_noise and label == -1)}
        return float(len(values))

    @staticmethod
    def _cluster_entropy(labels: list[int], exclude_noise: bool) -> float:
        filtered = [label for label in labels if not (exclude_noise and label == -1)]
        if not filtered:
            return 0.0
        counts = np.asarray(list(Counter(filtered).values()), dtype=np.float64)
        proportions = counts / counts.sum()
        return _normalized_entropy(proportions)

    @staticmethod
    def _score_weighted_cluster_entropy(labels: list[int], weights: np.ndarray, exclude_noise: bool) -> float:
        masses = Counter()
        total = 0.0
        for label, weight in zip(labels, weights, strict=True):
            if exclude_noise and label == -1:
                continue
            masses[int(label)] += float(weight)
            total += float(weight)
        if len(masses) < 2 or total == 0:
            return 0.0
        proportions = np.asarray([value / total for value in masses.values()], dtype=np.float64)
        return _normalized_entropy(proportions)

    @staticmethod
    def _largest_cluster_share(labels: list[int], exclude_noise: bool) -> float:
        filtered = [label for label in labels if not (exclude_noise and label == -1)]
        if not filtered:
            return 0.0
        counts = Counter(filtered)
        return float(max(counts.values()) / len(labels))

    def _query_nearest_cluster_share(self, query_embedding: np.ndarray, labels: list[int], artifacts: ClusterArtifacts) -> float:
        nearest_label = self._nearest_cluster_label(query_embedding, artifacts)
        if nearest_label is None:
            return 0.0
        return float(np.mean(np.asarray(labels) == nearest_label))

    def _query_nearest_cluster_score_mass(
        self,
        query_embedding: np.ndarray,
        labels: list[int],
        artifacts: ClusterArtifacts,
        weights: np.ndarray,
    ) -> float:
        nearest_label = self._nearest_cluster_label(query_embedding, artifacts)
        if nearest_label is None:
            return 0.0
        masses = [weight for label, weight in zip(labels, weights, strict=True) if label == nearest_label]
        return float(sum(masses))

    def _nearest_cluster_label(self, query_embedding: np.ndarray, artifacts: ClusterArtifacts) -> int | None:
        if artifacts.centroids.size == 0:
            return None
        metric = str(artifacts.summary.get("centroid_query_metric", "cosine"))
        if metric == "inverse_euclidean":
            distances = np.linalg.norm(artifacts.centroids - query_embedding, axis=1)
            index = int(np.argmin(distances))
        else:
            similarities = artifacts.centroids @ query_embedding
            index = int(np.argmax(similarities))
        return int(artifacts.centroid_labels[index])

    def _dominant_cluster_enrichment(self, labels: list[int], artifacts: ClusterArtifacts, exclude_noise: bool) -> float:
        filtered = [label for label in labels if not (exclude_noise and label == -1)]
        if not filtered:
            return 0.0
        top_counts = Counter(filtered)
        best_label = None
        best_value = float("-inf")
        for label, count in top_counts.items():
            collection_share = float(artifacts.label_share_lookup.get(int(label), 0.0))
            if collection_share == 0:
                continue
            enrichment = (count / len(labels)) / collection_share
            if enrichment > best_value:
                best_value = enrichment
                best_label = label
        return 0.0 if best_label is None else float(best_value)

    @staticmethod
    def _jaccard(left: list[int], right: list[int]) -> float:
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        if not union:
            return 0.0
        return float(len(left_set & right_set) / len(union))

    def _limited_queries(self, split_key: str) -> pd.DataFrame:
        frame = self.bundle.queries[split_key]
        if self.config.runtime.limit_queries is None:
            return frame
        return frame.head(self.config.runtime.limit_queries).copy()
