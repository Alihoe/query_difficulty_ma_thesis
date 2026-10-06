# src/query_difficulty/features_hit_at_1_lightweight.py
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.clustering import ClusterManager
from query_difficulty.config import REPO_ROOT, ProjectConfig
from query_difficulty.console import Console
from query_difficulty.dataset import DatasetBundle
from query_difficulty.features import (
    FeatureBuilder,
    _entropy,
    _pairwise_summary,
    _ratio,
    _safe_skew,
    _safe_slope,
    _stream_top_document_ids_by_query,
)
from query_difficulty.retrieval import RetrievalManager, is_lexical_retriever, retriever_document_scope
from query_difficulty.text import TextServices

LIGHTWEIGHT_MANIFEST_METADATA_VERSION = 1
LIGHTWEIGHT_MANIFEST_DEFAULT_PATH = Path("config/lightweight_hit_at_1_features.json")
LIGHTWEIGHT_FEATURE_TABLE_SCHEMA_VERSION = 1

CONSENSUS_MODE_FULL = "full"
CONSENSUS_MODE_BM25_AND_SELF = "bm25_and_self"
CONSENSUS_MODES = frozenset({CONSENSUS_MODE_FULL, CONSENSUS_MODE_BM25_AND_SELF})

SEMANTIC_CLUSTER_FEATURE_NAMES = frozenset({
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
})

SEMANTIC_SURPRISE_FEATURE_NAMES = frozenset({
    "boundary_semantic_surprise_score",
    "surprising_document_fraction",
})

SEMANTIC_GRAPH_FEATURE_NAMES = frozenset({
    "preceding_candidate_set_coherence_ratio",
    "query_adjusted_coherence_ratio",
    "average_neighbour_degree",
    "graph_density",
    "semantic_retrieval_coherency_graph_density",
    "largest_semantic_retrieval_coherency_component_share",
    "log_query_document_hyper_volume",
})

LEXICAL_GRAPH_FEATURE_NAMES = frozenset({
    "retrieval_coherency_graph_density",
    "largest_retrieval_coherency_component_share",
})

LEXICAL_SURPRISE_FEATURE_NAMES = frozenset({
    "boundary_surprise_score",
    "surprising_document_fraction",
})

LEXICAL_ENTITY_FEATURE_NAMES = frozenset({
    "top_1_title_abstract_entity_overlap",
    "minimum_top_k_pairwise_entity_similarity",
    "maximum_top_k_pairwise_entity_similarity",
    "mean_top_k_pairwise_entity_similarity",
})

LEXICAL_TFIDF_PAIRWISE_FEATURE_NAMES = frozenset({
    "minimum_top_k_pairwise_tfidf_similarity",
    "maximum_top_k_pairwise_tfidf_similarity",
    "mean_top_k_pairwise_tfidf_similarity",
})

SEMANTIC_AGREEMENT_FEATURE_NAMES = frozenset({
    "mean_semantic_top_1_agreement",
    "maximum_semantic_top_1_agreement",
    "mean_semantic_top_100_agreement",
    "maximum_semantic_top_100_agreement",
    "bm25_top_1_agreement",
    "bm25_top_100_agreement",
    "top_1_consensus_frequency",
    "top_100_document_consensus",
})

MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES = SEMANTIC_AGREEMENT_FEATURE_NAMES - frozenset({
    "bm25_top_1_agreement",
    "bm25_top_100_agreement",
})

SEMANTIC_GEOMETRY_FEATURE_NAMES = frozenset({
    "query_top_k_centroid_cosine",
    "top_k_pairwise_coherence",
    "mean_top_k_centroid_distance",
    "top_k_centroid_distance_standard_deviation",
    "maximum_top_k_centroid_distance",
    "minimum_top_k_pairwise_cosine_similarity",
    "maximum_top_k_pairwise_cosine_similarity",
    "top_k_pairwise_cosine_standard_deviation",
    "preceding_candidate_set_coherence_ratio",
    "query_adjusted_coherence_ratio",
    "average_neighbour_degree",
    "graph_density",
    "embedding_spatial_autocorrelation",
    "semantic_retrieval_coherency_graph_density",
    "largest_semantic_retrieval_coherency_component_share",
    "log_query_document_hyper_volume",
})

SEMANTIC_PERTURBATION_FEATURE_NAMES = frozenset({
    "perturbation_ranking_stability",
})

SEMANTIC_DISTANCE_WEIGHTED_TRAINING_RR_FEATURE_NAME = "distance_weighted_training_query_rr_at_k"

SEMANTIC_CLOSE_COMPETITOR_FEATURE_NAME = "close_competitor_fraction"

LEXICAL_CLARITY_SCORE_FEATURE_NAME = "clarity_score"

def _write_manifest_atomic(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def _compute_manifest_entry(
    importances_path: Path,
    threshold: float,
    exclude_multi_retriever_consensus: bool,
    retriever_label: str,
    console: Console | None,
) -> dict[str, Any]:
    frame = pd.read_parquet(importances_path)
    if exclude_multi_retriever_consensus:
        excluded = frame.loc[frame["feature"].isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
        if not excluded.empty and console is not None:
            console.log(
                f"{retriever_label}: excluding {len(excluded)} multi-retriever consensus feature(s) "
                f"from the candidate pool: {sorted(excluded['feature'])}"
            )
        frame = frame.loc[~frame["feature"].isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
    qualifying = frame.loc[frame["importance"] > threshold].sort_values("importance", ascending=False)
    if console is not None:
        console.log(
            f"{retriever_label}: {len(qualifying)}/{len(frame)} features above {threshold:.4f} "
            f"(coverage {float(qualifying['importance'].sum()):.4f})"
        )
    return {
        "qualifying_features": qualifying["feature"].tolist(),
        "source_feature_count": int(len(frame)),
        "qualifying_feature_count": int(len(qualifying)),
        "qualifying_importance_sum": float(qualifying["importance"].sum()),
        "pipeline_stage": "freeze",
    }


def refresh_manifest_from_full_models(
    store: ArtifactStore,
    manifest_path: Path,
    threshold: float = 0.005,
    exclude_multi_retriever_consensus: bool = False,
    console: Console | None = None,
) -> dict[str, Any]:
    source_root = store.root_dir / "models" / "hit_at_1"
    if not source_root.exists():
        raise FileNotFoundError(f"No full hit@1 models found at {source_root}. Train the full hit@1 predictor first.")
    retrievers: dict[str, dict] = {}
    for retriever_dir in sorted(p for p in source_root.iterdir() if p.is_dir() and p.name != "all_meaningful"):
        importances_path = retriever_dir / "feature_importances.parquet"
        if not importances_path.exists():
            if console is not None:
                console.log(f"Skipping {retriever_dir.name}: no feature_importances.parquet")
            continue
        retrievers[retriever_dir.name] = _compute_manifest_entry(
            importances_path, threshold, exclude_multi_retriever_consensus, retriever_dir.name, console
        )
    manifest = {
        "metadata_version": LIGHTWEIGHT_MANIFEST_METADATA_VERSION,
        "importance_threshold": threshold,
        "excluded_multi_retriever_consensus_features": (
            sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES) if exclude_multi_retriever_consensus else []
        ),
        "source": "artifacts/models/hit_at_1/<retriever>/feature_importances.parquet",
        "retrievers": retrievers,
    }
    _write_manifest_atomic(manifest_path, manifest)
    return manifest


def add_manifest_entry(
    store: ArtifactStore,
    manifest_path: Path,
    retriever_name: str,
    threshold: float = 0.005,
    exclude_multi_retriever_consensus: bool = True,
    console: Console | None = None,
) -> dict[str, Any]:
    importances_path = store.root_dir / "models" / "hit_at_1" / retriever_name / "feature_importances.parquet"
    if not importances_path.exists():
        raise FileNotFoundError(f"No feature importances found at {importances_path}. Train the full hit@1 predictor first.")
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "metadata_version": LIGHTWEIGHT_MANIFEST_METADATA_VERSION,
            "importance_threshold": threshold,
            "excluded_multi_retriever_consensus_features": [],
            "source": "artifacts/models/hit_at_1/<retriever>/feature_importances.parquet",
            "retrievers": {},
        }
    manifest["retrievers"][retriever_name] = _compute_manifest_entry(
        importances_path, threshold, exclude_multi_retriever_consensus, retriever_name, console
    )
    if exclude_multi_retriever_consensus:
        manifest["excluded_multi_retriever_consensus_features"] = sorted(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)
    _write_manifest_atomic(manifest_path, manifest)
    return manifest


_SHARED_ROW_COLUMNS = frozenset({
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
})


class _LightweightFeaturePlan:
    def __init__(self, qualifying_features: frozenset[str], is_lexical: bool) -> None:
        self.qualifying_features = qualifying_features
        if is_lexical:
            self.needs_cluster = False
            self.needs_semantic_surprise = False
            self.needs_semantic_graph = False
            self.needs_lexical_surprise = bool(qualifying_features & LEXICAL_SURPRISE_FEATURE_NAMES)
            self.needs_lexical_graph = bool(qualifying_features & LEXICAL_GRAPH_FEATURE_NAMES)
            self.needs_entity = bool(qualifying_features & LEXICAL_ENTITY_FEATURE_NAMES)
            self.needs_tfidf_pairwise = bool(qualifying_features & LEXICAL_TFIDF_PAIRWISE_FEATURE_NAMES)
            self.needs_agreement = False
            self.needs_geometry = False
            self.needs_perturbation = False
            self.needs_distance_weighted_rr = False
            self.needs_close_competitor = False
            self.needs_clarity_score = LEXICAL_CLARITY_SCORE_FEATURE_NAME in qualifying_features
        else:
            self.needs_cluster = bool(qualifying_features & SEMANTIC_CLUSTER_FEATURE_NAMES)
            self.needs_semantic_surprise = bool(qualifying_features & SEMANTIC_SURPRISE_FEATURE_NAMES)
            self.needs_semantic_graph = bool(qualifying_features & SEMANTIC_GRAPH_FEATURE_NAMES)
            self.needs_lexical_surprise = False
            self.needs_lexical_graph = False
            self.needs_entity = False
            self.needs_tfidf_pairwise = False
            self.needs_agreement = bool(qualifying_features & SEMANTIC_AGREEMENT_FEATURE_NAMES)
            self.needs_geometry = bool(qualifying_features & SEMANTIC_GEOMETRY_FEATURE_NAMES)
            self.needs_perturbation = bool(qualifying_features & SEMANTIC_PERTURBATION_FEATURE_NAMES)
            self.needs_distance_weighted_rr = SEMANTIC_DISTANCE_WEIGHTED_TRAINING_RR_FEATURE_NAME in qualifying_features
            self.needs_close_competitor = SEMANTIC_CLOSE_COMPETITOR_FEATURE_NAME in qualifying_features
            self.needs_clarity_score = False


def _select_qualifying(row: dict[str, Any], plan: _LightweightFeaturePlan) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key in _SHARED_ROW_COLUMNS or key in plan.qualifying_features
    }


class LightweightHitAtOnePredictorFeatureBuilder(FeatureBuilder):
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        bundle: DatasetBundle,
        retrieval: RetrievalManager,
        clusters: ClusterManager,
        text_services: TextServices,
        manifest_path: Path | str | None = None,
        consensus_mode: str = CONSENSUS_MODE_FULL,
    ) -> None:
        super().__init__(config, store, console, bundle, retrieval, clusters, text_services)
        raw_manifest_path = Path(manifest_path) if manifest_path is not None else LIGHTWEIGHT_MANIFEST_DEFAULT_PATH
        self.manifest_path = self._resolve_manifest_path(raw_manifest_path)
        self._manifest = self._load_manifest(self.manifest_path)
        self._plan_cache: dict[str, _LightweightFeaturePlan] = {}
        if consensus_mode not in CONSENSUS_MODES:
            raise ValueError(f"Unknown consensus_mode '{consensus_mode}', expected one of {sorted(CONSENSUS_MODES)}")
        self.consensus_mode = consensus_mode
        self._bm25_and_self_rankings_top100_cache: dict[tuple[str, str], dict[str, dict[int, list[int]]]] = {}

    @property
    def manifest_metadata_version(self) -> int:
        return int(self._manifest.get("metadata_version", 0))

    @staticmethod
    def _resolve_manifest_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        direct_path = path.resolve()
        if direct_path.exists():
            return direct_path
        repo_relative_path = (REPO_ROOT / path).resolve()
        if repo_relative_path.exists():
            return repo_relative_path
        return direct_path

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Lightweight hit@1 feature manifest not found at {path} (also tried resolving relative "
                f"to the repo root {REPO_ROOT}). Run "
                "scripts/freeze_lightweight_hit_at_1_feature_manifest.py after training the full hit@1 "
                "predictor for the retrievers you want a lightweight variant for."
            )
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("metadata_version", 0)) != LIGHTWEIGHT_MANIFEST_METADATA_VERSION:
            raise ValueError(
                f"Lightweight hit@1 feature manifest at {path} uses metadata version "
                f"{manifest.get('metadata_version')}, expected {LIGHTWEIGHT_MANIFEST_METADATA_VERSION}. "
                "Regenerate it with scripts/freeze_lightweight_hit_at_1_feature_manifest.py."
            )
        return manifest

    def has_manifest_entry(self, retriever_name: str) -> bool:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        return retriever_name in self._manifest.get("retrievers", {})

    def reload_manifest(self) -> None:
        self._manifest = self._load_manifest(self.manifest_path)
        self._plan_cache.clear()

    def _plan(self, retriever_name: str) -> _LightweightFeaturePlan:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        if retriever_name in self._plan_cache:
            return self._plan_cache[retriever_name]
        entry = self._manifest.get("retrievers", {}).get(retriever_name)
        if entry is None:
            raise KeyError(
                f"No frozen lightweight feature selection for retriever '{retriever_name}' in "
                f"{self.manifest_path}. Train the full hit@1 "
                "predictor for it, then regenerate the manifest with "
                "scripts/freeze_lightweight_hit_at_1_feature_manifest.py."
            )
        qualifying = frozenset(entry["qualifying_features"])
        plan = _LightweightFeaturePlan(qualifying, is_lexical_retriever(retriever_name))
        self._plan_cache[retriever_name] = plan
        return plan

    def qualifying_features(self, retriever_name: str) -> frozenset[str]:
        return self._plan(retriever_name).qualifying_features

    def _lightweight_feature_table_root(self, split_key: str, retriever_name: str) -> Path:
        table_name = (
            "descriptive_lightweight_hit_at_1"
            if self.consensus_mode == CONSENSUS_MODE_FULL
            else f"descriptive_lightweight_hit_at_1_{self.consensus_mode}"
        )
        return self.store.root_dir / "features" / table_name / split_key / retriever_name

    def _load_cached_lightweight_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int],
        plan: _LightweightFeaturePlan,
    ) -> dict[int, pd.DataFrame]:
        root = self._lightweight_feature_table_root(split_key, retriever_name)
        metadata_path = root / "metadata.json"
        if not metadata_path.exists():
            return {}
        metadata = self.store.read_json(metadata_path)
        if int(metadata.get("metadata_version", 0)) != LIGHTWEIGHT_FEATURE_TABLE_SCHEMA_VERSION:
            return {}
        if sorted(metadata.get("qualifying_features", [])) != sorted(plan.qualifying_features):
            return {}
        frames = {}
        for depth in candidate_sizes:
            path = root / f"depth_{int(depth):04d}.parquet"
            if not path.exists():
                continue
            frames[int(depth)] = self.store.read_parquet(path)
        return frames

    def _write_lightweight_feature_tables(
        self,
        split_key: str,
        retriever_name: str,
        frames: dict[int, pd.DataFrame],
        plan: _LightweightFeaturePlan,
    ) -> None:
        root = self._lightweight_feature_table_root(split_key, retriever_name)
        metadata_path = root / "metadata.json"
        existing_depths: set[int] = set()
        if metadata_path.exists():
            existing = self.store.read_json(metadata_path)
            if sorted(existing.get("qualifying_features", [])) == sorted(plan.qualifying_features):
                existing_depths = {int(depth) for depth in existing.get("candidate_sizes", [])}
        for depth, frame in frames.items():
            self.store.write_parquet(root / f"depth_{int(depth):04d}.parquet", frame)
        all_depths = sorted(existing_depths | {int(depth) for depth in frames})
        self.store.write_json(
            metadata_path,
            {
                "metadata_version": LIGHTWEIGHT_FEATURE_TABLE_SCHEMA_VERSION,
                "qualifying_features": sorted(plan.qualifying_features),
                "candidate_sizes": all_depths,
            },
        )

    def _lightweight_checkpoint_namespace(self) -> str:
        return (
            "lightweight_hit_at_1"
            if self.consensus_mode == CONSENSUS_MODE_FULL
            else f"lightweight_hit_at_1_{self.consensus_mode}"
        )

    def build_features(
        self,
        split_key: str,
        retriever_name: str,
        candidate_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> dict[int, pd.DataFrame]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        active_candidate_sizes = self._normalize_candidate_sizes(candidate_sizes)
        cache_key = (split_key, retriever_name, tuple(active_candidate_sizes))
        if cache_key in self._feature_table_cache:
            return self._feature_table_cache[cache_key]
        plan = self._plan(retriever_name)
        cached = self._load_cached_lightweight_features(split_key, retriever_name, active_candidate_sizes, plan)
        missing_candidate_sizes = [depth for depth in active_candidate_sizes if depth not in cached]
        if not missing_candidate_sizes:
            self.console.log(f"Using cached lightweight feature tables for {split_key} / {retriever_name}")
            self._feature_table_cache[cache_key] = cached
            return cached
        self.console.log(f"Generating lightweight feature tables for {split_key} / {retriever_name}")
        self._active_language = split_key.split("_", 1)[0]
        queries = self._limited_queries(split_key)
        query_ids = [int(query_id) for query_id in queries["query_id"].tolist()]
        checkpoint_namespace = self._lightweight_checkpoint_namespace()
        checkpoint = self._prepare_feature_checkpoint(
            split_key,
            retriever_name,
            query_ids,
            missing_candidate_sizes,
            checkpoint_namespace=checkpoint_namespace,
            feature_schema_version=LIGHTWEIGHT_FEATURE_TABLE_SCHEMA_VERSION,
            checkpoint_label="lightweight hit@1 feature",
            plan_identity=sorted(plan.qualifying_features),
        )
        if checkpoint["processed_queries"] >= checkpoint["total_queries"] and checkpoint["batches"]:
            self.console.log(
                f"Resuming lightweight hit@1 feature checkpoint for {split_key} / {retriever_name}: "
                f"all {checkpoint['total_queries']} queries already checkpointed, finalizing"
            )
        elif checkpoint["processed_queries"] > 0:
            self.console.log(
                f"Resuming lightweight hit@1 feature checkpoint for {split_key} / {retriever_name}: "
                f"{checkpoint['processed_queries']}/{checkpoint['total_queries']} queries already computed"
            )
        semantic = not is_lexical_retriever(retriever_name)
        if checkpoint["processed_queries"] < checkpoint["total_queries"]:
            ranking_frame = self.retrieval.ensure_rankings(split_key, retriever_name)
            rankings = self._rankings_by_query(split_key, retriever_name, ranking_frame=ranking_frame)
            query_signals = self._query_signal_lookup(split_key)
            all_rankings_top100: dict[str, dict[int, list[int]]] = {}
            semantic_state = None
            semantic_thresholds = None
            if semantic:
                self.console.log(
                    f"Preparing lightweight semantic prerequisites for {split_key} / {retriever_name} "
                    f"(cluster={plan.needs_cluster}, surprise={plan.needs_semantic_surprise}, "
                    f"graph={plan.needs_semantic_graph}, agreement={plan.needs_agreement}, "
                    f"geometry={plan.needs_geometry}, perturbation={plan.needs_perturbation}, "
                    f"distance_weighted_rr={plan.needs_distance_weighted_rr})"
                )
                if plan.needs_agreement:
                    if self.consensus_mode == CONSENSUS_MODE_BM25_AND_SELF:
                        all_rankings_top100 = self._bm25_and_self_rankings_top100(split_key, retriever_name)
                    else:
                        all_rankings_top100 = self._all_rankings_top100(split_key, retriever_document_scope(retriever_name))
                semantic_state = self._semantic_state(split_key, retriever_name)
                semantic_thresholds = self._semantic_feature_thresholds(retriever_name)
            lexical_artifacts = self.retrieval.ensure_lexical_artifacts(retriever_name) if not semantic else None
            if not semantic and plan.needs_entity:
                self._ensure_document_entities(retriever_name)
            pending_rows: dict[int, list[dict[str, Any]]] = {depth: [] for depth in missing_candidate_sizes}
            pending_query_ids: list[int] = []
            progress = tqdm(
                total=len(queries),
                desc=f"{split_key} {retriever_name} lightweight hit@1 features",
                unit="query",
                initial=checkpoint["processed_queries"],
                **self.console.progress_kwargs(len(queries)),
            )
            try:
                for query_index, query in enumerate(queries.itertuples(index=False)):
                    if query_index < checkpoint["processed_queries"]:
                        continue
                    query_id = int(query.query_id)
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
                            pending_rows[depth].append(_select_qualifying(row, plan))
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
                        agreement_features = (
                            self._retriever_agreement_features(retriever_name, semantic_context, all_rankings_top100)
                            if plan.needs_agreement
                            else {}
                        )
                        for depth in missing_candidate_sizes:
                            row = dict(shared)
                            row["candidate_size"] = depth
                            row["gold_in_top_k"] = float(gold_rank is not None and gold_rank <= depth)
                            row |= lexical_pre
                            row |= semantic_pre_static
                            if plan.needs_distance_weighted_rr:
                                row["distance_weighted_training_query_rr_at_k"] = self._distance_weighted_training_rr(
                                    query,
                                    retriever_name,
                                    depth,
                                    semantic_context["query_embedding"],
                                )
                            row |= self._semantic_score_features(semantic_context, depth, retriever_name)
                            if plan.needs_geometry:
                                row |= self._semantic_geometry_features(semantic_context, depth, retriever_name)
                            if plan.needs_cluster:
                                row |= self._semantic_cluster_features(semantic_context, depth, retriever_name)
                            if plan.needs_agreement:
                                row |= agreement_features
                            if plan.needs_perturbation:
                                row |= self._perturbation_features(query_id, depth, semantic_state, semantic_context)
                            pending_rows[depth].append(_select_qualifying(row, plan))
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
                            checkpoint_namespace=checkpoint_namespace,
                            checkpoint_label="lightweight hit@1 feature",
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
                    checkpoint_namespace=checkpoint_namespace,
                    checkpoint_label="lightweight hit@1 feature",
                )
                checkpoint["batches"].append(batch_info)
                checkpoint["processed_queries"] += batch_info["query_count"]
        generated = self._finalize_feature_checkpoint(
            split_key,
            retriever_name,
            checkpoint["batches"],
            checkpoint["total_queries"],
            missing_candidate_sizes,
            checkpoint_namespace=checkpoint_namespace,
            checkpoint_label="lightweight hit@1 feature",
            frame_writer=lambda sk, rn, frames: self._write_lightweight_feature_tables(sk, rn, frames, plan),
        )
        combined = dict(cached)
        combined.update(generated)
        self._feature_table_cache[cache_key] = combined
        return combined

    def _bm25_and_self_rankings_top100(self, split_key: str, retriever_name: str) -> dict[str, dict[int, list[int]]]:
        cache_key = (split_key, retriever_name)
        if cache_key in self._bm25_and_self_rankings_top100_cache:
            return self._bm25_and_self_rankings_top100_cache[cache_key]
        bm25_name = self.retrieval.paired_retriever("bm25", retriever_name)
        self.console.log(
            f"Preparing bm25-and-self top-100 agreement rankings for {split_key} / {retriever_name} "
            f"(peers: {bm25_name}, {retriever_name})"
        )
        cache = {}
        for name in {bm25_name, retriever_name}:
            frame = self.retrieval.ensure_rankings(split_key, name)
            cache[name] = _stream_top_document_ids_by_query(frame, limit=100)
        self._bm25_and_self_rankings_top100_cache[cache_key] = cache
        return cache

    def _semantic_state(self, split_key: str, retriever_name: str) -> dict[str, Any]:
        cache_key = (split_key, retriever_name)
        if cache_key in self._semantic_state_cache:
            return self._semantic_state_cache[cache_key]
        plan = self._plan(retriever_name)
        spec, document_embeddings = self.retrieval.ensure_semantic_embeddings(retriever_name)
        query_embeddings = self.retrieval.ensure_query_embeddings(split_key, retriever_name)
        index = self.retrieval.ensure_semantic_index_object(retriever_name) if plan.needs_perturbation else None
        if plan.needs_cluster:
            spherical = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "spherical_kmeans")
            hdbscan = self.clusters.ensure_cluster_artifacts(retriever_name, document_embeddings, "hdbscan")
        else:
            spherical = None
            hdbscan = None
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
        return state

    def _semantic_feature_thresholds(self, retriever_name: str) -> dict[str, float]:
        retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        if retriever_name in self._semantic_threshold_bundle_cache:
            return self._semantic_threshold_bundle_cache[retriever_name]
        plan = self._plan(retriever_name)
        bundle = {
            "semantic_close_competitor_threshold": (
                self._semantic_close_competitor_threshold(retriever_name) if plan.needs_close_competitor else math.nan
            ),
            "semantic_surprise_threshold": (
                self._semantic_surprise_threshold(retriever_name) if plan.needs_semantic_surprise else math.nan
            ),
            "semantic_graph_threshold": (
                self._semantic_graph_threshold(retriever_name) if plan.needs_semantic_graph else math.nan
            ),
        }
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

        plan = self._plan(retriever_name)
        query_position = state["query_positions"][query_id]
        query_embedding = state["query_embeddings"][query_position]
        collection_centroid = state["collection_centroid"]
        pre_features: dict[str, float] = {
            "query_collection_centroid_cosine": float(np.dot(query_embedding, collection_centroid)),
        }
        spherical_query_embedding = None
        hdbscan_query_embedding = None
        if plan.needs_cluster:
            spherical = state["spherical"]
            hdbscan = state["hdbscan"]
            spherical_query_embedding = self._cluster_view_query_embedding(query_embedding, spherical)
            hdbscan_query_embedding = self._cluster_view_query_embedding(query_embedding, hdbscan)
            nearest_spherical, spherical_margin = self._nearest_centroid_features(spherical_query_embedding, spherical)
            nearest_hdbscan, hdbscan_margin = self._nearest_centroid_features(hdbscan_query_embedding, hdbscan)
            pre_features.update({
                "nearest_centroid_cosine_spherical": nearest_spherical,
                "nearest_second_nearest_centroid_margin_spherical": spherical_margin,
                "nearest_centroid_cosine_hdbscan": nearest_hdbscan,
                "nearest_second_nearest_centroid_margin_hdbscan": hdbscan_margin,
                "cluster_weighted_corpus_familiarity": self._cluster_weighted_corpus_familiarity(spherical_query_embedding, spherical),
            })
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
        plan = self._plan(retriever_name)
        static_state = self._semantic_query_static_state(query, retriever_name, state)
        query_embedding = static_state["query_embedding"]
        top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
        if plan.needs_geometry:
            document_positions = np.array([self.document_position[int(document_id)] for document_id in top_document_ids], dtype=np.int64)
            top_embeddings = state["document_embeddings"][document_positions]
        else:
            document_positions = None
            top_embeddings = None
        similarities = ranking["score"].to_numpy(dtype=np.float64)
        if threshold_bundle is None:
            threshold_bundle = self._semantic_feature_thresholds(retriever_name)
        if plan.needs_semantic_surprise:
            semantic_surprise_scores = self._semantic_surprise_scores(int(query.query_id), retriever_name, similarities)
        else:
            semantic_surprise_scores = np.full(similarities.shape, math.nan)
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
            "spherical_labels": None,
            "hdbscan_labels": None,
            "semantic_rankings_top100": (
                {
                    name: all_rankings_top100[name][int(query.query_id)]
                    for name in self.retrieval.enabled_semantic_retrievers()
                    if name in all_rankings_top100
                }
                if plan.needs_agreement
                else {}
            ),
            "bm25_top100": (
                all_rankings_top100[self.retrieval.paired_retriever("bm25", retriever_name)][int(query.query_id)]
                if plan.needs_agreement
                else None
            ),
            "semantic_close_competitor_threshold": threshold_bundle["semantic_close_competitor_threshold"],
            "semantic_surprise_threshold": threshold_bundle["semantic_surprise_threshold"],
            "semantic_graph_threshold": threshold_bundle["semantic_graph_threshold"],
            "semantic_surprise_scores": semantic_surprise_scores,
            "retriever_name": retriever_name,
            "query_id": int(query.query_id),
            "index": state["index"],
            "_geometry_cache": {},
        }

    def _lexical_query_context(self, query: Any, ranking: pd.DataFrame, full_scores: np.ndarray, retriever_name: str) -> dict[str, Any]:
        top_document_ids = ranking["document_id"].to_numpy(dtype=np.int64)
        score_values = ranking["score"].to_numpy(dtype=np.float64)
        return {
            "top_document_ids": top_document_ids,
            "scores": score_values,
            "full_scores": np.asarray(full_scores, dtype=np.float64),
            "documents": self.document_lookup.loc[top_document_ids].reset_index(drop=True),
            "entity_sets": None,
        }

    def _lexical_post_features(self, query: Any, context: dict[str, Any], depth: int, retriever_name: str) -> dict[str, float]:
        plan = self._plan(retriever_name)
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
        if plan.needs_tfidf_pairwise or plan.needs_lexical_graph:
            tfidf_similarity = (stats.tfidf_matrix[positions] @ stats.tfidf_matrix[positions].T).toarray()
        else:
            tfidf_similarity = None
        if plan.needs_tfidf_pairwise:
            tfidf_min, tfidf_max, tfidf_mean, _ = _pairwise_summary(tfidf_similarity)
        else:
            tfidf_min = tfidf_max = tfidf_mean = math.nan
        if plan.needs_entity:
            entity_matrix = self._entity_similarity_matrix(top_ids, retriever_name)
            entity_min, entity_max, entity_mean, _ = _pairwise_summary(entity_matrix)
            entity_overlap_value = self._entity_overlap(query.query_id, query.split_key, top1["document_id"], retriever_name)
        else:
            entity_min = entity_max = entity_mean = math.nan
            entity_overlap_value = math.nan
        if plan.needs_lexical_graph:
            graph_threshold = self._lexical_graph_threshold(retriever_name)
            graph_density, component_share = self._graph_features(tfidf_similarity, graph_threshold)
        else:
            graph_density = component_share = math.nan
        clarity_score_value = self._clarity_score(retriever_name, top_ids, scores) if plan.needs_clarity_score else math.nan
        full_scores = context["full_scores"]
        positive_mass = full_scores[full_scores > 0].sum()
        if plan.needs_lexical_surprise:
            surprise_scores = self._lexical_surprise_scores(query.query_id, retriever_name, context["scores"])
            surprise_threshold = self._lexical_surprise_threshold(retriever_name)
        else:
            surprise_scores = np.full(context["scores"].shape, math.nan)
            surprise_threshold = math.nan
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
            "top_1_title_abstract_entity_overlap": entity_overlap_value,
            "top_1_query_term_coverage": _ratio(len(set(indexed_query_tokens) & top1_terms), len(set(indexed_query_tokens))) if indexed_query_tokens else math.nan,
            "top_1_unmatched_idf_mass_ratio": (idf_mass_missing / idf_mass_total) if indexed_query_tokens and idf_mass_total else math.nan,
            "minimum_top_k_pairwise_tfidf_similarity": tfidf_min,
            "maximum_top_k_pairwise_tfidf_similarity": tfidf_max,
            "mean_top_k_pairwise_tfidf_similarity": tfidf_mean,
            "minimum_top_k_pairwise_entity_similarity": entity_min,
            "maximum_top_k_pairwise_entity_similarity": entity_max,
            "mean_top_k_pairwise_entity_similarity": entity_mean,
            "clarity_score": clarity_score_value,
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


def ensure_manifest_entry(
    config: ProjectConfig,
    store: ArtifactStore,
    console: Console,
    full_features: FeatureBuilder,
    retrieval: RetrievalManager,
    language: str,
    retriever_name: str,
    lightweight_features: LightweightHitAtOnePredictorFeatureBuilder,
) -> None:
    from query_difficulty.modeling import PREDICTOR_MODE_LOCAL, HitAtOnePredictorRunner, hit_at_1_model_is_current

    retriever_name = retrieval.resolve_retriever_name(retriever_name)
    if lightweight_features.has_manifest_entry(retriever_name):
        return
    if not hit_at_1_model_is_current(store, retriever_name):
        console.log(
            f"No full hit@1 model for {retriever_name}; training it first so a lightweight feature "
            "manifest entry can be derived from its feature importances."
        )
        full_runner = HitAtOnePredictorRunner(config, store, console, full_features, retrieval)
        full_runner._run_retriever(language, retriever_name, PREDICTOR_MODE_LOCAL)
    threshold = float(lightweight_features._manifest.get("importance_threshold", 0.005))
    console.log(
        f"Freezing lightweight feature manifest entry for {retriever_name} from its full model "
        f"(threshold={threshold:g}, multi-retriever consensus features excluded from candidacy by default, "
        "matching the bm25/qwen3_embedding_0_6b adoption). Only this retriever's entry is added; every "
        "other retriever's existing manifest entry is left untouched."
    )
    add_manifest_entry(
        store,
        lightweight_features.manifest_path,
        retriever_name,
        threshold=threshold,
        exclude_multi_retriever_consensus=True,
        console=console,
    )
    lightweight_features.reload_manifest()
