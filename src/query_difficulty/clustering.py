from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.config import ProjectConfig
from query_difficulty.console import Console


def _patch_sklearn_check_array_for_umap() -> None:
    import sklearn.utils
    import sklearn.utils.validation

    check_array = sklearn.utils.check_array
    parameters = inspect.signature(check_array).parameters
    if "force_all_finite" in parameters or "ensure_all_finite" not in parameters:
        return

    original = check_array

    def _compat_check_array(*args: Any, force_all_finite: Any = None, **kwargs: Any) -> Any:
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return original(*args, **kwargs)

    sklearn.utils.check_array = _compat_check_array
    sklearn.utils.validation.check_array = _compat_check_array
    try:
        import umap.umap_ as umap_module
    except Exception:
        return
    umap_module.check_array = _compat_check_array


def _ensure_numba_cache_dir(root_dir: Any) -> None:
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = root_dir / "tmp" / "numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir.resolve())


@dataclass(slots=True)
class ClusterArtifacts:
    retriever_name: str
    algorithm: str
    assignments: pd.DataFrame
    centroids: np.ndarray
    centroid_labels: list[int]
    summary: dict[str, Any]
    view_name: str = "raw_l2"
    view_state: dict[str, Any] = field(default_factory=dict)
    label_lookup: dict[int, int] = field(default_factory=dict)
    label_share_lookup: dict[int, float] = field(default_factory=dict)


class ClusterManager:
    def __init__(
        self,
        config: ProjectConfig,
        store: ArtifactStore,
        console: Console,
        document_ids: np.ndarray,
    ) -> None:
        self.config = config
        self.store = store
        self.console = console
        self.document_ids = document_ids
        self._cluster_artifact_cache: dict[tuple[str, str], ClusterArtifacts] = {}
        _ensure_numba_cache_dir(self.store.root_dir)

    def ensure_cluster_artifacts(self, retriever_name: str, embeddings: np.ndarray, algorithm: str) -> ClusterArtifacts:
        cache_key = (retriever_name, algorithm)
        if cache_key in self._cluster_artifact_cache:
            self.console.log(f"Using in-memory {algorithm} clusters for {retriever_name}")
            return self._cluster_artifact_cache[cache_key]

        root = self.store.path("clusters", retriever_name, algorithm, "summary.json").parent
        summary_path = root / "summary.json"
        assignments_path = root / "assignments.parquet"
        centroids_path = root / "centroids.npy"
        centroid_labels_path = root / "centroid_labels.json"
        view_path = root / "view.json"
        view_state_path = root / "view_state.joblib"
        trace_path = root / "selection_trace.json"

        if (
            summary_path.exists()
            and assignments_path.exists()
            and centroids_path.exists()
            and centroid_labels_path.exists()
            and view_path.exists()
            and view_state_path.exists()
            and trace_path.exists()
        ):
            summary = self.store.read_json(summary_path)
            if self._summary_matches_algorithm_config(summary, algorithm):
                self.console.log(f"Using cached {algorithm} clusters for {retriever_name}")
                view_metadata = self.store.read_json(view_path)
                artifacts = ClusterArtifacts(
                    retriever_name=retriever_name,
                    algorithm=algorithm,
                    assignments=self.store.read_parquet(assignments_path),
                    centroids=self.store.read_numpy(centroids_path),
                    centroid_labels=[int(value) for value in self.store.read_json(centroid_labels_path)],
                    summary=summary,
                    view_name=str(view_metadata.get("name", "raw_l2")),
                    view_state=self.store.read_joblib(view_state_path),
                )
                artifacts.label_lookup = self._label_lookup(artifacts.assignments)
                artifacts.label_share_lookup = self._label_share_lookup(artifacts.assignments)
                self._cluster_artifact_cache[cache_key] = artifacts
                return artifacts

        if algorithm == "hdbscan":
            artifacts, trace = self._fit_hdbscan(retriever_name, embeddings)
        elif algorithm == "spherical_kmeans":
            artifacts, trace = self._fit_spherical_kmeans(retriever_name, embeddings)
        else:
            raise ValueError(f"Unsupported clustering algorithm '{algorithm}'")

        self.store.write_parquet(assignments_path, artifacts.assignments)
        self.store.write_numpy(centroids_path, artifacts.centroids.astype(np.float32, copy=False))
        self.store.write_json(centroid_labels_path, [int(value) for value in artifacts.centroid_labels])
        self.store.write_json(summary_path, artifacts.summary)
        self.store.write_json(view_path, {"name": artifacts.view_name})
        self.store.write_joblib(view_state_path, artifacts.view_state)
        self.store.write_json(trace_path, trace)
        self._cluster_artifact_cache[cache_key] = artifacts
        return artifacts

    def project_embeddings_for_cluster_view(self, embeddings: np.ndarray, artifacts: ClusterArtifacts) -> np.ndarray:
        return self._apply_embedding_view(embeddings, artifacts.view_name, artifacts.view_state)

    def _fit_hdbscan(self, retriever_name: str, embeddings: np.ndarray) -> tuple[ClusterArtifacts, dict[str, Any]]:
        self.console.log(f"Clustering {retriever_name} with HDBSCAN protocol {self.config.clustering.protocol_version}")
        normalized = self._normalize_rows(embeddings)
        reducer = self._fit_umap(normalized)
        projected = reducer.fit_transform(normalized).astype(np.float32, copy=False)
        labels = HDBSCAN(
            metric=self.config.clustering.hdbscan_metric,
            min_cluster_size=int(self.config.clustering.hdbscan_min_cluster_size),
            min_samples=self.config.clustering.hdbscan_min_samples,
            cluster_selection_method=self.config.clustering.hdbscan_cluster_selection_method,
            copy=False,
        ).fit_predict(projected)
        centroids, centroid_labels = self._cluster_centroids(projected, labels, normalize=False)
        metrics = self._cluster_distribution_metrics(labels, len(projected))
        summary = self._cluster_summary(labels, centroid_labels, "hdbscan")
        summary.update(
            {
                "protocol_version": self.config.clustering.protocol_version,
                "embedding_view": "umap",
                "selection_parameters": {
                    "umap_metric": self.config.clustering.hdbscan_umap_metric,
                    "n_neighbors": int(self.config.clustering.hdbscan_umap_n_neighbors),
                    "n_components": int(self.config.clustering.hdbscan_umap_n_components),
                    "min_dist": float(self.config.clustering.hdbscan_umap_min_dist),
                    "random_state": int(self.config.clustering.hdbscan_umap_random_state),
                    "hdbscan_metric": self.config.clustering.hdbscan_metric,
                    "hdbscan_min_cluster_size": int(self.config.clustering.hdbscan_min_cluster_size),
                    "hdbscan_min_samples": (
                        None
                        if self.config.clustering.hdbscan_min_samples is None
                        else int(self.config.clustering.hdbscan_min_samples)
                    ),
                    "cluster_selection_method": self.config.clustering.hdbscan_cluster_selection_method,
                },
                "analysis_metric": "euclidean",
                "centroid_query_metric": "inverse_euclidean",
                **metrics,
            }
        )
        assignments = pd.DataFrame({"document_id": self.document_ids, "label": labels.astype(np.int32)})
        trace = {
            "protocol_version": self.config.clustering.protocol_version,
            "algorithm": "hdbscan",
            "embedding_view": "umap",
            "selection_parameters": summary["selection_parameters"],
            "summary": summary,
        }
        return (
            ClusterArtifacts(
                retriever_name=retriever_name,
                algorithm="hdbscan",
                assignments=assignments,
                centroids=centroids,
                centroid_labels=centroid_labels,
                summary=summary,
                view_name="umap",
                view_state={
                    "reducer": reducer,
                    "input_normalization": "l2",
                    "analysis_metric": "euclidean",
                    "centroid_query_metric": "inverse_euclidean",
                },
                label_lookup=self._label_lookup(assignments),
                label_share_lookup=self._label_share_lookup(assignments),
            ),
            trace,
        )

    def _fit_spherical_kmeans(self, retriever_name: str, embeddings: np.ndarray) -> tuple[ClusterArtifacts, dict[str, Any]]:
        self.console.log(
            f"Clustering {retriever_name} with spherical k-means protocol {self.config.clustering.protocol_version}"
        )
        normalized = self._normalize_rows(embeddings)
        minimum_k, maximum_k = self._spherical_k_range(len(normalized))
        coarse_candidates = self._coarse_candidate_grid(minimum_k, maximum_k)
        effective_cap = self._spherical_effective_largest_cluster_share_cap(len(normalized))

        coarse_records = [
            self._evaluate_spherical_candidate(normalized, int(k), self._spherical_seed("coarse", int(k), offset=index))
            for index, k in enumerate(coarse_candidates)
        ]
        admissible_coarse = [record for record in coarse_records if record["largest_cluster_share_clustered"] <= effective_cap]
        best_coarse = max(admissible_coarse or coarse_records, key=self._spherical_candidate_sort_key)
        best_coarse_index = coarse_candidates.index(int(best_coarse["k"]))

        exhaustive_min_k, exhaustive_max_k = self._refined_interval(
            coarse_candidates,
            best_coarse_index,
            minimum_k,
            maximum_k,
        )
        exhaustive_records = [
            self._evaluate_spherical_candidate(
                normalized,
                int(k),
                self._spherical_seed("exhaustive", int(k), offset=index),
            )
            for index, k in enumerate(range(exhaustive_min_k, exhaustive_max_k + 1))
        ]
        admissible_exhaustive = [
            record for record in exhaustive_records if record["largest_cluster_share_clustered"] <= effective_cap
        ]
        validation_candidates = sorted(
            admissible_exhaustive or exhaustive_records,
            key=self._spherical_candidate_sort_key,
            reverse=True,
        )[: max(1, int(self.config.clustering.spherical_validation_candidates))]

        validation_records: list[dict[str, Any]] = []
        for candidate in validation_candidates:
            k = int(candidate["k"])
            runs = [
                self._evaluate_spherical_candidate(
                    normalized,
                    k,
                    self._spherical_seed("validation", k, offset=run_index),
                )
                for run_index in range(int(self.config.clustering.spherical_validation_runs))
            ]
            validation_records.append(
                {
                    "k": k,
                    "mean_silhouette_cosine": float(np.mean([run["silhouette_cosine"] for run in runs])),
                    "mean_largest_cluster_share_clustered": float(
                        np.mean([run["largest_cluster_share_clustered"] for run in runs])
                    ),
                    "seed_stability": self._label_stability([run["labels"] for run in runs]),
                    "runs": runs,
                }
            )

        ranked_validation = sorted(validation_records, key=self._spherical_validation_sort_key, reverse=True)
        selected_validation = ranked_validation[0]
        selected_run = max(selected_validation["runs"], key=self._spherical_candidate_sort_key)
        selected_k = int(selected_validation["k"])
        second_validation_score = (
            float(ranked_validation[1]["mean_silhouette_cosine"]) if len(ranked_validation) > 1 else math.nan
        )

        labels = selected_run["labels"]
        centroids = selected_run["centroids"]
        centroid_labels = selected_run["centroid_labels"]
        metrics = self._cluster_distribution_metrics(labels, len(normalized))
        summary = self._cluster_summary(labels, centroid_labels, "spherical_kmeans")
        summary.update(
            {
                "protocol_version": self.config.clustering.protocol_version,
                "embedding_view": "raw_l2",
                "selection_strategy": "paper_coarse_to_fine_with_largest_cluster_share_cap",
                "selection_parameters": {
                    "largest_cluster_share_cap": float(self.config.clustering.spherical_largest_cluster_share_cap),
                    "validation_candidates": int(self.config.clustering.spherical_validation_candidates),
                    "validation_runs": int(self.config.clustering.spherical_validation_runs),
                    "iterations": int(self.config.clustering.spherical_iterations),
                },
                "analysis_metric": "cosine",
                "centroid_query_metric": "cosine",
                "selected_k": selected_k,
                "selection_mean_score": float(selected_validation["mean_silhouette_cosine"]),
                "selection_second_mean_score": second_validation_score,
                "selection_margin": (
                    float(selected_validation["mean_silhouette_cosine"]) - second_validation_score
                    if math.isfinite(second_validation_score)
                    else math.nan
                ),
                "largest_cluster_share_cap": float(self.config.clustering.spherical_largest_cluster_share_cap),
                "effective_largest_cluster_share_cap": float(effective_cap),
                "seed_stability": float(selected_validation["seed_stability"]),
                **metrics,
            }
        )
        assignments = pd.DataFrame({"document_id": self.document_ids, "label": labels.astype(np.int32)})
        trace = {
            "protocol_version": self.config.clustering.protocol_version,
            "algorithm": "spherical_kmeans",
            "embedding_view": "raw_l2",
            "largest_cluster_share_cap": float(self.config.clustering.spherical_largest_cluster_share_cap),
            "effective_largest_cluster_share_cap": float(effective_cap),
            "coarse_candidates": [int(value) for value in coarse_candidates],
            "validation_records": [self._serialize_validation_record(record) for record in ranked_validation],
            "selected_k": selected_k,
        }
        return (
            ClusterArtifacts(
                retriever_name=retriever_name,
                algorithm="spherical_kmeans",
                assignments=assignments,
                centroids=centroids,
                centroid_labels=centroid_labels,
                summary=summary,
                view_name="raw_l2",
                view_state={},
                label_lookup=self._label_lookup(assignments),
                label_share_lookup=self._label_share_lookup(assignments),
            ),
            trace,
        )

    def _fit_umap(self, embeddings: np.ndarray) -> Any:
        _ensure_numba_cache_dir(self.store.root_dir)
        _patch_sklearn_check_array_for_umap()
        from umap.umap_ import UMAP

        return UMAP(
            n_neighbors=int(self.config.clustering.hdbscan_umap_n_neighbors),
            n_components=int(self.config.clustering.hdbscan_umap_n_components),
            min_dist=float(self.config.clustering.hdbscan_umap_min_dist),
            metric=self.config.clustering.hdbscan_umap_metric,
            random_state=int(self.config.clustering.hdbscan_umap_random_state),
            transform_seed=int(self.config.clustering.hdbscan_umap_random_state),
            low_memory=True,
        )

    def _apply_embedding_view(self, embeddings: np.ndarray, view_name: str, view_state: dict[str, Any]) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype(np.float32, copy=False)
        if view_name == "raw_l2":
            return self._normalize_rows(embeddings)
        if view_name == "umap":
            _patch_sklearn_check_array_for_umap()
            reducer = view_state["reducer"]
            rows = embeddings.astype(np.float32, copy=False)
            if view_state.get("input_normalization") == "l2":
                rows = self._normalize_rows(rows)
            return reducer.transform(rows).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported cluster embedding view '{view_name}'")

    def _evaluate_spherical_candidate(self, embeddings: np.ndarray, k: int, seed: int) -> dict[str, Any]:
        labels, centroids = self._run_spherical_kmeans(embeddings, k, seed)
        metrics = self._cluster_distribution_metrics(labels, len(embeddings))
        return {
            "k": int(k),
            "seed": int(seed),
            "labels": labels,
            "centroids": centroids,
            "centroid_labels": [int(value) for value in np.unique(labels)],
            "silhouette_cosine": self._full_silhouette_cosine(embeddings, labels),
            "largest_cluster_share_clustered": float(metrics["largest_cluster_share_clustered"]),
            "effective_cluster_count": float(metrics["effective_cluster_count"]),
        }

    def _run_spherical_kmeans(self, embeddings: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        trainer = faiss.Kmeans(
            embeddings.shape[1],
            int(k),
            niter=int(self.config.clustering.spherical_iterations),
            nredo=1,
            spherical=True,
            verbose=False,
            gpu=False,
            seed=int(seed),
        )
        embeddings32 = embeddings.astype(np.float32, copy=False)
        trainer.train(embeddings32)
        _, indices = trainer.index.search(embeddings32, 1)
        labels = indices.ravel().astype(np.int32)
        centroids, _ = self._cluster_centroids(embeddings32, labels, normalize=True)
        return labels, centroids

    def _summary_matches_algorithm_config(self, summary: dict[str, Any], algorithm: str) -> bool:
        if summary.get("protocol_version") != self.config.clustering.protocol_version:
            return False
        if summary.get("algorithm") != algorithm:
            return False
        parameters = summary.get("selection_parameters", {})
        if algorithm == "hdbscan":
            return (
                summary.get("embedding_view") == "umap"
                and parameters.get("umap_metric") == self.config.clustering.hdbscan_umap_metric
                and int(parameters.get("n_neighbors", -1)) == int(self.config.clustering.hdbscan_umap_n_neighbors)
                and int(parameters.get("n_components", -1)) == int(self.config.clustering.hdbscan_umap_n_components)
                and math.isclose(
                    float(parameters.get("min_dist", math.nan)),
                    float(self.config.clustering.hdbscan_umap_min_dist),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and int(parameters.get("random_state", -1)) == int(self.config.clustering.hdbscan_umap_random_state)
                and parameters.get("hdbscan_metric") == self.config.clustering.hdbscan_metric
                and int(parameters.get("hdbscan_min_cluster_size", -1))
                == int(self.config.clustering.hdbscan_min_cluster_size)
                and parameters.get("hdbscan_min_samples") == self.config.clustering.hdbscan_min_samples
                and parameters.get("cluster_selection_method") == self.config.clustering.hdbscan_cluster_selection_method
            )
        if algorithm == "spherical_kmeans":
            return (
                summary.get("selection_strategy") == "paper_coarse_to_fine_with_largest_cluster_share_cap"
                and math.isclose(
                    float(summary.get("largest_cluster_share_cap", math.nan)),
                    float(self.config.clustering.spherical_largest_cluster_share_cap),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and int(parameters.get("validation_candidates", -1))
                == int(self.config.clustering.spherical_validation_candidates)
                and int(parameters.get("validation_runs", -1))
                == int(self.config.clustering.spherical_validation_runs)
                and int(parameters.get("iterations", -1)) == int(self.config.clustering.spherical_iterations)
            )
        return False

    def _spherical_k_range(self, document_count: int) -> tuple[int, int]:
        maximum_k = max(1, min(document_count, int(self.config.clustering.spherical_k_max)))
        minimum_k = 1 if maximum_k == 1 else min(maximum_k, max(2, int(self.config.clustering.spherical_k_min)))
        return minimum_k, maximum_k

    def _spherical_effective_largest_cluster_share_cap(self, document_count: int) -> float:
        if document_count <= 0:
            return 1.0
        minimum_useful_share = self.config.clustering.useful_cluster_size_min / float(document_count)
        return min(
            1.0,
            max(
                float(self.config.clustering.spherical_largest_cluster_share_cap),
                minimum_useful_share,
                1.0 / float(document_count),
            ),
        )

    @staticmethod
    def _coarse_candidate_grid(minimum_k: int, maximum_k: int) -> list[int]:
        if minimum_k >= maximum_k:
            return [int(minimum_k)]
        coarse_step = max(1, int(round(maximum_k ** (2.0 / 3.0))))
        values = list(range(minimum_k, maximum_k + 1, coarse_step))
        if values[-1] != maximum_k:
            values.append(maximum_k)
        return values

    @staticmethod
    def _refined_interval(
        coarse_candidates: list[int],
        best_index: int,
        minimum_k: int,
        maximum_k: int,
    ) -> tuple[int, int]:
        left_midpoint = (
            minimum_k
            if best_index == 0
            else int(math.floor((coarse_candidates[best_index - 1] + coarse_candidates[best_index]) / 2.0)) + 1
        )
        right_midpoint = (
            maximum_k
            if best_index == len(coarse_candidates) - 1
            else int(math.floor((coarse_candidates[best_index] + coarse_candidates[best_index + 1]) / 2.0))
        )
        return max(minimum_k, left_midpoint), min(maximum_k, right_midpoint)

    def _spherical_seed(self, stage: str, k: int, offset: int = 0) -> int:
        stage_offsets = {
            "coarse": 0,
            "exhaustive": 4,
            "validation": 88,
        }
        if stage not in stage_offsets:
            raise ValueError(f"Unsupported spherical k-means stage '{stage}'")
        return int(self.config.runtime.random_seed + stage_offsets[stage] + k + offset)

    @staticmethod
    def _spherical_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(candidate["silhouette_cosine"]),
            -float(candidate["largest_cluster_share_clustered"]),
            float(candidate["effective_cluster_count"]),
            -int(candidate["k"]),
        )

    @staticmethod
    def _spherical_validation_sort_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(record["mean_silhouette_cosine"]),
            -float(record["mean_largest_cluster_share_clustered"]),
            float(record["seed_stability"]) if math.isfinite(float(record["seed_stability"])) else float("-inf"),
            -int(record["k"]),
        )

    @staticmethod
    def _serialize_validation_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "k": int(record["k"]),
            "mean_silhouette_cosine": float(record["mean_silhouette_cosine"]),
            "mean_largest_cluster_share_clustered": float(record["mean_largest_cluster_share_clustered"]),
            "seed_stability": float(record["seed_stability"]),
            "runs": [
                {
                    "seed": int(run["seed"]),
                    "silhouette_cosine": float(run["silhouette_cosine"]),
                    "largest_cluster_share_clustered": float(run["largest_cluster_share_clustered"]),
                    "effective_cluster_count": float(run["effective_cluster_count"]),
                }
                for run in record["runs"]
            ],
        }

    @staticmethod
    def _cluster_distribution_metrics(labels: np.ndarray, document_count: int) -> dict[str, float | int]:
        non_noise_labels = labels[labels != -1]
        clustered_document_count = int(len(non_noise_labels))
        if clustered_document_count == 0:
            return {
                "clustered_document_count": 0,
                "coverage": 0.0,
                "noise_fraction": float(np.mean(labels == -1)) if document_count else 0.0,
                "largest_cluster_share_all": 0.0,
                "largest_cluster_share_clustered": 0.0,
                "effective_cluster_count": 0.0,
                "normalized_cluster_entropy": 0.0,
                "clusters_leq_10": 0,
                "doc_share_in_clusters_leq_10": 0.0,
            }
        _, counts = np.unique(non_noise_labels, return_counts=True)
        proportions = counts.astype(np.float64) / float(clustered_document_count)
        entropy = float(-np.sum(proportions * np.log(proportions)))
        normalized_entropy = 0.0 if len(counts) <= 1 else float(entropy / np.log(len(counts)))
        effective_cluster_count = float(np.exp(entropy))
        tiny_mask = counts <= 10
        return {
            "clustered_document_count": clustered_document_count,
            "coverage": float(clustered_document_count / float(document_count)) if document_count else 0.0,
            "noise_fraction": float(np.mean(labels == -1)),
            "largest_cluster_share_all": float(counts.max() / float(document_count)) if document_count else 0.0,
            "largest_cluster_share_clustered": float(counts.max() / float(clustered_document_count)),
            "effective_cluster_count": effective_cluster_count,
            "normalized_cluster_entropy": normalized_entropy,
            "clusters_leq_10": int(np.sum(tiny_mask)),
            "doc_share_in_clusters_leq_10": float(counts[tiny_mask].sum() / float(clustered_document_count)),
        }

    @staticmethod
    def _full_silhouette_cosine(embeddings: np.ndarray, labels: np.ndarray) -> float:
        if len(labels) < 3 or len(np.unique(labels)) < 2:
            return float("nan")
        try:
            return float(silhouette_score(embeddings, labels, metric="cosine"))
        except ValueError:
            return float("nan")

    @staticmethod
    def _label_stability(label_sets: list[np.ndarray]) -> float:
        if len(label_sets) < 2:
            return math.nan
        scores = []
        for left_index in range(len(label_sets)):
            for right_index in range(left_index + 1, len(label_sets)):
                scores.append(float(adjusted_mutual_info_score(label_sets[left_index], label_sets[right_index])))
        return float(np.mean(scores)) if scores else math.nan

    @staticmethod
    def _cluster_centroids(
        embeddings: np.ndarray,
        labels: np.ndarray,
        normalize: bool,
    ) -> tuple[np.ndarray, list[int]]:
        centroids = []
        centroid_labels = []
        for label in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0).astype(np.float32, copy=False)
            if normalize:
                norm = float(np.linalg.norm(centroid))
                if norm == 0.0:
                    continue
                centroid = centroid / norm
            centroids.append(centroid.astype(np.float32, copy=False))
            centroid_labels.append(label)
        if not centroids:
            return np.zeros((0, embeddings.shape[1]), dtype=np.float32), []
        return np.vstack(centroids), centroid_labels

    @staticmethod
    def _cluster_summary(labels: np.ndarray, centroid_labels: list[int], algorithm: str) -> dict[str, Any]:
        counts = [int(np.sum(labels == label)) for label in centroid_labels]
        if not counts:
            return {
                "algorithm": algorithm,
                "cluster_count": 0,
                "minimum_cluster_size": 0,
                "mean_cluster_size": 0.0,
                "maximum_cluster_size": 0,
            }
        return {
            "algorithm": algorithm,
            "cluster_count": len(centroid_labels),
            "minimum_cluster_size": int(min(counts)),
            "mean_cluster_size": float(np.mean(counts)),
            "maximum_cluster_size": int(max(counts)),
        }

    @staticmethod
    def _label_lookup(assignments: pd.DataFrame) -> dict[int, int]:
        return {int(document_id): int(label) for document_id, label in zip(assignments["document_id"], assignments["label"], strict=True)}

    @staticmethod
    def _label_share_lookup(assignments: pd.DataFrame) -> dict[int, float]:
        total = float(len(assignments))
        if total == 0:
            return {}
        counts = assignments["label"].value_counts().to_dict()
        return {int(label): float(count) / total for label, count in counts.items()}

    @staticmethod
    def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype(np.float32, copy=False)
        rows = embeddings.astype(np.float64, copy=True)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (rows / norms).astype(np.float32)
