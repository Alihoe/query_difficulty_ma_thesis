from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from query_difficulty.schema import SemanticRetrieverSpec


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RuntimeConfig:
    random_seed: int
    n_jobs: int
    allow_resource_downloads: bool
    limit_queries: int | None
    feature_checkpoint_batch_size: int = 100


@dataclass(slots=True)
class OrdinalPredictorConfig:
    base_model: str


@dataclass(slots=True)
class QppPredictorConfig:
    failure_model: str
    regressor: str
    mapping: str
    score_source: str
    minimum_size: int
    maximum_size: int


@dataclass(slots=True)
class TunedQppPredictorConfig:
    failure_model: str
    regressor: str
    score_source: str
    minimum_size: int
    maximum_size: int


@dataclass(slots=True)
class PredictorConfig:
    ordinal: OrdinalPredictorConfig
    qpp: QppPredictorConfig
    tuned_qpp: TunedQppPredictorConfig


@dataclass(slots=True)
class ClusteringConfig:
    protocol_version: str = "paper_title_abstract_clusters_v1"
    useful_cluster_size_min: int = 10
    useful_cluster_size_max: int = 1000
    hdbscan_umap_metric: str = "cosine"
    hdbscan_umap_n_neighbors: int = 10
    hdbscan_umap_n_components: int = 20
    hdbscan_umap_min_dist: float = 0.0
    hdbscan_umap_random_state: int = 42
    hdbscan_metric: str = "euclidean"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = 5
    hdbscan_cluster_selection_method: str = "eom"
    spherical_k_min: int = 2
    spherical_k_max: int = 1000
    spherical_largest_cluster_share_cap: float = 0.03
    spherical_validation_candidates: int = 5
    spherical_validation_runs: int = 5
    spherical_iterations: int = 25
    silhouette_sample_size: int = 2048


@dataclass(slots=True)
class ProjectConfig:
    project_name: str
    root_dir: Path
    data_dir: Path
    artifact_dir: Path
    latex_table_dir: Path
    languages: list[str]
    enabled_document_scopes: list[str]
    predictor_feature_modes: list[str]
    candidate_sizes: list[int]
    max_ranking_depth: int
    enabled_feature_blocks: list[str]
    enabled_clusterers: list[str]
    enabled_lexical_retrievers: list[str]
    enabled_semantic_retrievers: list[str]
    runtime: RuntimeConfig
    predictors: PredictorConfig
    clustering: ClusteringConfig
    semantic_retrievers: dict[str, SemanticRetrieverSpec]
    full_scope_retrievers: list[str] = field(default_factory=list)

    def semantic_specs(self) -> list[SemanticRetrieverSpec]:
        return [self.semantic_retrievers[name] for name in self.enabled_semantic_retrievers]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_config_path(config_path: str | Path) -> Path:
    raw_path = Path(config_path)
    if raw_path.is_absolute():
        return raw_path
    direct_path = raw_path.resolve()
    if direct_path.exists():
        return direct_path
    repo_relative_path = (REPO_ROOT / raw_path).resolve()
    if repo_relative_path.exists():
        return repo_relative_path
    searched = [str(direct_path), str(repo_relative_path)]
    raise FileNotFoundError(
        f"Could not find config file '{config_path}'. Tried: {', '.join(searched)}"
    )


def load_project_config(config_path: str | Path = "config/default.yaml") -> ProjectConfig:
    path = _resolve_config_path(config_path)
    root_dir = path.parent.parent
    payload = _load_yaml(path)
    runtime = RuntimeConfig(**payload["runtime"])
    predictors = PredictorConfig(
        ordinal=OrdinalPredictorConfig(**payload["predictors"]["ordinal"]),
        qpp=QppPredictorConfig(**payload["predictors"]["qpp"]),
        tuned_qpp=TunedQppPredictorConfig(**payload["predictors"]["tuned_qpp"]),
    )
    clustering = ClusteringConfig(**payload.get("clustering", {}))
    semantic_specs = {
        name: SemanticRetrieverSpec(name=name, **spec)
        for name, spec in payload["semantic_retrievers"].items()
    }
    return ProjectConfig(
        project_name=payload["project_name"],
        root_dir=root_dir,
        data_dir=(root_dir / payload["data_dir"]).resolve(),
        artifact_dir=(root_dir / payload["artifact_dir"]).resolve(),
        latex_table_dir=(root_dir / payload["latex_table_dir"]).resolve(),
        languages=list(payload["languages"]),
        enabled_document_scopes=list(payload.get("enabled_document_scopes", ["title_abstract"])),
        predictor_feature_modes=list(payload.get("predictor_feature_modes", ["local"])),
        candidate_sizes=list(payload["candidate_sizes"]),
        max_ranking_depth=int(payload["max_ranking_depth"]),
        enabled_feature_blocks=list(payload["enabled_feature_blocks"]),
        enabled_clusterers=list(payload["enabled_clusterers"]),
        enabled_lexical_retrievers=list(payload["enabled_lexical_retrievers"]),
        enabled_semantic_retrievers=list(payload["enabled_semantic_retrievers"]),
        runtime=runtime,
        predictors=predictors,
        clustering=clustering,
        semantic_retrievers=semantic_specs,
        full_scope_retrievers=list(payload.get("full_scope_retrievers", [])),
    )
