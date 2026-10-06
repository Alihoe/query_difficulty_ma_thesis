from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from query_difficulty.artifacts import ArtifactStore
from query_difficulty.clustering import ClusterManager
from query_difficulty.config import ProjectConfig, load_project_config
from query_difficulty.console import Console
from query_difficulty.dataset import DatasetManager
from query_difficulty.features import FeatureBuilder
from query_difficulty.modeling import configure_random_forest_n_jobs
from query_difficulty.retrieval import RetrievalManager
from query_difficulty.text import TextServices


@dataclass(slots=True)
class ProjectRuntime:
    config: ProjectConfig
    console: Console
    store: ArtifactStore
    dataset: DatasetManager
    retrieval: RetrievalManager
    clustering: ClusterManager
    text_services: TextServices
    features: FeatureBuilder


def build_runtime(config_path: str = "config/default.yaml", verbose: bool = True) -> ProjectRuntime:
    overall_started_at = perf_counter()
    config_started_at = perf_counter()
    config = load_project_config(config_path)
    config_elapsed = perf_counter() - config_started_at
    console = Console(verbose=verbose)
    console.log_timing("Loaded project configuration", config_elapsed, details=f"path={config_path}")
    configure_random_forest_n_jobs(config.runtime.n_jobs)
    store_started_at = perf_counter()
    store = ArtifactStore(config.artifact_dir)
    console.log_timing("Initialized artifact store", perf_counter() - store_started_at, details=f"root={store.root_dir}")
    dataset_started_at = perf_counter()
    dataset = DatasetManager(config.data_dir, store, console)
    bundle = dataset.load()
    console.log_timing(
        "Loaded dataset bundle",
        perf_counter() - dataset_started_at,
        details=(
            f"publications={len(bundle.publications)}, "
            f"query_splits={','.join(sorted(bundle.queries.keys()))}"
        ),
    )
    retrieval_started_at = perf_counter()
    retrieval = RetrievalManager(config, store, console, bundle.publications, bundle.queries)
    console.log_timing(
        "Initialized retrieval manager",
        perf_counter() - retrieval_started_at,
        details=f"document_count={len(retrieval.document_ids)}",
    )
    clustering_started_at = perf_counter()
    clustering = ClusterManager(config, store, console, retrieval.document_ids)
    console.log_timing("Initialized cluster manager", perf_counter() - clustering_started_at)
    text_services_started_at = perf_counter()
    text_services = TextServices(config.runtime.allow_resource_downloads, console)
    console.log_timing(
        "Initialized text services",
        perf_counter() - text_services_started_at,
        details=f"allow_downloads={config.runtime.allow_resource_downloads}",
    )
    features_started_at = perf_counter()
    features = FeatureBuilder(config, store, console, bundle, retrieval, clustering, text_services)
    console.log_timing("Initialized feature builder", perf_counter() - features_started_at)
    console.log_timing("Project runtime ready", perf_counter() - overall_started_at)
    return ProjectRuntime(
        config=config,
        console=console,
        store=store,
        dataset=dataset,
        retrieval=retrieval,
        clustering=clustering,
        text_services=text_services,
        features=features,
    )
