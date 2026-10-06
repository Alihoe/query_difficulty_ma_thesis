from __future__ import annotations

from types import SimpleNamespace

import typer

import run_hit_at_1_gate_cost_benchmark as benchmark
from _hit_at_1_gate_cost_common import (
    DEFAULT_LANGUAGE,
    DEFAULT_METRIC_K,
    DEFAULT_PREDICTOR_NAME,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVER,
    DEFAULT_TOP_K,
    report_dir,
    reranker_spec,
)
from query_difficulty.config import load_project_config

SUBSET_SIZE = 100
WARMUP_REPETITIONS = 1
REPETITIONS = 5


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    fresh: bool = typer.Option(False, help="Discard previous raw timings of the 100-query run instead of resuming."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    project = load_project_config(config)
    spec = reranker_spec(DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_MAX_LENGTH, DEFAULT_RERANKER_BATCH_SIZE)
    parent = report_dir(
        SimpleNamespace(store=SimpleNamespace(root_dir=project.artifact_dir)),
        DEFAULT_LANGUAGE,
        DEFAULT_RETRIEVER,
        spec,
        DEFAULT_TOP_K,
        DEFAULT_PREDICTOR_NAME,
    )
    if not (parent / "subset_query_ids.json").exists():
        raise FileNotFoundError(f"The 500-query subset {parent / 'subset_query_ids.json'} is required; run scripts/verify_fast_hit_at_1_gate.py first")
    benchmark.main(
        config=config,
        language=DEFAULT_LANGUAGE,
        retriever=DEFAULT_RETRIEVER,
        predictor_name=DEFAULT_PREDICTOR_NAME,
        subset_size=SUBSET_SIZE,
        full_test=False,
        seed=None,
        warmup_repetitions=WARMUP_REPETITIONS,
        repetitions=REPETITIONS,
        reranker_model=DEFAULT_RERANKER_MODEL,
        reranker_max_length=DEFAULT_RERANKER_MAX_LENGTH,
        reranker_batch_size=DEFAULT_RERANKER_BATCH_SIZE,
        request_size=None,
        gate_batch_size=0,
        top_k=DEFAULT_TOP_K,
        metric_k=DEFAULT_METRIC_K,
        device="cpu",
        full_test_sanity=False,
        fresh=fresh,
        output_dir=None,
        parent_subset_dir=str(parent),
        write_tables=True,
        quiet=quiet,
    )


if __name__ == "__main__":
    typer.run(main)
