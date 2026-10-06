from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.multistep import (
    CrossEncoderRerankerSpec,
    RerankerPolicyComparisonExperiment,
    default_reranker_spec,
)
from query_difficulty.runtime import build_runtime


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: str = typer.Option("qwen3_embedding_0_6b", help="Retriever name to use as the first-stage ranking."),
    reranker: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name (scored via sentence-transformers CrossEncoder).",
    ),
    top_k: int = typer.Option(10, help="Number of top-ranked documents to rerank."),
    max_length: Optional[int] = typer.Option(None, help="Override the reranker max sequence length."),
    batch_size: Optional[int] = typer.Option(None, help="Override the reranker batch size."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    command_started_at = perf_counter()
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Qwen3 embedding reranker policy comparison")
    runtime.console.log(f"Loading project configuration from {config}")
    spec = default_reranker_spec(reranker)
    if max_length is not None:
        spec = _replace_cross_encoder_spec(spec, max_length=max_length)
    if batch_size is not None:
        spec = _replace_cross_encoder_spec(spec, batch_size=batch_size)

    runtime.console.log(
        f"Running {retriever} reranker policy comparison for {language} with {spec.model_name} "
        f"(reranker_kind={spec.method}, top_k={top_k}, "
        f"max_length={spec.max_length}, batch_size={spec.batch_size})"
    )
    experiment = RerankerPolicyComparisonExperiment.from_runtime(
        runtime,
        reranker=spec,
        language=language,
        retriever_name=retriever,
        top_k=top_k,
    )
    summary = experiment.run()
    runtime.console.log(json.dumps(summary, indent=2))
    runtime.console.log_timing(
        "run_qwen_multistep_retrieval_minilm_top10 command complete",
        perf_counter() - command_started_at,
        details=f"reranker_kind={spec.method}, top_k={top_k}",
    )


def _replace_cross_encoder_spec(
    spec: CrossEncoderRerankerSpec,
    *,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> CrossEncoderRerankerSpec:
    return replace(
        spec,
        max_length=spec.max_length if max_length is None else max_length,
        batch_size=spec.batch_size if batch_size is None else batch_size,
    )


if __name__ == "__main__":
    typer.run(main)
