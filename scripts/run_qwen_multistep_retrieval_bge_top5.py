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
    FlagEmbeddingRerankerSpec,
    RerankerPolicyComparisonExperiment,
    default_flagembedding_reranker_spec,
)
from query_difficulty.runtime import build_runtime


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: str = typer.Option("qwen3_embedding_0_6b", help="Retriever name to use as the first-stage ranking."),
    reranker: str = typer.Option(
        "BAAI/bge-reranker-v2-m3",
        help="FlagEmbedding reranker model name.",
    ),
    top_k: int = typer.Option(5, help="Number of top-ranked documents to rerank."),
    candidate_fields: Optional[str] = typer.Option(
        "title_abstract",
        help="Candidate text fields for reranking: title or title_abstract.",
    ),
    quantize: bool = typer.Option(
        True,
        help="Apply PyTorch dynamic int8 quantization to the reranker's Linear layers for faster CPU inference.",
    ),
    max_length: Optional[int] = typer.Option(None, help="Override the reranker max sequence length."),
    batch_size: Optional[int] = typer.Option(None, help="Override the reranker batch size."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    command_started_at = perf_counter()
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Qwen3 embedding reranker policy comparison")
    runtime.console.log(f"Loading project configuration from {config}")
    spec = default_flagembedding_reranker_spec(reranker, quantize=quantize)
    if candidate_fields is not None:
        spec = replace(spec, candidate_fields=candidate_fields)
    if max_length is not None:
        spec = _replace_flagembedding_spec(spec, max_length=max_length)
    if batch_size is not None:
        spec = _replace_flagembedding_spec(spec, batch_size=batch_size)

    runtime.console.log(
        f"Running {retriever} reranker policy comparison for {language} with {spec.model_name} "
        f"(reranker_kind={spec.method}, top_k={top_k}, "
        f"max_length={spec.max_length}, batch_size={spec.batch_size}, "
        f"candidate_fields={spec.candidate_fields}, quantize_dynamic_int8={spec.quantize_dynamic_int8})"
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
        "run_qwen_multistep_retrieval_bge_top5 command complete",
        perf_counter() - command_started_at,
        details=f"reranker_kind={spec.method}, top_k={top_k}",
    )


def _replace_flagembedding_spec(
    spec: FlagEmbeddingRerankerSpec,
    *,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> FlagEmbeddingRerankerSpec:
    return replace(
        spec,
        max_length=spec.max_length if max_length is None else max_length,
        batch_size=spec.batch_size if batch_size is None else batch_size,
    )


if __name__ == "__main__":
    typer.run(main)
