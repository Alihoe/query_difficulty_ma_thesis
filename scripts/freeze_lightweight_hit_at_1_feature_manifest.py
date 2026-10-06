# scripts/freeze_lightweight_hit_at_1_feature_manifest.py
from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.features_hit_at_1_lightweight import refresh_manifest_from_full_models
from query_difficulty.runtime import build_runtime


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest_path: str = typer.Option(
        "config/lightweight_hit_at_1_features.json",
        help="Output path for the frozen feature manifest.",
    ),
    threshold: float = typer.Option(
        0.005,
        help="Minimum feature importance (fraction of total) for a feature to qualify.",
    ),
    exclude_multi_retriever_consensus: bool = typer.Option(
        True,
        help="Drop the multi-retriever consensus features (top_1_consensus_frequency, "
        "top_100_document_consensus, mean/maximum_semantic_top_1/100_agreement) from the candidate pool "
        "before thresholding, so they can never be selected downstream. bm25_top_1_agreement and "
        "bm25_top_100_agreement (which only ever need BM25) are kept as candidates. Default matches the "
        "bm25/qwen3_embedding_0_6b adoption (see lightweight_hit_at_1_predictor_feature_selection.md "
        "section 9); pass --no-exclude-multi-retriever-consensus to restore the full-consensus candidate pool.",
    ),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Freeze lightweight hit@1 feature manifest")
    output_path = Path(manifest_path)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    refresh_manifest_from_full_models(
        runtime.store,
        output_path,
        threshold=threshold,
        exclude_multi_retriever_consensus=exclude_multi_retriever_consensus,
        console=runtime.console,
    )
    runtime.console.log(f"Wrote lightweight hit@1 feature manifest to {output_path}")


if __name__ == "__main__":
    typer.run(main)
