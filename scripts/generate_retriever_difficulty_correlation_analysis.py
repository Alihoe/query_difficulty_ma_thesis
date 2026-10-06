from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import (
    DIFFICULTY_CORRELATION_FEATURE_SET_ALL,
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
    retriever_difficulty_correlation_analysis_for_runtime,
    retriever_difficulty_correlation_top_table,
)
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, compose_retriever_name
from query_difficulty.runtime import build_runtime


def _selected_retrievers(runtime, family: str) -> list[str]:
    family_key = family.strip().lower()
    if family_key not in {"all", "lexical", "semantic"}:
        raise typer.BadParameter("family must be one of: all, lexical, semantic")
    names: list[str] = []
    if family_key in {"all", "lexical"}:
        names.extend(
            compose_retriever_name(base_name, DEFAULT_DOCUMENT_SCOPE)
            for base_name in runtime.config.enabled_lexical_retrievers
        )
    if family_key in {"all", "semantic"}:
        names.extend(
            compose_retriever_name(base_name, DEFAULT_DOCUMENT_SCOPE)
            for base_name in runtime.config.enabled_semantic_retrievers
        )
    return names


def _normalized_feature_sets(feature_set: list[str] | None) -> tuple[str, ...]:
    if not feature_set:
        return (
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
            DIFFICULTY_CORRELATION_FEATURE_SET_ALL,
        )
    normalized = tuple(dict.fromkeys(value.strip().lower() for value in feature_set if value.strip()))
    valid = {
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
        DIFFICULTY_CORRELATION_FEATURE_SET_ALL,
    }
    invalid = [value for value in normalized if value not in valid]
    if invalid:
        raise typer.BadParameter(
            "feature_set values must be drawn from: "
            + ", ".join(sorted(valid))
            + f". Invalid values: {invalid}"
        )
    return normalized


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    split: str = typer.Option("en_train", help="Query split, for example en_train or en_test."),
    retriever: list[str] = typer.Option(
        None,
        "--retriever",
        help="Restrict analysis to specific base title_abstract retrievers. Repeat for multiple retrievers.",
    ),
    family: str = typer.Option("all", help="Retriever family to analyse: all, lexical, semantic."),
    feature_set: list[str] = typer.Option(
        None,
        "--feature-set",
        help="Feature-set views to generate. Repeat for gold_independent, gold_aware, or all_features.",
    ),
    descriptive_candidate_size: list[int] = typer.Option(
        None,
        "--descriptive-candidate-size",
        help="Restrict gold-independent analysis to specific candidate sizes. Repeat for multiple values.",
    ),
    gold_aware_candidate_size: list[int] = typer.Option(
        None,
        "--gold-aware-candidate-size",
        help="Restrict gold-aware analysis to specific candidate sizes. Repeat for multiple values.",
    ),
    output_dir: str = typer.Option(
        "artifacts/reports/retriever_difficulty_correlations",
        help="Directory where the parquet, CSV, and summary CSV files are written.",
    ),
    top_k_per_retriever: int = typer.Option(
        8,
        help="Rows per retriever for the summary CSV views.",
        min=1,
    ),
    quiet: bool = typer.Option(False, help="Reduce runtime console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    selected_retrievers = retriever or _selected_retrievers(runtime, family)
    selected_feature_sets = _normalized_feature_sets(feature_set)

    output_root = ROOT / output_dir / split
    output_root.mkdir(parents=True, exist_ok=True)

    results = retriever_difficulty_correlation_analysis_for_runtime(
        runtime,
        split,
        selected_retrievers,
        feature_sets=selected_feature_sets,
        descriptive_candidate_sizes=descriptive_candidate_size or None,
        gold_aware_candidate_sizes=gold_aware_candidate_size or None,
    )
    parquet_path = output_root / "correlations.parquet"
    csv_path = output_root / "correlations.csv"
    results.to_parquet(parquet_path, index=False)
    results.to_csv(csv_path, index=False)
    print(parquet_path)
    print(csv_path)

    for current_feature_set in selected_feature_sets:
        for current_family in ("all", "lexical", "semantic"):
            table = retriever_difficulty_correlation_top_table(
                results,
                feature_set=current_feature_set,
                retriever_family=current_family,
                top_k_per_retriever=top_k_per_retriever,
            )
            summary_path = output_root / f"top_{current_feature_set}_{current_family}.csv"
            table.to_csv(summary_path, index=False)
            print(summary_path)


if __name__ == "__main__":
    typer.run(main)
