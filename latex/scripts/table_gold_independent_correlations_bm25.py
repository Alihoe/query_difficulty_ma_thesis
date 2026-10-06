from __future__ import annotations

import typer

from _gold_correlation_matrix_tables import write_depth_matrix_tables
from query_difficulty.analysis import DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT


def main(
    split: str = typer.Option("en_train", help="Query split."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    descriptive_candidate_size: list[int] = typer.Option(
        None,
        "--descriptive-candidate-size",
        help="Restrict the depth matrix to specific candidate sizes. Repeat for multiple values.",
    ),
) -> None:
    path = write_depth_matrix_tables(
        config=config,
        split=split,
        feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        retriever_base_name="bm25",
        output_name="difficulty_analysis/table_gold_independent_correlations_bm25_{split}.tex",
        static_caption_template=(
            "Gold-independent depth-independent feature correlations with retrieval difficulty "
            "for \\textit{{BM25}} on the {split} split."
        ),
        static_label_template="tab:gold_independent_correlations_bm25_static_{split}",
        dynamic_symbol_caption_template=(
            "Gold-independent depth-dependent feature correlations with retrieval difficulty "
            "for \\textit{{BM25}} on the {split} split, evaluated across candidate-set size $k$."
        ),
        dynamic_symbol_label_template="tab:gold_independent_correlations_bm25_dynamic_symbols_{split}",
        descriptive_candidate_size=descriptive_candidate_size or None,
    )
    print(path)


if __name__ == "__main__":
    typer.run(main)
