from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import sys
from pathlib import Path

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _gold_correlation_matrix_tables import _feature_lookup, _load_feature_groups
from _in_top_k_sweep_importance_tables import (
    aggregate_importance_by_base_feature,
    build_across_depth_importance_table,
    direction_from_training,
)
from _in_top_max_depth_importance_tables import retriever_display
from _ordinal_importance_tables import ordinal_importances_by_depth, ordinal_training_target

from query_difficulty.analysis import _difficulty_correlation_retriever_family
from query_difficulty.modeling import IDENTIFIER_COLUMNS
from query_difficulty.runtime import build_runtime

MAIN_AVERAGE_THRESHOLD = 0.015
APPENDIX_LABEL = "appendix:importances-ordinal"


def main(
    language: str = typer.Option("en", help="Language code."),
    retriever: str = typer.Option("qwen3_embedding_0_6b", help="Retriever name."),
) -> None:
    runtime = build_runtime("config/default.yaml", verbose=False)
    config = runtime.config
    store_root = runtime.store.root_dir

    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)
    retriever_family = _difficulty_correlation_retriever_family(retriever)

    importances_by_depth = ordinal_importances_by_depth(store_root, retriever)
    max_depth = max(importances_by_depth)

    per_depth_summed: dict[int, pd.DataFrame] = {}
    for depth, importances in sorted(importances_by_depth.items()):
        train_path = store_root / "features" / "descriptive" / f"{language}_train" / retriever / f"depth_{depth:04d}.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Training features not found: {train_path}")
        training_frame, target = ordinal_training_target(pd.read_parquet(train_path), depth, max_depth)
        directions = direction_from_training(training_frame, target, IDENTIFIER_COLUMNS)
        per_depth_summed[depth] = aggregate_importance_by_base_feature(importances, directions)
        print(f"Loaded {len(importances)} importances for {retriever} at k={depth}")

    retriever_label = retriever.replace("_", "-")
    name = retriever_display(retriever)

    def table(average_threshold: float, caption: str, label: str) -> str:
        latex = build_across_depth_importance_table(
            per_depth_summed,
            importance_threshold=0.0,
            retriever_family=retriever_family,
            feature_to_display=feature_to_display,
            feature_to_order=feature_to_order,
            group_display=group_display,
            group_order=group_order,
            caption=caption,
            label=label,
            include_average=True,
            group_rows=False,
            average_threshold=average_threshold,
        )
        if not latex:
            raise ValueError(f"No features passed the importance threshold for {retriever}.")
        return latex

    label = f"tab:calibrated_ordinal_importances_{retriever_label}"
    tables = {
        f"table_calibrated_ordinal_importances_{retriever}.tex": table(
            MAIN_AVERAGE_THRESHOLD,
            f"Ordinal predictor feature importances for {name}, showing features with an average importance above "
            f"{MAIN_AVERAGE_THRESHOLD}. The complete table is given in Appendix \\ref{{{APPENDIX_LABEL}}}.",
            label,
        ),
        f"table_calibrated_ordinal_importances_{retriever}_full.tex": table(
            0.0, f"Complete ordinal predictor feature importances for {name}.", f"{label}_full"
        ),
    }

    output_dir = config.latex_table_dir / "calibrated_ordinal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, latex in tables.items():
        out_file = output_dir / file_name
        out_file.write_text(apply_thesis_captions(latex + "\n"), encoding="utf-8")
        print(f"wrote {out_file} ({latex.count(chr(92) + 'arrayrulecolor{gray')} rows)")


if __name__ == "__main__":
    typer.run(main)
