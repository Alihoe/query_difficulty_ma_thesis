from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import sys
from pathlib import Path

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _gold_correlation_matrix_tables import _feature_lookup, _load_feature_groups
from _in_top_max_depth_importance_tables import (
    build_flat_dynamic_importance_table,
    build_static_importance_table,
    classify_importances,
    retriever_display,
)
from query_difficulty.analysis import _difficulty_correlation_retriever_family
from query_difficulty.modeling import IDENTIFIER_COLUMNS, InTopMaxDepthPredictorRunner
from query_difficulty.runtime import build_runtime

MAIN_TABLE_MIN_PERCENT = 0.5
IMPORTANCE_THRESHOLD = MAIN_TABLE_MIN_PERCENT / 100.0
APPENDIX_LABEL = "appendix:importances-max-depth"
DEFAULT_RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]


def _load_merged_multi_depth_frame(
    store_root: Path, language: str, retriever: str, feature_depths: list[int]
) -> pd.DataFrame:
    frames_by_depth: dict[int, pd.DataFrame] = {}
    for depth in feature_depths:
        depth_path = (
            store_root / "features" / "descriptive" / f"{language}_train" / retriever / f"depth_{depth:04d}.parquet"
        )
        if not depth_path.exists():
            raise FileNotFoundError(f"Training features not found: {depth_path}")
        frames_by_depth[depth] = pd.read_parquet(depth_path)
    return InTopMaxDepthPredictorRunner._merge_multi_depth_frames(frames_by_depth)


def _direction_from_training(training_frame: pd.DataFrame) -> dict[str, str]:
    target = ~training_frame["retrieval_failure_within_max_depth"].astype(bool)
    directions = {}
    for col in training_frame.columns:
        if col in IDENTIFIER_COLUMNS:
            continue
        series = training_frame[col]
        if series.nunique() <= 1:
            directions[col] = "flat"
            continue
        mean_1 = series[target].mean()
        mean_0 = series[~target].mean()
        directions[col] = "positive" if mean_1 > mean_0 else "negative"
    return directions


def _write_tables(runtime, language: str, retriever: str) -> None:
    config = runtime.config
    store_root = runtime.store.root_dir

    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)

    feature_depths = sorted({int(size) for size in config.candidate_sizes})
    resolved = retriever
    retriever_family = _difficulty_correlation_retriever_family(resolved)

    imp_path = store_root / "models" / "in_top_max_depth" / resolved / "feature_importances.parquet"
    if not imp_path.exists():
        raise FileNotFoundError(f"Importances file not found: {imp_path}")

    importances = pd.read_parquet(imp_path)

    train_frame = _load_merged_multi_depth_frame(store_root, language, resolved, feature_depths)
    directions = _direction_from_training(train_frame)

    classified = classify_importances(importances, directions, retriever_family)
    total_importance = float(classified["importance"].sum())

    retriever_label = resolved.replace("_", "-")
    name = retriever_display(resolved)
    appendix_note = (
        f", showing features above {MAIN_TABLE_MIN_PERCENT}% of the total importance. "
        f"The complete table is given in Appendix \\ref{{{APPENDIX_LABEL}}}."
    )

    def static_table(min_percent: float, caption: str, label_suffix: str) -> str:
        return build_static_importance_table(
            classified,
            total_importance=total_importance,
            feature_to_display=feature_to_display,
            feature_to_order=feature_to_order,
            group_display=group_display,
            group_order=group_order,
            caption=caption,
            label=f"tab:in_top_max_depth_importances_{retriever_label}_static{label_suffix}",
            min_percent=min_percent,
        )

    def dynamic_table(threshold: float, caption: str, label_suffix: str) -> str:
        return build_flat_dynamic_importance_table(
            classified,
            total_importance=total_importance,
            importance_threshold=threshold,
            feature_to_display=feature_to_display,
            caption=caption,
            label=f"tab:in_top_max_depth_importances_{retriever_label}_dynamic{label_suffix}",
        )

    outputs = {
        f"table_in_top_max_depth_importances_{resolved}.tex": [
            static_table(
                MAIN_TABLE_MIN_PERCENT,
                f"Maximum Depth depth-independent feature importances for {name}{appendix_note}",
                "",
            ),
            dynamic_table(
                IMPORTANCE_THRESHOLD,
                f"Maximum Depth depth-dependent feature importances for {name}{appendix_note}",
                "",
            ),
        ],
        f"table_in_top_max_depth_importances_{resolved}_full.tex": [
            static_table(0.0, f"Complete Maximum Depth depth-independent feature importances for {name}.", "_full"),
            dynamic_table(0.0, f"Complete Maximum Depth depth-dependent feature importances for {name}.", "_full"),
        ],
    }

    output_dir = config.latex_table_dir / "in_top_max_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, parts in outputs.items():
        latex = "\n\n".join(part for part in parts if part)
        out_file = output_dir / file_name
        out_file.write_text(apply_thesis_captions(latex + "\n"), encoding="utf-8")
        print(f"wrote {out_file} ({latex.count(chr(92) + 'arrayrulecolor{gray')} rows)")


def main(
    language: str = typer.Option("en", help="Language code."),
    retriever: str = typer.Option("", help="Retriever name; empty writes BM25 and Qwen."),
) -> None:
    runtime = build_runtime("config/default.yaml", verbose=False)
    for name in [retriever] if retriever else DEFAULT_RETRIEVERS:
        print(f"Building Maximum Depth importance tables for {name}")
        _write_tables(runtime, language, name)


if __name__ == "__main__":
    typer.run(main)
