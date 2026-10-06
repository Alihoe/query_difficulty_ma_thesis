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
from _in_top_max_depth_importance_tables import retriever_display, split_depth_prefixed_feature
from query_difficulty.analysis import _difficulty_correlation_retriever_family
from query_difficulty.modeling import IDENTIFIER_COLUMNS, InTopMaxDepthPredictorRunner
from query_difficulty.runtime import build_runtime

MAIN_AVERAGE_PERCENT = 1.0
APPENDIX_LABEL = "appendix:importances-binary"
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


def _depth_dependent_base_features(train_frame: pd.DataFrame) -> set[str]:
    copies: dict[str, list[str]] = {}
    for column in train_frame.columns:
        feature_depth, base = split_depth_prefixed_feature(column)
        if feature_depth is not None:
            copies.setdefault(base, []).append(column)
    return {
        base
        for base, columns in copies.items()
        if any(not train_frame[columns[0]].equals(train_frame[column]) for column in columns[1:])
    }


def _main_depths(per_depth_importances: dict[int, pd.DataFrame], depth_dependent: set[str]) -> dict[str, str]:
    totals: dict[str, dict[int, float]] = {}
    for importances in per_depth_importances.values():
        total = float(importances["importance"].sum())
        for record in importances.itertuples(index=False):
            feature_depth, base = split_depth_prefixed_feature(record.feature)
            if feature_depth is None or base not in depth_dependent:
                continue
            by_depth = totals.setdefault(base, {})
            by_depth[feature_depth] = by_depth.get(feature_depth, 0.0) + record.importance / total
    return {base: str(max(by_depth, key=by_depth.get)) for base, by_depth in totals.items()}


def _write_table(runtime, language: str, retriever: str) -> None:
    config = runtime.config
    store_root = runtime.store.root_dir

    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)

    feature_depths = sorted({int(size) for size in config.candidate_sizes})
    retriever_family = _difficulty_correlation_retriever_family(retriever)

    print(f"Loading {retriever} training features for k={feature_depths}")
    train_frame = _load_merged_multi_depth_frame(store_root, language, retriever, feature_depths)
    depth_dependent = _depth_dependent_base_features(train_frame)
    print(f"{retriever}: {len(depth_dependent)} depth-dependent base features")

    per_depth_summed: dict[int, pd.DataFrame] = {}
    per_depth_importances: dict[int, pd.DataFrame] = {}
    for depth in feature_depths:
        imp_path = store_root / "models" / f"in_top_k{depth}" / retriever / "feature_importances.parquet"
        if not imp_path.exists():
            raise FileNotFoundError(f"Importances file not found: {imp_path}")
        importances = pd.read_parquet(imp_path)
        per_depth_importances[depth] = importances
        target = train_frame["gold_rank"] <= depth
        directions = direction_from_training(train_frame, target, IDENTIFIER_COLUMNS)
        per_depth_summed[depth] = aggregate_importance_by_base_feature(importances, directions)
        print(f"{retriever}: loaded {len(importances)} importances for k={depth}")

    retriever_label = retriever.replace("_", "-")
    name = retriever_display(retriever)
    main_depths = _main_depths(per_depth_importances, depth_dependent)

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
            main_depths=main_depths,
            include_average=True,
            group_rows=False,
            average_threshold=average_threshold,
        )
        if not latex:
            raise ValueError(f"No features passed the importance threshold for {retriever}.")
        return latex

    label = f"tab:in_top_k_sweep_importances_{retriever_label}"
    tables = {
        f"table_in_top_k_sweep_importances_{retriever}.tex": table(
            MAIN_AVERAGE_PERCENT,
            f"Binary predictor feature importances for {name} across candidate-set sizes, showing features with an "
            f"average importance above {MAIN_AVERAGE_PERCENT:g}% of the importance per $k$. The complete table is given "
            f"in Appendix \\ref{{{APPENDIX_LABEL}}}.",
            label,
        ),
        f"table_in_top_k_sweep_importances_{retriever}_full.tex": table(
            0.0, f"Complete binary predictor feature importances for {name} across candidate-set sizes.", f"{label}_full"
        ),
    }

    output_dir = config.latex_table_dir / "in_top_k_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, latex in tables.items():
        out_file = output_dir / file_name
        out_file.write_text(apply_thesis_captions(latex + "\n"), encoding="utf-8")
        print(f"wrote {out_file} ({latex.count(chr(92) + 'arrayrulecolor{gray')} rows)")


def main(
    language: str = typer.Option("en", help="Language code."),
    retriever: str = typer.Option("", help="Retriever name; empty writes BM25 and Qwen."),
) -> None:
    runtime = build_runtime("config/default.yaml", verbose=False)
    for name in [retriever] if retriever else DEFAULT_RETRIEVERS:
        _write_table(runtime, language, name)


if __name__ == "__main__":
    typer.run(main)
