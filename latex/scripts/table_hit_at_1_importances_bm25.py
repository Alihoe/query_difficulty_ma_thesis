from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import sys
from pathlib import Path

import pandas as pd
import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _in_top_max_depth_importance_tables import build_flat_importance_table, retriever_display
from query_difficulty.runtime import build_runtime

MAIN_TABLE_MIN_PERCENT = 0.5
APPENDIX_LABEL = "appendix:importances-hit-at-1"
DEFAULT_RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]


def _direction_from_training(training_frame: pd.DataFrame) -> dict[str, str]:
    target = training_frame["gold_rank"].fillna(-1.0).eq(1.0)
    directions = {}
    for col in training_frame.columns:
        if col in {
            "query_id",
            "document_id",
            "language",
            "split_name",
            "retriever_name",
            "candidate_size",
            "gold_rank",
            "smallest_sufficient_size",
            "retrieval_failure_within_max_depth",
            "rr_at_max_depth",
            "gold_in_top_k",
        }:
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

    with open(ROOT / "config" / "feature_groups.yaml", "r", encoding="utf-8") as f:
        groups_config = yaml.safe_load(f)

    feature_to_display = {}
    for group_key, group_data in groups_config["feature_groups"].items():
        for column_name, display_name in group_data["features"].items():
            feature_to_display[column_name] = display_name

    feature_depth = min(config.candidate_sizes)
    resolved = retriever

    imp_path = store_root / "models" / "hit_at_1" / resolved / "feature_importances.parquet"
    if not imp_path.exists():
        raise FileNotFoundError(f"Importances file not found: {imp_path}")

    importances = pd.read_parquet(imp_path)

    train_path = store_root / "features" / "descriptive" / f"{language}_train" / resolved / f"depth_{feature_depth:04d}.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")

    train_frame = pd.read_parquet(train_path)

    directions = _direction_from_training(train_frame)

    imp = importances.copy()
    imp["direction"] = imp["feature"].map(directions).fillna("unknown")
    imp["display_name"] = imp["feature"].map(feature_to_display).fillna(imp["feature"])

    label = f"tab:hit_at_1_importances_{resolved.replace('_', '-')}"
    name = retriever_display(resolved)
    tables = {
        f"table_hit_at_1_importances_{resolved}.tex": build_flat_importance_table(
            imp,
            min_percent=MAIN_TABLE_MIN_PERCENT,
            caption=(
                f"Hit-at-1 feature importances for {name}, showing features above {MAIN_TABLE_MIN_PERCENT}% of the "
                f"total importance. The complete table is given in Appendix \\ref{{{APPENDIX_LABEL}}}."
            ),
            label=label,
        ),
        f"table_hit_at_1_importances_{resolved}_full.tex": build_flat_importance_table(
            imp, min_percent=0.0, caption=f"Complete Hit-at-1 feature importances for {name}.", label=f"{label}_full"
        ),
    }

    output_dir = config.latex_table_dir / "hit_at_1"
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
        print(f"Building Hit-at-1 importance tables for {name}")
        _write_tables(runtime, language, name)


if __name__ == "__main__":
    typer.run(main)