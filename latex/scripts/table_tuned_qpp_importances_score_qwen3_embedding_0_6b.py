from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
import sys
from pathlib import Path

import pandas as pd
import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import _difficulty_correlation_retriever_family, _gold_independent_feature_group
from _in_top_max_depth_importance_tables import build_flat_importance_table, retriever_display
from query_difficulty.runtime import build_runtime

MAIN_TABLE_MIN_PERCENT = 0.5
APPENDIX_LABEL = "appendix:importances-qpp"


def _direction_from_training(training_frame: pd.DataFrame) -> dict[str, str]:
    target = training_frame["rr_at_max_depth"].astype(float)
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
        correlation = series.corr(target)
        if pd.isna(correlation):
            directions[col] = "flat"
        else:
            directions[col] = "positive" if correlation > 0 else "negative"
    return directions


def main(
    language: str = typer.Option("en", help="Language code."),
    retriever: str = typer.Option("qwen3_embedding_0_6b", help="Retriever name."),
) -> None:
    runtime = build_runtime("config/default.yaml", verbose=False)
    config = runtime.config
    store_root = runtime.store.root_dir

    with open(ROOT / "config" / "feature_groups.yaml", "r", encoding="utf-8") as f:
        groups_config = yaml.safe_load(f)

    feature_to_display = {}
    for group_key, group_data in groups_config["feature_groups"].items():
        for column_name, display_name in group_data["features"].items():
            feature_to_display[column_name] = display_name

    max_depth = max(int(size) for size in config.candidate_sizes)
    resolved = retriever

    imp_path = store_root / "models" / "tuned_qpp" / resolved / "score_importances.parquet"
    if not imp_path.exists():
        raise FileNotFoundError(f"Importances file not found: {imp_path}")

    importances = pd.read_parquet(imp_path)

    train_path = store_root / "features" / "descriptive" / f"{language}_train" / resolved / f"depth_{max_depth:04d}.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")

    train_frame = pd.read_parquet(train_path)

    directions = _direction_from_training(train_frame)

    imp = importances.copy()
    imp["direction"] = imp["feature"].map(directions).fillna("unknown")

    manifest = json.loads((imp_path.parent / "manifest.json").read_text(encoding="utf-8"))
    feature_depth = int(manifest["feature_depth"])
    retriever_family = _difficulty_correlation_retriever_family(resolved)
    imp["depth_dependent"] = imp["feature"].map(
        lambda name: bool(_gold_independent_feature_group(name, retriever_family)[1])
    )
    imp["display_name"] = [
        feature_to_display.get(feature, feature) + (f" ($k={feature_depth}$)" if dependent else "")
        for feature, dependent in zip(imp["feature"], imp["depth_dependent"])
    ]

    label = f"tab:tuned_qpp_importances_score_{resolved.replace('_', '-')}"
    name = retriever_display(resolved)
    tables = {
        f"table_tuned_qpp_importances_score_{resolved}.tex": build_flat_importance_table(
            imp,
            min_percent=MAIN_TABLE_MIN_PERCENT,
            caption=(
                f"QPP regressor feature importances for {name}, showing features above {MAIN_TABLE_MIN_PERCENT}% of "
                f"the total importance. The complete table is given in Appendix \\ref{{{APPENDIX_LABEL}}}."
            ),
            label=label,
        ),
        f"table_tuned_qpp_importances_score_{resolved}_full.tex": build_flat_importance_table(
            imp, min_percent=0.0, caption=f"Complete QPP regressor feature importances for {name}.", label=f"{label}_full"
        ),
    }

    output_dir = config.latex_table_dir / "tuned_qpp"
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_name, latex in tables.items():
        out_file = output_dir / file_name
        out_file.write_text(apply_thesis_captions(latex + "\n"), encoding="utf-8")
        print(f"wrote {out_file} ({latex.count(chr(92) + 'arrayrulecolor{gray')} rows)")


if __name__ == "__main__":
    typer.run(main)
