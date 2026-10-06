from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.modeling import QppPredictorRunner
from query_difficulty.runtime import build_runtime

RETRIEVER_NAME = "qwen3_embedding_0_6b"
REGRESSOR_TARGET_COLUMN = "rr_at_max_depth"


def log_rank_discount(gold_rank: pd.Series) -> pd.Series:
    ranks = pd.to_numeric(gold_rank, errors="coerce").to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(ranks), 1.0 / np.log2(ranks + 1.0), 0.0)
    return pd.Series(values, index=gold_rank.index, dtype=np.float64)


class LogRankQppPredictorRunner(QppPredictorRunner):
    predictor_name = "qpp_log_rank"
    predictor_label = "QPP (log-rank target)"

    def _fit_score_context(
        self,
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        display_name: str,
        score_feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        target_frame = training_frame.copy()
        target_frame[REGRESSOR_TARGET_COLUMN] = log_rank_discount(target_frame["gold_rank"])
        target = target_frame[REGRESSOR_TARGET_COLUMN]
        self.console.log(
            f"Replaced regression target with 1/log2(gold_rank+1) for {display_name}: "
            f"n={len(target)}, mean={target.mean():.4f}, median={target.median():.4f}, "
            f"zero (gold not retrieved)={int(target.eq(0.0).sum())}, one (gold at rank 1)={int(target.eq(1.0).sum())}"
        )
        return super()._fit_score_context(
            target_frame, evaluation_frame, display_name, score_feature_columns=score_feature_columns
        )


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("QPP predictor (log-rank target)")
    runtime.console.log(f"Loading project configuration from {config}")
    runtime.console.log(f"Running log-rank QPP predictor for {language} / {RETRIEVER_NAME}")
    runner = LogRankQppPredictorRunner(runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)
    summary = runner.run(language, retrievers=[RETRIEVER_NAME])
    summary_path = runtime.store.path("predictions", LogRankQppPredictorRunner.predictor_name, language, "summary.json")
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved log-rank QPP summary to {summary_path}")
    runtime.console.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
