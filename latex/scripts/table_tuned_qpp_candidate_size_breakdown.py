from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import candidate_size_class_breakdown_table
from query_difficulty.runtime import build_runtime

PREDICTOR_NAME = "tuned_qpp"
PREDICTOR_DISPLAY_NAME = "tuned QPP"

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "BM25",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
RETRIEVER_ORDER = ["bm25", "qwen3_embedding_0_6b"]


def _snap_selected_size_to_grid(frame: pd.DataFrame, ordered_sizes: list[int]) -> pd.DataFrame:
    grid = np.asarray(sorted(ordered_sizes), dtype=np.int64)
    selected_raw = frame["selected_size"].to_numpy(dtype=np.float64)
    snapped = np.full(len(frame), np.nan, dtype=np.float64)
    finite_mask = np.isfinite(selected_raw)
    positions = np.searchsorted(grid, selected_raw[finite_mask], side="left")
    positions = np.clip(positions, 0, len(grid) - 1)
    snapped[finite_mask] = grid[positions].astype(np.float64)
    snapped_frame = frame.copy()
    snapped_frame["selected_size"] = snapped
    return snapped_frame


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    candidate_sizes = sorted(int(size) for size in runtime.config.candidate_sizes)

    frames_by_retriever = {}
    for retriever in RETRIEVER_ORDER:
        prediction_path = (
            runtime.store.root_dir / "predictions" / PREDICTOR_NAME / language / retriever / "predictions.parquet"
        )
        if prediction_path.exists():
            frame = runtime.store.read_parquet(prediction_path)
            frames_by_retriever[retriever] = _snap_selected_size_to_grid(frame, candidate_sizes)
        else:
            print(f"Skipping {retriever}: no {PREDICTOR_NAME} predictions at {prediction_path}")
    print(f"Found {PREDICTOR_NAME} predictions for: {list(frames_by_retriever)}")

    latex = candidate_size_class_breakdown_table(
        frames_by_retriever,
        candidate_sizes,
        RETRIEVER_DISPLAY_NAMES,
        RETRIEVER_ORDER,
        caption=(
            f"Candidate-size class breakdown for the {PREDICTOR_DISPLAY_NAME} predictor on the {language} split: "
            "predicted (snapped up to the nearest evaluated candidate size) vs. actual (smallest sufficient) "
            "candidate size, with per-class F1."
        ),
        label=f"tab:{PREDICTOR_NAME}_candidate_size_breakdown_{language}",
    )
    output_path = runtime.config.latex_table_dir / "tuned_qpp" / f"table_{PREDICTOR_NAME}_candidate_size_breakdown_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
