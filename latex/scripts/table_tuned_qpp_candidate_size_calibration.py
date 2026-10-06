from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.runtime import build_runtime

PREDICTOR_NAME = "tuned_qpp"
PREDICTOR_DISPLAY_NAME = "tuned QPP"

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "BM25",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
RETRIEVER_ORDER = ["bm25", "qwen3_embedding_0_6b"]


def _snap_up(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    snapped = np.full(values.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(values)
    positions = np.searchsorted(grid, values[finite_mask], side="left")
    positions = np.clip(positions, 0, len(grid) - 1)
    snapped[finite_mask] = grid[positions].astype(np.float64)
    return snapped


def _calibration_rows(frame: pd.DataFrame, candidate_sizes: list[int]) -> list[dict[str, float | str | int]]:
    ordered_sizes = sorted(candidate_sizes)
    grid = np.asarray(ordered_sizes, dtype=np.int64)
    predicted_failure = frame["predicted_failure"].astype(bool).to_numpy()
    selected_size = frame["selected_size"].astype(np.float64).to_numpy()
    actual_size = frame["smallest_sufficient_size"].astype(np.float64).to_numpy()

    snapped_predicted = np.full(len(frame), np.nan, dtype=np.float64)
    snapped_predicted[~predicted_failure] = _snap_up(selected_size[~predicted_failure], grid)

    rows = []
    for size in ordered_sizes:
        mask = actual_size == size
        n = int(mask.sum())
        if n == 0:
            rows.append({"class": str(size), "n": 0, "exact_match": float("nan"), "median": float("nan"), "coverage": float("nan")})
            continue
        covered = (~predicted_failure[mask]) & (selected_size[mask] >= size)
        exact_match = snapped_predicted[mask] == size
        non_failure_predicted = selected_size[mask][~predicted_failure[mask]]
        rows.append(
            {
                "class": str(size),
                "n": n,
                "exact_match": float(np.mean(exact_match)),
                "median": float(np.median(non_failure_predicted)) if non_failure_predicted.size else float("nan"),
                "coverage": float(np.mean(covered)),
            }
        )

    unreachable_mask = np.isnan(actual_size)
    n_unreachable = int(unreachable_mask.sum())
    unreachable_exact_match = float(np.mean(predicted_failure[unreachable_mask])) if n_unreachable else float("nan")
    rows.append({"class": "Unreachable", "n": n_unreachable, "exact_match": unreachable_exact_match, "median": float("nan"), "coverage": unreachable_exact_match})
    return rows


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    candidate_sizes = sorted(int(size) for size in runtime.config.candidate_sizes)

    rows_by_retriever: dict[str, list[dict]] = {}
    for retriever in RETRIEVER_ORDER:
        prediction_path = (
            runtime.store.root_dir / "predictions" / PREDICTOR_NAME / language / retriever / "predictions.parquet"
        )
        if not prediction_path.exists():
            print(f"Skipping {retriever}: no {PREDICTOR_NAME} predictions at {prediction_path}")
            continue
        frame = runtime.store.read_parquet(prediction_path)
        rows_by_retriever[retriever] = _calibration_rows(frame, candidate_sizes)
    available_retrievers = [r for r in RETRIEVER_ORDER if r in rows_by_retriever]
    print(f"Found {PREDICTOR_NAME} predictions for: {available_retrievers}")

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Candidate-size calibration for the {PREDICTOR_DISPLAY_NAME} predictor on the {language} split, "
        "grouped by the true smallest sufficient candidate size. All columns describe the same $N$ queries in "
        "each row: Exact Match is the fraction predicted (snapped up) into this same class, Median Pred. is "
        "their median raw predicted size, and Coverage is the fraction whose predicted size was $\\geq$ actual.}"
    )
    lines.append(f"\\label{{tab:{PREDICTOR_NAME}_candidate_size_calibration_{language}}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    column_spec = "l|" + "|".join(["cccc"] * len(available_retrievers))
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    header_groups = " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{RETRIEVER_DISPLAY_NAMES[r]}}}" for r in available_retrievers
    )
    lines.append(f" & {header_groups} \\\\")
    header_sub = " & ".join("$N$ & Exact Match & Median Pred. & Coverage" for _ in available_retrievers)
    lines.append(f"Actual Class & {header_sub} \\\\")
    lines.append("\\midrule")

    n_rows = len(rows_by_retriever[available_retrievers[0]])
    for i in range(n_rows):
        cells = [rows_by_retriever[available_retrievers[0]][i]["class"]]
        for retriever in available_retrievers:
            row = rows_by_retriever[retriever][i]
            exact_match_fmt = "-" if np.isnan(row["exact_match"]) else f"{row['exact_match']:.4f}"
            median_fmt = "-" if np.isnan(row["median"]) else f"{row['median']:.1f}"
            coverage_fmt = "-" if np.isnan(row["coverage"]) else f"{row['coverage']:.4f}"
            cells.extend([str(row["n"]), exact_match_fmt, median_fmt, coverage_fmt])
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "tuned_qpp" / f"table_{PREDICTOR_NAME}_candidate_size_calibration_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
