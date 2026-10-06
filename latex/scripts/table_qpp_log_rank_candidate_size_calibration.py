from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import numpy as np
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.runtime import build_runtime
from table_tuned_qpp_candidate_size_calibration import _calibration_rows

PREDICTOR_NAME = "qpp_log_rank"
PREDICTOR_DISPLAY_NAME = "QPP predictor with log-rank regression target"

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "BM25",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
RETRIEVER_ORDER = ["bm25", "qwen3_embedding_0_6b"]


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
        print(f"Reading {prediction_path}")
        frame = runtime.store.read_parquet(prediction_path)
        rows_by_retriever[retriever] = _calibration_rows(frame, candidate_sizes)
    available_retrievers = [r for r in RETRIEVER_ORDER if r in rows_by_retriever]
    print(f"Found {PREDICTOR_NAME} predictions for: {available_retrievers}")
    if not available_retrievers:
        raise SystemExit(f"No {PREDICTOR_NAME} predictions found for {language}")

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Candidate-set-size prediction on the English dev split for the {PREDICTOR_DISPLAY_NAME} "
        "($1/\\log_2(\\mathrm{rank}+1)$), grouped by the smallest sufficient candidate-set size.}"
    )
    lines.append(f"\\label{{tab:{PREDICTOR_NAME}_candidate_size_calibration_{language}}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    column_spec = "l|" + "|".join(["cccc"] * len(available_retrievers))
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    header_groups = " & ".join(
        f"\\multicolumn{{4}}{{c{'|' if index < len(available_retrievers) - 1 else ''}}}{{{RETRIEVER_DISPLAY_NAMES[r]}}}"
        for index, r in enumerate(available_retrievers)
    )
    lines.append(f" & {header_groups} \\\\")
    header_sub = " & ".join("$N$ & Exact Match & Median Pred. & Coverage" for _ in available_retrievers)
    lines.append(f"Candidate Size & {header_sub} \\\\")
    lines.append("\\midrule")

    n_rows = len(rows_by_retriever[available_retrievers[0]])
    for i in range(n_rows):
        cells = [rows_by_retriever[available_retrievers[0]][i]["class"]]
        for retriever in available_retrievers:
            row = rows_by_retriever[retriever][i]
            exact_match_fmt = "--" if np.isnan(row["exact_match"]) else f"{row['exact_match']:.4f}"
            median_fmt = "--" if np.isnan(row["median"]) else f"{row['median']:.1f}"
            coverage_fmt = "--" if np.isnan(row["coverage"]) else f"{row['coverage']:.4f}"
            cells.extend([str(row["n"]), exact_match_fmt, median_fmt, coverage_fmt])
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "qpp_log_rank" / f"table_{PREDICTOR_NAME}_candidate_size_calibration_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved table to {output_path}")


if __name__ == "__main__":
    typer.run(main)
