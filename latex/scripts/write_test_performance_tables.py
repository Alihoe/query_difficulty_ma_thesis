from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _thesis_captions import apply_thesis_captions
from write_lightweight_tables import MANIFESTS, _source_candidate_count, _stage_counts
from query_difficulty.analysis import (
    candidate_size_class_breakdown_table,
    hit_at_1_predictor_metric_summary,
    in_top_max_depth_predictor_metric_summary,
    ordinal_predictor_metric_summary,
)
from query_difficulty.modeling import _selection_metrics
from query_difficulty.runtime import build_runtime

RETRIEVERS = ["qwen3_embedding_0_6b", "bm25", "gte_large_en_v1_5", "e5_large_v2"]
LIGHTWEIGHT_RETRIEVERS = ["qwen3_embedding_0_6b", "bm25"]
RETRIEVER_DISPLAY = {
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
    "bm25": "BM25",
    "gte_large_en_v1_5": "GTE-large-en-v1.5",
    "e5_large_v2": "E5-large-v2",
}
IN_TOP_K_DEPTHS = [5, 10, 20, 30, 50, 60, 100, 120, 200, 1000]
PREDICTION_FILE = "test_predictions.parquet"
MAX_DEPTH = 1000


def _frame(store_root: Path, predictor: str, language: str, retriever: str) -> pd.DataFrame:
    path = store_root / "predictions" / predictor / language / retriever / PREDICTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing test predictions: {path}")
    frame = pd.read_parquet(path)
    base = predictor.split("/")[-1]
    if "in_top_max_depth" not in frame.columns and (base == "in_top_max_depth" or base.startswith("in_top_k")):
        depth = MAX_DEPTH if base == "in_top_max_depth" else int(base[len("in_top_k"):])
        frame["in_top_max_depth"] = frame["gold_rank"].notna() & (frame["gold_rank"] <= depth)
    if "predicted_in_top_max_depth" not in frame.columns and "predicted_in_top_k" in frame.columns:
        frame["predicted_in_top_max_depth"] = frame["predicted_in_top_k"]
    return frame


def _mrr_at_5(frame: pd.DataFrame) -> float:
    gold_rank = frame["gold_rank"]
    return float(np.mean(np.where(gold_rank.notna() & (gold_rank <= 5), 1.0 / gold_rank, 0.0)))


def _count(path: Path) -> str:
    return str(len(pd.read_parquet(path)))


def _head_count_range(model_dir: Path) -> str:
    paths = sorted(model_dir.glob("depth_*_importances.parquet")) + [model_dir / "solvability_importances.parquet"]
    counts = [len(pd.read_parquet(path)) for path in paths if path.exists()]
    return str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"


def _better_pair(full: float, light: float, higher_is_better: bool, digits: int) -> tuple[str, str]:
    texts = [f"{full:.{digits}f}", f"{light:.{digits}f}"]
    full_r, light_r = round(full, digits), round(light, digits)
    if full_r == light_r:
        return texts[0], texts[1]
    index = 0 if ((full_r > light_r) if higher_is_better else (full_r < light_r)) else 1
    texts[index] = f"\\textbf{{{texts[index]}}}"
    return texts[0], texts[1]


def _bold_best_per_column(rows: list[list[float]], higher_is_better: list[bool], digits: list[int]) -> list[list[str]]:
    columns = list(zip(*rows))
    best = [
        (max if higher else min)(round(v, d) for v in column)
        for column, higher, d in zip(columns, higher_is_better, digits)
    ]
    out = []
    for row in rows:
        cells = []
        for value, top, d in zip(row, best, digits):
            text = f"{value:.{d}f}"
            cells.append(f"\\textbf{{{text}}}" if round(value, d) == top else text)
        out.append(cells)
    return out


def _table(label: str, column_spec: str, header_lines: list[str], rows: list[str]) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{label}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        *header_lines,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _ordered(store_root: Path, language: str, retrievers: list[str]) -> list[str]:
    mrr = {r: _mrr_at_5(_frame(store_root, "hit_at_1", language, r)) for r in retrievers}
    return sorted(retrievers, key=lambda r: mrr[r], reverse=True)


def hit_at_1_tables(store_root: Path, language: str, order: list[str], light_order: list[str]) -> tuple[str, str]:
    metrics = ("precision", "recall", "f0_5")
    values = []
    for r in order:
        full_frame = _frame(store_root, "hit_at_1", language, r)
        full = hit_at_1_predictor_metric_summary(full_frame)
        values.append([_mrr_at_5(full_frame), *[full[m] for m in metrics]])
    cells = _bold_best_per_column(values, [True] * 4, [4] * 4)
    full_lines = [
        " & ".join([RETRIEVER_DISPLAY[r], c[0], _count(store_root / "models" / "hit_at_1" / r / "feature_importances.parquet"), *c[1:]]) + " \\\\"
        for r, c in zip(order, cells)
    ]
    full_table = _table(
        f"tab:hit_at_1_test_performance_{language}", "l|c|c|ccc",
        [" & Retrieval & & & & \\\\", "Retriever & MRR@5 & \\#Feat. & Prec. & Rec. & F0.5 \\\\"], full_lines,
    )
    light_lines = []
    for r in light_order:
        full = hit_at_1_predictor_metric_summary(_frame(store_root, "hit_at_1", language, r))
        light = hit_at_1_predictor_metric_summary(_frame(store_root, "lightweight/hit_at_1", language, r))
        light_meta = store_root / "predictions" / "lightweight" / "hit_at_1" / language / r / "metadata.json"
        pairs = [_better_pair(float(full[m]), float(light[m]), True, 4) for m in metrics]
        light_lines.append(" & ".join([
            RETRIEVER_DISPLAY[r], _count(store_root / "models" / "hit_at_1" / r / "feature_importances.parquet"),
            *[p[0] for p in pairs], *[p[1] for p in pairs],
            str(len(json.loads(light_meta.read_text(encoding="utf-8"))["qualifying_features"])),
        ]) + " \\\\")
    light_table = _table(
        f"tab:hit_at_1_lightweight_test_performance_{language}", "l|c|ccc|ccc|c",
        [" & & \\multicolumn{3}{c|}{Full} & \\multicolumn{3}{c|}{Lightweight} & \\\\",
         "Retriever & \\#Feat. (Full) & Prec. & Rec. & F0.5 & Prec. & Rec. & F0.5 & \\#Feat. (Light) \\\\"],
        light_lines,
    )
    return full_table, light_table


def max_depth_tables(store_root: Path, language: str, order: list[str], light_order: list[str]) -> tuple[str, str]:
    metrics = ("precision", "recall", "f0_5", "mcc")
    values, extras = [], []
    for r in order:
        frame = _frame(store_root, "in_top_max_depth", language, r)
        summary = in_top_max_depth_predictor_metric_summary(frame)
        values.append([_mrr_at_5(frame), *[summary[m] for m in metrics]])
        extras.append(_count(store_root / "models" / "in_top_max_depth" / r / "feature_importances.parquet"))
    cells = _bold_best_per_column(values, [True] * 5, [4] * 5)
    full_lines = [" & ".join([RETRIEVER_DISPLAY[r], c[0], n, *c[1:]]) + " \\\\" for r, c, n in zip(order, cells, extras)]
    full_table = _table(
        f"tab:in_top_max_depth_test_performance_{language}", "l|c|c|cccc",
        [" & Retrieval & & \\multicolumn{4}{c}{Full} \\\\", "Retriever & MRR@5 & \\#Feat. & Prec. & Rec. & F0.5 & MCC \\\\"],
        full_lines,
    )
    light_lines = []
    for r in light_order:
        full = in_top_max_depth_predictor_metric_summary(_frame(store_root, "in_top_max_depth", language, r))
        light = in_top_max_depth_predictor_metric_summary(_frame(store_root, "lightweight/in_top_max_depth", language, r))
        pairs = [_better_pair(full[m], light[m], True, 4) for m in metrics]
        light_lines.append(" & ".join([
            RETRIEVER_DISPLAY[r], _count(store_root / "models" / "in_top_max_depth" / r / "feature_importances.parquet"),
            *[p[0] for p in pairs], *[p[1] for p in pairs],
            _count(store_root / "models" / "lightweight" / "in_top_max_depth" / r / "feature_importances.parquet"),
        ]) + " \\\\")
    light_table = _table(
        f"tab:in_top_max_depth_lightweight_test_performance_{language}", "l|c|cccc|cccc|c",
        [" & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
         "Retriever & \\#Feat. (Full) & Prec. & Rec. & F0.5 & MCC & Prec. & Rec. & F0.5 & MCC & \\#Feat. (Light) \\\\"],
        light_lines,
    )
    return full_table, light_table


def max_depth_confusion_table(store_root: Path, language: str, order: list[str]) -> str:
    lines = []
    for r in order:
        frame = _frame(store_root, "in_top_max_depth", language, r)
        actual = frame["in_top_max_depth"].astype(bool)
        predicted = frame["predicted_in_top_max_depth"].astype(bool)
        counts = [(~actual).sum(), actual.sum(), (~predicted & ~actual).sum(), (~predicted & actual).sum(), (predicted & ~actual).sum()]
        lines.append(f"{RETRIEVER_DISPLAY[r]} & " + " & ".join(str(int(v)) for v in counts) + " \\\\")
    header = [
        "& \\multicolumn{2}{c|}{Actual} & \\multicolumn{3}{c}{Predicted Not-in-Top-k} \\\\",
        "Retriever & Not-k & In-k & Correct & Incorrect & Missed \\\\",
    ]
    return _table(f"tab:in_top_max_depth_confusion_breakdown_{language}", "l|cc|ccc", header, lines)


def ordinal_tables(store_root: Path, language: str, order: list[str], light_order: list[str], sizes: list[int]) -> tuple[str, str]:
    keys = [("candidate_recall", True), ("exact_class_accuracy", True), ("quadratic_weighted_cohen_kappa", True), ("ordinal_mae", False)]
    values = []
    for r in order:
        summary = ordinal_predictor_metric_summary(_frame(store_root, "calibrated_ordinal", language, r), sizes)
        values.append([float(summary[k]) for k, _ in keys])
    cells = _bold_best_per_column(values, [h for _, h in keys], [4] * 4)
    full_lines = [
        " & ".join([RETRIEVER_DISPLAY[r], _head_count_range(store_root / "models" / "calibrated_ordinal" / r), *c]) + " \\\\"
        for r, c in zip(order, cells)
    ]
    full_table = _table(
        f"tab:calibrated_ordinal_test_performance_{language}", "l|c|cccc",
        ["Retriever & \\#Feat. & Recall & Exact Acc. & QWK & MAE \\\\"], full_lines,
    )
    light_lines = []
    for r in light_order:
        full = ordinal_predictor_metric_summary(_frame(store_root, "calibrated_ordinal", language, r), sizes)
        light = ordinal_predictor_metric_summary(_frame(store_root, "lightweight/calibrated_ordinal", language, r), sizes)
        pairs = [_better_pair(float(full[k]), float(light[k]), higher, 4) for k, higher in keys]
        light_lines.append(" & ".join([
            RETRIEVER_DISPLAY[r], _head_count_range(store_root / "models" / "calibrated_ordinal" / r),
            *[p[0] for p in pairs], *[p[1] for p in pairs],
            _head_count_range(store_root / "models" / "lightweight" / "calibrated_ordinal" / r),
        ]) + " \\\\")
    light_table = _table(
        f"tab:calibrated_ordinal_lightweight_test_performance_{language}", "l|c|cccc|cccc|c",
        [" & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
         "Retriever & \\#Feat. (Full) & Recall & Exact Acc. & QWK & MAE & Recall & Exact Acc. & QWK & MAE & \\#Feat. (Light) \\\\"],
        light_lines,
    )
    return full_table, light_table


def qpp_tables(store_root: Path, language: str, order: list[str], light_order: list[str]) -> tuple[str, str]:
    keys = [("candidate_recall", True, 4), ("mean_selected_size", False, 2), ("median_selected_size", False, 1), ("failure_rate", False, 4)]
    values = []
    for r in order:
        summary = _selection_metrics(_frame(store_root, "tuned_qpp", language, r))
        values.append([float(summary[k]) for k, _, _ in keys])
    cells = _bold_best_per_column(values, [h for _, h, _ in keys], [d for _, _, d in keys])
    full_lines = [
        " & ".join([RETRIEVER_DISPLAY[r], _count(store_root / "models" / "tuned_qpp" / r / "score_importances.parquet"), *c]) + " \\\\"
        for r, c in zip(order, cells)
    ]
    full_table = _table(
        f"tab:tuned_qpp_test_performance_{language}", "l|c|cccc",
        ["Retriever & \\#Feat. & Recall & Mean & Median & Failure \\\\"], full_lines,
    )
    light_lines = []
    for r in light_order:
        full = _selection_metrics(_frame(store_root, "tuned_qpp", language, r))
        light = _selection_metrics(_frame(store_root, "lightweight/tuned_qpp", language, r))
        pairs = [_better_pair(float(full[k]), float(light[k]), higher, digits) for k, higher, digits in keys]
        light_lines.append(" & ".join([
            RETRIEVER_DISPLAY[r], _count(store_root / "models" / "tuned_qpp" / r / "score_importances.parquet"),
            *[p[0] for p in pairs], *[p[1] for p in pairs],
            _count(store_root / "models" / "lightweight" / "tuned_qpp" / r / "score_importances.parquet"),
        ]) + " \\\\")
    light_table = _table(
        f"tab:tuned_qpp_lightweight_test_performance_{language}", "l|c|cccc|cccc|c",
        [" & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
         "Retriever & \\#Feat. (Full) & Recall & Mean & Median & Failure & Recall & Mean & Median & Failure & \\#Feat. (Light) \\\\"],
        light_lines,
    )
    return full_table, light_table


def binary_tables(store_root: Path, language: str, order: list[str], light_order: list[str]) -> tuple[str, str]:
    values = []
    for depth in IN_TOP_K_DEPTHS:
        values.append([in_top_max_depth_predictor_metric_summary(_frame(store_root, f"in_top_k{depth}", language, r))["mcc"] for r in order])
    lines = []
    for depth, row in zip(IN_TOP_K_DEPTHS, values):
        top = max(round(v, 4) for v in row)
        cells = [f"\\textbf{{{v:.4f}}}" if round(v, 4) == top else f"{v:.4f}" for v in row]
        lines.append(" & ".join([str(depth), *cells]) + " \\\\")
    full_table = _table(
        f"tab:in_top_k_sweep_test_performance_{language}", "c|" + "c" * len(order),
        ["$k$ & " + " & ".join(RETRIEVER_DISPLAY[r] for r in order) + " \\\\"], lines,
    )
    light_lines = []
    for depth in IN_TOP_K_DEPTHS:
        if light_lines:
            light_lines.append("\\midrule")
        for index, r in enumerate(light_order):
            full = in_top_max_depth_predictor_metric_summary(_frame(store_root, f"in_top_k{depth}", language, r))
            light = in_top_max_depth_predictor_metric_summary(_frame(store_root, f"lightweight/in_top_k{depth}", language, r))
            light_lines.append(" & ".join([
                str(depth) if index == 0 else "", RETRIEVER_DISPLAY[r],
                _count(store_root / "models" / f"in_top_k{depth}" / r / "feature_importances.parquet"),
                *_better_pair(full["mcc"], light["mcc"], True, 4),
                _count(store_root / "models" / "lightweight" / f"in_top_k{depth}" / r / "feature_importances.parquet"),
            ]) + " \\\\")
    light_table = _table(
        f"tab:in_top_k_sweep_lightweight_test_performance_{language}", "cl|c|c|c|c",
        [" & & & Full & Lightweight & \\\\", "$k$ & Retriever & \\#Feat. (Full) & MCC & MCC & \\#Feat. (Light) \\\\"],
        light_lines,
    )
    return full_table, light_table


def _snap(frame: pd.DataFrame, sizes: list[int]) -> pd.DataFrame:
    grid = np.asarray(sorted(sizes), dtype=np.int64)
    raw = frame["selected_size"].to_numpy(dtype=np.float64)
    snapped = np.full(len(frame), np.nan, dtype=np.float64)
    finite = np.isfinite(raw)
    snapped[finite] = grid[np.clip(np.searchsorted(grid, raw[finite], side="left"), 0, len(grid) - 1)]
    out = frame.copy()
    out["selected_size"] = snapped
    return out


def breakdown_table(store_root: Path, language: str, order: list[str], predictor: str, sizes: list[int], label: str, snap: bool) -> str:
    frames = {}
    for r in order:
        frame = _frame(store_root, predictor, language, r)
        frames[r] = _snap(frame, sizes) if snap else frame
    return candidate_size_class_breakdown_table(frames, sizes, RETRIEVER_DISPLAY, order, caption=label, label=label)


def selection_steps_table(store_root: Path, language: str, light_order: list[str]) -> str:
    lines = []
    for r in light_order:
        full = hit_at_1_predictor_metric_summary(_frame(store_root, "hit_at_1", language, r))["f0_5"]
        light = hit_at_1_predictor_metric_summary(_frame(store_root, "lightweight/hit_at_1", language, r))["f0_5"]
        lines.append(" & ".join([
            RETRIEVER_DISPLAY[r], str(_source_candidate_count(store_root, "hit_at_1", r, "feature_importances.parquet", False)),
            *map(str, _stage_counts(MANIFESTS["hit_at_1"], r)), *_better_pair(full, light, True, 4),
        ]) + " \\\\")
    header = ["Retriever & Full & Importance & $|\\rho|\\geq0.80$ & $|\\rho|\\geq0.70$ & RFE & F0.5 Full & F0.5 Light \\\\"]
    return _table(f"tab:hit_at_1_lightweight_feature_selection_steps_{language}", "l|ccccc|cc", header, lines)


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir
    table_dir = runtime.config.latex_table_dir
    order = _ordered(store_root, language, RETRIEVERS)
    light_order = _ordered(store_root, language, LIGHTWEIGHT_RETRIEVERS)
    print(f"Retrievers ordered by test MRR@5: {order}")
    candidate_sizes = sorted(int(s) for s in runtime.config.candidate_sizes)
    ordinal_sizes = sorted(int(s) for s in json.loads((store_root / "models" / "calibrated_ordinal" / "bm25" / "manifest.json").read_text(encoding="utf-8"))["candidate_sizes"])

    tables: dict[tuple[str, str], str] = {}
    full, light = hit_at_1_tables(store_root, language, order, light_order)
    tables[("hit_at_1", f"table_hit_at_1_test_performance_{language}.tex")] = full
    tables[("hit_at_1", f"table_hit_at_1_lightweight_test_performance_{language}.tex")] = light
    tables[("hit_at_1", f"table_hit_at_1_lightweight_feature_selection_steps_{language}.tex")] = selection_steps_table(store_root, language, light_order)
    full, light = max_depth_tables(store_root, language, order, light_order)
    tables[("in_top_max_depth", f"table_in_top_max_depth_test_performance_{language}.tex")] = full
    tables[("in_top_max_depth", f"table_in_top_max_depth_lightweight_test_performance_{language}.tex")] = light
    tables[("in_top_max_depth", f"table_in_top_max_depth_confusion_breakdown_{language}.tex")] = max_depth_confusion_table(store_root, language, order)
    full, light = ordinal_tables(store_root, language, order, light_order, candidate_sizes)
    tables[("calibrated_ordinal", f"table_calibrated_ordinal_test_performance_{language}.tex")] = full
    tables[("calibrated_ordinal", f"table_calibrated_ordinal_lightweight_test_performance_{language}.tex")] = light
    tables[("calibrated_ordinal", f"table_calibrated_ordinal_candidate_size_breakdown_{language}.tex")] = breakdown_table(
        store_root, language, order, "calibrated_ordinal", ordinal_sizes, f"tab:calibrated_ordinal_candidate_size_breakdown_{language}", False)
    full, light = qpp_tables(store_root, language, order, light_order)
    tables[("tuned_qpp", f"table_tuned_qpp_test_performance_{language}.tex")] = full
    tables[("tuned_qpp", f"table_tuned_qpp_lightweight_test_performance_{language}.tex")] = light
    tables[("tuned_qpp", f"table_tuned_qpp_candidate_size_breakdown_{language}.tex")] = breakdown_table(
        store_root, language, order, "tuned_qpp", candidate_sizes, f"tab:tuned_qpp_candidate_size_breakdown_{language}", True)
    full, light = binary_tables(store_root, language, order, light_order)
    tables[("in_top_k_sweep", f"table_in_top_k_sweep_test_performance_{language}.tex")] = full
    tables[("in_top_k_sweep", f"table_in_top_k_sweep_lightweight_test_performance_{language}.tex")] = light

    for (category, name), latex in tables.items():
        path = table_dir / category / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(apply_thesis_captions(latex) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    typer.run(main)
