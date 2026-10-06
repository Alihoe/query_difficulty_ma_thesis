from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.runtime import build_runtime

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "\\textit{BM25}",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
DEFAULT_RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]
PREDICTION_FILE = "test_predictions.parquet"
MAX_RANKING_DEPTH = 1000

PREDICTOR_BLOCKS = [
    ("hit_at_1", "Hit-at-1 Prediction"),
    ("in_top_max_depth", "Maximum Depth Prediction"),
    ("in_top_k_sweep", "Binary Candidate-Set-Size Prediction"),
    ("calibrated_ordinal", "Ordinal Prediction"),
    ("tuned_qpp", "QPP Prediction"),
]

PREDICTOR_TUNED_METRIC = {
    "hit_at_1": "f0_5",
    "in_top_max_depth": "mcc",
    "in_top_k_sweep": "mcc",
    "calibrated_ordinal": "recall",
    "tuned_qpp": "f1",
}


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision == 0.0 and recall == 0.0):
        return math.nan
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return math.nan
    return (1.0 + beta_sq) * precision * recall / denominator


def _confusion_metrics(predicted_positive: np.ndarray, actual_positive: np.ndarray) -> dict[str, float]:
    predicted_positive = np.asarray(predicted_positive, dtype=bool)
    actual_positive = np.asarray(actual_positive, dtype=bool)
    tp = int(np.sum(predicted_positive & actual_positive))
    fp = int(np.sum(predicted_positive & ~actual_positive))
    fn = int(np.sum(~predicted_positive & actual_positive))
    tn = int(np.sum(~predicted_positive & ~actual_positive))
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    balanced_accuracy = (
        (recall + specificity) / 2.0 if np.isfinite(recall) and np.isfinite(specificity) else math.nan
    )
    if len(np.unique(actual_positive)) < 2 or len(np.unique(predicted_positive)) < 2:
        mcc = math.nan
    else:
        mcc = float(matthews_corrcoef(actual_positive, predicted_positive))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _fbeta(precision, recall, 1.0),
        "f0_5": _fbeta(precision, recall, 0.5),
        "f2": _fbeta(precision, recall, 2.0),
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
    }


def _load_predictions(store_root: Path, predictor_name: str, language: str, retriever: str) -> pd.DataFrame:
    path = store_root / "predictions" / predictor_name / language / retriever / PREDICTION_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {predictor_name} predictions for {language}/{retriever}. Expected {path}.")
    frame = pd.read_parquet(path)
    if "predicted_in_top_max_depth" not in frame.columns and "predicted_in_top_k" in frame.columns:
        frame["predicted_in_top_max_depth"] = frame["predicted_in_top_k"]
    if predictor_name == "in_top_max_depth" and "in_top_max_depth" not in frame.columns:
        frame["in_top_max_depth"] = frame["gold_rank"].notna() & (frame["gold_rank"] <= MAX_RANKING_DEPTH)
    return frame


def _actual_positive(gold_rank: pd.Series, k: int) -> np.ndarray:
    return (gold_rank <= k).to_numpy(dtype=bool)


def _hit_at_1_rows(store_root: Path, language: str, retriever: str) -> list[tuple[int, dict[str, float]]]:
    frame = _load_predictions(store_root, "hit_at_1", language, retriever)
    actual_positive = frame["hit_at_1"].astype(bool).to_numpy()
    predicted_positive = frame["predicted_hit_at_1"].astype(bool).to_numpy()
    return [(1, _confusion_metrics(predicted_positive, actual_positive))]


def _in_top_max_depth_best_rows(
    store_root: Path, language: str, retriever: str, max_ranking_depth: int
) -> list[tuple[int, dict[str, float]]]:
    frame = _load_predictions(store_root, "in_top_max_depth", language, retriever)
    actual_positive = frame["in_top_max_depth"].astype(bool).to_numpy()
    predicted_positive = frame["predicted_in_top_max_depth"].astype(bool).to_numpy()
    return [(max_ranking_depth, _confusion_metrics(predicted_positive, actual_positive))]


def _in_top_k_sweep_rows(store_root: Path, language: str, retriever: str, candidate_sizes: list[int]) -> list[tuple[int, dict[str, float]]]:
    rows = []
    for k in candidate_sizes:
        frame = _load_predictions(store_root, f"in_top_k{k}", language, retriever)
        gold_rank = frame["gold_rank"]
        predicted_positive = frame["predicted_in_top_max_depth"].astype(bool).to_numpy()
        rows.append((k, _confusion_metrics(predicted_positive, _actual_positive(gold_rank, k))))
    return rows


def _selected_size_rows(
    store_root: Path, predictor_name: str, language: str, retriever: str, candidate_sizes: list[int]
) -> list[tuple[int, dict[str, float]]]:
    frame = _load_predictions(store_root, predictor_name, language, retriever)
    gold_rank = frame["gold_rank"]
    predicted_failure = frame["predicted_failure"].astype(bool).to_numpy()
    selected_size = frame["selected_size"].astype(np.float64).to_numpy()
    predicted_size = np.where(predicted_failure, np.inf, selected_size)
    return [
        (k, _confusion_metrics(predicted_size <= k, _actual_positive(gold_rank, k)))
        for k in candidate_sizes
    ]


def _predictor_rows(
    predictor_key: str,
    store_root: Path,
    language: str,
    retriever: str,
    candidate_sizes: list[int],
    max_ranking_depth: int,
) -> list[tuple[int, dict[str, float]]]:
    if predictor_key == "hit_at_1":
        return _hit_at_1_rows(store_root, language, retriever)
    if predictor_key == "in_top_max_depth":
        return _in_top_max_depth_best_rows(store_root, language, retriever, max_ranking_depth)
    if predictor_key == "in_top_k_sweep":
        return _in_top_k_sweep_rows(store_root, language, retriever, candidate_sizes)
    if predictor_key == "calibrated_ordinal":
        return _selected_size_rows(store_root, "calibrated_ordinal", language, retriever, candidate_sizes)
    if predictor_key == "tuned_qpp":
        return _selected_size_rows(store_root, "tuned_qpp", language, retriever, candidate_sizes)
    raise ValueError(f"Unknown predictor key: {predictor_key}")


METRIC_COLUMNS = ["precision", "recall", "f1", "f0_5", "f2", "mcc", "balanced_accuracy"]
METRIC_HEADER_LABELS = {
    "precision": "Prec.",
    "recall": "Rec.",
    "f1": "F1",
    "f0_5": "F0.5",
    "f2": "F2",
    "mcc": "MCC",
    "balanced_accuracy": "BalAcc",
}


def _format_metric(
    value: float,
    is_best_in_k: bool = False,
    is_tuned_objective: bool = False,
    is_predictor_best: bool = False,
) -> str:
    if not np.isfinite(value):
        return "--"
    text = f"{value:.3f}"
    if is_best_in_k:
        text = f"\\textbf{{{text}}}"
    if is_tuned_objective:
        text = f"\\underline{{{text}}}"
    if is_predictor_best:
        text = f"\\fbox{{{text}}}"
    return text


def _build_table(
    language: str,
    retriever: str,
    retriever_display: str,
    store_root: Path,
    candidate_sizes: list[int],
    max_ranking_depth: int,
) -> str:
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append(
        "\\caption{Candidate-size confusion breakdown and metrics per predictor, grouped by $k$, "
        f"{retriever_display} retriever, {language} dev split. Bold: best value per metric within a "
        "$k$ group. Underline: each predictor's tuned objective. Box: a predictor's own best value "
        "across its $k$ rows.}"
    )
    lines.append(f"\\label{{tab:candidate_size_confusion_{retriever}_{language}}}")

    rows_by_k: dict[int, list[tuple[str, str, dict[str, float]]]] = {}
    rows_by_predictor: dict[str, list[dict[str, float]]] = {}
    for predictor_key, predictor_label in PREDICTOR_BLOCKS:
        rows = _predictor_rows(predictor_key, store_root, language, retriever, candidate_sizes, max_ranking_depth)
        for k, metrics in rows:
            rows_by_k.setdefault(k, []).append((predictor_key, predictor_label, metrics))
            rows_by_predictor.setdefault(predictor_key, []).append(metrics)

    predictor_own_best: dict[str, dict[str, float]] = {
        predictor_key: {
            column: max(
                (metrics[column] for metrics in predictor_rows if np.isfinite(metrics[column])),
                default=math.nan,
            )
            for column in METRIC_COLUMNS
        }
        for predictor_key, predictor_rows in rows_by_predictor.items()
        if len(predictor_rows) > 1
    }

    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{l|rr|rrrr|rrrrrrr}")
    lines.append("\\toprule")
    header_metrics = " & ".join(METRIC_HEADER_LABELS[column] for column in METRIC_COLUMNS)
    lines.append(f"Predictor & Actual $\\le k$ & Predicted $\\le k$ & TP & FP & TN & FN & {header_metrics} \\\\")
    for k in sorted(rows_by_k):
        entries = rows_by_k[k]
        best_per_column = {
            column: max(
                (metrics[column] for _, _, metrics in entries if np.isfinite(metrics[column])),
                default=math.nan,
            )
            for column in METRIC_COLUMNS
        }
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{14}}{{l}}{{\\textit{{k={k}}}}} \\\\")
        for predictor_key, predictor_label, metrics in entries:
            tuned_column = PREDICTOR_TUNED_METRIC.get(predictor_key)
            own_best = predictor_own_best.get(predictor_key)
            formatted = {
                column: _format_metric(
                    metrics[column],
                    is_best_in_k=metrics[column] == best_per_column[column],
                    is_tuned_objective=column == tuned_column,
                    is_predictor_best=(
                        own_best is not None and metrics[column] == own_best[column] and np.isfinite(metrics[column])
                    ),
                )
                for column in METRIC_COLUMNS
            }
            actual_count = int(metrics["tp"] + metrics["fn"])
            predicted_count = int(metrics["tp"] + metrics["fp"])
            lines.append(
                f"{predictor_label} & {actual_count} & {predicted_count} & {metrics['tp']} & {metrics['fp']} & "
                f"{metrics['tn']} & {metrics['fn']} & {formatted['precision']} & {formatted['recall']} & "
                f"{formatted['f1']} & {formatted['f0_5']} & {formatted['f2']} & {formatted['mcc']} & "
                f"{formatted['balanced_accuracy']} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: list[str] = typer.Option(
        None,
        "--retriever",
        help="Retriever(s) to include. Repeat for multiple. Defaults to bm25 and qwen3_embedding_0_6b.",
    ),
) -> None:
    runtime = build_runtime(config, verbose=False)
    retrievers = retriever or DEFAULT_RETRIEVERS
    candidate_sizes = sorted(int(size) for size in runtime.config.candidate_sizes)
    max_ranking_depth = int(runtime.config.max_ranking_depth)
    store_root = runtime.store.root_dir

    output_dir = runtime.config.latex_table_dir / "candidate_size_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    for retriever_name in retrievers:
        display_name = RETRIEVER_DISPLAY_NAMES.get(retriever_name, retriever_name)
        latex = _build_table(language, retriever_name, display_name, store_root, candidate_sizes, max_ranking_depth)
        output_path = output_dir / f"table_candidate_size_confusion_{retriever_name}_{language}.tex"
        output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
        print(output_path)


if __name__ == "__main__":
    typer.run(main)
