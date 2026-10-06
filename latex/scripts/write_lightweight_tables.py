# latex/scripts/write_lightweight_tables.py
from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _lightweight_calibrated_ordinal_per_head_common import build_frames_by_depth, head_target, head_training_rows
from _lightweight_in_top_max_depth_common import build_merged_frame
from _lightweight_tuned_qpp_common import FAILURE_TARGET_COLUMN, REGRESSOR_TARGET_COLUMN, build_qpp_frame

from query_difficulty.analysis import (
    hit_at_1_predictor_metric_summary,
    in_top_max_depth_predictor_metric_summary,
    ordinal_predictor_metric_summary,
)
from query_difficulty.features_hit_at_1_lightweight import MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES
from query_difficulty.lightweight_selection import base_feature_name, selection_source_predictor
from query_difficulty.modeling import (
    IDENTIFIER_COLUMNS,
    CalibratedOrdinalPredictorRunner,
    HitAtOnePredictorRunner,
    InTopMaxDepthPredictorRunner,
    TunedQppPredictorRunner,
)
from query_difficulty.modeling_calibrated_ordinal_lightweight import SOLVABILITY_HEAD
from query_difficulty.runtime import build_runtime

RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]
RETRIEVER_DISPLAY = {
    "bm25": "BM25",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
    "bge_m3": "BGE-M3",
    "bge_large_en_v1_5": "BGE-large-en-v1.5",
    "e5_large_v2": "E5-large-v2",
    "gtr_t5_xl": "GTR-T5-XL",
    "multilingual_e5_large": "multilingual-E5-large",
    "multilingual_e5_base": "multilingual-E5-base",
}
HIT_AT_1_TABLE_RETRIEVERS = list(RETRIEVER_DISPLAY)
IN_TOP_K_DEPTHS = [5, 10, 20, 30, 50, 60, 100, 120, 200, 1000]
ORDINAL_STAGE_DEPTHS = [5, 10, 20, 50, 100, 200]
MANIFESTS = {
    "hit_at_1": "config/lightweight_hit_at_1_features.reselected",
    "in_top_max_depth": "config/lightweight_in_top_max_depth_features.grouped",
    "in_top_k_sweep": "config/lightweight_in_top_k_sweep_features.reselected",
    "calibrated_ordinal": "config/lightweight_calibrated_ordinal_features.reselected",
    "tuned_qpp_failure": "config/lightweight_tuned_qpp_failure_features.reselected",
    "tuned_qpp_regressor": "config/lightweight_tuned_qpp_regressor_features.reselected",
}


def _read(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def _display_names() -> dict[str, str]:
    groups = yaml.safe_load((ROOT / "config" / "feature_groups.yaml").read_text(encoding="utf-8"))
    names = {}
    for group in groups["feature_groups"].values():
        names.update(group["features"])
    return names


def _direction(series: pd.Series, target: pd.Series) -> str:
    series = pd.to_numeric(series, errors="coerce")
    if series.nunique(dropna=True) <= 1:
        return ""
    if target.dtype == bool or set(pd.unique(target.dropna())) <= {0, 1}:
        positive = target.astype(bool)
        higher = series[positive].mean() > series[~positive].mean()
    else:
        correlation = series.corr(pd.to_numeric(target, errors="coerce"))
        if pd.isna(correlation):
            return ""
        higher = correlation > 0
    return "+" if higher else "$-$"


def _is_depth_dependent(frame: pd.DataFrame, feature: str) -> bool:
    if "::" not in feature:
        return False
    base = base_feature_name(feature)
    copies = [column for column in frame.columns if "::" in column and base_feature_name(column) == base and column != feature]
    if not copies:
        return True
    reference = frame[feature]
    return any(not reference.equals(frame[column]) for column in copies)


def _feature_label(feature: str, frame: pd.DataFrame, names: dict[str, str]) -> str:
    label = _escape(names.get(base_feature_name(feature), base_feature_name(feature)))
    if "::" in feature and _is_depth_dependent(frame, feature):
        label += f" ($k={int(feature.split('::', 1)[0].removeprefix('depth_'))}$)"
    return label


def _importance_table(caption: str, label: str, importances: pd.DataFrame, frame: pd.DataFrame, target: pd.Series, names: dict[str, str]) -> str:
    table = importances.loc[importances["importance"] > 0].sort_values("importance", ascending=False)
    total = float(importances["importance"].sum())
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{0.85}",
        "\\begin{tabular}{p{0.55\\linewidth}clrr}",
        "\\toprule",
        "Feature Name & Direction & Importance (raw) & Importance (\\% of total) \\\\",
        "\\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{_feature_label(row.feature, frame, names)} & {_direction(frame[row.feature], target)} & "
            f"{row.importance:.6f} & {row.importance / total * 100:.2f}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _ordinal_combined_importance_table(store_root: Path, retriever: str, ordinal_frames: dict, names: dict[str, str]) -> str:
    model_root = store_root / "models" / "lightweight" / "calibrated_ordinal" / retriever
    heads = [SOLVABILITY_HEAD] + [str(depth) for depth in ORDINAL_STAGE_DEPTHS]
    shares: dict[str, dict[str, float]] = {}
    signs: dict[str, dict[str, str]] = {}
    for head in heads:
        file_name = "solvability_importances.parquet" if head == SOLVABILITY_HEAD else f"depth_{int(head):04d}_importances.parquet"
        importances = pd.read_parquet(model_root / file_name)
        rows = head_training_rows(ordinal_frames, head)
        target = head_target(rows, head)
        total = float(importances["importance"].sum())
        for record in importances.loc[importances["importance"] > 0].itertuples(index=False):
            shares.setdefault(record.feature, {})[head] = record.importance / total * 100
            signs.setdefault(record.feature, {})[head] = _direction(rows[record.feature], target)
    order = sorted(shares, key=lambda feature: -sum(shares[feature].values()) / len(heads))
    column_best = {head: max(shares[feature].get(head, 0.0) for feature in shares) for head in heads}

    def cell(feature: str, head: str) -> str:
        if head not in shares[feature]:
            return ""
        text = f"{signs[feature][head]}{shares[feature][head]:.1f}"
        return f"\\textbf{{{text}}}" if shares[feature][head] == column_best[head] else text

    header = ["Feature", "Solv."] + [f"$k={head}$" for head in heads[1:]] + ["Average"]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Lightweight ordinal predictor feature importances (\\% of each head; sign = direction; bold = most important feature per head) for {_escape(retriever)}.}}",
        f"\\label{{tab:calibrated_ordinal_lightweight_importances_{retriever}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{0.85}",
        "\\begin{tabular}{p{0.38\\linewidth}" + "r" * (len(heads) + 1) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for feature in order:
        cells = [_escape(names.get(feature, feature))]
        for head in heads:
            cells.append(cell(feature, head))
        cells.append(f"{sum(shares[feature].values()) / len(heads):.1f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _in_top_k_combined_importance_table(store_root: Path, retriever: str, merged_frame: pd.DataFrame, names: dict[str, str]) -> str:
    shares: dict[str, dict[int, float]] = {}
    cells: dict[str, dict[int, list[tuple[float, str, int | None]]]] = {}
    for depth in IN_TOP_K_DEPTHS:
        importances = pd.read_parquet(store_root / "models" / "lightweight" / f"in_top_k{depth}" / retriever / "feature_importances.parquet")
        total = float(importances["importance"].sum())
        within = merged_frame["gold_rank"].le(depth).fillna(False)
        for record in importances.loc[importances["importance"] > 0].itertuples(index=False):
            base = base_feature_name(record.feature)
            share = record.importance / total * 100
            feature_depth = int(record.feature.split("::", 1)[0].removeprefix("depth_")) if _is_depth_dependent(merged_frame, record.feature) else None
            shares.setdefault(base, {})[depth] = shares.get(base, {}).get(depth, 0.0) + share
            cells.setdefault(base, {}).setdefault(depth, []).append((share, _direction(merged_frame[record.feature], within), feature_depth))
    order = sorted(shares, key=lambda base: -sum(shares[base].values()) / len(IN_TOP_K_DEPTHS))
    column_best = {depth: max(shares[base].get(depth, 0.0) for base in shares) for depth in IN_TOP_K_DEPTHS}

    def main_depth(base: str) -> str:
        totals: dict[int, float] = {}
        for parts in cells[base].values():
            for share, _, feature_depth in parts:
                if feature_depth is not None:
                    totals[feature_depth] = totals.get(feature_depth, 0.0) + share
        return str(max(totals, key=totals.get)) if totals else ""

    def cell(base: str, depth: int) -> str:
        if depth not in cells[base]:
            return ""
        sign = max(cells[base][depth], key=lambda part: part[0])[1]
        text = f"{sign}{shares[base][depth]:.1f}"
        return f"\\textbf{{{text}}}" if shares[base][depth] == column_best[depth] else text

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Lightweight binary predictor feature importances (\\% of each $k$; sign = direction; bold = most important feature per $k$; Depth = feature depth with the highest total importance) for {_escape(retriever)}.}}",
        f"\\label{{tab:in_top_k_sweep_lightweight_importances_{retriever}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\renewcommand{\\arraystretch}{0.85}",
        "\\begin{tabular}{p{0.28\\linewidth}c|" + "r" * len(IN_TOP_K_DEPTHS) + "|r}",
        "\\toprule",
        " & ".join(["Feature", "Depth"] + [f"$k={depth}$" for depth in IN_TOP_K_DEPTHS] + ["Average"]) + " \\\\",
        "\\midrule",
    ]
    for base in order:
        row = [_escape(names.get(base, base)), main_depth(base)] + [cell(base, depth) for depth in IN_TOP_K_DEPTHS]
        row.append(f"{sum(shares[base].values()) / len(IN_TOP_K_DEPTHS):.1f}")
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _write(table_dir: Path, category: str, name: str, content: str) -> None:
    path = table_dir / category / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(apply_thesis_captions(content), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def _metadata(store_root: Path, predictor_name: str, language: str, retriever: str) -> dict:
    path = store_root / "predictions" / predictor_name / language / retriever / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mcc(metrics: dict) -> float:
    tp, fp = metrics["true_positives"], metrics["false_positives"]
    tn, fn = metrics["true_negatives"], metrics["false_negatives"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else math.nan


def _importance_count(store_root: Path, predictor_name: str, retriever: str, file_name: str) -> int:
    return int(len(pd.read_parquet(store_root / "models" / predictor_name / retriever / file_name)))


def _bold_better(full: float, light: float, higher_is_better: bool, digits: int) -> tuple[str, str]:
    texts = [f"{full:.{digits}f}", f"{light:.{digits}f}"]
    full, light = round(full, digits), round(light, digits)
    better = (full > light) if higher_is_better else (full < light)
    if full == light:
        return texts[0], texts[1]
    index = 0 if better else 1
    texts[index] = f"\\textbf{{{texts[index]}}}"
    return texts[0], texts[1]


def _tabular(caption: str, label: str, column_spec: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _mrr_at_5(frame: pd.DataFrame) -> float:
    gold_rank = frame["gold_rank"]
    return float(np.mean(np.where(gold_rank.notna() & (gold_rank <= 5), 1.0 / gold_rank, 0.0)))


BASELINE_REPORT_SUBDIR = Path("reports") / "predictor_baselines"
MAJORITY_BASELINE_NOTE = "Majority: always predicts the majority class of the train split."
HIT_AT_1_BASELINE_NOTE = f"{MAJORITY_BASELINE_NOTE} Margin: top-1/top-2 retrieval-score margin with a threshold tuned on dev for maximum F0.5."
IN_TOP_K_BASELINE_NOTE = (
    f"{MAJORITY_BASELINE_NOTE} Margin: rank-$k$/rank-$(k+1)$ retrieval-score margin with threshold and direction tuned on dev for maximum MCC; "
    "not available for $k=1000$ because only the top 1000 retrieval scores are cached."
)
IN_TOP_MAX_DEPTH_BASELINE_NOTE = (
    f"{MAJORITY_BASELINE_NOTE} The rank-1000/rank-1001 margin baseline is not available because only the top 1000 retrieval scores are cached."
)


def _load_baselines(store_root: Path, language: str) -> dict:
    path = store_root / BASELINE_REPORT_SUBDIR / f"predictor_baselines_{language}.json"
    if not path.exists():
        print(f"No baselines at {path}; run scripts/compute_predictor_baselines.py. Baseline cells are left as '-'.")
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(record["baseline"], record["family"], record["retriever"], record["k"]): record for record in payload["records"]}


def _baseline_cells(baselines: dict, baseline: str, family: str, retriever: str, k: int | None, split: str, metrics: tuple[str, ...]) -> list[str]:
    record = baselines.get((baseline, family, retriever, k))
    values = (record or {}).get(f"{split}_metrics") or {}
    cells = []
    for metric in metrics:
        value = values.get(metric)
        cells.append("-" if value is None or not math.isfinite(value) else f"{value:.4f}")
    return cells


def _hit_at_1_full_vs_lightweight_table(runtime, store_root: Path, language: str, baselines: dict) -> str:
    rows = []
    for retriever in HIT_AT_1_TABLE_RETRIEVERS:
        full_frame = runtime.store.read_parquet(store_root / "predictions" / "hit_at_1" / language / retriever / "predictions.parquet")
        light_root = store_root / "predictions" / "lightweight" / "hit_at_1" / language / retriever
        light_frame = runtime.store.read_parquet(light_root / "predictions.parquet")
        full, light = hit_at_1_predictor_metric_summary(full_frame), hit_at_1_predictor_metric_summary(light_frame)
        rows.append({
            "key": retriever,
            "retriever": RETRIEVER_DISPLAY[retriever],
            "values": {
                "mrr": _mrr_at_5(full_frame),
                "precision_full": full["precision"], "recall_full": full["recall"], "f0_5_full": full["f0_5"],
                "precision_light": light["precision"], "recall_light": light["recall"], "f0_5_light": light["f0_5"],
            },
            "features_full": len(pd.read_parquet(store_root / "models" / "hit_at_1" / retriever / "feature_importances.parquet")),
            "features_light": len(json.loads((light_root / "metadata.json").read_text(encoding="utf-8"))["qualifying_features"]),
        })
    rows.sort(key=lambda row: row["values"]["mrr"], reverse=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Hit@1 predictor performance on the English dev split. {HIT_AT_1_BASELINE_NOTE}}}",
        f"\\label{{tab:hit_at_1_full_vs_lightweight_dev_performance_{language}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l|c|c|ccc|ccc|c|ccc|ccc}",
        "\\toprule",
        " & Retrieval & & \\multicolumn{3}{c|}{Full} & \\multicolumn{3}{c|}{Lightweight} & & \\multicolumn{3}{c|}{Majority} & \\multicolumn{3}{c}{Margin} \\\\",
        "Retriever & MRR@5 & \\#Feat. (Full) & Prec. & Rec. & F0.5 & Prec. & Rec. & F0.5 & \\#Feat. (Light) & Prec. & Rec. & F0.5 & Prec. & Rec. & F0.5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        pairs = [_bold_better(row["values"][f"{metric}_full"], row["values"][f"{metric}_light"], True, 4) for metric in ("precision", "recall", "f0_5")]
        cells = [row["retriever"], f"{row['values']['mrr']:.4f}", str(row["features_full"])]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + [str(row["features_light"])]
        for baseline in ("majority", "margin"):
            cells += _baseline_cells(baselines, baseline, "hit_at_1", row["key"], None, "dev", ("precision", "recall", "f0_5"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _in_top_max_depth_full_vs_lightweight_table(runtime, store_root: Path, language: str, baselines: dict) -> str:
    rows = []
    for retriever in RETRIEVERS:
        full_frame = runtime.store.read_parquet(store_root / "predictions" / "in_top_max_depth" / language / retriever / "predictions.parquet")
        light_frame = runtime.store.read_parquet(store_root / "predictions" / "lightweight" / "in_top_max_depth" / language / retriever / "predictions.parquet")
        full, light = in_top_max_depth_predictor_metric_summary(full_frame), in_top_max_depth_predictor_metric_summary(light_frame)
        values = {"mrr": _mrr_at_5(full_frame)}
        for metric in ("precision", "recall", "f0_5", "mcc"):
            values[f"{metric}_full"], values[f"{metric}_light"] = full[metric], light[metric]
        rows.append({
            "key": retriever,
            "retriever": RETRIEVER_DISPLAY[retriever],
            "values": values,
            "features_full": _importance_count(store_root, "in_top_max_depth", retriever, "feature_importances.parquet"),
            "features_light": _importance_count(store_root, "lightweight/in_top_max_depth", retriever, "feature_importances.parquet"),
        })
    rows.sort(key=lambda row: row["values"]["mrr"], reverse=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Performance on the English dev split, comparing the full versus lightweight variants of the \\textit{{In-Max-Depth}} predictor with max depth=1000. {IN_TOP_MAX_DEPTH_BASELINE_NOTE}}}",
        f"\\label{{tab:in_top_max_depth_full_vs_lightweight_dev_performance_{language}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l|c|c|cccc|cccc|c|cccc}",
        "\\toprule",
        " & Retrieval & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & & \\multicolumn{4}{c}{Majority} \\\\",
        "Retriever & MRR@5 & \\#Feat. (Full) & Prec. & Rec. & F0.5 & MCC & Prec. & Rec. & F0.5 & MCC & \\#Feat. (Light) & Prec. & Rec. & F0.5 & MCC \\\\",
        "\\midrule",
    ]
    for row in rows:
        pairs = [_bold_better(row["values"][f"{metric}_full"], row["values"][f"{metric}_light"], True, 4) for metric in ("precision", "recall", "f0_5", "mcc")]
        cells = [row["retriever"], f"{row['values']['mrr']:.4f}", str(row["features_full"])]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + [str(row["features_light"])]
        cells += _baseline_cells(baselines, "majority", "in_top_max_depth", row["key"], 1000, "dev", ("precision", "recall", "f0_5", "mcc"))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _in_top_k_full_vs_lightweight_table(runtime, store_root: Path, language: str, baselines: dict) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Performance of the binary candidate-set-size predictors on the English dev split. {IN_TOP_K_BASELINE_NOTE}}}",
        f"\\label{{tab:in_top_k_sweep_full_vs_lightweight_dev_performance_{language}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{cl|c|c|c|c|c|c}",
        "\\toprule",
        " & & & Full & Lightweight & & Majority & Margin \\\\",
        "$k$ & Retriever & \\#Feat. (Full) & MCC & MCC & \\#Feat. (Light) & MCC & MCC \\\\",
    ]
    for depth in IN_TOP_K_DEPTHS:
        lines.append("\\midrule")
        for index, retriever in enumerate(RETRIEVERS):
            full = in_top_max_depth_predictor_metric_summary(runtime.store.read_parquet(store_root / "predictions" / f"in_top_k{depth}" / language / retriever / "predictions.parquet"))
            light = in_top_max_depth_predictor_metric_summary(runtime.store.read_parquet(store_root / "predictions" / "lightweight" / f"in_top_k{depth}" / language / retriever / "predictions.parquet"))
            cells = [str(depth) if index == 0 else "", RETRIEVER_DISPLAY[retriever],
                     str(_importance_count(store_root, f"in_top_k{depth}", retriever, "feature_importances.parquet"))]
            cells += list(_bold_better(full["mcc"], light["mcc"], True, 4))
            cells.append(str(_importance_count(store_root, f"lightweight/in_top_k{depth}", retriever, "feature_importances.parquet")))
            for baseline in ("majority", "margin"):
                cells += _baseline_cells(baselines, baseline, "in_top_k", retriever, depth, "dev", ("mcc",))
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _head_feature_count_range(model_dir: Path) -> str:
    paths = sorted(model_dir.glob("depth_*_importances.parquet")) + [model_dir / "solvability_importances.parquet"]
    counts = [len(pd.read_parquet(path)) for path in paths if path.exists()]
    if not counts:
        raise FileNotFoundError(f"No per-head feature importances under {model_dir}")
    return str(counts[0]) if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"


def _calibrated_ordinal_full_vs_lightweight_table(runtime, store_root: Path, language: str) -> str:
    candidate_sizes = runtime.config.candidate_sizes
    metric_keys = [("recall", "candidate_recall", True), ("exact", "exact_class_accuracy", True), ("qwk", "quadratic_weighted_cohen_kappa", True), ("mae", "ordinal_mae", False)]
    rows = []
    for retriever in RETRIEVERS:
        full = ordinal_predictor_metric_summary(pd.read_parquet(store_root / "predictions" / "calibrated_ordinal" / language / retriever / "predictions.parquet"), candidate_sizes)
        light = ordinal_predictor_metric_summary(pd.read_parquet(store_root / "predictions" / "lightweight" / "calibrated_ordinal" / language / retriever / "predictions.parquet"), candidate_sizes)
        values = {}
        for short, key, _ in metric_keys:
            values[f"{short}_full"], values[f"{short}_light"] = float(full[key]), float(light[key])
        rows.append({
            "retriever": RETRIEVER_DISPLAY[retriever],
            "values": values,
            "features_full": _head_feature_count_range(store_root / "models" / "calibrated_ordinal" / retriever),
            "features_light": _head_feature_count_range(store_root / "models" / "lightweight" / "calibrated_ordinal" / retriever),
        })
    rows.sort(key=lambda row: row["values"]["recall_full"], reverse=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Performance on the English dev split, comparing the full versus lightweight variants of the calibrated ordinal predictor.}",
        f"\\label{{tab:calibrated_ordinal_lightweight_dev_performance_{language}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l|c|cccc|cccc|c}",
        "\\toprule",
        " & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
        "Retriever & \\#Feat. (Full) & Recall & Exact Acc. & QWK & MAE & Recall & Exact Acc. & QWK & MAE & \\#Feat. (Light) \\\\",
        "\\midrule",
    ]
    for row in rows:
        pairs = [_bold_better(row["values"][f"{short}_full"], row["values"][f"{short}_light"], higher, 4) for short, _, higher in metric_keys]
        cells = [row["retriever"], row["features_full"]]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + [row["features_light"]]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _tuned_qpp_full_vs_lightweight_table(store_root: Path, language: str) -> str:
    metric_keys = [("candidate_recall", True, 4), ("mean_selected_size", False, 2), ("median_selected_size", False, 1), ("failure_rate", False, 4)]
    rows = []
    for retriever in RETRIEVERS:
        full = _metadata(store_root, "tuned_qpp", language, retriever)["dev_metrics"]
        light = _metadata(store_root, "lightweight/tuned_qpp", language, retriever)["dev_metrics"]
        regressor = len(_read(f"{MANIFESTS['tuned_qpp_regressor']}.rfe_pruned.json")["retrievers"][retriever]["qualifying_features"])
        rows.append({
            "retriever": RETRIEVER_DISPLAY[retriever],
            "recall_full": full["candidate_recall"],
            "pairs": [_bold_better(full[key], light[key], higher, digits) for key, higher, digits in metric_keys],
            "features_full": str(_importance_count(store_root, "tuned_qpp", retriever, "score_importances.parquet")),
            "features_light": str(regressor),
        })
    rows.sort(key=lambda row: row["recall_full"], reverse=True)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Performance on the English dev split, comparing the full versus lightweight variants of the QPP predictor.}",
        f"\\label{{tab:tuned_qpp_lightweight_dev_performance_{language}}}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l|c|cccc|cccc|c}",
        "\\toprule",
        " & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
        "Retriever & \\#Feat. (Full) & Recall & Mean & Median & Failure & Recall & Mean & Median & Failure & \\#Feat. (Light) \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["retriever"], row["features_full"]]
        cells += [pair[0] for pair in row["pairs"]] + [pair[1] for pair in row["pairs"]] + [row["features_light"]]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _source_candidate_count(store_root: Path, family: str, retriever: str, file_name: str, grouped: bool) -> int:
    source = selection_source_predictor(store_root, family, retriever, file_name)
    features = pd.read_parquet(store_root / "models" / source / retriever / file_name)["feature"]
    bases = features.map(base_feature_name)
    features = features.loc[~bases.isin(MULTI_RETRIEVER_CONSENSUS_FEATURE_NAMES)]
    return int(features.map(base_feature_name).nunique()) if grouped else int(len(features))


def _stage_counts(prefix: str, retriever: str, head: str | None = None, depth: int | None = None) -> list[int]:
    counts = []
    for stage in ("freeze", "correlation_pruned", "correlation_pruned_round2", "rfe_pruned"):
        manifest = _read(f"{prefix}.{stage}.json")
        if depth is not None:
            entry = manifest["depths"][str(depth)]["retrievers"][retriever]
        else:
            entry = manifest["retrievers"][retriever]
        if head is not None:
            entry = entry["heads"][head]
        counts.append(len(entry["qualifying_features"]))
    return counts


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir
    table_dir = runtime.config.latex_table_dir
    names = _display_names()
    baselines = _load_baselines(store_root, language)
    split = f"{language}_train"

    hit_runner = HitAtOnePredictorRunner(runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)
    depth_runner = InTopMaxDepthPredictorRunner(runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)
    qpp_runner = TunedQppPredictorRunner(runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)
    ordinal_runner = CalibratedOrdinalPredictorRunner(runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval)
    hit_depth = hit_runner._feature_depth()

    for retriever in RETRIEVERS:
        escaped = _escape(retriever)
        print(f"Building {split} frames for {retriever}")
        hit_frame = runtime.features.build_features(split, retriever, candidate_sizes=[hit_depth])[hit_depth]
        merged_frame = build_merged_frame(depth_runner, split, retriever)
        qpp_frame = build_qpp_frame(qpp_runner, split, retriever)
        ordinal_frames = build_frames_by_depth(ordinal_runner, split, retriever)

        importances = pd.read_parquet(store_root / "models" / "lightweight" / "hit_at_1" / retriever / "feature_importances.parquet")
        _write(table_dir, "hit_at_1", f"table_hit_at_1_lightweight_importances_{retriever}.tex", _importance_table(
            f"Lightweight hit@1 predictor feature importances for {escaped}.", f"tab:hit_at_1_lightweight_importances_{retriever}",
            importances, hit_frame, hit_frame["gold_rank"].fillna(-1.0).eq(1.0), names))

        reachable = ~merged_frame["retrieval_failure_within_max_depth"].astype(bool)
        importances = pd.read_parquet(store_root / "models" / "lightweight" / "in_top_max_depth" / retriever / "feature_importances.parquet")
        _write(table_dir, "in_top_max_depth", f"table_in_top_max_depth_lightweight_importances_{retriever}.tex", _importance_table(
            f"Lightweight in-max-depth predictor feature importances for {escaped}.", f"tab:in_top_max_depth_lightweight_importances_{retriever}",
            importances, merged_frame, reachable, names))

        _write(table_dir, "in_top_k_sweep", f"table_in_top_k_sweep_lightweight_importances_{retriever}.tex",
               _in_top_k_combined_importance_table(store_root, retriever, merged_frame, names))

        _write(table_dir, "calibrated_ordinal", f"table_calibrated_ordinal_lightweight_importances_{retriever}.tex",
               _ordinal_combined_importance_table(store_root, retriever, ordinal_frames, names))

        model_root = store_root / "models" / "lightweight" / "tuned_qpp" / retriever
        _write(table_dir, "tuned_qpp", f"table_tuned_qpp_lightweight_importances_failure_{retriever}.tex", _importance_table(
            f"Lightweight QPP failure classifier feature importances for {escaped}.", f"tab:tuned_qpp_lightweight_importances_failure_{retriever}",
            pd.read_parquet(model_root / "failure_importances.parquet"), qpp_frame, ~qpp_frame[FAILURE_TARGET_COLUMN].astype(bool), names))
        _write(table_dir, "tuned_qpp", f"table_tuned_qpp_lightweight_importances_score_{retriever}.tex", _importance_table(
            f"Lightweight QPP score regressor feature importances for {escaped}.", f"tab:tuned_qpp_lightweight_importances_score_{retriever}",
            pd.read_parquet(model_root / "score_importances.parquet"), qpp_frame, qpp_frame[REGRESSOR_TARGET_COLUMN].astype(float), names))

    _write(table_dir, "hit_at_1", f"table_hit_at_1_full_vs_lightweight_dev_performance_{language}.tex", _hit_at_1_full_vs_lightweight_table(runtime, store_root, language, baselines))

    _write(table_dir, "in_top_max_depth", f"table_in_top_max_depth_full_vs_lightweight_dev_performance_{language}.tex", _in_top_max_depth_full_vs_lightweight_table(runtime, store_root, language, baselines))

    _write(table_dir, "in_top_k_sweep", f"table_in_top_k_sweep_full_vs_lightweight_dev_performance_{language}.tex", _in_top_k_full_vs_lightweight_table(runtime, store_root, language, baselines))

    _write(table_dir, "calibrated_ordinal", f"table_calibrated_ordinal_lightweight_dev_performance_{language}.tex", _calibrated_ordinal_full_vs_lightweight_table(runtime, store_root, language))

    _write(table_dir, "tuned_qpp", f"table_tuned_qpp_lightweight_dev_performance_{language}.tex", _tuned_qpp_full_vs_lightweight_table(store_root, language))

    steps_header = ["Full", "Importance", "$|\\rho|\\geq0.80$", "$|\\rho|\\geq0.70$", "RFE"]
    rows = []
    for r in RETRIEVERS:
        full_f0_5 = hit_at_1_predictor_metric_summary(runtime.store.read_parquet(store_root / "predictions" / "hit_at_1" / language / r / "predictions.parquet"))["f0_5"]
        light_f0_5 = hit_at_1_predictor_metric_summary(runtime.store.read_parquet(store_root / "predictions" / "lightweight" / "hit_at_1" / language / r / "predictions.parquet"))["f0_5"]
        rows.append([RETRIEVER_DISPLAY[r], str(_source_candidate_count(store_root, "hit_at_1", r, "feature_importances.parquet", False)),
                     *map(str, _stage_counts(MANIFESTS["hit_at_1"], r)), *_bold_better(full_f0_5, light_f0_5, True, 4)])
    _write(table_dir, "hit_at_1", f"table_hit_at_1_lightweight_feature_selection_steps_{language}.tex", _tabular(
        "Hit@1 lightweight feature selection (number of features) and dev F0.5.", f"tab:hit_at_1_lightweight_feature_selection_steps_{language}",
        "l|ccccc|cc", ["Retriever", *steps_header, "F0.5 Full", "F0.5 Light"], rows))

    rows = []
    for retriever in RETRIEVERS:
        freeze = _read(f"{MANIFESTS['in_top_max_depth']}.json")["retrievers"][retriever]
        later = [len(_read(f"{MANIFESTS['in_top_max_depth']}.{stage}.json")["retrievers"][retriever]["qualifying_features"])
                 for stage in ("correlation_pruned", "correlation_pruned_round2", "rfe_pruned")]
        rows.append([RETRIEVER_DISPLAY[retriever], str(freeze["source_group_count"]), str(freeze["qualifying_feature_count"]), *map(str, later)])
    _write(table_dir, "in_top_max_depth", f"table_in_top_max_depth_lightweight_feature_selection_steps_{language}.tex", _tabular(
        "In-max-depth lightweight feature selection (number of features; full and importance counted per feature over all depths).",
        f"tab:in_top_max_depth_lightweight_feature_selection_steps_{language}", "l|ccccc", ["Retriever", *steps_header], rows))

    rows = [[f"${depth}$", RETRIEVER_DISPLAY[r], str(_source_candidate_count(store_root, f"in_top_k{depth}", r, "feature_importances.parquet", True)),
             *map(str, _stage_counts(MANIFESTS["in_top_k_sweep"], r, depth=depth))] for depth in IN_TOP_K_DEPTHS for r in RETRIEVERS]
    _write(table_dir, "in_top_k_sweep", f"table_in_top_k_sweep_lightweight_feature_selection_steps_{language}.tex", _tabular(
        "Binary predictor lightweight feature selection (number of features; full and importance counted per feature over all depths).",
        f"tab:in_top_k_sweep_lightweight_feature_selection_steps_{language}", "cl|ccccc", ["$k$", "Retriever", *steps_header], rows))

    rows = []
    for retriever in RETRIEVERS:
        for head in [SOLVABILITY_HEAD] + [str(depth) for depth in ORDINAL_STAGE_DEPTHS]:
            file_name = "solvability_importances.parquet" if head == SOLVABILITY_HEAD else f"depth_{int(head):04d}_importances.parquet"
            rows.append([RETRIEVER_DISPLAY[retriever], "Solvability" if head == SOLVABILITY_HEAD else f"$k={head}$",
                         str(_source_candidate_count(store_root, "calibrated_ordinal", retriever, file_name, False)),
                         *map(str, _stage_counts(MANIFESTS["calibrated_ordinal"], retriever, head=head))])
    _write(table_dir, "calibrated_ordinal", f"table_calibrated_ordinal_lightweight_feature_selection_steps_{language}.tex", _tabular(
        "Ordinal lightweight feature selection per head (number of features).", f"tab:calibrated_ordinal_lightweight_feature_selection_steps_{language}",
        "ll|ccccc", ["Retriever", "Head", *steps_header], rows))

    rows = []
    for retriever in RETRIEVERS:
        for head, key, file_name in (("Failure", "tuned_qpp_failure", "failure_importances.parquet"), ("Regressor", "tuned_qpp_regressor", "score_importances.parquet")):
            rows.append([RETRIEVER_DISPLAY[retriever], head, str(_source_candidate_count(store_root, "tuned_qpp", retriever, file_name, False)),
                         *map(str, _stage_counts(MANIFESTS[key], retriever))])
    _write(table_dir, "tuned_qpp", f"table_tuned_qpp_lightweight_feature_selection_steps_{language}.tex", _tabular(
        "QPP lightweight feature selection per head (number of features).", f"tab:tuned_qpp_lightweight_feature_selection_steps_{language}",
        "ll|ccccc", ["Retriever", "Head", *steps_header], rows))


if __name__ == "__main__":
    typer.run(main)
