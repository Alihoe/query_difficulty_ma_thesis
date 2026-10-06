# latex/scripts/write_frozen_predictor_test_tables.py
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "latex" / "scripts"))

from write_lightweight_tables import (
    HIT_AT_1_BASELINE_NOTE,
    HIT_AT_1_TABLE_RETRIEVERS,
    IN_TOP_K_BASELINE_NOTE,
    IN_TOP_K_DEPTHS,
    IN_TOP_MAX_DEPTH_BASELINE_NOTE,
    MANIFESTS,
    RETRIEVER_DISPLAY,
    RETRIEVERS,
    _baseline_cells,
    _bold_better,
    _head_feature_count_range,
    _importance_count,
    _load_baselines,
    _read,
    _write,
)

from query_difficulty.runtime import build_runtime

REPORT_SUBDIR = Path("reports") / "lightweight" / "frozen_test_evaluation"
LANGUAGE_DISPLAY = {"en": "English"}
CAPTION_PREFIX = "Test-set evaluation of frozen predictors"


def _number(value, digits: int) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _pair(full, light, higher_is_better: bool, digits: int) -> tuple[str, str]:
    if full is None or light is None:
        return _number(full, digits), _number(light, digits)
    return _bold_better(float(full), float(light), higher_is_better, digits)


def _header(caption: str, label: str, column_spec: str, scriptsize: bool = True) -> list[str]:
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if scriptsize:
        lines += ["\\scriptsize", "\\setlength{\\tabcolsep}{3pt}"]
    return lines + [f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule"]


def _footer() -> list[str]:
    return ["\\bottomrule", "\\end{tabular}", "\\end{table}"]


def _metric(records: dict, variant: str, family: str, retriever: str, k, key: str):
    record = records.get((variant, family, retriever, k))
    return None if record is None else record["test_metrics"].get(key)


def _first(*values):
    return next((value for value in values if value is not None), None)


def _hit_at_1_table(records: dict, baselines: dict, store_root: Path, language: str, language_name: str) -> str:
    rows = []
    for retriever in HIT_AT_1_TABLE_RETRIEVERS:
        metadata_path = store_root / "predictions" / "lightweight" / "hit_at_1" / language / retriever / "metadata.json"
        rows.append({
            "retriever": retriever,
            "mrr": _first(_metric(records, "full", "hit_at_1", retriever, None, "retrieval_mrr_at_5"), _metric(records, "lightweight", "hit_at_1", retriever, None, "retrieval_mrr_at_5")),
            "features_full": _importance_count(store_root, "hit_at_1", retriever, "feature_importances.parquet"),
            "features_light": len(json.loads(metadata_path.read_text(encoding="utf-8"))["qualifying_features"]),
        })
    rows.sort(key=lambda row: -1.0 if row["mrr"] is None else row["mrr"], reverse=True)
    lines = _header(
        f"{CAPTION_PREFIX}: Hit@1 predictor performance on the {language_name} test split. {HIT_AT_1_BASELINE_NOTE}",
        f"tab:hit_at_1_full_vs_lightweight_test_performance_{language}",
        "l|c|c|ccc|ccc|c|ccc|ccc",
    )
    lines += [
        " & Retrieval & & \\multicolumn{3}{c|}{Full} & \\multicolumn{3}{c|}{Lightweight} & & \\multicolumn{3}{c|}{Majority} & \\multicolumn{3}{c}{Margin} \\\\",
        "Retriever & MRR@5 & \\#Feat. (Full) & Prec. & Rec. & F0.5 & Prec. & Rec. & F0.5 & \\#Feat. (Light) & Prec. & Rec. & F0.5 & Prec. & Rec. & F0.5 \\\\",
        "\\midrule",
    ]
    for row in rows:
        pairs = [_pair(_metric(records, "full", "hit_at_1", row["retriever"], None, key), _metric(records, "lightweight", "hit_at_1", row["retriever"], None, key), True, 4) for key in ("precision", "recall", "f0_5")]
        cells = [RETRIEVER_DISPLAY[row["retriever"]], _number(row["mrr"], 4), str(row["features_full"])]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + [str(row["features_light"])]
        for baseline in ("majority", "margin"):
            cells += _baseline_cells(baselines, baseline, "hit_at_1", row["retriever"], None, "test", ("precision", "recall", "f0_5"))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _in_top_max_depth_table(records: dict, baselines: dict, store_root: Path, language: str, language_name: str) -> str:
    rows = []
    for retriever in RETRIEVERS:
        rows.append({
            "retriever": retriever,
            "mrr": _first(_metric(records, "full", "in_top_max_depth", retriever, 1000, "retrieval_mrr_at_5"), _metric(records, "lightweight", "in_top_max_depth", retriever, 1000, "retrieval_mrr_at_5")),
        })
    rows.sort(key=lambda row: -1.0 if row["mrr"] is None else row["mrr"], reverse=True)
    lines = _header(
        f"{CAPTION_PREFIX}: performance on the {language_name} test split, comparing the full versus lightweight variants of the \\textit{{In-Max-Depth}} predictor with max depth=1000. {IN_TOP_MAX_DEPTH_BASELINE_NOTE}",
        f"tab:in_top_max_depth_full_vs_lightweight_test_performance_{language}",
        "l|c|c|cccc|cccc|c|cccc",
    )
    lines += [
        " & Retrieval & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & & \\multicolumn{4}{c}{Majority} \\\\",
        "Retriever & MRR@5 & \\#Feat. (Full) & Prec. & Rec. & F0.5 & MCC & Prec. & Rec. & F0.5 & MCC & \\#Feat. (Light) & Prec. & Rec. & F0.5 & MCC \\\\",
        "\\midrule",
    ]
    for row in rows:
        retriever = row["retriever"]
        pairs = [_pair(_metric(records, "full", "in_top_max_depth", retriever, 1000, key), _metric(records, "lightweight", "in_top_max_depth", retriever, 1000, key), True, 4) for key in ("precision", "recall", "f0_5", "mcc")]
        cells = [RETRIEVER_DISPLAY[retriever], _number(row["mrr"], 4), str(_importance_count(store_root, "in_top_max_depth", retriever, "feature_importances.parquet"))]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        cells.append(str(_importance_count(store_root, "lightweight/in_top_max_depth", retriever, "feature_importances.parquet")))
        cells += _baseline_cells(baselines, "majority", "in_top_max_depth", retriever, 1000, "test", ("precision", "recall", "f0_5", "mcc"))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _in_top_k_table(records: dict, baselines: dict, store_root: Path, language: str, language_name: str) -> str:
    lines = _header(
        f"{CAPTION_PREFIX}: performance of the binary candidate-set-size predictors on the {language_name} test split. {IN_TOP_K_BASELINE_NOTE}",
        f"tab:in_top_k_sweep_full_vs_lightweight_test_performance_{language}",
        "cl|c|c|c|c|c|c",
    )
    lines += [
        " & & & Full & Lightweight & & Majority & Margin \\\\",
        "$k$ & Retriever & \\#Feat. (Full) & MCC & MCC & \\#Feat. (Light) & MCC & MCC \\\\",
    ]
    for depth in IN_TOP_K_DEPTHS:
        lines.append("\\midrule")
        for index, retriever in enumerate(RETRIEVERS):
            cells = [str(depth) if index == 0 else "", RETRIEVER_DISPLAY[retriever], str(_importance_count(store_root, f"in_top_k{depth}", retriever, "feature_importances.parquet"))]
            cells += list(_pair(_metric(records, "full", "in_top_k", retriever, depth, "mcc"), _metric(records, "lightweight", "in_top_k", retriever, depth, "mcc"), True, 4))
            cells.append(str(_importance_count(store_root, f"lightweight/in_top_k{depth}", retriever, "feature_importances.parquet")))
            for baseline in ("majority", "margin"):
                cells += _baseline_cells(baselines, baseline, "in_top_k", retriever, depth, "test", ("mcc",))
            lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _calibrated_ordinal_table(records: dict, store_root: Path, language: str, language_name: str) -> str:
    metric_keys = [("candidate_recall", True), ("exact_class_accuracy", True), ("quadratic_weighted_cohen_kappa", True), ("ordinal_mae", False)]
    rows = [{"retriever": r, "recall": _first(_metric(records, "full", "calibrated_ordinal", r, None, "candidate_recall"), _metric(records, "lightweight", "calibrated_ordinal", r, None, "candidate_recall"))} for r in RETRIEVERS]
    rows.sort(key=lambda row: -1.0 if row["recall"] is None else row["recall"], reverse=True)
    lines = _header(
        f"{CAPTION_PREFIX}: performance on the {language_name} test split, comparing the full versus lightweight variants of the calibrated ordinal predictor.",
        f"tab:calibrated_ordinal_lightweight_test_performance_{language}",
        "l|c|cccc|cccc|c",
    )
    lines += [
        " & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
        "Retriever & \\#Feat. (Full) & Recall & Exact Acc. & QWK & MAE & Recall & Exact Acc. & QWK & MAE & \\#Feat. (Light) \\\\",
        "\\midrule",
    ]
    for row in rows:
        retriever = row["retriever"]
        pairs = [_pair(_metric(records, "full", "calibrated_ordinal", retriever, None, key), _metric(records, "lightweight", "calibrated_ordinal", retriever, None, key), higher, 4) for key, higher in metric_keys]
        cells = [RETRIEVER_DISPLAY[retriever], _head_feature_count_range(store_root / "models" / "calibrated_ordinal" / retriever)]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        cells.append(_head_feature_count_range(store_root / "models" / "lightweight" / "calibrated_ordinal" / retriever))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _tuned_qpp_table(records: dict, store_root: Path, language: str, language_name: str) -> str:
    metric_keys = [("candidate_recall", True, 4), ("mean_selected_size", False, 2), ("median_selected_size", False, 1), ("failure_rate", False, 4)]
    rows = [{"retriever": r, "recall": _first(_metric(records, "full", "tuned_qpp", r, None, "candidate_recall"), _metric(records, "lightweight", "tuned_qpp", r, None, "candidate_recall"))} for r in RETRIEVERS]
    rows.sort(key=lambda row: -1.0 if row["recall"] is None else row["recall"], reverse=True)
    lines = _header(
        f"{CAPTION_PREFIX}: performance on the {language_name} test split, comparing the full versus lightweight variants of the QPP predictor.",
        f"tab:tuned_qpp_lightweight_test_performance_{language}",
        "l|c|cccc|cccc|c",
    )
    lines += [
        " & & \\multicolumn{4}{c|}{Full} & \\multicolumn{4}{c|}{Lightweight} & \\\\",
        "Retriever & \\#Feat. (Full) & Recall & Mean & Median & Failure & Recall & Mean & Median & Failure & \\#Feat. (Light) \\\\",
        "\\midrule",
    ]
    for row in rows:
        retriever = row["retriever"]
        pairs = [_pair(_metric(records, "full", "tuned_qpp", retriever, None, key), _metric(records, "lightweight", "tuned_qpp", retriever, None, key), higher, digits) for key, higher, digits in metric_keys]
        cells = [RETRIEVER_DISPLAY[retriever], str(_importance_count(store_root, "tuned_qpp", retriever, "score_importances.parquet"))]
        cells += [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
        cells.append(str(len(_read(f"{MANIFESTS['tuned_qpp_regressor']}.rfe_pruned.json")["retrievers"][retriever]["qualifying_features"])))
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _qpp_available(records: dict, variant: str) -> list[str]:
    return [retriever for retriever in RETRIEVERS if records.get((variant, "tuned_qpp", retriever, None))]


def _tuned_qpp_calibration_table(records: dict, variant: str, language: str, language_name: str) -> str:
    available = _qpp_available(records, variant)
    name = "tuned QPP" if variant == "full" else "lightweight QPP"
    stem = "tuned_qpp" if variant == "full" else "tuned_qpp_lightweight"
    caption = (
        f"{CAPTION_PREFIX}: candidate-size calibration for the {name} predictor on the {language_name} test split, "
        "grouped by the true smallest sufficient candidate size. All columns describe the same $N$ queries in "
        "each row: Exact Match is the fraction predicted (snapped up) into this same class, Median Pred. is "
        "their median raw predicted size, and Coverage is the fraction whose predicted size was $\\geq$ actual."
    )
    lines = _header(caption, f"tab:{stem}_candidate_size_calibration_test_{language}", "l|" + "|".join(["cccc"] * len(available)))
    lines.append(" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{RETRIEVER_DISPLAY[r]}}}" for r in available) + " \\\\")
    lines.append("Actual Class & " + " & ".join("$N$ & Exact Match & Median Pred. & Coverage" for _ in available) + " \\\\")
    lines.append("\\midrule")
    calibrations = {r: records[(variant, "tuned_qpp", r, None)]["test_qpp_details"]["calibration"] for r in available}
    for row_index in range(len(calibrations[available[0]])):
        cells = [calibrations[available[0]][row_index]["class"]]
        for retriever in available:
            row = calibrations[retriever][row_index]
            cells += [str(row["n"]), _number(row["exact_match"], 4), _number(row["median"], 1), _number(row["coverage"], 4)]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def _tuned_qpp_breakdown_table(records: dict, variant: str, language: str, language_name: str) -> str:
    available = _qpp_available(records, variant)
    name = "tuned QPP" if variant == "full" else "lightweight QPP"
    stem = "tuned_qpp" if variant == "full" else "tuned_qpp_lightweight"
    caption = (
        f"{CAPTION_PREFIX}: candidate-size class breakdown for the {name} predictor on the {language_name} test split: "
        "predicted (snapped up to the nearest evaluated candidate size) vs. actual (smallest sufficient) "
        "candidate size, with per-class F1."
    )
    lines = _header(caption, f"tab:{stem}_candidate_size_breakdown_test_{language}", "l" + "|ccc" * len(available), scriptsize=False)
    lines.append(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{RETRIEVER_DISPLAY[r]}}}" for r in available) + " \\\\")
    lines.append("Candidate Size & " + " & ".join("Predicted & Actual & F1" for _ in available) + " \\\\")
    lines.append("\\midrule")
    breakdowns = {r: records[(variant, "tuned_qpp", r, None)]["test_qpp_details"]["breakdown"] for r in available}
    for row_index in range(len(breakdowns[available[0]])):
        cells = [breakdowns[available[0]][row_index]["class"]]
        for retriever in available:
            row = breakdowns[retriever][row_index]
            cells += [str(row["predicted"]), str(row["actual"]), _number(row["f1"], 4)]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + _footer())


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir
    table_dir = runtime.config.latex_table_dir
    language_name = LANGUAGE_DISPLAY.get(language, language)
    metrics_path = store_root / REPORT_SUBDIR / f"test_metrics_{language}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run scripts/evaluate_frozen_predictors_test.py first.")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    baselines = _load_baselines(store_root, language)
    records = {(record["variant"], record["family"], record["retriever"], record["k"]): record for record in payload["records"]}
    counts = {variant: sum(1 for key in records if key[0] == variant) for variant in ("full", "lightweight")}
    print(f"Read {len(records)} frozen-predictor test record(s) (full {counts['full']}, lightweight {counts['lightweight']}) from {metrics_path.relative_to(ROOT)} (generated {payload['generated_at']})")

    _write(table_dir, "hit_at_1", f"table_hit_at_1_full_vs_lightweight_test_performance_{language}.tex", _hit_at_1_table(records, baselines, store_root, language, language_name))
    _write(table_dir, "in_top_max_depth", f"table_in_top_max_depth_full_vs_lightweight_test_performance_{language}.tex", _in_top_max_depth_table(records, baselines, store_root, language, language_name))
    _write(table_dir, "in_top_k_sweep", f"table_in_top_k_sweep_full_vs_lightweight_test_performance_{language}.tex", _in_top_k_table(records, baselines, store_root, language, language_name))
    _write(table_dir, "calibrated_ordinal", f"table_calibrated_ordinal_lightweight_test_performance_{language}.tex", _calibrated_ordinal_table(records, store_root, language, language_name))
    _write(table_dir, "tuned_qpp", f"table_tuned_qpp_lightweight_test_performance_{language}.tex", _tuned_qpp_table(records, store_root, language, language_name))
    for variant, stem in (("full", "tuned_qpp"), ("lightweight", "tuned_qpp_lightweight")):
        if not _qpp_available(records, variant):
            print(f"Skipping {variant} QPP calibration and breakdown tables: no evaluated {variant} QPP records")
            continue
        _write(table_dir, "tuned_qpp", f"table_{stem}_candidate_size_calibration_test_{language}.tex", _tuned_qpp_calibration_table(records, variant, language, language_name))
        _write(table_dir, "tuned_qpp", f"table_{stem}_candidate_size_breakdown_test_{language}.tex", _tuned_qpp_breakdown_table(records, variant, language, language_name))


if __name__ == "__main__":
    typer.run(main)
