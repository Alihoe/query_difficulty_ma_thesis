# scripts/evaluate_frozen_predictors_test.py
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "latex" / "scripts"))

from table_tuned_qpp_candidate_size_breakdown import _snap_selected_size_to_grid
from table_tuned_qpp_candidate_size_calibration import _calibration_rows

from query_difficulty.analysis import (
    _candidate_size_classes,
    hit_at_1_predictor_metric_summary,
    in_top_max_depth_predictor_metric_summary,
    ordinal_predictor_metric_summary,
)
from query_difficulty.modeling import (
    _calibrated_ordinal_model_signature,
    _prediction_model_dir,
    _selection_metrics,
    _sha256_path,
)
from query_difficulty.modeling_tuned_qpp_lightweight import _lightweight_tuned_qpp_model_signature
from query_difficulty.runtime import build_runtime

RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]
HIT_AT_1_RETRIEVERS = [
    "bm25",
    "qwen3_embedding_0_6b",
    "bge_m3",
    "bge_large_en_v1_5",
    "e5_large_v2",
    "gtr_t5_xl",
    "multilingual_e5_large",
    "multilingual_e5_base",
]
IN_TOP_K_DEPTHS = [5, 10, 20, 30, 50, 60, 100, 120, 200, 1000]
MAX_DEPTH = 1000
OUTPUT_SUBDIR = Path("reports") / "lightweight" / "frozen_test_evaluation"
HEADLINE_METRICS = {
    "hit_at_1": ["precision", "recall", "f0_5"],
    "in_top_max_depth": ["precision", "recall", "f0_5", "mcc"],
    "in_top_k": ["mcc"],
    "calibrated_ordinal": ["candidate_recall", "exact_class_accuracy", "quadratic_weighted_cohen_kappa", "ordinal_mae"],
    "tuned_qpp": ["candidate_recall", "failure_rate", "mean_selected_size", "median_selected_size"],
}
THRESHOLD_KEYS = {
    "hit_at_1": "threshold",
    "in_top_max_depth": "threshold",
    "in_top_k": "threshold",
    "calibrated_ordinal": "solvability_threshold",
    "tuned_qpp": "failure_threshold",
}


def _jobs() -> list[dict]:
    jobs = []
    for variant, prefix in (("full", ""), ("lightweight", "lightweight/")):
        jobs += [{"variant": variant, "family": "hit_at_1", "predictor_name": f"{prefix}hit_at_1", "retriever": r, "k": None} for r in HIT_AT_1_RETRIEVERS]
        jobs += [{"variant": variant, "family": "in_top_max_depth", "predictor_name": f"{prefix}in_top_max_depth", "retriever": r, "k": MAX_DEPTH} for r in RETRIEVERS]
        jobs += [{"variant": variant, "family": "in_top_k", "predictor_name": f"{prefix}in_top_k{k}", "retriever": r, "k": k} for k in IN_TOP_K_DEPTHS for r in RETRIEVERS]
        jobs += [{"variant": variant, "family": "calibrated_ordinal", "predictor_name": f"{prefix}calibrated_ordinal", "retriever": r, "k": None} for r in RETRIEVERS]
        jobs += [{"variant": variant, "family": "tuned_qpp", "predictor_name": f"{prefix}tuned_qpp", "retriever": r, "k": None} for r in RETRIEVERS]
    return jobs


def _current_signature(store, job: dict) -> str:
    model_dir = _prediction_model_dir(store, job["predictor_name"], job["retriever"])
    if job["family"] == "calibrated_ordinal":
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        return _calibrated_ordinal_model_signature(store, job["retriever"], manifest, predictor_name=job["predictor_name"])
    if job["family"] == "tuned_qpp":
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        return _lightweight_tuned_qpp_model_signature(store, job["predictor_name"], job["retriever"], manifest)
    return _sha256_path(model_dir / "model.joblib")


def _binary_frame(frame: pd.DataFrame, family: str, k: int | None) -> pd.DataFrame:
    if family == "hit_at_1":
        return frame
    predicted_column = "predicted_in_top_k" if "predicted_in_top_k" in frame.columns else "predicted_in_top_max_depth"
    output = frame[["query_id", "gold_rank", "confidence"]].copy()
    output["in_top_max_depth"] = frame["gold_rank"].notna() & (frame["gold_rank"] <= k)
    output["predicted_in_top_max_depth"] = frame[predicted_column].astype(bool)
    return output


def _positive_rate(frame: pd.DataFrame, family: str) -> float:
    if family == "hit_at_1":
        return float(frame["hit_at_1"].astype(bool).mean())
    if family in ("in_top_max_depth", "in_top_k"):
        return float(frame["in_top_max_depth"].astype(bool).mean())
    return float(frame["gold_rank"].notna().mean())


def _mrr_at_5(frame: pd.DataFrame) -> float:
    gold_rank = frame["gold_rank"]
    return float(np.mean(np.where(gold_rank.notna() & (gold_rank <= 5), 1.0 / gold_rank, 0.0)))


def _metrics(frame: pd.DataFrame, family: str, k: int | None, candidate_sizes: list[int]) -> tuple[dict, pd.DataFrame]:
    if family == "hit_at_1":
        prepared = _binary_frame(frame, family, k)
        metrics = hit_at_1_predictor_metric_summary(prepared)
    elif family in ("in_top_max_depth", "in_top_k"):
        prepared = _binary_frame(frame, family, k)
        metrics = in_top_max_depth_predictor_metric_summary(prepared)
    elif family == "calibrated_ordinal":
        prepared = frame
        metrics = ordinal_predictor_metric_summary(frame, candidate_sizes)
    else:
        prepared = frame
        metrics = _selection_metrics(frame)
    metrics = {str(key): float(value) for key, value in metrics.items()}
    metrics["retrieval_mrr_at_5"] = _mrr_at_5(frame)
    return metrics, prepared


def _verify(store, job: dict, test_meta: dict, dev_meta: dict, test_frame: pd.DataFrame) -> str | None:
    if test_meta.get("evaluation_split") != "test":
        return f"test metadata evaluation_split is {test_meta.get('evaluation_split')!r}"
    current = _current_signature(store, job)
    if test_meta.get("model_signature") != current:
        return "test predictions were produced by a different model file than the current frozen model (stale)"
    key = THRESHOLD_KEYS[job["family"]]
    if key not in test_meta or key not in dev_meta:
        return f"threshold key {key!r} missing from dev or test metadata"
    if float(test_meta[key]) != float(dev_meta[key]):
        return f"test {key} {test_meta[key]} differs from dev-selected {key} {dev_meta[key]}"
    if job["family"] in ("hit_at_1", "in_top_max_depth", "in_top_k"):
        predicted_column = {"hit_at_1": "predicted_hit_at_1", "in_top_max_depth": "predicted_in_top_max_depth", "in_top_k": "predicted_in_top_k"}[job["family"]]
        recomputed = test_frame["confidence"].astype(float) >= float(test_meta[key])
        if not bool((recomputed == test_frame[predicted_column].astype(bool)).all()):
            return "stored test labels do not equal confidence >= frozen threshold"
    if test_frame["query_id"].duplicated().any():
        return "duplicate query_id rows in test predictions"
    return None


def _qpp_details(frame: pd.DataFrame, candidate_sizes: list[int]) -> dict:
    ordered_sizes = sorted(candidate_sizes)
    calibration = _calibration_rows(frame, ordered_sizes)
    snapped = _snap_selected_size_to_grid(frame, ordered_sizes)
    actual, predicted = _candidate_size_classes(snapped, ordered_sizes)
    labels = [str(size) for size in ordered_sizes] + ["Unreachable"]
    indices = list(range(len(labels)))
    f1 = f1_score(actual, predicted, labels=indices, average=None, zero_division=0)
    breakdown = [
        {
            "class": labels[index],
            "predicted": int((predicted == index).sum()),
            "actual": int((actual == index).sum()),
            "f1": float(f1[index]),
        }
        for index in indices
    ]
    return {"calibration": calibration, "breakdown": breakdown}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format(value: float) -> str:
    return "nan" if value is None or not math.isfinite(value) else f"{value:.4f}"


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store = runtime.store
    candidate_sizes = [int(size) for size in runtime.config.candidate_sizes]
    output_dir = store.root_dir / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()

    print(f"Test-set evaluation of frozen full and lightweight predictors ({language}_test)")
    print("Read-only: models, features, thresholds and policies are taken as frozen from train/dev.")
    print(f"Candidate sizes: {candidate_sizes}")
    print(f"{len(jobs)} predictor/retriever combination(s)")

    records = []
    coverage = []
    consistency = []
    for index, job in enumerate(jobs, start=1):
        tag = f"{job['predictor_name']} / {job['retriever']}"
        prediction_root = store.root_dir / "predictions" / job["predictor_name"] / language / job["retriever"]
        paths = {name: prediction_root / name for name in ("metadata.json", "predictions.parquet", "test_metadata.json", "test_predictions.parquet")}
        absent = [name for name, path in paths.items() if not path.exists()]
        if absent:
            reason = f"missing {', '.join(absent)}"
            print(f"[{index}/{len(jobs)}] {tag}: NOT EVALUATED ({reason})")
            coverage.append({**job, "status": "not_evaluated", "reason": reason})
            continue
        dev_meta = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
        test_meta = json.loads(paths["test_metadata.json"].read_text(encoding="utf-8"))
        test_frame = pd.read_parquet(paths["test_predictions.parquet"])
        dev_frame = pd.read_parquet(paths["predictions.parquet"])
        problem = _verify(store, job, test_meta, dev_meta, test_frame)
        if problem is not None:
            print(f"[{index}/{len(jobs)}] {tag}: NOT EVALUATED ({problem})")
            coverage.append({**job, "status": "not_evaluated", "reason": problem})
            continue

        test_metrics, test_prepared = _metrics(test_frame, job["family"], job["k"], candidate_sizes)
        dev_metrics, dev_prepared = _metrics(dev_frame, job["family"], job["k"], candidate_sizes)
        if job["family"] == "tuned_qpp":
            stored = dev_meta.get("dev_metrics", {})
            for key in HEADLINE_METRICS["tuned_qpp"]:
                if key in stored and not math.isclose(float(stored[key]), dev_metrics[key], rel_tol=0.0, abs_tol=1e-9):
                    raise RuntimeError(f"{tag}: recomputed dev {key} {dev_metrics[key]} differs from stored {stored[key]}")
        record = {
            **job,
            "split": f"{language}_test",
            "n_queries": int(len(test_frame)),
            "positive_rate": _positive_rate(test_prepared, job["family"]),
            "dev_n_queries": int(len(dev_frame)),
            "dev_positive_rate": _positive_rate(dev_prepared, job["family"]),
            "frozen_threshold": float(test_meta[THRESHOLD_KEYS[job["family"]]]),
            "model_signature": test_meta["model_signature"],
            "test_metrics": test_metrics,
            "dev_metrics": dev_metrics,
        }
        if job["family"] == "tuned_qpp":
            record["test_qpp_details"] = _qpp_details(test_frame, candidate_sizes)
            record["dev_qpp_details"] = _qpp_details(dev_frame, candidate_sizes)
        records.append(record)
        coverage.append({**job, "status": "evaluated", "reason": ""})
        for metric in HEADLINE_METRICS[job["family"]]:
            consistency.append({
                "variant": job["variant"],
                "family": job["family"],
                "retriever": job["retriever"],
                "k": job["k"],
                "metric": metric,
                "dev": dev_metrics[metric],
                "test": test_metrics[metric],
                "test_minus_dev": test_metrics[metric] - dev_metrics[metric],
                "dev_positive_rate": record["dev_positive_rate"],
                "test_positive_rate": record["positive_rate"],
                "dev_n": record["dev_n_queries"],
                "test_n": record["n_queries"],
            })
        summary = ", ".join(f"{metric} dev {_format(dev_metrics[metric])} test {_format(test_metrics[metric])}" for metric in HEADLINE_METRICS[job["family"]])
        print(f"[{index}/{len(jobs)}] {tag}: n={record['n_queries']} | {summary}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Test-set evaluation of frozen full and lightweight predictors. Models, features, thresholds and policies were selected on train/dev and are not changed here.",
        "language": language,
        "split": f"{language}_test",
        "candidate_sizes": candidate_sizes,
        "records": records,
    }
    json_path = output_dir / f"test_metrics_{language}.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    long_rows = []
    for record in records:
        for split_key, metrics_key in (("test", "test_metrics"), ("dev", "dev_metrics")):
            for metric, value in record[metrics_key].items():
                long_rows.append({"variant": record["variant"], "family": record["family"], "retriever": record["retriever"], "k": record["k"], "split": split_key, "metric": metric, "value": value})
    pd.DataFrame(long_rows).to_csv(output_dir / f"test_metrics_{language}.csv", index=False)
    consistency_frame = pd.DataFrame(consistency)
    consistency_frame.to_csv(output_dir / f"dev_vs_test_{language}.csv", index=False)
    coverage_frame = pd.DataFrame(coverage)[["variant", "family", "predictor_name", "retriever", "k", "status", "reason"]]
    coverage_frame.to_csv(output_dir / f"coverage_{language}.csv", index=False)

    lines = [
        f"# Test-set evaluation of frozen predictors ({language}_test)",
        "",
        "Full and lightweight predictors. Models, features, thresholds and policies were fixed on train/dev; nothing was refit or retuned on test.",
        "",
        "## Dev vs test (headline metrics)",
        "",
        "| Variant | Family | Retriever | k | Metric | Dev | Test | Test - Dev | Dev pos. rate | Test pos. rate |",
        "|-|-|-|-|-|-|-|-|-|-|",
    ]
    for row in consistency:
        k_text = "" if row["k"] is None else str(row["k"])
        lines.append(
            f"| {row['variant']} | {row['family']} | {row['retriever']} | {k_text} | {row['metric']} | {_format(row['dev'])} | {_format(row['test'])} | "
            f"{_format(row['test_minus_dev'])} | {_format(row['dev_positive_rate'])} | {_format(row['test_positive_rate'])} |"
        )
    lines += ["", "## Not evaluated", ""]
    not_evaluated = [row for row in coverage if row["status"] != "evaluated"]
    if not_evaluated:
        for row in not_evaluated:
            k_text = "" if row["k"] is None else f" k={row['k']}"
            lines.append(f"- {row['predictor_name']} / {row['retriever']}{k_text}: {row['reason']}")
    else:
        lines.append("- none")
    (output_dir / f"summary_{language}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    print(f"Evaluated {len(records)} of {len(jobs)}; not evaluated {len(not_evaluated)}")
    for row in not_evaluated:
        print(f"  not evaluated: {row['predictor_name']} / {row['retriever']}: {row['reason']}")
    if not consistency_frame.empty:
        print("")
        print("Mean test - dev per family and metric:")
        grouped = consistency_frame.groupby(["variant", "family", "metric"], sort=False)["test_minus_dev"].agg(["mean", "min", "max", "count"])
        print(grouped.to_string(float_format=lambda value: f"{value:.4f}"))
    for name in (f"test_metrics_{language}.json", f"test_metrics_{language}.csv", f"dev_vs_test_{language}.csv", f"coverage_{language}.csv", f"summary_{language}.md"):
        print(f"wrote {(output_dir / name).relative_to(ROOT)}")


if __name__ == "__main__":
    typer.run(main)
