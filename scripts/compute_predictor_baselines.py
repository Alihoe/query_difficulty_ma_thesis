# scripts/compute_predictor_baselines.py
from __future__ import annotations

from datetime import datetime
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import _binary_predictor_metric_summary
from query_difficulty.modeling import (
    HIT_AT_1_THRESHOLD_POLICY_NAME,
    HitAtOnePredictorRunner,
    InTopMaxDepthPredictorRunner,
)
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
RANKING_DEPTH = 1000
OUTPUT_SUBDIR = Path("reports") / "predictor_baselines"
IN_TOP_K_RUNNER_DEFAULTS = inspect.signature(InTopMaxDepthPredictorRunner.__init__).parameters
IN_TOP_K_OBJECTIVE = IN_TOP_K_RUNNER_DEFAULTS["threshold_objective"].default
IN_TOP_K_BETA = IN_TOP_K_RUNNER_DEFAULTS["threshold_beta"].default


def _split_data(store_root: Path, split: str, retriever: str, with_scores: bool) -> tuple[pd.Series, pd.DataFrame | None]:
    queries = pd.read_parquet(store_root / "queries" / f"{split}.parquet", columns=["query_id", "document_id"])
    if not queries["query_id"].is_unique:
        raise ValueError(f"{split} queries are not unique per query_id")
    ranking = pd.read_parquet(store_root / "rankings" / split / f"{retriever}_top_{RANKING_DEPTH}.parquet")
    gold = queries.merge(ranking[["query_id", "document_id", "rank"]], on=["query_id", "document_id"], how="left")
    gold_rank = gold.set_index("query_id")["rank"].astype(float)
    if not with_scores:
        return gold_rank, None
    scores = ranking.pivot(index="query_id", columns="rank", values="score").reindex(queries["query_id"])
    if scores.shape[1] != RANKING_DEPTH or scores.isna().any().any():
        raise ValueError(f"{split}/{retriever}: expected complete top-{RANKING_DEPTH} scores for every query")
    return gold_rank, scores


def _labels(gold_rank: pd.Series, family: str, k: int | None) -> np.ndarray:
    if family == "hit_at_1":
        return (gold_rank == 1.0).to_numpy(dtype=bool)
    return (gold_rank.notna() & (gold_rank <= k)).to_numpy(dtype=bool)


def _margin(scores: pd.DataFrame, rank: int) -> np.ndarray | None:
    if rank + 1 > RANKING_DEPTH:
        return None
    return (scores[rank] - scores[rank + 1]).to_numpy(dtype=np.float64)


def _summary(actual: np.ndarray, predicted: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"actual": actual, "predicted": predicted, "confidence": confidence})
    return {str(key): float(value) for key, value in _binary_predictor_metric_summary(frame, "actual", "predicted").items()}


def _majority_record(family: str, retriever: str, k: int | None, train_labels: np.ndarray, splits: dict) -> dict:
    train_rate = float(train_labels.mean())
    if train_rate == 0.5:
        raise ValueError(f"{family}/{retriever}/k={k}: train classes are exactly balanced, majority class undefined")
    majority_positive = train_rate > 0.5
    record = {
        "baseline": "majority",
        "family": family,
        "retriever": retriever,
        "k": k,
        "available": True,
        "reason": "",
        "train_positive_rate": train_rate,
        "majority_class": "positive" if majority_positive else "negative",
    }
    for split_key, labels in splits.items():
        constant = np.full(labels.shape, majority_positive, dtype=bool)
        record[f"{split_key}_metrics"] = _summary(labels, constant, constant.astype(np.float64))
        record[f"{split_key}_positive_rate"] = float(labels.mean())
    return record


def _margin_record(family: str, retriever: str, k: int | None, margins: dict, splits: dict) -> dict:
    rank = 1 if family == "hit_at_1" else k
    record = {"baseline": "margin", "family": family, "retriever": retriever, "k": k, "margin_ranks": [rank, rank + 1]}
    if margins["dev"] is None:
        record.update({
            "available": False,
            "reason": f"rank {rank + 1} score is not cached (rankings store the top {RANKING_DEPTH}); computing it would require rerunning retrieval",
        })
        return record
    dev_margin, dev_labels = margins["dev"], splits["dev"]
    if family == "hit_at_1":
        threshold, _ = HitAtOnePredictorRunner._select_threshold(dev_margin, pd.Series(dev_labels))
        direction = 1.0
        objective = HIT_AT_1_THRESHOLD_POLICY_NAME
        dev_objective = None
    else:
        options = []
        for direction_option in (1.0, -1.0):
            option_threshold, option_metrics = InTopMaxDepthPredictorRunner._select_optimal_threshold(
                direction_option * dev_margin, pd.Series(dev_labels), IN_TOP_K_OBJECTIVE, IN_TOP_K_BETA
            )
            options.append((option_metrics["score"], direction_option, option_threshold))
        best_score = max(score for score, _, _ in options)
        _, direction, threshold = next(option for option in options if option[0] == best_score)
        objective = f"maximize_{IN_TOP_K_OBJECTIVE}"
        dev_objective = float(best_score)
    record.update({
        "available": True,
        "reason": "",
        "threshold_split": "dev",
        "objective": objective,
        "direction": "margin >= threshold" if direction > 0 else "margin <= threshold",
        "threshold": float(direction * threshold),
        "dev_objective_at_threshold": dev_objective,
    })
    for split_key, labels in splits.items():
        signed = direction * margins[split_key]
        predicted = signed >= threshold
        record[f"{split_key}_metrics"] = _summary(labels, predicted, signed)
        record[f"{split_key}_positive_rate"] = float(labels.mean())
    return record


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format(value) -> str:
    return "nan" if value is None or not math.isfinite(value) else f"{value:.4f}"


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir
    output_dir = store_root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Predictor baselines for {language} from cached top-{RANKING_DEPTH} rankings (no retrieval is rerun)")
    print("Majority: majority class of the train split. Margin: single retrieval-score margin, threshold tuned on dev.")
    print(f"Hit@1 margin objective: {HIT_AT_1_THRESHOLD_POLICY_NAME}; binary margin objective: {IN_TOP_K_OBJECTIVE} (direction also chosen on dev)")

    retrievers = list(dict.fromkeys(HIT_AT_1_RETRIEVERS + RETRIEVERS))
    records = []
    for retriever in retrievers:
        print(f"Loading rankings for {retriever}")
        train_gold, _ = _split_data(store_root, f"{language}_train", retriever, with_scores=False)
        dev_gold, dev_scores = _split_data(store_root, f"{language}_dev", retriever, with_scores=True)
        test_gold, test_scores = _split_data(store_root, f"{language}_test", retriever, with_scores=True)
        targets = [("hit_at_1", None)] if retriever in HIT_AT_1_RETRIEVERS else []
        if retriever in RETRIEVERS:
            targets += [("in_top_max_depth", MAX_DEPTH)] + [("in_top_k", k) for k in IN_TOP_K_DEPTHS]
        for family, k in targets:
            splits = {"dev": _labels(dev_gold, family, k), "test": _labels(test_gold, family, k)}
            rank = 1 if family == "hit_at_1" else k
            margins = {"dev": _margin(dev_scores, rank), "test": _margin(test_scores, rank)}
            majority = _majority_record(family, retriever, k, _labels(train_gold, family, k), splits)
            margin = _margin_record(family, retriever, k, margins, splits)
            records += [majority, margin]
            key = "f0_5" if family == "hit_at_1" else "mcc"
            k_text = "" if k is None else f" k={k}"
            margin_text = (
                f"margin {key} dev {_format(margin['dev_metrics'][key])} test {_format(margin['test_metrics'][key])} ({margin['direction']}, t={margin['threshold']:.6g})"
                if margin["available"] else f"margin unavailable: {margin['reason']}"
            )
            print(
                f"  {family}{k_text}: majority={majority['majority_class']} (train rate {majority['train_positive_rate']:.4f}) "
                f"{key} dev {_format(majority['dev_metrics'][key])} test {_format(majority['test_metrics'][key])} | {margin_text}"
            )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Majority-class (train) and single-score margin (threshold tuned on dev) baselines from cached rankings.",
        "language": language,
        "ranking_depth": RANKING_DEPTH,
        "records": records,
    }
    json_path = output_dir / f"predictor_baselines_{language}.json"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    rows = []
    for record in records:
        for split_key in ("dev", "test"):
            for metric, value in (record.get(f"{split_key}_metrics") or {}).items():
                rows.append({"baseline": record["baseline"], "family": record["family"], "retriever": record["retriever"], "k": record["k"], "split": split_key, "metric": metric, "value": value})
    csv_path = output_dir / f"predictor_baselines_{language}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    typer.run(main)
