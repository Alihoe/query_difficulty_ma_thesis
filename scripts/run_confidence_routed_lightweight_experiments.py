from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
)
from query_difficulty.modeling import _predictor_completed_retrievers
from query_difficulty.modeling_hit_at_1_lightweight import load_or_create_lightweight_hit_at_1_test_predictions
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, split_retriever_name
from query_difficulty.runtime import build_runtime

LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME = "lightweight/hit_at_1"
LIGHTWEIGHT_MANIFEST_PATH = "config/lightweight_hit_at_1_features.json"
MRR_CUTOFF = 5
EXCLUDED_RETRIEVERS = ["qwen3_embedding_0_6b_finetuned"]


def _reciprocal_rank_at_cutoff(gold_rank: pd.Series, cutoff: int) -> np.ndarray:
    ranks = gold_rank.to_numpy(dtype=np.float64)
    within_cutoff = np.isfinite(ranks) & (ranks <= cutoff)
    return np.where(within_cutoff, 1.0 / ranks, 0.0)


def _route(confidence_frame: pd.DataFrame, mrr_frame: pd.DataFrame, hit_frame: pd.DataFrame, columns: list[str]) -> dict:
    confidence_matrix = confidence_frame[columns].to_numpy()
    mrr_matrix = mrr_frame[columns].to_numpy()
    hit_matrix = hit_frame[columns].to_numpy()
    selected_index = confidence_matrix.argmax(axis=1)
    routed_mrr = mrr_matrix[np.arange(len(selected_index)), selected_index]
    routed_hit = hit_matrix[np.arange(len(selected_index)), selected_index]
    oracle_mrr = mrr_matrix.max(axis=1)
    oracle_hit = hit_matrix.any(axis=1)
    return {
        "candidates": list(columns),
        "num_candidates": len(columns),
        "routed_mrr_at_5": float(routed_mrr.mean()),
        "oracle_mrr_at_5": float(oracle_mrr.mean()),
        "routed_hit_at_1": float(routed_hit.mean()),
        "oracle_hit_at_1": float(oracle_hit.mean()),
    }


def _enumerate_all_subsets(
    confidence_frame: pd.DataFrame, mrr_frame: pd.DataFrame, candidates: list[str], console
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(candidates)
    confidence_matrix = confidence_frame[candidates].to_numpy()
    mrr_matrix = mrr_frame[candidates].to_numpy()
    num_subsets = 1 << n
    subset_routed = np.zeros(num_subsets, dtype=np.float64)
    subset_oracle = np.zeros(num_subsets, dtype=np.float64)
    subset_size = np.zeros(num_subsets, dtype=np.int64)
    console.log(f"Enumerating all {num_subsets} subsets of {n} candidates (routed and oracle MRR@5 for each)")
    for mask in tqdm(range(1, num_subsets), desc="subset enumeration", unit="subset"):
        columns = [i for i in range(n) if mask & (1 << i)]
        sub_confidence = confidence_matrix[:, columns]
        sub_mrr = mrr_matrix[:, columns]
        selected = sub_confidence.argmax(axis=1)
        subset_routed[mask] = sub_mrr[np.arange(sub_mrr.shape[0]), selected].mean()
        subset_oracle[mask] = sub_mrr.max(axis=1).mean()
        subset_size[mask] = len(columns)
    return subset_routed, subset_oracle, subset_size


def _shapley_values_from_subsets(
    subset_routed: np.ndarray, candidates: list[str]
) -> dict[str, dict[str, float]]:
    n = len(candidates)
    num_subsets = 1 << n
    weight_by_size = [1.0 / (n * math.comb(n - 1, size)) for size in range(n)] + [0.0]
    shapley = np.zeros(n, dtype=np.float64)
    min_contribution = np.full(n, np.inf, dtype=np.float64)
    max_contribution = np.full(n, -np.inf, dtype=np.float64)
    for mask in range(num_subsets):
        size = bin(mask).count("1")
        weight = weight_by_size[size]
        value_without = subset_routed[mask]
        for i in range(n):
            bit = 1 << i
            if mask & bit:
                continue
            contribution = subset_routed[mask | bit] - value_without
            shapley[i] += weight * contribution
            if contribution < min_contribution[i]:
                min_contribution[i] = contribution
            if contribution > max_contribution[i]:
                max_contribution[i] = contribution

    return {
        candidates[i]: {
            "shapley_value": float(shapley[i]),
            "min_marginal_contribution": float(min_contribution[i]),
            "max_marginal_contribution": float(max_contribution[i]),
        }
        for i in range(n)
    }


def _best_configuration_per_size(
    subset_routed: np.ndarray, subset_oracle: np.ndarray, subset_size: np.ndarray, candidates: list[str]
) -> tuple[list[dict], dict]:
    n = len(candidates)

    def _members(mask: int) -> list[str]:
        return [candidates[i] for i in range(n) if mask & (1 << i)]

    by_size = []
    for size in range(1, n + 1):
        indices = np.where(subset_size == size)[0]
        if len(indices) == 0:
            continue
        best_index = int(indices[np.argmax(subset_routed[indices])])
        by_size.append(
            {
                "size": size,
                "routed_mrr_at_5": float(subset_routed[best_index]),
                "oracle_mrr_at_5": float(subset_oracle[best_index]),
                "candidates": _members(best_index),
            }
        )
    global_best_index = int(np.argmax(subset_routed))
    global_best = {
        "size": int(subset_size[global_best_index]),
        "routed_mrr_at_5": float(subset_routed[global_best_index]),
        "oracle_mrr_at_5": float(subset_oracle[global_best_index]),
        "candidates": _members(global_best_index),
    }
    return by_size, global_best


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Confidence-routed MRR@5 experiments (lightweight only) - test split")
    runtime.console.log(
        "Auto-discovers every retriever/scope with a current lightweight hit@1 model, no fixed "
        "retriever list, so this picks up new retrievers/scopes automatically as they land. Retrievers "
        "whose feature manifest has collapsed to <=1 qualifying feature (gtr_t5_xl, multilingual_e5_base) "
        "are included but flagged: confidence_routed_retrieval_experiments.md found them badly overconfident "
        "when selected in the dev-split analysis; Experiment 5 (Shapley values) below measures whether that "
        "holds up on the test split and across every pool constellation, not just the full pool."
    )
    runtime.console.log(
        "Routing decision is still argmax over each retriever's hit@1 classifier confidence (unchanged); "
        f"the reported/optimized metric is MRR@{MRR_CUTOFF} (reciprocal rank of the gold document, 0 if "
        f"beyond rank {MRR_CUTOFF} or not found), not hit@1 accuracy. Hit@1 numbers are still reported "
        "alongside for reference."
    )

    lightweight_features = LightweightHitAtOnePredictorFeatureBuilder(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features.bundle,
        runtime.retrieval,
        runtime.clustering,
        runtime.text_services,
        manifest_path=LIGHTWEIGHT_MANIFEST_PATH,
        consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
    )

    discovered = sorted(
        _predictor_completed_retrievers(runtime.store, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language)
    )
    runtime.console.log(
        f"Discovered {len(discovered)} retriever/scope(s) with lightweight predictions: {discovered}"
    )

    candidates: list[str] = []
    for name in discovered:
        if name in EXCLUDED_RETRIEVERS:
            runtime.console.log(f"Excluding {name} from the routing pool (explicitly excluded)")
            continue
        document_scope = split_retriever_name(name)[1]
        if document_scope != DEFAULT_DOCUMENT_SCOPE:
            runtime.console.log(f"Excluding {name} from the routing pool (field-specific document scope '{document_scope}')")
            continue
        feature_count = len(lightweight_features.qualifying_features(name))
        if feature_count <= 1:
            runtime.console.log(f"Including {name} despite only {feature_count} qualifying feature(s) (flagged, not excluded)")
        candidates.append(name)
    if not candidates:
        raise RuntimeError("No retrievers with a current, non-degenerate lightweight hit@1 model were found.")
    runtime.console.log(f"Routing pool: {len(candidates)} candidate(s): {candidates}")

    per_retriever_frames: dict[str, pd.DataFrame] = {}
    for name in candidates:
        runtime.console.section(f"Scoring lightweight test split for {name}")
        output = load_or_create_lightweight_hit_at_1_test_predictions(
            config=runtime.config,
            store=runtime.store,
            console=runtime.console,
            features=lightweight_features,
            retrieval=runtime.retrieval,
            language=language,
            retriever_name=name,
            predictor_name=LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
        )
        output = output.set_index("query_id")
        mrr5 = _reciprocal_rank_at_cutoff(output["gold_rank"], MRR_CUTOFF)
        hit_rate = float(output["hit_at_1"].astype(bool).mean())
        mrr5_mean = float(mrr5.mean())
        runtime.console.log(
            f"{name}: test hit@1 = {hit_rate:.4f}, test MRR@{MRR_CUTOFF} = {mrr5_mean:.4f} over {len(output)} queries"
        )
        per_retriever_frames[name] = pd.DataFrame(
            {"confidence": output["confidence"], "hit_at_1": output["hit_at_1"], "mrr5": mrr5}, index=output.index
        )

    query_id_sets = {name: set(frame.index) for name, frame in per_retriever_frames.items()}
    common_query_ids = sorted(set.intersection(*query_id_sets.values()))
    for name, ids in query_id_sets.items():
        missing = len(ids) - len(common_query_ids)
        if missing:
            runtime.console.log(f"Warning: {name} is missing {missing} of the common test queries")
    runtime.console.log(f"{len(common_query_ids)} queries common to all {len(candidates)} candidates")

    confidence_frame = pd.DataFrame(
        {name: per_retriever_frames[name].loc[common_query_ids, "confidence"] for name in candidates}
    )
    mrr_frame = pd.DataFrame(
        {name: per_retriever_frames[name].loc[common_query_ids, "mrr5"] for name in candidates}
    )
    hit_frame = pd.DataFrame(
        {name: per_retriever_frames[name].loc[common_query_ids, "hit_at_1"] for name in candidates}
    )
    single_retriever_mrr5 = {name: float(mrr_frame[name].mean()) for name in candidates}
    single_retriever_hit_rates = {name: float(hit_frame[name].mean()) for name in candidates}

    runtime.console.section("Experiment 1: full pool vs. best single retriever")
    full_pool = _route(confidence_frame, mrr_frame, hit_frame, candidates)
    best_single = max(single_retriever_mrr5, key=single_retriever_mrr5.get)
    runtime.console.log(
        f"Full pool ({len(candidates)} candidates): routed MRR@{MRR_CUTOFF}={full_pool['routed_mrr_at_5']:.4f}, "
        f"oracle MRR@{MRR_CUTOFF}={full_pool['oracle_mrr_at_5']:.4f}, best single ({best_single})="
        f"{single_retriever_mrr5[best_single]:.4f}"
    )

    runtime.console.section("Experiment 2: incremental pool, best individual retriever added first")
    ranked_by_quality = sorted(candidates, key=lambda name: single_retriever_mrr5[name], reverse=True)
    incremental_by_quality = []
    for k in range(1, len(ranked_by_quality) + 1):
        subset = ranked_by_quality[:k]
        result = _route(confidence_frame, mrr_frame, hit_frame, subset)
        result["added"] = subset[-1]
        incremental_by_quality.append(result)
        runtime.console.log(
            f"  k={k:2d} (+{subset[-1]}): routed={result['routed_mrr_at_5']:.4f} oracle={result['oracle_mrr_at_5']:.4f}"
        )

    runtime.console.section("Experiment 3: greedy pool, retriever with the largest oracle gain added first")
    remaining = list(candidates)
    chosen: list[str] = []
    greedy_by_oracle_gain = []
    while remaining:
        best_addition = None
        best_result = None
        for name in remaining:
            trial_result = _route(confidence_frame, mrr_frame, hit_frame, chosen + [name])
            if best_result is None or trial_result["oracle_mrr_at_5"] > best_result["oracle_mrr_at_5"]:
                best_addition = name
                best_result = trial_result
        chosen.append(best_addition)
        remaining.remove(best_addition)
        best_result["added"] = best_addition
        greedy_by_oracle_gain.append(best_result)
        runtime.console.log(
            f"  k={len(chosen):2d} (+{best_addition}): routed={best_result['routed_mrr_at_5']:.4f} "
            f"oracle={best_result['oracle_mrr_at_5']:.4f}"
        )

    runtime.console.section("Experiment 4: leave-one-out from the full pool")
    leave_one_out = []
    for name in candidates:
        subset = [c for c in candidates if c != name]
        result = _route(confidence_frame, mrr_frame, hit_frame, subset)
        result["removed"] = name
        result["routed_mrr_at_5_delta"] = result["routed_mrr_at_5"] - full_pool["routed_mrr_at_5"]
        leave_one_out.append(result)
    leave_one_out.sort(key=lambda entry: -entry["routed_mrr_at_5_delta"])
    for entry in leave_one_out:
        runtime.console.log(
            f"  without {entry['removed']}: routed={entry['routed_mrr_at_5']:.4f} "
            f"(delta {entry['routed_mrr_at_5_delta']:+.4f})"
        )

    runtime.console.section(f"Enumerating subsets ({len(candidates)} candidates, {1 << len(candidates)} subsets) for Experiments 5-6")
    subset_routed, subset_oracle, subset_size = _enumerate_all_subsets(confidence_frame, mrr_frame, candidates, runtime.console)

    runtime.console.section("Experiment 5: exact Shapley values")
    runtime.console.log(
        "Mean marginal contribution to routed MRR@5 averaged across every possible pool constellation "
        "(exact Shapley value, Shapley 1953), plus the worst (min) and best (max) constellation-specific "
        "contribution seen for each candidate. A retriever whose shapley_value and max_marginal_contribution "
        "are both near zero or negative brings no gain in any constellation, not just the one full pool."
    )
    shapley_values = _shapley_values_from_subsets(subset_routed, candidates)
    for name in sorted(shapley_values, key=lambda n: shapley_values[n]["shapley_value"], reverse=True):
        entry = shapley_values[name]
        runtime.console.log(
            f"  {name}: shapley={entry['shapley_value']:+.5f}, "
            f"min={entry['min_marginal_contribution']:+.5f}, max={entry['max_marginal_contribution']:+.5f}"
        )

    runtime.console.section("Experiment 6: best configuration per pool size (exhaustive search)")
    runtime.console.log(
        "Best achievable routed MRR@5 at each pool size, found by exhaustive search over all subsets, "
        "with the oracle MRR@5 and exact membership of that same configuration."
    )
    best_configuration_by_size, global_best_configuration = _best_configuration_per_size(
        subset_routed, subset_oracle, subset_size, candidates
    )
    for entry in best_configuration_by_size:
        runtime.console.log(
            f"  size={entry['size']:2d}: routed={entry['routed_mrr_at_5']:.4f} "
            f"oracle={entry['oracle_mrr_at_5']:.4f} {entry['candidates']}"
        )
    runtime.console.log(
        f"Global best: size={global_best_configuration['size']}, "
        f"routed={global_best_configuration['routed_mrr_at_5']:.4f}, "
        f"oracle={global_best_configuration['oracle_mrr_at_5']:.4f}, "
        f"members={global_best_configuration['candidates']}"
    )

    summary = {
        "language": language,
        "num_queries": len(common_query_ids),
        "mrr_cutoff": MRR_CUTOFF,
        "candidates": candidates,
        "single_retriever_mrr_at_5": single_retriever_mrr5,
        "single_retriever_hit_at_1": single_retriever_hit_rates,
        "full_pool": full_pool,
        "incremental_by_quality": incremental_by_quality,
        "greedy_by_oracle_gain": greedy_by_oracle_gain,
        "leave_one_out": leave_one_out,
        "shapley_values": shapley_values,
        "best_configuration_by_size": best_configuration_by_size,
        "global_best_configuration": global_best_configuration,
    }
    summary_path = runtime.store.path(
        "predictions", "lightweight", "hit_at_1", language, "confidence_routed_experiments", "summary.json"
    )
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved experiment summary to {summary_path}")

    runtime.console.section("Confidence-routed lightweight experiments summary")
    runtime.console.log(
        f"Full pool ({len(candidates)} candidates): routed MRR@{MRR_CUTOFF}={full_pool['routed_mrr_at_5']:.4f}, "
        f"oracle MRR@{MRR_CUTOFF}={full_pool['oracle_mrr_at_5']:.4f}"
    )
    runtime.console.log(f"Best single retriever: {best_single} = {single_retriever_mrr5[best_single]:.4f}")
    most_harmful = leave_one_out[0]
    if most_harmful["routed_mrr_at_5_delta"] > 0:
        runtime.console.log(
            f"Most harmful candidate: {most_harmful['removed']} (removing it improves routed MRR@{MRR_CUTOFF} by "
            f"{most_harmful['routed_mrr_at_5_delta']:+.4f})"
        )
    else:
        runtime.console.log(f"No candidate's removal improves routed MRR@{MRR_CUTOFF}: every candidate contributes positively")


if __name__ == "__main__":
    typer.run(main)
