from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from query_difficulty.runtime import build_runtime
from table_multistage_reranking_full_comparison import (
    POLICY_DISPLAY_NAMES,
    POLICY_ORDER,
    RERANKERS,
    RETRIEVER_DISPLAY_NAMES,
    TOP_K_VALUES,
    _summary_path,
)

PAIRS = [
    ("bm25", "minilm"),
    ("qwen3_embedding_0_6b", "minilm"),
    ("bm25", "jina"),
    ("qwen3_embedding_0_6b", "jina"),
    ("bm25", "bge"),
    ("qwen3_embedding_0_6b", "bge"),
]
ADAPTIVE_POLICIES = [policy for policy in POLICY_ORDER if policy not in {"no_rerank", "rerank_all"}]
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
CHUNK_SIZE = 500
TOLERANCE = 1e-9


def _load_run(store_root: Path, language: str, retriever: str, reranker: str, top_k: int) -> dict | None:
    summary_path = _summary_path(store_root, language, retriever, reranker, top_k)
    if summary_path is None:
        return None
    outcomes_path = summary_path.parent / "test_query_outcomes.parquet"
    if not outcomes_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["test"]
    outcomes = pd.read_parquet(outcomes_path).sort_values("query_id").reset_index(drop=True)
    return {"summary": summary, "outcomes": outcomes, "path": outcomes_path}


def _verified_policy(run: dict, policy: str, top_k: int) -> dict | None:
    outcomes = run["outcomes"]
    summary = run["summary"]
    entry = summary.get(policy)
    if entry is None or f"rr_{policy}" not in outcomes or f"depth_{policy}" not in outcomes:
        return None
    rr = outcomes[f"rr_{policy}"].to_numpy(dtype=np.float64)
    depth = outcomes[f"depth_{policy}"].to_numpy(dtype=np.int64)
    mrr = float(rr.mean())
    calls = int(depth.sum())
    reranked = int((depth > 0).sum())
    if abs(mrr - float(entry["mrr_at_5"])) > TOLERANCE:
        raise ValueError(f"{run['path']}: {policy} MRR@5 {mrr:.6f} != summary {entry['mrr_at_5']:.6f}")
    if calls != int(entry["reranked_pairs"]):
        raise ValueError(f"{run['path']}: {policy} calls {calls} != summary {entry['reranked_pairs']}")
    if reranked != int(entry["reranked_queries"]):
        raise ValueError(f"{run['path']}: {policy} reranked {reranked} != summary {entry['reranked_queries']}")
    return {"policy": policy, "top_k": top_k, "rr": rr, "mrr": mrr, "calls": calls, "reranked": reranked}


def _verified_rerank_all(run: dict, top_k: int) -> dict:
    outcomes = run["outcomes"]
    entry = run["summary"][f"rerank_all_top{top_k}"]
    rr = outcomes["rr_rerank_all"].to_numpy(dtype=np.float64)
    mrr = float(rr.mean())
    calls = int(len(outcomes) * top_k)
    if abs(mrr - float(entry["mrr_at_5"])) > TOLERANCE:
        raise ValueError(f"{run['path']}: rerank all MRR@5 {mrr:.6f} != summary {entry['mrr_at_5']:.6f}")
    if calls != int(entry["reranked_pairs"]):
        raise ValueError(f"{run['path']}: rerank all calls {calls} != summary {entry['reranked_pairs']}")
    return {"rr": rr, "mrr": mrr, "calls": calls}


def _paired_bootstrap_ci(adaptive_rr: np.ndarray, baseline_rr: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    query_count = len(adaptive_rr)
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, query_count, size=(stop - start, query_count))
        deltas[start:stop] = adaptive_rr[indices].mean(axis=1) - baseline_rr[indices].mean(axis=1)
        print(f"    resamples {stop:,}/{BOOTSTRAP_SAMPLES:,}")
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir
    seed = int(runtime.config.runtime.random_seed)
    reranker_labels = {reranker["key"]: reranker["label"] for reranker in RERANKERS}
    print(
        f"Best adaptive configuration vs rerank all: B={BOOTSTRAP_SAMPLES:,}, "
        f"{CONFIDENCE_LEVEL:.0%} percentile CI, seed={seed}"
    )

    rows = []
    missing = []
    for retriever, reranker in PAIRS:
        print(f"\n{retriever} + {reranker}")
        runs = {}
        for top_k in TOP_K_VALUES:
            run = _load_run(store_root, language, retriever, reranker, top_k)
            if run is None:
                print(f"  k={top_k}: no policy comparison outcomes found")
                continue
            runs[top_k] = run
            print(f"  k={top_k}: {len(run['outcomes']):,} queries from {run['path']}")
        candidates = []
        for top_k, run in runs.items():
            for policy in ADAPTIVE_POLICIES:
                candidate = _verified_policy(run, policy, top_k)
                if candidate is None:
                    print(f"  k={top_k}: {policy} missing in cached outcomes, skipped")
                    continue
                candidates.append(candidate)
        if not candidates:
            missing.append(f"{retriever}/{reranker}")
            rows.append(
                f"{RETRIEVER_DISPLAY_NAMES[retriever]} & {reranker_labels[reranker]} & "
                + " & ".join(["n/a"] * 10)
                + " \\\\"
            )
            continue
        ranked = sorted(candidates, key=lambda candidate: (candidate["mrr"], -candidate["calls"]), reverse=True)
        best = ranked[0]
        summary_grid = [
            (float(run["summary"][policy]["mrr_at_5"]), -int(run["summary"][policy]["reranked_pairs"]), policy, grid_k)
            for grid_k, run in runs.items()
            for policy in ADAPTIVE_POLICIES
            if policy in run["summary"]
        ]
        _, _, summary_policy, summary_k = max(summary_grid)
        if (summary_policy, summary_k) != (best["policy"], best["top_k"]):
            raise ValueError(
                f"{retriever}/{reranker}: outcome-based selection {best['policy']}@{best['top_k']} differs from "
                f"summary-based selection {summary_policy}@{summary_k}"
            )
        print(f"  evaluated {len(candidates)} adaptive policy/k configurations, highest observed test MRR@5:")
        for rank, candidate in enumerate(ranked[:5], start=1):
            print(
                f"    {rank}. {candidate['policy']} k={candidate['top_k']}: MRR@5={candidate['mrr']:.6f}, "
                f"calls={candidate['calls']:,}"
            )
        top_k = best["top_k"]
        baseline = _verified_rerank_all(runs[top_k], top_k)
        query_count = len(best["rr"])
        print(
            f"  best adaptive: {best['policy']} at k={top_k}, MRR@5={best['mrr']:.4f}, "
            f"rerank all MRR@5={baseline['mrr']:.4f}, calls {best['calls']:,} vs {baseline['calls']:,}"
        )
        lower, upper = _paired_bootstrap_ci(best["rr"], baseline["rr"], seed)
        delta = best["mrr"] - baseline["mrr"]
        saved = 100.0 * (1.0 - best["calls"] / baseline["calls"])
        print(f"  delta={delta:+.4f} CI=[{lower:+.4f}, {upper:+.4f}], calls saved={saved:.1f}%")
        cells = [
            RETRIEVER_DISPLAY_NAMES[retriever],
            reranker_labels[reranker],
            POLICY_DISPLAY_NAMES[best["policy"]],
            str(top_k),
            f"{100.0 * best['reranked'] / query_count:.1f}\\%",
            f"{baseline['mrr']:.4f}",
            f"{best['mrr']:.4f}",
            f"{delta:+.4f}",
            f"[{lower:+.4f}, {upper:+.4f}]",
            f"{baseline['calls']:,}",
            f"{best['calls']:,}",
            f"{saved:.1f}\\%",
        ]
        rows.append(" & ".join(cells) + " \\\\")

    confidence_label = f"{CONFIDENCE_LEVEL:.0%}".replace("%", "\\%")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Post-hoc paired bootstrap analysis of the best observed adaptive reranking configurations on the English test set.}",
        f"\\label{{tab:multistage_reranking_best_adaptive_bootstrap_{language}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllrr|ccc|c|rrr}",
        "\\toprule",
        " & & & & & \\multicolumn{3}{c|}{MRR@5} & & \\multicolumn{3}{c}{Reranker calls} \\\\",
        f"Retriever & Reranker & Policy & $k$ & \\% reranked & Rerank all & Adaptive & $\\Delta$ & {confidence_label} CI & Rerank all & Adaptive & Saved \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\par\\smallskip",
        "{\\footnotesize Post-hoc analysis. For each retriever and reranker pair, the adaptive configuration is the "
        "adaptive policy and depth $k$ with the highest observed test MRR@5, selected on the test grid itself. "
        "The baseline reranks the full top-$k$ at the same depth.}",
        "\\end{table}",
    ]
    output_path = (
        runtime.config.latex_table_dir
        / "multistage_reranking"
        / f"table_multistage_reranking_best_adaptive_bootstrap_{language}.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(f"\n{output_path}")
    if missing:
        print("Pairs without policy comparison outcomes (shown as n/a): " + ", ".join(missing))


if __name__ == "__main__":
    typer.run(main)
