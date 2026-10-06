from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from query_difficulty.multistep import (
    RerankerPolicyComparisonExperiment,
    _combined_selector_depth,
    _condition_name,
    _policy_comparison_combinations,
)
from query_difficulty.runtime import build_runtime
from table_multistage_reranking_full_comparison import (
    POLICY_DISPLAY_NAMES,
    POLICY_ORDER,
    RETRIEVER_DISPLAY_NAMES,
    TOP_K_VALUES,
    _bold_best_mrr,
    _mark_overall_best,
    _percent_reranked_text,
)

RETRIEVER = "qwen3_embedding_0_6b"
METRIC_K = 5


def _selector_frame(runtime, language: str, top_k: int) -> pd.DataFrame:
    loader = RerankerPolicyComparisonExperiment.__new__(RerankerPolicyComparisonExperiment)
    loader.store = runtime.store
    loader.console = runtime.console
    loader.retrieval = runtime.retrieval
    loader.features = runtime.features
    loader.language = language
    loader.retriever_name = runtime.retrieval.resolve_retriever_name(RETRIEVER)
    loader.top_k = int(top_k)
    return loader._load_selector_frame()


def _policy_depths(selector_frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    depths: dict[str, list[int]] = {_condition_name(combo): [] for combo in _policy_comparison_combinations()}
    for row in selector_frame.itertuples(index=False):
        gate_pass = {
            "hit1": bool(row.hit1_gate_pass),
            "max_depth": bool(row.max_depth_gate_pass),
            "ordinal": bool(row.ordinal_gate_pass),
            "qpp": bool(row.qpp_gate_pass),
            "in_top_k": bool(row.in_top_k_gate_pass),
            "in_top_k_depth": bool(row.in_top_k_depth_gate_pass),
        }
        depth = {
            "hit1": None,
            "max_depth": None,
            "ordinal": None if pd.isna(row.ordinal_selected_size) else int(float(row.ordinal_selected_size)),
            "qpp": None if pd.isna(row.qpp_selected_size) else int(float(row.qpp_selected_size)),
            "in_top_k": None,
            "in_top_k_depth": (
                None if pd.isna(row.in_top_k_depth_selected_size) else int(float(row.in_top_k_depth_selected_size))
            ),
        }
        for combo in _policy_comparison_combinations():
            selected, combined_depth = _combined_selector_depth(combo, gate_pass, depth, top_k)
            depths[_condition_name(combo)].append(combined_depth if selected else 0)
    frame = pd.DataFrame(depths)
    frame.insert(0, "query_id", selector_frame["query_id"].astype(int).to_numpy())
    frame["no_rerank"] = 0
    frame["rerank_all"] = int(top_k)
    return frame


def _oracle_metrics(gold_rank: pd.Series, depth_frame: pd.DataFrame, top_k: int) -> dict[str, dict]:
    merged = depth_frame.merge(gold_rank.rename("gold_rank").reset_index(), on="query_id", how="left")
    rank = merged["gold_rank"].to_numpy(dtype=np.float64)
    rr_no_rerank = np.where(rank <= METRIC_K, 1.0 / rank, 0.0)
    metrics: dict[str, dict] = {}
    for policy in POLICY_ORDER:
        depth = merged[policy].to_numpy(dtype=np.float64)
        rr = np.where((depth > 0) & (rank <= depth), 1.0, rr_no_rerank)
        reranked = int((depth > 0).sum())
        metrics[policy] = {
            "mrr_at_5": float(rr.mean()),
            "reranked_queries": reranked,
            "queries_reranked_pct": 100.0 * reranked / len(merged) if len(merged) else 0.0,
        }
    print(f"  top_k={top_k}: " + ", ".join(f"{policy}={metrics[policy]['mrr_at_5']:.4f}" for policy in POLICY_ORDER))
    return metrics


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    retriever_name = runtime.retrieval.resolve_retriever_name(RETRIEVER)
    split_key = f"{language}_test"

    print(f"Loading {split_key} queries and {retriever_name} rankings")
    queries = runtime.retrieval.queries[split_key][["query_id", "document_id"]].astype(int)
    ranking = runtime.retrieval.ensure_rankings(split_key, retriever_name)[["query_id", "rank", "document_id"]].astype(int)
    gold_rank = queries.merge(ranking, on=["query_id", "document_id"], how="left").set_index("query_id")["rank"]
    print(f"  queries={len(queries)}, gold found in ranking={int(gold_rank.notna().sum())}")

    metrics_by_top_k: dict[int, dict[str, dict]] = {}
    for top_k in TOP_K_VALUES:
        print(f"Loading current lightweight selector predictions for top_k={top_k}")
        selector_frame = _selector_frame(runtime, language, top_k)
        print(f"  selector rows={len(selector_frame)}")
        depth_frame = _policy_depths(selector_frame, top_k)
        missing_queries = sorted(set(queries["query_id"]) - set(depth_frame["query_id"]))
        if missing_queries:
            raise KeyError(f"Missing selector predictions for {len(missing_queries)} queries, e.g. {missing_queries[:10]}")
        metrics_by_top_k[top_k] = _oracle_metrics(gold_rank, depth_frame, top_k)

    rows = []
    for policy in POLICY_ORDER:
        display_name = "Perfect rerank all" if policy == "rerank_all" else POLICY_DISPLAY_NAMES[policy]
        row: dict = {"policy": display_name, "percent_reranked": _percent_reranked_text(metrics_by_top_k, policy)}
        for top_k in TOP_K_VALUES:
            entry = metrics_by_top_k[top_k][policy]
            row[top_k] = {
                "mrr_at_5": entry["mrr_at_5"],
                "mrr_text": f"{entry['mrr_at_5']:.4f}",
            }
        rows.append(row)

    marked_rows = [row for policy, row in zip(POLICY_ORDER, rows) if policy != "rerank_all"]
    for top_k in TOP_K_VALUES:
        _bold_best_mrr([row[top_k] for row in marked_rows])
    _mark_overall_best(marked_rows, TOP_K_VALUES)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{MRR@5 of {RETRIEVER_DISPLAY_NAMES[retriever_name]} with perfect reranking of the candidate set "
        "selected by each policy.}"
    )
    lines.append(f"\\label{{tab:multistage_reranking_oracle_comparison_{retriever_name}_{language}}}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    column_spec = "l|c|" + "c" * len(TOP_K_VALUES)
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    top_k_headers = " & ".join(f"$k$={top_k}" for top_k in TOP_K_VALUES)
    recall_cells = " & ".join(f"{float((gold_rank <= top_k).mean()):.4f}" for top_k in TOP_K_VALUES)
    lines.append(f"Policy & \\% reranked & {top_k_headers} \\\\")
    lines.append("\\midrule")
    lines.append(f"{RETRIEVER_DISPLAY_NAMES[retriever_name]} Recall@$k$ & - & {recall_cells} \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = []
        for top_k in TOP_K_VALUES:
            cells.append(row[top_k]["mrr_text"])
        lines.append(f"{row['policy']} & {row['percent_reranked']} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    output_path = (
        runtime.config.latex_table_dir
        / "multistage_reranking"
        / f"table_multistage_reranking_oracle_comparison_{retriever_name}_{language}.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
