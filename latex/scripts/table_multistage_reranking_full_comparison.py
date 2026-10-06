from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.multistep import canonical_multistage_reranker_specs
from query_difficulty.runtime import build_runtime

RETRIEVER_DISPLAY_NAMES = {
    "bm25": "BM25",
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
RETRIEVER_ORDER = ["bm25", "qwen3_embedding_0_6b"]
TOP_K_VALUES = [5, 10, 20, 30, 50]

RERANKERS = [
    {
        "key": "bge",
        "label": "bge-reranker-v2-m3",
        "run_script_hint": "run_{retriever_prefix}_multistep_retrieval_bge_top{top_k}.py",
    },
    {
        "key": "jina",
        "label": "jina-reranker-v2-base-multilingual",
        "run_script_hint": "run_{retriever_prefix}_multistep_retrieval_jina_top{top_k}.py",
    },
    {
        "key": "minilm",
        "label": "ms-marco-MiniLM-L-6-v2",
        "run_script_hint": "run_{retriever_prefix}_multistep_retrieval_minilm_top{top_k}.py",
    },
]

POLICY_ORDER = [
    "no_rerank",
    "rerank_all",
    "hit1",
    "max_depth",
    "in_top_k",
    "hit1_and_max_depth",
    "hit1_and_in_top_k",
    "ordinal",
    "qpp",
    "in_top_k_depth",
    "hit1_and_ordinal",
    "hit1_and_qpp",
    "hit1_and_in_top_k_depth",
]
POLICY_DISPLAY_NAMES = {
    "no_rerank": "No rerank",
    "rerank_all": "Rerank all",
    "hit1": "Hit-at-1",
    "max_depth": "Maximum Depth",
    "hit1_and_max_depth": "Hit-at-1 $\\wedge$ Maximum Depth",
    "ordinal": "Ordinal",
    "qpp": "QPP",
    "in_top_k": "Binary (fixed $k$)",
    "in_top_k_depth": "Binary (adaptive $k$)",
    "hit1_and_ordinal": "Hit-at-1 $\\wedge$ Ordinal",
    "hit1_and_qpp": "Hit-at-1 $\\wedge$ QPP",
    "hit1_and_in_top_k": "Hit-at-1 $\\wedge$ Binary (fixed $k$)",
    "hit1_and_in_top_k_depth": "Hit-at-1 $\\wedge$ Binary (adaptive $k$)",
}
REFERENCE_ROWS = {"No rerank", "Rerank all", "Perfect reranking"}
COMBINED_RERANKER_ORDER = ["minilm", "jina", "bge"]
APPENDIX_LABEL = "appendix:reranking-full"


def _summary_path(store_root: Path, language: str, retriever: str, reranker_key: str, top_k: int) -> Path | None:
    spec = canonical_multistage_reranker_specs()[reranker_key]
    path = (
        store_root
        / "multistep"
        / "comparison_lightweight_selectors"
        / language
        / retriever
        / spec.method
        / spec.key
        / f"top_{top_k}"
        / spec.cache_dir_name
        / "summary.json"
    )
    return path if path.exists() else None


def _load_policy_metrics(summary_path: Path, top_k: int) -> tuple[dict[str, dict], list[str]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    test = payload["test"]
    rerank_all_key = f"rerank_all_top{top_k}"
    metrics = dict(test)
    if rerank_all_key in metrics:
        metrics["rerank_all"] = metrics.pop(rerank_all_key)
    evaluated_queries = int(payload["evaluation"]["evaluated_queries"])
    missing_policies: list[str] = []
    for policy in POLICY_ORDER:
        entry = metrics.get(policy)
        if entry is None:
            missing_policies.append(policy)
            continue
        entry["queries_reranked_pct"] = (
            100.0 * entry["reranked_queries"] / evaluated_queries if evaluated_queries else 0.0
        )
    return metrics, missing_policies


def _percent_reranked_text(metrics_by_top_k: dict[int, dict | None], policy: str) -> str:
    values = sorted(
        {
            f"{metrics[policy]['queries_reranked_pct']:.1f}"
            for metrics in metrics_by_top_k.values()
            if metrics is not None and metrics.get(policy) is not None
        },
        key=float,
    )
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]}\\%"
    return f"{values[0]}-{values[-1]}\\%"


def _perfect_reranking_row(runtime, language: str, retriever: str) -> dict:
    split_key = f"{language}_test"
    queries = runtime.retrieval.queries[split_key][["query_id", "document_id"]].astype(int)
    ranking = runtime.retrieval.ensure_rankings(split_key, retriever)[["query_id", "rank", "document_id"]].astype(int)
    gold_rank = queries.merge(ranking, on=["query_id", "document_id"], how="left").set_index("query_id")["rank"]
    row: dict = {"policy": "Perfect reranking", "percent_reranked": "100.0\\%"}
    for top_k in TOP_K_VALUES:
        mrr_at_5 = float((gold_rank <= top_k).mean())
        row[top_k] = {
            "mrr_at_5": mrr_at_5,
            "mrr_text": f"{mrr_at_5:.4f}",
            "calls_text": f"{len(queries) * top_k:,}",
        }
    return row


def _bold_best_mrr(cells: list[dict]) -> None:
    scored = [cell for cell in cells if cell is not None]
    if not scored:
        return
    best = max(cell["mrr_at_5"] for cell in scored)
    for cell in scored:
        if cell["mrr_at_5"] == best:
            cell["mrr_text"] = f"\\textbf{{{cell['mrr_text']}}}"


def _mark_overall_best(rows: list[dict], top_k_values: list[int]) -> None:
    all_cells = [row[top_k] for row in rows for top_k in top_k_values if row[top_k] is not None]
    if not all_cells:
        return
    best = max(cell["mrr_at_5"] for cell in all_cells)
    for row in rows:
        best_cells = [
            row[top_k] for top_k in top_k_values if row[top_k] is not None and row[top_k]["mrr_at_5"] == best
        ]
        for cell in best_cells:
            cell["mrr_text"] = f"\\underline{{{cell['mrr_text']}}}"
        if best_cells:
            row["policy"] = f"\\textbf{{{row['policy']}}}"


def _tabular_header() -> list[str]:
    column_spec = "l|c|" + "|".join("cr" for _ in TOP_K_VALUES)
    top_k_headers = " & ".join(
        f"\\multicolumn{{2}}{{c{'|' if index < len(TOP_K_VALUES) - 1 else ''}}}{{$k$={top_k}}}"
        for index, top_k in enumerate(TOP_K_VALUES)
    )
    metric_headers = " & ".join("MRR@5 & \\#Calls" for _ in TOP_K_VALUES)
    return [
        "\\setlength{\\tabcolsep}{5pt}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        f" & & {top_k_headers} \\\\",
        f"Policy & \\% reranked & {metric_headers} \\\\",
        "\\midrule",
    ]


def _row_line(row: dict) -> str:
    cells = []
    for top_k in TOP_K_VALUES:
        cell = row[top_k]
        cells.extend(["n/a", "n/a"] if cell is None else [cell["mrr_text"], cell["calls_text"]])
    return f"{row['policy']} & {row['percent_reranked']} & " + " & ".join(cells) + " \\\\"


def _has_best_value(row: dict) -> bool:
    return any(row[top_k] is not None and "\\textbf{" in row[top_k]["mrr_text"] for top_k in TOP_K_VALUES)


def _combined_table(rows_by_retriever: dict[str, list[dict]], reranker: dict, language: str) -> str:
    n_columns = 2 + 2 * len(TOP_K_VALUES)
    names = " and ".join(f"\\textit{{{RETRIEVER_DISPLAY_NAMES[r]}}}" for r in RETRIEVER_ORDER)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{names} reranked with \\textit{{{reranker['label']}}}. Besides the reference policies, only "
        f"policies that reach the best MRR@5 for at least one $k$ are shown. The complete tables are given in "
        f"Appendix \\ref{{{APPENDIX_LABEL}}}.}}",
        f"\\label{{tab:multistage_reranking_comparison_{reranker['key']}_{language}}}",
    ]
    lines += _tabular_header()
    for index, retriever in enumerate(RETRIEVER_ORDER):
        if index:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{n_columns}}}{{l}}{{\\textit{{{RETRIEVER_DISPLAY_NAMES[retriever]}}}}} \\\\")
        for row in rows_by_retriever[retriever]:
            if row["policy"] in REFERENCE_ROWS or _has_best_value(row):
                lines.append(_row_line(row))
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir

    missing: list[str] = []
    perfect_rows = {retriever: _perfect_reranking_row(runtime, language, retriever) for retriever in RETRIEVER_ORDER}
    rows_by_reranker: dict[str, dict[str, list[dict]]] = {}

    for retriever in RETRIEVER_ORDER:
        retriever_prefix = "qwen" if retriever == "qwen3_embedding_0_6b" else retriever
        for reranker in RERANKERS:
            metrics_by_top_k: dict[int, dict[str, dict] | None] = {}
            for top_k in TOP_K_VALUES:
                summary_path = _summary_path(store_root, language, retriever, reranker["key"], top_k)
                if summary_path is None:
                    metrics_by_top_k[top_k] = None
                    hint = reranker["run_script_hint"].format(retriever_prefix=retriever_prefix, top_k=top_k)
                    missing.append(f"{retriever}/{reranker['key']}/top_{top_k} -- run {hint} first")
                    continue
                metrics, missing_policies = _load_policy_metrics(summary_path, top_k)
                metrics_by_top_k[top_k] = metrics
                if missing_policies:
                    policy_labels = ", ".join(missing_policies)
                    missing.append(
                        f"{retriever}/{reranker['key']}/top_{top_k} -- cached summary predates these "
                        f"selectors ({policy_labels}); shown as n/a for those rows. Re-run to refresh."
                    )

            rows = []
            for policy in POLICY_ORDER:
                row: dict = {
                    "policy": POLICY_DISPLAY_NAMES[policy],
                    "percent_reranked": _percent_reranked_text(metrics_by_top_k, policy),
                }
                for top_k in TOP_K_VALUES:
                    metrics = metrics_by_top_k[top_k]
                    entry = metrics.get(policy) if metrics is not None else None
                    if entry is None:
                        row[top_k] = None
                        continue
                    row[top_k] = {
                        "mrr_at_5": entry["mrr_at_5"],
                        "mrr_text": f"{entry['mrr_at_5']:.4f}",
                        "calls_text": f"{int(entry['reranked_pairs']):,}",
                    }
                rows.append(row)

            for top_k in TOP_K_VALUES:
                _bold_best_mrr([row[top_k] for row in rows])
            _mark_overall_best(rows, TOP_K_VALUES)
            rows.insert(POLICY_ORDER.index("rerank_all") + 1, perfect_rows[retriever])
            rows_by_reranker.setdefault(reranker["key"], {})[retriever] = rows

            lines = []
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append(
                f"\\caption{{\\textit{{{RETRIEVER_DISPLAY_NAMES[retriever]}}} reranked with "
                f"\\textit{{{reranker['label']}}}.}}"
            )
            lines.append(
                f"\\label{{tab:multistage_reranking_full_comparison_{retriever}_{reranker['key']}_{language}}}"
            )
            lines += _tabular_header()
            lines += [_row_line(row) for row in rows]
            lines.append("\\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("}")
            lines.append("\\end{table}")

            output_path = (
                runtime.config.latex_table_dir
                / "multistage_reranking"
                / f"table_multistage_reranking_full_comparison_{retriever}_{reranker['key']}_{language}.tex"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
            print(f"wrote {output_path} ({len(rows)} rows)")

    reranker_by_key = {reranker["key"]: reranker for reranker in RERANKERS}
    for key in COMBINED_RERANKER_ORDER:
        rows_by_retriever = rows_by_reranker.get(key, {})
        if set(rows_by_retriever) != set(RETRIEVER_ORDER):
            missing.append(f"combined table for {key} skipped -- not all retrievers available")
            continue
        latex = _combined_table(rows_by_retriever, reranker_by_key[key], language)
        output_path = (
            runtime.config.latex_table_dir
            / "multistage_reranking"
            / f"table_multistage_reranking_comparison_{key}_{language}.tex"
        )
        output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
        kept = {r: sum(1 for row in rows_by_retriever[r] if row["policy"] in REFERENCE_ROWS or _has_best_value(row)) for r in RETRIEVER_ORDER}
        print(f"wrote {output_path} (rows kept per retriever: {kept})")

    if missing:
        print("\nMissing combinations (shown as n/a in the tables above):")
        for entry in missing:
            print(f"  - {entry}")


if __name__ == "__main__":
    typer.run(main)
