from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.runtime import build_runtime


RETRIEVER_NAME = "qwen3_embedding_0_6b"
RETRIEVER_LABEL = "Qwen3-Emb-0.6B"
RECALL_SIZES = [100, 200, 500, 1000, 2000, 5000]
BOLD_SIZE = 1000
RECOVERY_WINDOWS = {1000: 200, 2000: 1000, 5000: 2000}


def gold_ranks(rankings: pd.DataFrame, queries: pd.DataFrame) -> pd.Series:
    gold_by_query = queries.set_index("query_id")["document_id"]
    predicted_gold = rankings["query_id"].map(gold_by_query)
    hits = rankings[rankings["document_id"] == predicted_gold]
    best_rank = hits.groupby("query_id")["rank"].min()
    return gold_by_query.index.to_series().map(best_rank)


def bold(text: str, size: int) -> str:
    return f"\\textbf{{{text}}}" if size == BOLD_SIZE else text


def build_latex(recalls: dict[int, float], recovered: dict[int, int], split_key: str, n_queries: int) -> str:
    header = " & ".join(bold(f"@{size}", size) for size in RECALL_SIZES)
    recall_row = " & ".join(bold(f"{recalls[size]:.3f}", size) for size in RECALL_SIZES)
    recovered_row = " & ".join(
        bold(f"+{recovered[size]}", size) if size in recovered else "--" for size in RECALL_SIZES
    )
    windows = " and ".join(f"{low}--{high}" for high, low in RECOVERY_WINDOWS.items())
    caption = (
        f"{RETRIEVER_LABEL} Recall@$k$ on the English development set ($n={n_queries}$). "
        f"Recovered: additional queries whose gold publication enters the candidate set between ranks {windows}."
    )
    column_spec = "l" + "r" * len(RECALL_SIZES)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:qwen_deep_recall_{split_key}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        f" & {header} \\\\",
        "\\midrule",
        f"Recall & {recall_row} \\\\",
        f"Recovered & {recovered_row} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(
    language: str = typer.Option("en", help="Language code."),
    split: str = typer.Option("dev", help="Query split name."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    split_key = f"{language}_{split}"
    max_depth = max(RECALL_SIZES)
    print(f"Loading {split_key} queries")
    queries = runtime.dataset.load().queries[split_key][["query_id", "document_id"]].astype(int)
    print(f"Loaded {len(queries)} queries")

    query_frame = runtime.retrieval.queries[split_key]
    print(f"Loading {RETRIEVER_NAME} query embeddings for {split_key}")
    query_embeddings = runtime.retrieval.ensure_query_embeddings(split_key, RETRIEVER_NAME)
    print(f"Searching {RETRIEVER_NAME} to depth {max_depth}")
    rankings = runtime.retrieval._rank_semantic_frame(
        query_frame, RETRIEVER_NAME, max_depth, query_embeddings=query_embeddings
    )
    print(f"Retrieved {len(rankings)} ranking rows")

    ranks = gold_ranks(rankings, queries)
    recalls = {size: float((ranks <= size).mean()) for size in RECALL_SIZES}
    recovered = {
        high: int(((ranks > low) & (ranks <= high)).sum()) for high, low in RECOVERY_WINDOWS.items()
    }
    for size in RECALL_SIZES:
        print(f"Recall@{size}: {recalls[size]:.4f} ({int((ranks <= size).sum())}/{len(ranks)})")
    for high, low in RECOVERY_WINDOWS.items():
        print(f"Recovered between ranks {low} and {high}: {recovered[high]}")
    print(f"Gold beyond rank {max_depth}: {int(ranks.isna().sum())}")

    latex = build_latex(recalls, recovered, split_key, len(queries))
    output_path = runtime.config.latex_table_dir / "candidate_size_comparison" / f"table_qwen_deep_recall_{split_key}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
