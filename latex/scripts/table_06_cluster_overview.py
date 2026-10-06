from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import cluster_overview_table
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, split_retriever_name
from query_difficulty.runtime import build_runtime

ALGORITHM_ORDER = ["hdbscan", "spherical_kmeans"]
ALGORITHM_LABELS = {"hdbscan": "HDBSCAN", "spherical_kmeans": "Spherical $k$-means"}
EXCLUDED_RETRIEVERS = ["qwen3_embedding_0_6b_finetuned"]


def main(config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file.")) -> None:
    runtime = build_runtime(config, verbose=False)
    frame = cluster_overview_table(runtime.config.artifact_dir)

    retrievers = sorted(
        name
        for name in frame["retriever"].unique()
        if split_retriever_name(name)[1] == DEFAULT_DOCUMENT_SCOPE and name not in EXCLUDED_RETRIEVERS
    )
    rows = []
    for retriever in retrievers:
        retriever_rows = frame[frame["retriever"] == retriever].set_index("algorithm")
        cells = [retriever.replace("_", "\\_")]
        for algorithm in ALGORITHM_ORDER:
            if algorithm not in retriever_rows.index:
                cells += ["--", "--", "--", "--"]
                continue
            row = retriever_rows.loc[algorithm]
            cells += [
                str(int(row["cluster_count"])),
                str(int(row["minimum_cluster_size"])),
                f"{row['mean_cluster_size']:.2f}",
                str(int(row["maximum_cluster_size"])),
            ]
        rows.append(cells)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Cluster overview per semantic retriever and clustering algorithm.}")
    lines.append("\\label{tab:cluster_overview}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l|cccc|cccc}")
    lines.append("\\toprule")
    lines.append(
        " & \\multicolumn{4}{c|}{" + ALGORITHM_LABELS["hdbscan"] + "} & \\multicolumn{4}{c}{"
        + ALGORITHM_LABELS["spherical_kmeans"] + "} \\\\"
    )
    lines.append("Retriever & Clusters & Min & Mean & Max & Clusters & Min & Mean & Max \\\\")
    lines.append("\\midrule")
    for cells in rows:
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "difficulty_analysis" / "table_06_cluster_overview.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
