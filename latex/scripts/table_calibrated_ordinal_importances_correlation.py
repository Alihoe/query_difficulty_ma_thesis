from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import pandas as pd
import typer
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _ordinal_importance_tables import mean_importance_across_depths, ordinal_importances_by_depth

from query_difficulty.runtime import build_runtime

RETRIEVER_DISPLAY_NAMES = {
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
    "bge_m3": "BGE-M3",
}

RETRIEVER_HEADER_LABELS = {
    "qwen3_embedding_0_6b": "\\makecell{Qwen3-Emb-\\\\0.6B}",
    "bge_m3": "BGE-M3",
}

RETRIEVER_ORDER = ["qwen3_embedding_0_6b", "bge_m3"]
RETRIEVER_DISPLAY_NAMES.update({
    "gte_large_en_v1_5": "GTE-large-en-v1.5",
    "e5_large_v2": "E5-large-v2",
})
RETRIEVER_HEADER_LABELS.update({
    "qwen3_embedding_0_6b": r"\makecell{Qwen3-Emb-\\0.6B}",
    "gte_large_en_v1_5": r"\makecell{GTE-large-\\en-v1.5}",
    "e5_large_v2": r"\makecell{E5-large-\\v2}",
})
RETRIEVER_ORDER = ["qwen3_embedding_0_6b", "gte_large_en_v1_5", "e5_large_v2"]


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir

    importances_by_retriever: dict[str, pd.Series] = {}
    for retriever in RETRIEVER_ORDER:
        importances_by_depth = ordinal_importances_by_depth(store_root, retriever)
        importances_by_retriever[retriever] = mean_importance_across_depths(importances_by_depth)
        print(f"Averaged importances for {retriever} over k={sorted(importances_by_depth)}")

    all_features = sorted(set().union(*(series.index for series in importances_by_retriever.values())))
    aligned = pd.DataFrame(
        {
            retriever: importances_by_retriever[retriever].reindex(all_features).fillna(0.0)
            for retriever in RETRIEVER_ORDER
        }
    )

    n = len(RETRIEVER_ORDER)
    rho_matrix = pd.DataFrame(index=RETRIEVER_ORDER, columns=RETRIEVER_ORDER, dtype=float)
    for i in range(n):
        for j in range(n):
            retriever_i = RETRIEVER_ORDER[i]
            retriever_j = RETRIEVER_ORDER[j]
            rho, _ = spearmanr(aligned[retriever_i], aligned[retriever_j])
            rho_matrix.loc[retriever_i, retriever_j] = rho

    row_max_partner = {}
    for retriever in RETRIEVER_ORDER:
        off_diagonal = rho_matrix.loc[retriever, [r for r in RETRIEVER_ORDER if r != retriever]]
        row_max_partner[retriever] = off_diagonal.idxmax()

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Ordinal predictor feature importance correlation ($\\rho$) across semantic retrievers, using "
        "each retriever's mean importance across the predictor's binary classifiers for all $k$ (not separated by "
        "$k$). Bold marks each row's largest off-diagonal value.}"
    )
    lines.append("\\label{tab:calibrated_ordinal_importances_correlation}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(RETRIEVER_ORDER) + "}")
    lines.append("\\toprule")
    header = "Retriever & " + " & ".join(RETRIEVER_HEADER_LABELS[r] for r in RETRIEVER_ORDER) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for retriever in RETRIEVER_ORDER:
        cells = [RETRIEVER_DISPLAY_NAMES[retriever]]
        for other in RETRIEVER_ORDER:
            value = f"{rho_matrix.loc[retriever, other]:.3f}"
            if other == row_max_partner[retriever]:
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "calibrated_ordinal" / "table_calibrated_ordinal_importances_correlation.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
