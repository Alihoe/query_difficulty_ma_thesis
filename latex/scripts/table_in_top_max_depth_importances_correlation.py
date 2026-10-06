from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import pandas as pd
import typer
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.runtime import build_runtime

RETRIEVER_DISPLAY_NAMES = {
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
    "bge_m3": "BGE-M3",
    "bge_large_en_v1_5": "BGE-large-en-v1.5",
    "e5_large_v2": "E5-large-v2",
    "gtr_t5_xl": "GTR-T5-XL",
    "multilingual_e5_large": "multilingual-E5-large",
    "multilingual_e5_base": "multilingual-E5-base",
}

RETRIEVER_HEADER_LABELS = {
    "qwen3_embedding_0_6b": "\\makecell{Qwen3-Emb-\\\\0.6B}",
    "bge_m3": "BGE-M3",
    "bge_large_en_v1_5": "\\makecell{BGE-large-\\\\en-v1.5}",
    "e5_large_v2": "\\makecell{E5-large-\\\\v2}",
    "gtr_t5_xl": "GTR-T5-XL",
    "multilingual_e5_large": "\\makecell{mE5-\\\\large}",
    "multilingual_e5_base": "\\makecell{mE5-\\\\base}",
}

RETRIEVER_ORDER = [
    "qwen3_embedding_0_6b",
    "bge_m3",
    "bge_large_en_v1_5",
    "e5_large_v2",
    "gtr_t5_xl",
    "multilingual_e5_large",
    "multilingual_e5_base",
]
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
    predictor_name = "in_top_max_depth"

    available_retrievers = []
    for retriever in RETRIEVER_ORDER:
        imp_path = store_root / "models" / predictor_name / retriever / "feature_importances.parquet"
        if imp_path.exists():
            available_retrievers.append(retriever)
        else:
            print(f"Skipping {retriever}: no feature importances at {imp_path}")
    print(f"Found {predictor_name} feature importances for: {available_retrievers}")

    importances_by_retriever: dict[str, pd.Series] = {}
    for retriever in available_retrievers:
        imp_path = store_root / "models" / predictor_name / retriever / "feature_importances.parquet"
        importances = pd.read_parquet(imp_path)
        importances_by_retriever[retriever] = importances.set_index("feature")["importance"]

    all_features = sorted(set().union(*(series.index for series in importances_by_retriever.values())))
    aligned = pd.DataFrame(
        {
            retriever: importances_by_retriever[retriever].reindex(all_features).fillna(0.0)
            for retriever in available_retrievers
        }
    )

    n = len(available_retrievers)
    rho_matrix = pd.DataFrame(index=available_retrievers, columns=available_retrievers, dtype=float)
    for i in range(n):
        for j in range(n):
            retriever_i = available_retrievers[i]
            retriever_j = available_retrievers[j]
            rho, _ = spearmanr(aligned[retriever_i], aligned[retriever_j])
            rho_matrix.loc[retriever_i, retriever_j] = rho

    row_max_partner = {}
    for retriever in available_retrievers:
        off_diagonal = rho_matrix.loc[retriever, [r for r in available_retrievers if r != retriever]]
        if not off_diagonal.empty:
            row_max_partner[retriever] = off_diagonal.idxmax()

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{In-top-max-depth feature importance correlation ($\\rho$) across semantic retrievers "
        "(identical for the ungated and gated variants, which share the same trained classifier). Bold "
        "marks each row's largest off-diagonal value.}"
    )
    lines.append("\\label{tab:in_top_max_depth_importances_correlation}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l" + "c" * len(available_retrievers) + "}")
    lines.append("\\toprule")
    header = "Retriever & " + " & ".join(RETRIEVER_HEADER_LABELS[r] for r in available_retrievers) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for retriever in available_retrievers:
        cells = [RETRIEVER_DISPLAY_NAMES[retriever]]
        for other in available_retrievers:
            value = f"{rho_matrix.loc[retriever, other]:.3f}"
            if row_max_partner.get(retriever) == other:
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "in_top_max_depth" / "table_in_top_max_depth_importances_correlation.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
