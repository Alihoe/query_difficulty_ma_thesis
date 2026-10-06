from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
import math
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _routing_retriever_names import routing_retriever_display_name
from query_difficulty.runtime import build_runtime


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    summary_path = runtime.store.path(
        "predictions", "lightweight", "hit_at_1", language, "confidence_routed_experiments", "summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No confidence-routed experiments summary found at {summary_path}. Run "
            "scripts/run_confidence_routed_lightweight_experiments.py first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    shapley_values = payload["shapley_values"]
    single_retriever_mrr5 = payload["single_retriever_mrr_at_5"]

    rows = sorted(shapley_values.items(), key=lambda item: item[1]["shapley_value"], reverse=True)
    best_own_mrr5_name = max(single_retriever_mrr5, key=lambda name: single_retriever_mrr5[name])
    best_shapley_name = max(shapley_values, key=lambda name: shapley_values[name]["shapley_value"])
    best_min_name = max(shapley_values, key=lambda name: shapley_values[name]["min_marginal_contribution"])
    for name, entry in shapley_values.items():
        if not math.isclose(entry["max_marginal_contribution"], single_retriever_mrr5[name], abs_tol=1e-12):
            raise ValueError(
                f"Best-case contribution of {name} differs from its own MRR@5; the omitted column would carry information"
            )

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Shapley value of each retriever's contribution to the confidence-routed MRR@5, with its own "
        "MRR@5 and its worst-case contribution to a single retriever combination. The best-case contribution equals "
        "the own MRR@5 for every retriever and is therefore not shown.}"
    )
    lines.append(f"\\label{{tab:confidence_routed_shapley_values_{language}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l|r|rr}")
    lines.append("\\toprule")
    lines.append("Retriever & Own MRR@5 & Shapley value & Min.\\ contribution \\\\")
    lines.append("\\midrule")
    for name, entry in rows:
        own_mrr5_text = f"{single_retriever_mrr5[name]:.4f}"
        shapley_text = f"{entry['shapley_value']:+.4f}"
        min_text = f"{entry['min_marginal_contribution']:+.4f}"
        if name == best_own_mrr5_name:
            own_mrr5_text = f"\\textbf{{{own_mrr5_text}}}"
        if name == best_shapley_name:
            shapley_text = f"\\textbf{{{shapley_text}}}"
        if name == best_min_name:
            min_text = f"\\textbf{{{min_text}}}"
        lines.append(f"{routing_retriever_display_name(name)} & {own_mrr5_text} & {shapley_text} & {min_text} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = runtime.config.latex_table_dir / "confidence_routed" / f"table_confidence_routed_shapley_values_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
