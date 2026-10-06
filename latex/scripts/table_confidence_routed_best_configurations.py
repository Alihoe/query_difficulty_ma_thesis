from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
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
    global_best = payload["global_best_configuration"]
    by_size = sorted(payload["best_configuration_by_size"], key=lambda entry: entry["size"])

    caption = (
        "Best confidence-routed MRR@5 and MRR@5 with perfect routing at each pool size, found by exhaustive search "
        f"over all combinations of the {len(payload['candidates'])} candidate retrievers. Perfect routing selects the "
        "best retriever for each query. A dot marks the retrievers in the best combination of each size."
    )
    label = f"tab:confidence_routed_best_configurations_{language}"

    members: list[str] = []
    for entry in by_size:
        for name in entry["candidates"]:
            if name not in members:
                members.append(name)
    missing = sorted(set(payload["candidates"]) - set(members))
    if missing:
        raise ValueError(f"Candidates never selected in any best combination: {missing}")

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{r|rr|" + "c" * len(members) + "}")
    lines.append("\\toprule")
    header = ["Size", "Routed", "\\makecell{Perfect\\\\routing}"] + [
        f"\\rotatebox{{90}}{{{routing_retriever_display_name(name)}}}" for name in members
    ]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for entry in by_size:
        size = entry["size"]
        routed_text = f"{entry['routed_mrr_at_5']:.4f}"
        perfect_text = f"{entry['oracle_mrr_at_5']:.4f}"
        size_text = str(size)
        if size == global_best["size"]:
            size_text = f"\\textbf{{{size_text}}}"
            routed_text = f"\\textbf{{{routed_text}}}"
        marks = ["$\\bullet$" if name in entry["candidates"] else "" for name in members]
        lines.append(" & ".join([size_text, routed_text, perfect_text] + marks) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path = (
        runtime.config.latex_table_dir / "confidence_routed" / f"table_confidence_routed_best_configurations_{language}.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
