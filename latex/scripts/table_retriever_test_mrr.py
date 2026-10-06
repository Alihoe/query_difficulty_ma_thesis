from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _routing_retriever_names import routing_retriever_display_name
from _thesis_captions import apply_thesis_captions
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
    mrr = payload["single_retriever_mrr_at_5"]
    print(f"{len(mrr)} retrievers evaluated on {payload['num_queries']} {language} test queries")
    label = f"tab:retriever_test_mrr_{language}"
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{label}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{l|r}",
        "\\toprule",
        "Retriever & MRR@5 \\\\",
        "\\midrule",
    ]
    for name, value in sorted(mrr.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"{routing_retriever_display_name(name)} & {value:.4f} \\\\")
        print(f"  {routing_retriever_display_name(name):<26} {value:.4f}")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    output_path = runtime.config.latex_table_dir / "feature_generation" / f"table_retriever_test_mrr_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    typer.run(main)
