from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import difficulty_distribution_table, frame_to_latex
from query_difficulty.runtime import build_runtime


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    query_key = f"{language}_test"
    frame = difficulty_distribution_table(runtime.config.root_dir / "data" / "submissions", runtime.dataset.load().queries[query_key], language)
    latex = frame_to_latex(frame, f"Query difficulty distribution for {language}.", f"tab:query_difficulty_distribution_{language}")
    output_path = runtime.config.latex_table_dir / "difficulty_analysis" / f"table_14_query_difficulty_distribution_{language}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
