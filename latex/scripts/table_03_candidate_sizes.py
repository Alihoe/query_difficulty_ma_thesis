from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import candidate_size_grid_table, frame_to_latex
from query_difficulty.runtime import build_runtime


def main(config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file.")) -> None:
    runtime = build_runtime(config, verbose=False)
    frame = candidate_size_grid_table(runtime.config.candidate_sizes)
    latex = frame_to_latex(frame, "Evaluated candidate-set sizes used by the project.", "tab:candidate_size_grid")
    output_path = runtime.config.latex_table_dir / "submission" / "table_03_candidate_sizes.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
