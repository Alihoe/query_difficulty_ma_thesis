from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import read_submission_archive, recall_at_k
from query_difficulty.runtime import build_runtime

TOP_K = 5
SOLVABILITY_QUARTILES = (0.25, 0.5, 0.75)

INK = "#1f1f1f"
MUTED = "#8a8a8a"
MISSED = "#eeeeec"
SOLVED = "#2b2b2b"
STRIP = "#e3e3e0"
BAR = "#6b6b6b"


def success_matrix(submission_dir: Path, gold_queries: pd.DataFrame, language: str) -> pd.DataFrame:
    gold = gold_queries[["query_id", "document_id"]].astype(int)
    rows: dict[str, pd.Series] = {}
    archive_paths = sorted(submission_dir.glob("*.zip"))
    for position, archive_path in enumerate(archive_paths, start=1):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            print(f"[{position}/{len(archive_paths)}] {archive_path.stem}: no '{language}' run, skipped")
            continue
        frame = archives[language].rename(columns={"index": "query_id"})
        frame["query_id"] = frame["query_id"].astype(int)
        merged = gold.merge(frame, on="query_id", how="left")
        hits = [
            int(isinstance(preds, list) and recall_at_k(preds, gold_id, TOP_K) == 1.0)
            for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)
        ]
        rows[archive_path.stem] = pd.Series(hits, index=merged["query_id"].to_numpy())
        print(
            f"[{position}/{len(archive_paths)}] {archive_path.stem}: "
            f"{sum(hits)}/{len(hits)} solved at top {TOP_K}, {int(merged['preds'].isna().sum())} queries missing"
        )
    return pd.DataFrame(rows).T


def solvability_regions(system_count: int) -> list[tuple[str, int, int]]:
    rare_max, minority_max, majority_max = (int(system_count * quartile) for quartile in SOLVABILITY_QUARTILES)
    return [
        ("Unsolved", 0, 0),
        ("Rarely solved", 1, rare_max),
        ("Minority solved", rare_max + 1, minority_max),
        ("Majority solved", minority_max + 1, majority_max),
        ("Widely solved", majority_max + 1, system_count - 1),
        ("Universally solved", system_count, system_count),
    ]


def order_matrix(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    solved_per_system = matrix.sum(axis=1)
    system_order = sorted(matrix.index, key=lambda system: (-int(solved_per_system[system]), system))
    ordered = matrix.loc[system_order]
    values = ordered.to_numpy()
    solved_per_query = values.sum(axis=0)
    lexicographic_keys = [-values[row] for row in reversed(range(values.shape[0]))]
    column_order = np.lexsort(lexicographic_keys + [solved_per_query])
    return ordered.iloc[:, column_order], solved_per_query[column_order]


def region_label(name: str, low: int, high: int) -> str:
    short_name = name.split()[0]
    count_text = str(low) if low == high else f"{low}–{high}"
    return f"{short_name}\n{count_text}"


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    query_key = f"{language}_test"
    gold_queries = runtime.dataset.load().queries[query_key]
    submission_dir = runtime.config.root_dir / "data" / "submissions"
    print(f"Building success matrix for {query_key} ({len(gold_queries)} queries) from {submission_dir}")

    matrix = success_matrix(submission_dir, gold_queries, language)
    system_count = matrix.shape[0]
    ordered, solved_per_query = order_matrix(matrix)
    solved_per_system = ordered.sum(axis=1).to_numpy()
    regions = solvability_regions(system_count)
    print(f"Matrix: {system_count} systems x {ordered.shape[1]} queries")

    boundaries = []
    spans = []
    for name, low, high in regions:
        in_region = np.flatnonzero((solved_per_query >= low) & (solved_per_query <= high))
        start, stop = (int(in_region[0]), int(in_region[-1]) + 1) if in_region.size else (None, None)
        spans.append((name, low, high, start, stop))
        print(f"  {name:<20} {low:>2}-{high:<2} systems: {in_region.size:>5} queries ({in_region.size / ordered.shape[1]:.1%})")
        if start is not None and start > 0:
            boundaries.append(start)

    figure = plt.figure(figsize=(6.3, 3.6))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 0.13], height_ratios=[0.09, 1.0], wspace=0.03, hspace=0.04)
    strip_axis = figure.add_subplot(grid[0, 0])
    matrix_axis = figure.add_subplot(grid[1, 0], sharex=strip_axis)
    bar_axis = figure.add_subplot(grid[1, 1], sharey=matrix_axis)

    matrix_axis.imshow(
        ordered.to_numpy(),
        cmap=ListedColormap([MISSED, SOLVED]),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="none",
    )
    for boundary in boundaries:
        matrix_axis.axvline(boundary - 0.5, color=MUTED, linewidth=0.6)
        strip_axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
    matrix_axis.set_yticks(range(system_count))
    matrix_axis.set_yticklabels([str(rank) for rank in range(1, system_count + 1)], fontsize=4.5, color=INK)
    matrix_axis.tick_params(axis="y", length=0, pad=1.5)
    matrix_axis.set_ylabel("System (ranked by queries solved)", fontsize=7.5, color=INK)
    matrix_axis.set_xticks([])
    matrix_axis.set_xlabel(
        f"Queries ({ordered.shape[1]:,}), ordered from unsolved to universally solved",
        fontsize=7.5,
        color=INK,
    )
    for spine in matrix_axis.spines.values():
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)

    for name, low, high, start, stop in spans:
        if start is None:
            continue
        strip_axis.axvspan(start - 0.5, stop - 0.5, color=STRIP, linewidth=0)
        strip_axis.text(
            (start + stop - 1) / 2,
            0.5,
            region_label(name, low, high),
            ha="center",
            va="center",
            fontsize=6,
            color=INK,
            transform=strip_axis.get_xaxis_transform(),
        )
    strip_axis.set_xlim(-0.5, ordered.shape[1] - 0.5)
    strip_axis.set_ylim(0, 1)
    strip_axis.axis("off")

    bar_axis.barh(range(system_count), solved_per_system, height=0.75, color=BAR, linewidth=0)
    bar_axis.set_xlim(0, ordered.shape[1])
    bar_axis.tick_params(axis="y", left=False, labelleft=False)
    bar_axis.tick_params(axis="x", labelsize=5.5, colors=INK, length=2, width=0.5)
    bar_axis.set_xlabel("Solved", fontsize=7.5, color=INK)
    bar_axis.grid(True, axis="x", color="#e6e6e6", linewidth=0.5)
    bar_axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bar_axis.spines[side].set_visible(False)
    bar_axis.spines["bottom"].set_color(MUTED)
    bar_axis.spines["bottom"].set_linewidth(0.6)

    output_path = ROOT / "latex" / "figures" / "difficulty_analysis" / f"cross_system_success_matrix_{query_key}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close(figure)
    print(output_path)


if __name__ == "__main__":
    typer.run(main)
