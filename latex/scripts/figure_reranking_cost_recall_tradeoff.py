from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.multistep import canonical_multistage_reranker_specs
from query_difficulty.runtime import build_runtime

RETRIEVER_DISPLAY_NAMES = {
    "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B",
}
TOP_K_VALUES = [10, 30, 50]
REFERENCE_RERANKER = "minilm"

FIXED_DEPTH_POLICIES = ["max_depth", "in_top_k", "hit1_and_max_depth", "hit1_and_in_top_k"]
ADAPTIVE_DEPTH_POLICIES = [
    "ordinal",
    "qpp",
    "in_top_k_depth",
    "hit1_and_ordinal",
    "hit1_and_qpp",
    "hit1_and_in_top_k_depth",
]
POLICY_DISPLAY_NAMES = {
    "no_rerank": "No rerank",
    "rerank_all": "Full reranking",
    "hit1": "Hit-at-1",
    "max_depth": "Maximum Depth",
    "in_top_k": "Binary (fixed $k$)",
    "hit1_and_max_depth": "Hit-at-1 $\\wedge$ Maximum Depth",
    "hit1_and_in_top_k": "Hit-at-1 $\\wedge$ Binary (fixed $k$)",
    "ordinal": "Ordinal",
    "qpp": "QPP",
    "in_top_k_depth": "Binary (adaptive $k$)",
    "hit1_and_ordinal": "Hit-at-1 $\\wedge$ Ordinal",
    "hit1_and_qpp": "Hit-at-1 $\\wedge$ QPP",
    "hit1_and_in_top_k_depth": "Hit-at-1 $\\wedge$ Binary (adaptive $k$)",
}
SELECTOR_MARKERS = {
    "max_depth": "s",
    "in_top_k": "^",
    "in_top_k_depth": "v",
    "ordinal": "p",
    "qpp": "h",
}

INK = "#1f1f1f"
MUTED = "#8a8a8a"
LIGHT = "#bdbdbd"
OTHER = "#6f6f6f"
GRID = "#e6e6e6"
HIT1_COLOR = "#2a78d6"


def _run_dir(store_root: Path, language: str, retriever: str, top_k: int) -> Path:
    spec = canonical_multistage_reranker_specs()[REFERENCE_RERANKER]
    return (
        store_root
        / "multistep"
        / "comparison_lightweight_selectors"
        / language
        / retriever
        / spec.method
        / spec.key
        / f"top_{top_k}"
        / spec.cache_dir_name
    )


def _gold_ranks(runtime, language: str, retriever: str) -> pd.Series:
    split_key = f"{language}_test"
    queries = runtime.retrieval.queries[split_key][["query_id", "document_id"]].astype(int)
    ranking = runtime.retrieval.ensure_rankings(split_key, retriever)[["query_id", "rank", "document_id"]].astype(int)
    return queries.merge(ranking, on=["query_id", "document_id"], how="left").set_index("query_id")["rank"]


def _policy_points(outcomes: pd.DataFrame, gold_rank: pd.Series, top_k: int, summary: dict) -> pd.DataFrame:
    rank = outcomes["query_id"].map(gold_rank)
    n_queries = len(outcomes)
    full_pairs = n_queries * top_k
    full_recall = float((rank <= top_k).mean())
    summary_full_pairs = int(summary["test"][f"rerank_all_top{top_k}"]["reranked_pairs"])
    if summary_full_pairs != full_pairs:
        print(f"    warning: full reranking pairs {full_pairs:,} differ from summary {summary_full_pairs:,}")

    rows = [
        {"policy": "no_rerank", "pairs": 0, "recall": float((rank <= 1).mean())},
        {"policy": "rerank_all", "pairs": full_pairs, "recall": full_recall},
    ]
    for policy in ["hit1", *FIXED_DEPTH_POLICIES, *ADAPTIVE_DEPTH_POLICIES]:
        column = f"depth_{policy}"
        if column not in outcomes.columns:
            print(f"    warning: {column} missing, policy skipped")
            continue
        depth = outcomes[column].astype(int)
        retained_depth = depth.where(depth > 0, 1)
        pairs = int(depth.sum())
        summary_pairs = summary["test"].get(policy, {}).get("reranked_pairs")
        if summary_pairs is not None and int(summary_pairs) != pairs:
            print(f"    warning: {policy} pairs {pairs:,} differ from summary {int(summary_pairs):,}")
        rows.append({"policy": policy, "pairs": pairs, "recall": float((rank <= retained_depth).mean())})

    points = pd.DataFrame(rows)
    points["relative_cost"] = points["pairs"] / full_pairs
    points["retained_recall"] = points["recall"] / full_recall
    points["display"] = points["policy"].map(POLICY_DISPLAY_NAMES)
    return points


def _other_marker_style(policy: str) -> dict:
    combined = policy.startswith("hit1_and_")
    selector = policy.removeprefix("hit1_and_")
    return {
        "s": 28,
        "marker": SELECTOR_MARKERS[selector],
        "facecolor": "white" if combined else OTHER,
        "edgecolor": OTHER,
        "linewidth": 0.9 if combined else 0.0,
    }


def _legend_handle(policy: str) -> Line2D:
    style = _other_marker_style(policy)
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=style["marker"],
        markersize=5,
        markerfacecolor=style["facecolor"],
        markeredgecolor=style["edgecolor"],
        markeredgewidth=style["linewidth"],
        label=POLICY_DISPLAY_NAMES[policy],
    )


def _draw_panel(axis, points: pd.DataFrame, top_k: int) -> None:
    by_policy = points.set_index("policy")
    for policy in [*FIXED_DEPTH_POLICIES, *ADAPTIVE_DEPTH_POLICIES]:
        if policy not in by_policy.index:
            continue
        axis.scatter(
            by_policy.loc[policy, "relative_cost"],
            by_policy.loc[policy, "retained_recall"],
            **_other_marker_style(policy),
            zorder=3,
        )

    no_rerank = by_policy.loc["no_rerank"]
    axis.scatter(no_rerank["relative_cost"], no_rerank["retained_recall"], s=26, marker="X", facecolor=MUTED,
                 edgecolor="white", linewidth=0.5, zorder=2)
    axis.annotate("No rerank", (no_rerank["relative_cost"], no_rerank["retained_recall"]), xytext=(5, 0),
                  textcoords="offset points", ha="left", va="center", fontsize=7, color=MUTED)

    full = by_policy.loc["rerank_all"]
    axis.scatter(full["relative_cost"], full["retained_recall"], s=44, marker="o", facecolor=INK,
                 edgecolor="white", linewidth=0.6, zorder=4)
    axis.annotate("Full reranking", (full["relative_cost"], full["retained_recall"]), xytext=(-12, 11),
                  textcoords="offset points", ha="right", va="center", fontsize=7.5, color=INK,
                  fontweight="bold",
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "relpos": (1, 0.5),
                              "shrinkA": 1, "shrinkB": 4})

    hit1 = by_policy.loc["hit1"]
    axis.scatter(hit1["relative_cost"], hit1["retained_recall"], s=44, marker="o", facecolor=HIT1_COLOR,
                 edgecolor="white", linewidth=0.6, zorder=5)
    axis.annotate("Hit-at-1", (hit1["relative_cost"], hit1["retained_recall"]),
                  xytext=(hit1["relative_cost"], 0.45), textcoords=("data", "axes fraction"), ha="center",
                  va="center", fontsize=7.5, color=INK, fontweight="bold",
                  arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "relpos": (0.5, 1),
                              "shrinkA": 1, "shrinkB": 5})

    axis.text(0.03, 0.97, "better ↖", transform=axis.transAxes, ha="left", va="top", fontsize=6.5,
              color=LIGHT)
    axis.set_title(f"$k$ = {top_k}", fontsize=8.5, color=INK, pad=4, loc="left")
    axis.grid(True, color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.6)
        axis.spines[side].set_color(MUTED)
    axis.tick_params(labelsize=7, width=0.6, length=2.5, color=MUTED)


def _print_points(points: pd.DataFrame) -> None:
    for row in points.itertuples():
        print(
            f"    {row.display:<34} pairs={row.pairs:>8,}  cost={row.relative_cost:.3f}  "
            f"recall={row.recall:.4f}  retained={row.retained_recall:.4f}"
        )


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    store_root = runtime.store.root_dir

    for retriever, retriever_label in RETRIEVER_DISPLAY_NAMES.items():
        print(f"\n=== {retriever_label} ({retriever}) ===")
        gold_rank = _gold_ranks(runtime, language, retriever)
        print(f"  test queries: {len(gold_rank):,}")

        points_by_k: dict[int, pd.DataFrame] = {}
        for top_k in TOP_K_VALUES:
            run_dir = _run_dir(store_root, language, retriever, top_k)
            outcomes_path = run_dir / "test_query_outcomes.parquet"
            summary_path = run_dir / "summary.json"
            if not outcomes_path.exists() or not summary_path.exists():
                print(f"  k={top_k}: missing {run_dir}, skipped")
                continue
            print(f"  k={top_k}: reading {outcomes_path}")
            outcomes = pd.read_parquet(outcomes_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            points = _policy_points(outcomes, gold_rank, top_k, summary)
            _print_points(points)
            points_by_k[top_k] = points

        if len(points_by_k) != len(TOP_K_VALUES):
            print(f"  figure for {retriever} skipped, not all k values available")
            continue

        figure, axes = plt.subplots(1, len(TOP_K_VALUES), figsize=(6.3, 2.3), sharex=True, sharey=True)
        for axis, top_k in zip(axes, TOP_K_VALUES):
            _draw_panel(axis, points_by_k[top_k], top_k)
        axes[0].set_ylabel("Retained candidate recall", fontsize=8, color=INK)
        for axis in axes:
            axis.set_xlabel("Relative reranking cost", fontsize=8, color=INK)

        legend_handles = [
            _legend_handle(policy)
            for selector in SELECTOR_MARKERS
            for policy in (selector, f"hit1_and_{selector}")
        ]
        figure.legend(handles=legend_handles, loc="upper center", ncol=len(SELECTOR_MARKERS), frameon=False,
                      fontsize=6.5, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.9)
        figure.tight_layout(rect=(0, 0, 1, 0.86))

        output_path = (
            ROOT
            / "latex"
            / "figures"
            / "multistage_reranking"
            / f"reranking_cost_recall_tradeoff_{retriever}_{language}.pdf"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    typer.run(main)
