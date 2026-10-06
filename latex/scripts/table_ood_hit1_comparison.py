from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _ood_generalization_common import HIT1_COMPARISON_NAMESPACE
from query_difficulty.config import load_project_config

CLEF_CONFIG_PATH = "config/default.yaml"
DATASETS = [
    ("clef2025", "CLEF 2025", ROOT / "artifacts_clef2025"),
    ("clef2022", "CLEF 2022", ROOT / "artifacts_clef2022"),
    ("scifact", "SciFact", ROOT / "artifacts_scifact"),
    ("climatecheck", "ClimateCheck", ROOT / "artifacts_climatecheck"),
]
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
CHUNK_SIZE = 500


def _paired_bootstrap_ci(differences: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1)
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def main() -> None:
    config = load_project_config(CLEF_CONFIG_PATH)
    seed = int(config.runtime.random_seed)
    print(f"OOD Hit@1 table: paired bootstrap B={BOOTSTRAP_SAMPLES:,}, {CONFIDENCE_LEVEL:.0%} percentile CI, seed={seed}")
    rows = []
    for key, label, artifact_dir in DATASETS:
        output_dir = artifact_dir / HIT1_COMPARISON_NAMESPACE
        summary_path = output_dir / "summary.json"
        outcomes_path = output_dir / "outcomes.parquet"
        if not summary_path.exists() or not outcomes_path.exists():
            raise FileNotFoundError(f"Missing {summary_path} or {outcomes_path}; run scripts/run_all_ood_hit1_comparison.py first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        outcomes = pd.read_parquet(outcomes_path)
        query_count = int(summary["evaluated_queries"])
        if len(outcomes) != query_count:
            raise ValueError(f"{key}: {len(outcomes)} outcome rows but summary reports {query_count} queries")
        print(f"{label}: {query_count} queries from {outcomes_path}")
        for index, (top_k, metrics) in enumerate(summary["results_by_top_k"].items()):
            no_rerank = outcomes["rr_no_rerank"].to_numpy(dtype=np.float64)
            rerank_all = outcomes[f"rr_rerank_all_top{top_k}"].to_numpy(dtype=np.float64)
            hit1 = outcomes[f"rr_hit1_top{top_k}"].to_numpy(dtype=np.float64)
            for name, values in (("no_rerank", no_rerank), ("rerank_all", rerank_all), ("hit1", hit1)):
                if abs(float(values.mean()) - float(metrics[name]["mrr_at_5"])) > 1e-9:
                    raise ValueError(f"{key} k={top_k}: {name} MRR@5 differs between outcomes and summary")
            delta = float(hit1.mean() - rerank_all.mean())
            lower, upper = _paired_bootstrap_ci(hit1 - rerank_all, seed)
            reranked = 100.0 * metrics["hit1"]["reranked_queries"] / query_count
            saved = 100.0 * (1.0 - metrics["hit1"]["reranked_pairs"] / metrics["rerank_all"]["reranked_pairs"])
            print(
                f"  k={top_k}: no rerank={no_rerank.mean():.4f}, rerank all={rerank_all.mean():.4f}, "
                f"Hit@1={hit1.mean():.4f}, delta={delta:+.4f} CI=[{lower:+.4f}, {upper:+.4f}], "
                f"reranked={reranked:.1f}%, calls saved={saved:.1f}%"
            )
            dataset_cell = f"{label} ({query_count:,})" if index == 0 else ""
            mrr_texts = [f"{value:.4f}" for value in (no_rerank.mean(), rerank_all.mean(), hit1.mean())]
            best_text = max(mrr_texts, key=float)
            mrr_cells = " & ".join(f"\\textbf{{{text}}}" if text == best_text else text for text in mrr_texts)
            rows.append(
                f"{dataset_cell} & {top_k} & {mrr_cells} & "
                f"{delta:+.4f} & [{lower:+.4f}, {upper:+.4f}] & {reranked:.1f}\\% & {saved:.1f}\\% \\\\"
            )
        rows.append("\\midrule")
    rows.pop()

    confidence_label = f"{CONFIDENCE_LEVEL:.0%}".replace("%", "\\%")
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Zero-shot transfer of the lightweight Hit@1 gate to out-of-distribution datasets. For retrieval \\textit{Qwen3-Embedding-0.6B} is used and for reranking \\textit{jina-reranker-v2-base-multilingual}.}",
        "\\label{tab:ood_hit1_comparison_en}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lr|ccc|cc|rr}",
        "\\toprule",
        f"Dataset (queries) & $k$ & No rerank & Rerank all & Hit@1 & $\\Delta$ & {confidence_label} CI & \\% reranked & Calls saved \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\par\\smallskip",
        "{\\footnotesize $\\Delta$ is Hit@1 minus Rerank all with a paired bootstrap confidence interval over queries. "
        "The Hit@1 predictor is trained on CLEF 2026 only and applied without retraining. "
        "ClimateCheck is evaluated against full gold sets.}",
        "\\end{table}",
    ]
    output_path = ROOT / config.latex_table_dir / "ood_generalization" / "table_ood_hit1_comparison_en.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions("\n".join(lines)), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
