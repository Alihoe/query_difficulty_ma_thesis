from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _ood_generalization_common import (
    CLEF_CONFIG_PATH,
    HIT1_COMPARISON_TOP_K_VALUES,
    hit1_comparison_inputs,
    hit1_comparison_summary_path,
    run_hit1_comparison,
)
from query_difficulty.runtime import build_runtime

CLEF_ARTIFACT_DIR = ROOT / "artifacts"
DATASETS = [
    ("clef2025", "config/clef2025.yaml", ROOT / "artifacts_clef2025", None),
    ("clef2022", "config/clef2022.yaml", ROOT / "artifacts_clef2022", None),
    ("scifact", "config/scifact.yaml", ROOT / "artifacts_scifact", None),
    ("climatecheck", "config/climatecheck.yaml", ROOT / "artifacts_climatecheck", "data/ClimateCheck/gold_sets_en_test.json"),
]
BANNER = "#" * 100


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"


def _status(artifact_dir: Path, gold_sets_path: str | None) -> tuple[bool, str]:
    summary_path = hit1_comparison_summary_path(artifact_dir)
    if not summary_path.exists():
        return False, "no previous Hit@1 comparison summary"
    current = hit1_comparison_inputs(artifact_dir, CLEF_ARTIFACT_DIR, gold_sets_path)
    if current["reranker_score_signature"] is None:
        return False, "reranker score cache missing"
    stored = json.loads(summary_path.read_text(encoding="utf-8")).get("inputs", {})
    changed = sorted(key for key in current if stored.get(key) != current[key])
    if changed:
        return False, f"inputs changed: {', '.join(changed)}"
    return True, "up to date (same Hit@1 model, reranker scores, gold sets and settings)"


def _print_summary(dataset: str, summary: dict) -> None:
    confusion = summary["hit1_confusion"]
    print(
        f"[run_all_ood_hit1] {dataset}: {summary['evaluated_queries']} queries, skipped {summary['skipped_queries']}, "
        f"Hit@1 TP={confusion['tp']} FP={confusion['fp']} TN={confusion['tn']} FN={confusion['fn']}",
        flush=True,
    )
    print(f"    {'k':>3}  {'No rerank':>9}  {'Rerank all':>10}  {'Hit@1':>7}  {'% reranked':>10}  {'calls all':>10}  {'calls Hit@1':>11}", flush=True)
    for top_k, metrics in summary["results_by_top_k"].items():
        share = 100.0 * metrics["hit1"]["reranked_queries"] / summary["evaluated_queries"]
        print(
            f"    {top_k:>3}  {metrics['no_rerank']['mrr_at_5']:>9.4f}  {metrics['rerank_all']['mrr_at_5']:>10.4f}  "
            f"{metrics['hit1']['mrr_at_5']:>7.4f}  {share:>9.1f}%  {metrics['rerank_all']['reranked_pairs']:>10,}  "
            f"{metrics['hit1']['reranked_pairs']:>11,}",
            flush=True,
        )


def main() -> None:
    started_at = perf_counter()
    print(
        f"[run_all_ood_hit1] {len(DATASETS)} OOD datasets: {', '.join(name for name, _, _, _ in DATASETS)}; "
        f"policies No rerank, Rerank all and lightweight Hit@1 at k={list(HIT1_COMPARISON_TOP_K_VALUES)}",
        flush=True,
    )
    plan = []
    for dataset, config_path, artifact_dir, gold_sets_path in DATASETS:
        up_to_date, reason = _status(artifact_dir, gold_sets_path)
        plan.append((dataset, config_path, artifact_dir, gold_sets_path, up_to_date))
        print(f"    {dataset:<13} {'skip' if up_to_date else 'run':<5} {reason}", flush=True)

    clef_runtime = None
    results = []
    for index, (dataset, config_path, artifact_dir, gold_sets_path, up_to_date) in enumerate(plan, start=1):
        print(f"\n{BANNER}\n# ({index}/{len(plan)}) {dataset}\n{BANNER}", flush=True)
        step_started_at = perf_counter()
        if up_to_date:
            summary = json.loads(hit1_comparison_summary_path(artifact_dir).read_text(encoding="utf-8"))
            print(f"[run_all_ood_hit1] {dataset} is up to date; showing the stored results", flush=True)
            _print_summary(dataset, summary)
            results.append((dataset, "SKIPPED (up to date)", 0.0))
            continue
        try:
            if clef_runtime is None:
                print(f"[run_all_ood_hit1] loading the CLEF runtime from {CLEF_CONFIG_PATH} (frozen predictor and reference pool)", flush=True)
                clef_runtime = build_runtime(CLEF_CONFIG_PATH)
            summary = run_hit1_comparison(config_path, dataset, clef_runtime, gold_sets_path=gold_sets_path)
            status = "OK"
            _print_summary(dataset, summary)
        except Exception:
            traceback.print_exc()
            status = "FAILED"
        elapsed = perf_counter() - step_started_at
        print(
            f"[run_all_ood_hit1] ({index}/{len(plan)}) {status} {dataset} in {_format_seconds(elapsed)}; "
            f"total elapsed {_format_seconds(perf_counter() - started_at)}",
            flush=True,
        )
        results.append((dataset, status, elapsed))

    print("\n[run_all_ood_hit1] summary:", flush=True)
    for dataset, status, elapsed in results:
        print(f"  {status:<22} {_format_seconds(elapsed):>12}  {dataset}", flush=True)
    failures = [dataset for dataset, status, _ in results if status == "FAILED"]
    print(
        f"[run_all_ood_hit1] finished in {_format_seconds(perf_counter() - started_at)}; "
        f"{len(failures)} failed{': ' + ', '.join(failures) if failures else ''}",
        flush=True,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
