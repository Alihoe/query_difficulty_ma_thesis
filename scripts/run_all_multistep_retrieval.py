from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.multistep import canonical_multistage_reranker_specs

MULTISTEP_ROOT = ROOT / "artifacts" / "multistep"
SELECTION_NAMESPACE = "comparison_lightweight_selectors"
LANGUAGE = "en"

RETRIEVERS = {
    "bm25": "bm25",
    "qwen": "qwen3_embedding_0_6b",
}
RERANKER_ORDER = ["jina", "minilm", "bge"]
TOP_K_VALUES = [50, 30, 20, 10, 5]
TABLE_SCRIPTS = [
    "latex/scripts/table_multistage_reranking_full_comparison.py",
]
BANNER = "#" * 100


def _script_path(prefix: str, reranker: str, top_k: int) -> str:
    path = ROOT / "scripts" / f"run_{prefix}_multistep_retrieval_{reranker}_top{top_k}.py"
    if not path.exists():
        raise FileNotFoundError(f"Expected multistage script is missing: {path}")
    return str(path.relative_to(ROOT))


def _config_dir(retriever_name: str, reranker: str, top_k: int) -> Path:
    spec = canonical_multistage_reranker_specs()[reranker]
    return (
        MULTISTEP_ROOT
        / SELECTION_NAMESPACE
        / LANGUAGE
        / retriever_name
        / spec.method
        / spec.key
        / f"top_{top_k}"
        / spec.cache_dir_name
    )


def _cached_scores(retriever_name: str, reranker: str, top_k: int) -> Path | None:
    for cached_top_k in sorted(k for k in TOP_K_VALUES if k >= top_k):
        path = _config_dir(retriever_name, reranker, cached_top_k) / f"test_top{cached_top_k}_scores.parquet"
        if path.exists():
            return path
    return None


def _checkpoint_count(retriever_name: str, reranker: str, top_k: int) -> int:
    checkpoint_dir = _config_dir(retriever_name, reranker, top_k) / "_score_checkpoints"
    return len(list(checkpoint_dir.glob("chunk_*.parquet"))) if checkpoint_dir.exists() else 0


def _combinations() -> list[dict]:
    combinations = []
    for reranker in RERANKER_ORDER:
        for prefix, retriever_name in RETRIEVERS.items():
            combinations.append(
                {
                    "reranker": reranker,
                    "retriever": retriever_name,
                    "steps": [
                        {
                            "script": _script_path(prefix, reranker, top_k),
                            "reranker": reranker,
                            "retriever": retriever_name,
                            "top_k": top_k,
                        }
                        for top_k in TOP_K_VALUES
                    ],
                }
            )
    return combinations


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"


def _describe(step: dict) -> str:
    return f"{step['reranker']} / {step['retriever']} / top_k={step['top_k']}"


def _combination_name(combination: dict) -> str:
    return f"{combination['reranker']} / {combination['retriever']}"


def _run_step(index: int, total: int, step: dict, started_at: float) -> tuple[str, int, float]:
    if "top_k" in step:
        cache = _cached_scores(step["retriever"], step["reranker"], step["top_k"])
        if cache is None:
            mode = "NEW RERANKING"
            checkpoints = _checkpoint_count(step["retriever"], step["reranker"], step["top_k"])
            print(
                f"\n{BANNER}\n# NEW RERANKING: {_describe(step)}\n"
                f"# No complete score cache for top_k>={step['top_k']} in "
                f"{_config_dir(step['retriever'], step['reranker'], step['top_k']).relative_to(ROOT)}\n"
                f"# Reranker inference will run now (resuming from {checkpoints} checkpoint files).\n{BANNER}",
                flush=True,
            )
        else:
            mode = "cached"
            print(
                f"\n[run_all_multistep] using cached reranker scores for {_describe(step)}: "
                f"{cache.relative_to(ROOT)}",
                flush=True,
            )
    else:
        mode = "table"
    print(f"[run_all_multistep] ({index}/{total}) starting {step['script']} [{mode}]", flush=True)
    step_started_at = perf_counter()
    completed = subprocess.run([sys.executable, step["script"]], cwd=ROOT)
    elapsed = perf_counter() - step_started_at
    status = "OK" if completed.returncode == 0 else f"FAILED (exit {completed.returncode})"
    print(
        f"[run_all_multistep] ({index}/{total}) {status} {step['script']} [{mode}] in "
        f"{_format_seconds(elapsed)}; total elapsed {_format_seconds(perf_counter() - started_at)}",
        flush=True,
    )
    return mode, completed.returncode, elapsed


def main() -> None:
    started_at = perf_counter()
    combinations = _combinations()
    cached_combinations = [c for c in combinations if _cached_scores(c["retriever"], c["reranker"], 50) is not None]
    new_combinations = [c for c in combinations if _cached_scores(c["retriever"], c["reranker"], 50) is None]
    total = sum(len(c["steps"]) for c in combinations) + len(TABLE_SCRIPTS)

    specs = canonical_multistage_reranker_specs()
    print("[run_all_multistep] reranker configurations (cache folder names):", flush=True)
    for reranker in RERANKER_ORDER:
        print(f"    {reranker:<8} {specs[reranker].method}/{specs[reranker].key}/top_K/{specs[reranker].cache_dir_name}", flush=True)
    print(f"[run_all_multistep] {total} steps queued; top_k order within each combination: {TOP_K_VALUES}", flush=True)
    print(f"[run_all_multistep] phase 1: {len(cached_combinations)} combinations with a cached top_k=50 score file", flush=True)
    for combination in cached_combinations:
        print(f"    {_combination_name(combination)}", flush=True)
    print(
        f"[run_all_multistep] phase 2: {len(new_combinations)} combinations needing NEW top_k=50 reranking "
        "(smaller top_k then reuse the top_k=50 scores; skipped if top_k=50 fails)",
        flush=True,
    )
    for combination in new_combinations:
        checkpoints = _checkpoint_count(combination["retriever"], combination["reranker"], 50)
        print(f"    {_combination_name(combination):<40} top_k=50 checkpoint files: {checkpoints}", flush=True)
    print(f"[run_all_multistep] phase 3: LaTeX tables {TABLE_SCRIPTS}", flush=True)

    results: list[tuple[str, str, int, float]] = []
    index = 0
    for phase_title, phase_combinations in (
        ("PHASE 1: COMBINATIONS WITH CACHED TOP_K=50 RERANKER SCORES", cached_combinations),
        ("PHASE 2: COMBINATIONS NEEDING NEW TOP_K=50 RERANKING", new_combinations),
    ):
        print(f"\n{BANNER}\n# {phase_title}\n{BANNER}", flush=True)
        for combination in phase_combinations:
            print(f"\n[run_all_multistep] ===== combination {_combination_name(combination)} =====", flush=True)
            top_50_failed = False
            for step in combination["steps"]:
                index += 1
                if top_50_failed:
                    print(
                        f"[run_all_multistep] ({index}/{total}) SKIPPED {step['script']}: top_k=50 failed for "
                        f"{_combination_name(combination)}, so smaller top_k would trigger new reranking",
                        flush=True,
                    )
                    results.append((step["script"], "skipped", -1, 0.0))
                    continue
                mode, returncode, elapsed = _run_step(index, total, step, started_at)
                results.append((step["script"], mode, returncode, elapsed))
                if step["top_k"] == 50 and returncode != 0:
                    top_50_failed = True

    print(f"\n{BANNER}\n# PHASE 3: LATEX TABLES\n{BANNER}", flush=True)
    for script in TABLE_SCRIPTS:
        index += 1
        mode, returncode, elapsed = _run_step(index, total, {"script": script}, started_at)
        results.append((script, mode, returncode, elapsed))

    print("\n[run_all_multistep] summary:", flush=True)
    for script, mode, returncode, elapsed in results:
        if returncode == -1:
            status = "SKIPPED"
        elif returncode == 0:
            status = "OK"
        else:
            status = f"FAILED (exit {returncode})"
        print(f"  {status:<18} {mode:<14} {_format_seconds(elapsed):>12}  {script}", flush=True)
    failures = [script for script, _, returncode, _ in results if returncode != 0]
    print(
        f"[run_all_multistep] finished {len(results) - len(failures)}/{len(results)} steps successfully "
        f"in {_format_seconds(perf_counter() - started_at)}",
        flush=True,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
