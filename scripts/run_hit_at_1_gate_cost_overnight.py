from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import typer

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
STEPS = (
    ("repeated fixed 100-query workload", "run_hit_at_1_gate_cost_benchmark_subset_100.py"),
    ("robustness across independent 100-query subsets", "run_hit_at_1_gate_cost_robustness_subsets.py"),
)
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _keep_awake(enabled: bool) -> None:
    if sys.platform != "win32":
        return
    import ctypes

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if enabled else ES_CONTINUOUS
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def _other_python_processes() -> list[str]:
    try:
        import psutil
    except ImportError:
        return []
    own = {os.getpid(), os.getppid()}
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (process.info.get("name") or "").lower()
        if not name.startswith("python") or process.info["pid"] in own:
            continue
        command = " ".join(process.info.get("cmdline") or [])
        found.append(f"PID {process.info['pid']}: {command}")
    return found


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    skip_robustness: bool = typer.Option(False, help="Only run the repeated fixed-workload benchmark."),
) -> None:
    _log("Overnight cost benchmark: two timing experiments, run one after the other")
    others = _other_python_processes()
    if others:
        _log("WARNING: other Python processes are running and will distort the timings (nothing is stopped):")
        for line in others:
            _log(f"  {line}")
    _keep_awake(True)
    results = []
    try:
        steps = STEPS[:1] if skip_robustness else STEPS
        for index, (label, script) in enumerate(steps, start=1):
            _log(f"Step {index}/{len(steps)}: {label} ({script})")
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, "-u", str(SCRIPTS_DIR / script), "--config", config], cwd=ROOT)
            elapsed = time.perf_counter() - started
            results.append((label, completed.returncode, elapsed))
            _log(f"Step {index}/{len(steps)} finished with exit code {completed.returncode} after {elapsed / 3600:.2f} h")
    finally:
        _keep_awake(False)
    _log("Summary:")
    for label, code, elapsed in results:
        _log(f"  {label}: {'OK' if code == 0 else f'FAILED (exit code {code})'}, {elapsed / 3600:.2f} h")
    if any(code != 0 for _, code, _ in results):
        _log("At least one step failed; rerun this script to resume (completed runs are kept).")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
