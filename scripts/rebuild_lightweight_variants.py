# scripts/rebuild_lightweight_variants.py
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import typer

ROOT = Path(__file__).resolve().parents[1]
STATE_DIRECTORY = ROOT / "artifacts" / "logs" / "rebuild_lightweight_variants"
STATE_PATH = STATE_DIRECTORY / "state.json"
LANES = {
    "all": None,
    "multi_depth": {"in_top_max_depth_full_no_consensus", "in_top_max_depth_reselection", "in_top_k_full_no_consensus", "in_top_k_reselection"},
    "single_depth": {
        "tuned_qpp_full_no_consensus", "tuned_qpp_reselection", "calibrated_ordinal_full_no_consensus",
        "calibrated_ordinal_reselection", "hit_at_1_full_no_consensus", "hit_at_1_reselection",
    },
}
IN_TOP_K_FAMILIES = [f"in_top_k{depth}" for depth in (5, 10, 20, 30, 50, 60, 100, 120, 200, 1000)]

STEPS = [
    ("in_top_max_depth_full_no_consensus", ["scripts/train_full_predictors_without_consensus.py", "--family", "in_top_max_depth"],
     "consensus-free full In-Max-Depth model (Qwen)"),
    ("in_top_max_depth_reselection", ["scripts/reselect_lightweight_in_top_max_depth_features_grouped.py"],
     "lightweight In-Max-Depth selection and training"),
    ("in_top_k_full_no_consensus", ["scripts/train_full_predictors_without_consensus.py", *[argument for family in IN_TOP_K_FAMILIES for argument in ("--family", family)]],
     "consensus-free full in-top-k models for all k (Qwen)"),
    ("in_top_k_reselection", ["scripts/reselect_lightweight_in_top_k_sweep_features.py"],
     "lightweight binary (in-top-k) selection and training for all k"),
    ("tuned_qpp_full_no_consensus", ["scripts/train_full_predictors_without_consensus.py", "--family", "tuned_qpp"],
     "consensus-free full tuned QPP model (Qwen)"),
    ("tuned_qpp_reselection", ["scripts/reselect_lightweight_tuned_qpp_features.py"],
     "lightweight tuned QPP selection (both heads) and training"),
    ("calibrated_ordinal_full_no_consensus", ["scripts/train_full_predictors_without_consensus.py", "--family", "calibrated_ordinal"],
     "consensus-free full calibrated ordinal model (Qwen)"),
    ("calibrated_ordinal_reselection", ["scripts/reselect_lightweight_calibrated_ordinal_features.py"],
     "lightweight calibrated ordinal selection (one feature set per head) and training"),
    ("hit_at_1_full_no_consensus", ["scripts/train_full_predictors_without_consensus.py", "--family", "hit_at_1"],
     "consensus-free full Hit@1 model (Qwen)"),
    ("hit_at_1_reselection", ["scripts/reselect_lightweight_hit_at_1_features.py"],
     "lightweight Hit@1 selection, adoption in the Hit@1 manifest and training"),
]


def _state_path(lane: str) -> Path:
    return STATE_PATH if lane == "all" else STATE_DIRECTORY / f"state_{lane}.json"


def _load_state(lane: str) -> dict:
    path = _state_path(lane)
    if not path.exists():
        return {"completed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _completed_in_any_lane() -> dict:
    completed = {}
    for lane in LANES:
        completed.update(_load_state(lane).get("completed", {}))
    return completed


def _save_state(lane: str, state: dict) -> None:
    STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = _state_path(lane)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _run_step(name: str, command: list[str]) -> int:
    log_path = STATE_DIRECTORY / f"{name}.log"
    STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} start: {' '.join(command)} =====\n")
        process = subprocess.Popen(
            [sys.executable, *command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def main(
    lane: str = typer.Option(
        "all",
        help="Which steps to run: 'all' (default), 'multi_depth' (In-Max-Depth and in-top-k) or 'single_depth' "
        "(QPP, calibrated ordinal, Hit@1). Two lanes can run in parallel; each keeps its own progress file and a step "
        "completed in any lane counts as completed.",
    ),
    restart: bool = typer.Option(False, help="Forget the recorded progress of this lane and run its steps again."),
) -> None:
    if lane not in LANES:
        raise typer.BadParameter(f"Unknown lane '{lane}'; choose one of {sorted(LANES)}")
    steps = [step for step in STEPS if LANES[lane] is None or step[0] in LANES[lane]]
    state = {"completed": {}} if restart else _load_state(lane)
    _log("Rebuild of all lightweight variants (one process at a time, resumable)")
    _log(f"Lane: {lane}; progress file: {_state_path(lane)}; one log per step in {STATE_DIRECTORY}")
    _log(
        "Memory: each step needs up to about 2-3 GB. Run this without other training or reranking jobs in "
        "parallel; the machine has 15 GB and previous parallel runs were stopped for lack of memory."
    )
    already_completed = {} if restart else _completed_in_any_lane()
    for index, (name, command, description) in enumerate(steps, start=1):
        status = "done" if name in already_completed else "pending"
        _log(f"  {index:2d}/{len(steps)} [{status}] {name}: {description}")
    total_started = time.perf_counter()
    for index, (name, command, description) in enumerate(steps, start=1):
        completed_elsewhere = {} if restart else _completed_in_any_lane()
        if name in state["completed"] or name in completed_elsewhere:
            finished = (state["completed"].get(name) or completed_elsewhere[name])["finished"]
            _log(f"Skipping step {index}/{len(steps)} {name}: completed at {finished}")
            continue
        _log(f"===== Step {index}/{len(steps)}: {name} - {description} =====")
        _log(f"Command: python {' '.join(command)}")
        started = time.perf_counter()
        return_code = _run_step(name, command)
        elapsed_hours = (time.perf_counter() - started) / 3600
        if return_code != 0:
            state.setdefault("failed", {})[name] = {"finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "return_code": return_code}
            _save_state(lane, state)
            _log(
                f"Step {name} failed with exit code {return_code} after {elapsed_hours:.2f} h. See "
                f"{STATE_DIRECTORY / (name + '.log')}. Fix the cause and run this script again; completed steps are "
                "skipped and the selection scripts resume from their checkpoints."
            )
            raise typer.Exit(code=return_code)
        state["completed"][name] = {"finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "hours": round(elapsed_hours, 3)}
        state.get("failed", {}).pop(name, None)
        _save_state(lane, state)
        _log(f"Step {index}/{len(steps)} {name} finished in {elapsed_hours:.2f} h")
    _log(f"All {len(steps)} steps of lane {lane} completed ({(time.perf_counter() - total_started) / 3600:.2f} h in this run)")
    _log(
        "Next: regenerate the lightweight tables and rerun the multistage and OOD experiments "
        "(see lightweight_predictor_variants.md, section 'Rebuild procedure')."
    )


if __name__ == "__main__":
    typer.run(main)
