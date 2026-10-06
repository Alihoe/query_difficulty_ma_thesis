# scripts/score_frozen_full_predictors_test.py
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
import traceback

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_frozen_predictors_test import HIT_AT_1_RETRIEVERS, IN_TOP_K_DEPTHS, RETRIEVERS, _current_signature

from query_difficulty.modeling import (
    _hit_at_1_model_artifact_status,
    _prediction_split_output_paths,
    load_or_create_calibrated_ordinal_test_predictions,
    load_or_create_hit_at_1_test_predictions,
    load_or_create_in_top_k_test_predictions,
    load_or_create_max_depth_test_predictions,
)
from query_difficulty.modeling_tuned_qpp_lightweight import load_or_create_lightweight_tuned_qpp_test_predictions
from query_difficulty.runtime import build_runtime


def _jobs(include_other_hit_at_1_retrievers: bool = True) -> list[dict]:
    jobs = []
    for retriever in RETRIEVERS:
        jobs.append({"family": "hit_at_1", "predictor_name": "hit_at_1", "retriever": retriever, "k": None})
        jobs.append({"family": "in_top_max_depth", "predictor_name": "in_top_max_depth", "retriever": retriever, "k": 1000})
        jobs += [{"family": "in_top_k", "predictor_name": f"in_top_k{k}", "retriever": retriever, "k": k} for k in IN_TOP_K_DEPTHS]
        jobs.append({"family": "calibrated_ordinal", "predictor_name": "calibrated_ordinal", "retriever": retriever, "k": None})
        jobs.append({"family": "tuned_qpp", "predictor_name": "tuned_qpp", "retriever": retriever, "k": None})
    if include_other_hit_at_1_retrievers:
        jobs += [{"family": "hit_at_1", "predictor_name": "hit_at_1", "retriever": r, "k": None} for r in HIT_AT_1_RETRIEVERS if r not in RETRIEVERS]
    return jobs


def _score(runtime, job: dict, language: str):
    common = {"store": runtime.store, "console": runtime.console, "features": runtime.features, "retrieval": runtime.retrieval, "language": language, "retriever_name": job["retriever"]}
    family = job["family"]
    if family == "hit_at_1":
        payload, reason = _hit_at_1_model_artifact_status(runtime.store, job["retriever"])
        if payload is None:
            raise RuntimeError(f"full hit@1 model is not current and would be retrained ({reason}); skipped")
        return load_or_create_hit_at_1_test_predictions(**common)
    if family == "in_top_max_depth":
        return load_or_create_max_depth_test_predictions(**common, predictor_name="in_top_max_depth")
    if family == "in_top_k":
        return load_or_create_in_top_k_test_predictions(**common, target_depth=job["k"], predictor_name_prefix="in_top_k")
    if family == "calibrated_ordinal":
        return load_or_create_calibrated_ordinal_test_predictions(**common, predictor_name="calibrated_ordinal")
    return load_or_create_lightweight_tuned_qpp_test_predictions(**common, predictor_name="tuned_qpp")


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
    include_other_hit_at_1_retrievers: bool = typer.Option(True, help="Also score full Hit@1 for the six other dense retrievers."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    store = runtime.store
    jobs = _jobs(include_other_hit_at_1_retrievers)

    print(f"Frozen full-predictor test scoring for {language}: {len(jobs)} predictor(s) for {', '.join(sorted({job['retriever'] for job in jobs}))}")
    print("Models are only loaded and applied; no training, tuning, calibration or threshold change happens here.")
    print("Tuned QPP rebuilds its isotonic score-to-size mapping from train-split artifacts only, as for the lightweight predictor.")
    signatures = {}
    for job in jobs:
        try:
            signatures[(job["predictor_name"], job["retriever"])] = _current_signature(store, job)
        except FileNotFoundError as error:
            signatures[(job["predictor_name"], job["retriever"])] = None
            print(f"  missing frozen model for {job['predictor_name']} / {job['retriever']}: {error}")

    failures = []
    overall_started_at = perf_counter()
    for index, job in enumerate(jobs, start=1):
        tag = f"{job['predictor_name']} / {job['retriever']}"
        before = signatures[(job["predictor_name"], job["retriever"])]
        if before is None:
            failures.append((tag, "frozen model file missing"))
            print(f"[{index}/{len(jobs)}] {tag}: SKIPPED (frozen model file missing)")
            continue
        paths = _prediction_split_output_paths(store, job["predictor_name"], language, job["retriever"], "test")
        state = "cached file present" if paths["prediction"].exists() else "no cached file"
        print(f"[{index}/{len(jobs)}] {tag} ({state})")
        started_at = perf_counter()
        try:
            frame = _score(runtime, job, language)
        except Exception as error:
            failures.append((tag, f"{type(error).__name__}: {error}"))
            print(f"    FAILED: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue
        if _current_signature(store, job) != before:
            raise RuntimeError(f"Frozen model changed during scoring: {tag}")
        print(f"    {len(frame)} test queries, {perf_counter() - started_at:.1f}s -> {paths['prediction'].relative_to(ROOT)}")

    print("")
    print(f"Scored {len(jobs) - len(failures)} of {len(jobs)} in {perf_counter() - overall_started_at:.1f}s. All frozen model files are unchanged.")
    for tag, reason in failures:
        print(f"  not scored: {tag}: {reason}")
    if failures:
        print("Unscored predictors are reported as not evaluated by scripts/evaluate_frozen_predictors_test.py.")


if __name__ == "__main__":
    typer.run(main)
