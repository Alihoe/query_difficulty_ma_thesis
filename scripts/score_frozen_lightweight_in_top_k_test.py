# scripts/score_frozen_lightweight_in_top_k_test.py
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.modeling import _in_top_k_model_path, _prediction_split_output_paths, _sha256_path
from query_difficulty.modeling_in_top_k_sweep_lightweight import (
    LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX,
    load_or_create_lightweight_in_top_k_test_predictions,
)
from query_difficulty.runtime import build_runtime

RETRIEVERS = ["bm25", "qwen3_embedding_0_6b"]
DEPTHS = [5, 10, 20, 30, 50, 60, 100, 120, 200, 1000]


def main(
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    store = runtime.store
    jobs = [(retriever, depth) for retriever in RETRIEVERS for depth in DEPTHS]

    print(f"Frozen lightweight in-top-k test scoring for {language}: {len(jobs)} predictor(s)")
    print("Models are only loaded and applied; no training, tuning, calibration or threshold change happens here.")
    signatures = {}
    missing = []
    for retriever, depth in jobs:
        model_path = _in_top_k_model_path(store, retriever, depth, predictor_name_prefix=LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX)
        if not model_path.exists():
            missing.append(str(model_path))
            continue
        signatures[(retriever, depth)] = _sha256_path(model_path)
    if missing:
        print("Aborting before any scoring, frozen model file(s) missing:")
        for path in missing:
            print(f"  {path}")
        raise typer.Exit(code=1)

    for index, (retriever, depth) in enumerate(jobs, start=1):
        predictor_name = f"{LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX}{depth}"
        paths = _prediction_split_output_paths(store, predictor_name, language, retriever, "test")
        state = "cached file present" if paths["prediction"].exists() else "no cached file"
        print(f"[{index}/{len(jobs)}] {predictor_name} / {retriever} ({state})")
        started_at = perf_counter()
        frame = load_or_create_lightweight_in_top_k_test_predictions(
            store=store,
            console=runtime.console,
            features=runtime.features,
            retrieval=runtime.retrieval,
            language=language,
            retriever_name=retriever,
            target_depth=depth,
        )
        model_path = _in_top_k_model_path(store, retriever, depth, predictor_name_prefix=LIGHTWEIGHT_IN_TOP_K_PREDICTOR_NAME_PREFIX)
        if _sha256_path(model_path) != signatures[(retriever, depth)]:
            raise RuntimeError(f"Frozen model changed during scoring: {model_path}")
        print(
            f"    {len(frame)} test queries, {int(frame['predicted_in_top_k'].sum())} predicted positive, "
            f"{perf_counter() - started_at:.1f}s -> {paths['prediction'].relative_to(ROOT)}"
        )

    print("Done. All frozen model files are unchanged.")


if __name__ == "__main__":
    typer.run(main)
