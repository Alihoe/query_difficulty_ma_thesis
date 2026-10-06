from __future__ import annotations

import itertools
import math
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import typer

from _hit_at_1_gate_cost_common import (
    DEFAULT_LANGUAGE,
    DEFAULT_PREDICTOR_NAME,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVER,
    DEFAULT_TOP_K,
    append_jsonl,
    default_request_size,
    length_representativeness,
    load_deployed_gate,
    pair_token_lengths,
    read_json,
    read_jsonl,
    report_dir,
    reranker_spec,
    run_environment_snapshot,
    split_workload,
    stratified_subset,
    table_paths,
    write_json,
    write_robustness_table,
)
from run_hit_at_1_gate_cost_benchmark import STRATEGY_ADAPTIVE, STRATEGY_FULL, StrategyRunner
from query_difficulty.hit_at_1_gate_fast import (
    COMPONENT_GATE_TOTAL,
    COMPONENT_ONLINE_TOTAL,
    COMPONENT_RERANK,
    environment_metadata,
    load_jina_reranker,
    run_gate,
    score_pairs,
)
from query_difficulty.runtime import build_runtime

DEFAULT_ROBUSTNESS_SUBSET_SIZE = 100
DEFAULT_ROBUSTNESS_SUBSET_COUNT = 5
DEFAULT_ROBUSTNESS_WARMUP_REPETITIONS = 1


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else math.nan,
        "sd": float(array.std(ddof=1)) if array.size > 1 else math.nan,
        "min": float(array.min()) if array.size else math.nan,
        "max": float(array.max()) if array.size else math.nan,
    }


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Language code."),
    retriever: str = typer.Option(DEFAULT_RETRIEVER, help="First-stage dense retriever."),
    predictor_name: str = typer.Option(DEFAULT_PREDICTOR_NAME, help="Deployed predictor namespace."),
    subset_size: int = typer.Option(DEFAULT_ROBUSTNESS_SUBSET_SIZE, help="Queries per independently sampled stratified subset."),
    subset_count: int = typer.Option(DEFAULT_ROBUSTNESS_SUBSET_COUNT, help="Number of subsets when --seed is not given."),
    seed: Optional[list[int]] = typer.Option(None, "--seed", help="Subset seeds (repeatable). Default: runtime.random_seed and the following consecutive integers."),
    warmup_repetitions: int = typer.Option(DEFAULT_ROBUSTNESS_WARMUP_REPETITIONS, help="Untimed warm-up runs of both strategies on the first subset."),
    reranker_model: str = typer.Option(DEFAULT_RERANKER_MODEL, help="Jina reranker model."),
    reranker_max_length: int = typer.Option(DEFAULT_RERANKER_MAX_LENGTH, help="Reranker maximum sequence length."),
    reranker_batch_size: int = typer.Option(DEFAULT_RERANKER_BATCH_SIZE, help="Reranker batch size."),
    request_size: Optional[int] = typer.Option(None, help="Pairs per compute_score request (default: thesis setting)."),
    gate_batch_size: int = typer.Option(0, help="Queries per gate batch; 0 processes the whole subset in one batch."),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Reranking depth."),
    device: str = typer.Option("cpu", help="Reranker device (thesis setup: CPU with dynamic int8)."),
    fresh: bool = typer.Option(False, help="Discard previous robustness results instead of resuming."),
    output_dir: Optional[str] = typer.Option(None, help="Override the output directory."),
    write_tables: bool = typer.Option(True, help="Write the LaTeX robustness table to latex/tables/efficiency."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    log = console.log
    console.section("Robustness of the online saving across independently sampled stratified subsets")
    retriever = runtime.retrieval.resolve_retriever_name(retriever)
    spec = reranker_spec(reranker_model, reranker_max_length, reranker_batch_size)
    request_size = int(request_size or default_request_size(reranker_batch_size))
    base_seed = int(runtime.config.runtime.random_seed)
    seeds = [int(value) for value in seed] if seed else [base_seed + offset for offset in range(int(subset_count))]
    parent = report_dir(runtime, language, retriever, spec, top_k, predictor_name)
    target_dir = Path(output_dir).resolve() if output_dir else parent / f"robustness_subsets_{int(subset_size)}"
    test_key = f"{language}_test"
    threads_at_start = int(torch.get_num_threads())
    environment = environment_metadata(device)
    log(f"Output directory: {target_dir}")
    log(f"Seeds: {seeds}; {subset_size} queries per subset; {warmup_repetitions} warm-up run(s); one Full and one Adaptive run per subset")

    settings = {
        "predictor_name": predictor_name,
        "subset_size": int(subset_size),
        "seeds": seeds,
        "warmup_repetitions": int(warmup_repetitions),
        "batch_size": int(reranker_batch_size),
        "max_length": int(reranker_max_length),
        "request_size": request_size,
        "gate_batch_size": int(gate_batch_size),
        "top_k": int(top_k),
        "device": device,
        "torch_num_threads": threads_at_start,
    }
    records_path = target_dir / "subset_runs.jsonl"
    settings_path = target_dir / "settings.json"
    if fresh and target_dir.exists():
        shutil.rmtree(target_dir)
    if settings_path.exists():
        previous = read_json(settings_path)
        comparable = {key: value for key, value in settings.items() if key not in ("seeds", "warmup_repetitions")}
        if {key: value for key, value in previous.items() if key not in ("seeds", "warmup_repetitions")} != comparable:
            raise RuntimeError(f"Existing results in {target_dir} were recorded with different settings {previous}; rerun with --fresh")
    write_json(settings_path, settings)
    write_json(target_dir / "environment.json", environment)

    console.section("Preparation (untimed)")
    gate = load_deployed_gate(runtime, language, retriever, predictor_name, test_key, log)
    full_workload = split_workload(runtime, test_key, retriever, top_k)
    full_result = run_gate(gate.state, full_workload.inputs)
    population_skip_rate = float(full_result.skip.mean())
    log(f"Full-test gate skip rate: {population_skip_rate:.4f} ({int(full_result.skip.sum())}/{len(full_result.skip)})")
    query_text_by_id = full_workload.queries.set_index("query_id")["text"].astype(str).to_dict()
    document_text_by_id = {
        int(document_id): str(text)
        for document_id, text in zip(runtime.retrieval.document_ids.tolist(), runtime.retrieval.document_scope_bundle(retriever)["texts"], strict=True)
    }
    texts = (query_text_by_id, document_text_by_id)
    log(f"Loading {spec.model_name} on {device} (dynamic int8={spec.quantize_dynamic_int8}, batch={reranker_batch_size}, max_length={reranker_max_length})")
    reranker = load_jina_reranker(spec.model_name, spec.trust_remote_code, spec.use_flash_attn, spec.quantize_dynamic_int8, device)
    first_id = int(full_workload.inputs.query_ids[0])
    score_pairs(reranker, [[query_text_by_id[first_id], document_text_by_id[int(full_workload.top_document_ids[0, 0])]]], reranker_batch_size, reranker_max_length, request_size)

    lengths_path = target_dir / "full_test_pair_token_lengths.npy"
    if lengths_path.exists():
        per_pair = np.load(lengths_path)
    else:
        per_pair = pair_token_lengths(reranker._tokenizer, full_workload, query_text_by_id, document_text_by_id, top_k, reranker_max_length, log)
        target_dir.mkdir(parents=True, exist_ok=True)
        np.save(lengths_path, per_pair)

    subsets: dict[int, dict[str, Any]] = {}
    for subset_seed in seeds:
        path = target_dir / "subsets" / f"seed_{subset_seed}.json"
        existing = read_json(path) if path.exists() else None
        if existing is not None and existing.get("model_signature") == gate.model_signature and existing.get("requested_size") == int(subset_size):
            subsets[subset_seed] = existing
            continue
        subset = stratified_subset(full_workload.inputs.query_ids, full_result.skip, subset_size, subset_seed)
        subset["model_signature"] = gate.model_signature
        mask = np.isin(full_workload.inputs.query_ids, np.asarray(subset["query_ids"], dtype=np.int64))
        subset["representativeness"] = length_representativeness(per_pair, mask, ~full_result.skip)
        write_json(path, subset)
        subsets[subset_seed] = subset
        log(
            f"  seed {subset_seed}: {subset['size']} queries, skip rate {subset['subset_skip_rate']:.3f}, mean tokens/query "
            f"{subset['representativeness']['all_queries']['subset_per_query_total']['mean']:.0f} "
            f"(KS p={subset['representativeness']['all_queries']['ks_per_query_total']['p_value']:.3f})"
        )
    overlaps = {
        f"{a}-{b}": len(set(subsets[a]["query_ids"]) & set(subsets[b]["query_ids"]))
        for a, b in itertools.combinations(seeds, 2)
    }
    log(f"Pairwise subset overlaps (shared queries): {overlaps}")

    records = read_jsonl(records_path)
    completed = {(record["subset_seed"], record["strategy"]) for record in records if record["phase"] == "measured"}
    pending = [(subset_seed, strategy) for subset_seed in seeds for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE) if (subset_seed, strategy) not in completed]
    log(f"{len(pending)} measured strategy runs pending")
    runner_settings = {
        "gate_batch_size": int(gate_batch_size),
        "device": device,
        "top_k": int(top_k),
        "batch_size": int(reranker_batch_size),
        "max_length": int(reranker_max_length),
        "request_size": request_size,
    }
    full_positions = pd.Index(full_workload.inputs.query_ids)

    def subset_runner(subset_seed: int) -> tuple[StrategyRunner, np.ndarray, Any]:
        query_ids = np.asarray(subsets[subset_seed]["query_ids"], dtype=np.int64)
        workload = split_workload(runtime, test_key, retriever, top_k, query_ids=query_ids)
        expected_skip = full_result.skip[full_positions.get_indexer(query_ids)]
        return StrategyRunner(gate, workload, reranker, texts, runner_settings, log), expected_skip, workload

    def execute(phase: str, subset_seed: int, strategy: str, sequence_index: int, position: int) -> None:
        runner, expected_skip, workload = subset_runner(subset_seed)
        label = f"{phase} seed {subset_seed} {strategy}"
        log(f"Starting {label}")
        if strategy == STRATEGY_FULL:
            timings, details = runner.run_full(label)
        else:
            timings, details = runner.run_adaptive(label)
            if not np.array_equal(~details["rerank_mask"], expected_skip):
                raise RuntimeError("The gate produced different skip decisions than in preparation; aborting")
        query_count = len(workload.inputs.query_ids)
        append_jsonl(
            records_path,
            {
                "experiment": "independent_stratified_subsets",
                "phase": phase,
                "subset_seed": subset_seed,
                "subset_file": str(target_dir / "subsets" / f"seed_{subset_seed}.json"),
                "sequence_index": sequence_index,
                "position_in_pair": position,
                "strategy": strategy,
                "queries": query_count,
                "skipped_queries": query_count - int(details["reranked_queries"]),
                "reranked_queries": int(details["reranked_queries"]),
                "reranker_calls": int(details["reranker_calls"]),
                "gate_seconds": timings.get(COMPONENT_GATE_TOTAL, 0.0),
                "rerank_seconds": timings[COMPONENT_RERANK],
                "online_seconds": timings[COMPONENT_ONLINE_TOTAL],
                "components_seconds": timings,
                "environment": run_environment_snapshot(),
            },
        )
        log(f"Finished {label}: online {timings[COMPONENT_ONLINE_TOTAL]:.1f}s, reranked {details['reranked_queries']}/{query_count}")

    if pending:
        console.section("Warm-up (untimed for the analysis)")
        for warmup_index in range(int(warmup_repetitions)):
            order = (STRATEGY_FULL, STRATEGY_ADAPTIVE) if warmup_index % 2 == 0 else (STRATEGY_ADAPTIVE, STRATEGY_FULL)
            for position, strategy in enumerate(order, start=1):
                execute("warmup", seeds[0], strategy, -1 - warmup_index, position)
        console.section("Measured runs: one Full and one Adaptive run per subset, alternating order")
        for index, subset_seed in enumerate(seeds):
            order = (STRATEGY_FULL, STRATEGY_ADAPTIVE) if index % 2 == 0 else (STRATEGY_ADAPTIVE, STRATEGY_FULL)
            for position, strategy in enumerate(order, start=1):
                if (subset_seed, strategy) in completed:
                    continue
                execute("measured", subset_seed, strategy, index, position)

    if int(torch.get_num_threads()) != threads_at_start:
        raise RuntimeError("The torch thread configuration changed during the robustness runs")

    console.section("Summary")
    measured = [record for record in read_jsonl(records_path) if record["phase"] == "measured"]
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for record in measured:
        by_seed.setdefault(int(record["subset_seed"]), {})[record["strategy"]] = record
    rows = []
    for subset_seed in seeds:
        entry = by_seed.get(subset_seed, {})
        if STRATEGY_FULL not in entry or STRATEGY_ADAPTIVE not in entry:
            continue
        full, adaptive = entry[STRATEGY_FULL], entry[STRATEGY_ADAPTIVE]
        queries = int(full["queries"])
        rows.append(
            {
                "subset_seed": subset_seed,
                "queries": queries,
                "skipped_queries": int(adaptive["skipped_queries"]),
                "reranked_queries": int(adaptive["reranked_queries"]),
                "full_reranker_calls": int(full["reranker_calls"]),
                "adaptive_reranker_calls": int(adaptive["reranker_calls"]),
                "full_online_seconds": float(full["online_seconds"]),
                "adaptive_gate_seconds": float(adaptive["gate_seconds"]),
                "adaptive_rerank_seconds": float(adaptive["rerank_seconds"]),
                "adaptive_online_seconds": float(adaptive["online_seconds"]),
                "full_online_seconds_per_query": float(full["online_seconds"]) / queries,
                "adaptive_online_seconds_per_query": float(adaptive["online_seconds"]) / queries,
                "adaptive_gate_seconds_per_query": float(adaptive["gate_seconds"]) / queries,
                "adaptive_reranked_share": int(adaptive["reranked_queries"]) / queries,
                "relative_online_saving": 1.0 - float(adaptive["online_seconds"]) / float(full["online_seconds"]),
                "full_first": int(full["position_in_pair"]) == 1,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(target_dir / "robustness_per_subset.csv", index=False)
    statistics = {
        key: _statistics(frame[key].tolist() if len(frame) else [])
        for key in (
            "full_online_seconds_per_query",
            "adaptive_online_seconds_per_query",
            "adaptive_gate_seconds_per_query",
            "adaptive_reranked_share",
            "relative_online_saving",
        )
    }
    summary = {
        "experiment": "independent stratified subsets (separate from the repeated fixed-workload timing)",
        "variability": "across subsets: combines workload composition and execution variability",
        "subset_size": int(subset_size),
        "seeds": seeds,
        "completed_subsets": int(len(frame)),
        "warmup_repetitions": int(warmup_repetitions),
        "top_k": int(top_k),
        "population_skip_rate": population_skip_rate,
        "pairwise_subset_overlaps": overlaps,
        "statistics": statistics,
        "settings": settings,
    }
    write_json(target_dir / "robustness_summary.json", summary)
    if len(frame):
        saving = statistics["relative_online_saving"]
        log(
            f"Relative online saving across {len(frame)} subsets: mean {100 * saving['mean']:.2f}%, SD {100 * saving['sd']:.2f}%, "
            f"min {100 * saving['min']:.2f}%, max {100 * saving['max']:.2f}%"
        )
    if write_tables and len(frame) >= 2:
        path = table_paths(runtime, language, retriever, predictor_name)["robustness"]
        write_robustness_table(path, summary, retriever)
        log(f"Wrote robustness table {path}")
    log(f"Outputs in {target_dir}")


if __name__ == "__main__":
    typer.run(main)
