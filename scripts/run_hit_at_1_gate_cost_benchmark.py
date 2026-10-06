from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import typer

from _hit_at_1_gate_cost_common import (
    DEFAULT_LANGUAGE,
    DEFAULT_METRIC_K,
    DEFAULT_PREDICTOR_NAME,
    DEFAULT_REPETITIONS,
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RETRIEVER,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_WARMUP_REPETITIONS,
    cached_top_k_scores,
    default_request_size,
    effectiveness,
    load_deployed_gate,
    load_or_create_subset,
    multistep_run_dir,
    read_json,
    report_dir,
    reranker_spec,
    split_workload,
    VARIABILITY_LABEL,
    append_jsonl,
    length_representativeness,
    pair_token_lengths,
    run_environment_snapshot,
    write_available_tables,
    write_cost_summary,
    write_json,
)
from query_difficulty.hit_at_1_gate_fast import (
    COMPONENT_GATE_TOTAL,
    COMPONENT_ONLINE_TOTAL,
    COMPONENT_RERANK,
    GATE_COMPONENTS,
    GateInputs,
    build_pairs,
    environment_metadata,
    gold_reciprocal_ranks,
    load_jina_reranker,
    reranked_reciprocal_ranks,
    run_gate,
    score_pairs,
    synchronize_device,
)
from query_difficulty.runtime import build_runtime

STRATEGY_FULL = "full"
STRATEGY_ADAPTIVE = "adaptive"
RAW_FIELDS = [
    "phase",
    "repetition",
    "sequence_index",
    "position_in_pair",
    "strategy",
    "component",
    "seconds",
    "queries",
    "reranked_queries",
    "reranker_calls",
]


def _slice_inputs(inputs: GateInputs, rows: np.ndarray) -> GateInputs:
    return GateInputs(
        query_ids=inputs.query_ids[rows],
        query_texts=[inputs.query_texts[row] for row in rows],
        query_keys=[inputs.query_keys[row] for row in rows],
        query_embeddings=np.ascontiguousarray(inputs.query_embeddings[rows]),
        retrieval_document_ids=inputs.retrieval_document_ids[rows],
        retrieval_scores=inputs.retrieval_scores[rows],
    )


class StrategyRunner:
    def __init__(self, gate, workload, reranker, texts, settings: dict[str, Any], log) -> None:
        self.gate = gate
        self.workload = workload
        self.reranker = reranker
        self.query_text_by_id, self.document_text_by_id = texts
        self.settings = settings
        self.log = log
        query_count = len(workload.inputs.query_ids)
        batch_size = int(settings["gate_batch_size"]) or query_count
        self.gate_batches = [
            (np.arange(start, min(start + batch_size, query_count)), _slice_inputs(workload.inputs, np.arange(start, min(start + batch_size, query_count))))
            for start in range(0, query_count, batch_size)
        ]

    def _progress(self, label: str):
        started = time.perf_counter()

        def report(done: int, total: int) -> None:
            elapsed = time.perf_counter() - started
            remaining = elapsed / done * (total - done) if done else math.nan
            self.log(f"    {label}: {done:,}/{total:,} pairs scored, {elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min remaining")

        return report

    def run_full(self, label: str) -> tuple[dict[str, float], dict[str, Any]]:
        device = self.settings["device"]
        top_k = self.settings["top_k"]
        progress = self._progress(label)
        synchronize_device(device)
        started = time.perf_counter()
        pairs = build_pairs(self.workload.inputs.query_ids, self.workload.top_document_ids[:, :top_k], self.query_text_by_id, self.document_text_by_id)
        scores = score_pairs(self.reranker, pairs, self.settings["batch_size"], self.settings["max_length"], self.settings["request_size"], progress)
        synchronize_device(device)
        elapsed = time.perf_counter() - started
        query_count = len(self.workload.inputs.query_ids)
        timings = {COMPONENT_RERANK: elapsed, COMPONENT_ONLINE_TOTAL: elapsed}
        details = {
            "reranked_queries": query_count,
            "reranker_calls": len(pairs),
            "scores": scores.reshape(query_count, top_k),
            "rerank_mask": np.ones(query_count, dtype=bool),
        }
        return timings, details

    def run_adaptive(self, label: str) -> tuple[dict[str, float], dict[str, Any]]:
        device = self.settings["device"]
        top_k = self.settings["top_k"]
        progress = self._progress(label)
        component_seconds = {component: 0.0 for component in GATE_COMPONENTS}
        skip = np.empty(len(self.workload.inputs.query_ids), dtype=bool)
        synchronize_device(device)
        started = time.perf_counter()
        for rows, batch_inputs in self.gate_batches:
            result = run_gate(self.gate.state, batch_inputs)
            skip[rows] = result.skip
            for component in GATE_COMPONENTS:
                component_seconds[component] += result.timings[component]
        rerank_mask = ~skip
        synchronize_device(device)
        gate_finished = time.perf_counter()
        pairs = build_pairs(
            self.workload.inputs.query_ids[rerank_mask],
            self.workload.top_document_ids[rerank_mask, :top_k],
            self.query_text_by_id,
            self.document_text_by_id,
        )
        scores = score_pairs(self.reranker, pairs, self.settings["batch_size"], self.settings["max_length"], self.settings["request_size"], progress)
        synchronize_device(device)
        finished = time.perf_counter()
        timings = dict(component_seconds)
        timings[COMPONENT_GATE_TOTAL] = gate_finished - started
        timings[COMPONENT_RERANK] = finished - gate_finished
        timings[COMPONENT_ONLINE_TOTAL] = finished - started
        details = {
            "reranked_queries": int(rerank_mask.sum()),
            "reranker_calls": len(pairs),
            "scores": scores.reshape(int(rerank_mask.sum()), top_k),
            "rerank_mask": rerank_mask,
        }
        return timings, details


def _read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_FIELDS)
    return pd.read_csv(path)


def _append_raw(path: Path, rows: list[dict[str, Any]]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _representativeness(tokenizer, full_workload, subset_rows: np.ndarray, full_rerank: np.ndarray, texts, top_k: int, max_length: int, log) -> dict[str, Any]:
    query_text_by_id, document_text_by_id = texts
    per_pair = pair_token_lengths(tokenizer, full_workload, query_text_by_id, document_text_by_id, top_k, max_length, log)
    subset_mask = np.zeros(len(per_pair), dtype=bool)
    subset_mask[subset_rows] = True
    report = length_representativeness(per_pair, subset_mask, full_rerank)
    for name in ("all_queries", "reranked_queries"):
        entry = report[name]
        log(
            f"  {name}: mean tokens/query full {entry['full_test_per_query_total']['mean']:.1f} vs subset "
            f"{entry['subset_per_query_total']['mean']:.1f}; KS(query totals) D={entry['ks_per_query_total']['statistic']:.4f}, "
            f"p={entry['ks_per_query_total']['p_value']:.4f}"
        )
    return report


def _score_agreement(
    fresh: np.ndarray,
    cached: np.ndarray,
    top_document_ids: np.ndarray,
    gold: np.ndarray,
    top_k: int,
    metric_k: int,
) -> dict[str, Any]:
    fresh = np.asarray(fresh, dtype=np.float64)
    cached = np.asarray(cached, dtype=np.float64)
    ranks = np.broadcast_to(np.arange(1, top_k + 1), fresh.shape)
    fresh_order = np.lexsort((ranks, -fresh), axis=1)
    cached_order = np.lexsort((ranks, -cached), axis=1)
    fresh_rr = reranked_reciprocal_ranks(top_document_ids[:, :top_k], fresh, gold, metric_k)
    cached_rr = reranked_reciprocal_ranks(top_document_ids[:, :top_k], cached, gold, metric_k)
    difference = np.abs(fresh - cached)
    return {
        "queries": int(len(fresh)),
        "pairs": int(fresh.size),
        "identical_score_share": float(np.mean(fresh == cached)),
        "max_abs_score_difference": float(difference.max()) if difference.size else 0.0,
        "mean_abs_score_difference": float(difference.mean()) if difference.size else 0.0,
        "queries_with_changed_top_k_order": int(np.sum(np.any(fresh_order != cached_order, axis=1))),
        "queries_with_changed_top_1": int(np.sum(fresh_order[:, 0] != cached_order[:, 0])),
        "queries_with_changed_rr_at_metric_k": int(np.sum(fresh_rr != cached_rr)),
        "mrr_fresh": float(fresh_rr.mean()) if len(fresh_rr) else math.nan,
        "mrr_cached": float(cached_rr.mean()) if len(cached_rr) else math.nan,
    }


def _aggregate(raw: pd.DataFrame, query_count: int) -> dict[str, Any]:
    measured = raw.loc[raw["phase"] == "measured"]
    strategies: dict[str, Any] = {}
    for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE):
        frame = measured.loc[measured["strategy"] == strategy]
        pivot = frame.pivot_table(index="repetition", columns="component", values="seconds", aggfunc="first")
        entry: dict[str, Any] = {"repetitions": int(len(pivot))}
        for component in pivot.columns:
            values = pivot[component].to_numpy(dtype=np.float64)
            entry[f"{component}_total_mean"] = float(values.mean())
            entry[f"{component}_total_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            entry[f"{component}_per_query_mean"] = float(values.mean() / query_count)
            entry[f"{component}_per_query_sd"] = float(values.std(ddof=1) / query_count) if len(values) > 1 else 0.0
        entry["online_total_mean"] = entry[f"{COMPONENT_ONLINE_TOTAL}_total_mean"]
        entry["online_total_sd"] = entry[f"{COMPONENT_ONLINE_TOTAL}_total_sd"]
        entry["online_per_query_mean"] = entry[f"{COMPONENT_ONLINE_TOTAL}_per_query_mean"]
        entry["online_per_query_sd"] = entry[f"{COMPONENT_ONLINE_TOTAL}_per_query_sd"]
        entry["rerank_per_query_mean"] = entry[f"{COMPONENT_RERANK}_per_query_mean"]
        entry["rerank_per_query_sd"] = entry[f"{COMPONENT_RERANK}_per_query_sd"]
        if f"{COMPONENT_GATE_TOTAL}_per_query_mean" in entry:
            entry["gate_per_query_mean"] = entry[f"{COMPONENT_GATE_TOTAL}_per_query_mean"]
            entry["gate_per_query_sd"] = entry[f"{COMPONENT_GATE_TOTAL}_per_query_sd"]
        calls = frame["reranker_calls"].dropna().unique()
        entry["reranker_calls"] = int(calls[0]) if len(calls) else 0
        reranked = frame["reranked_queries"].dropna().unique()
        entry["reranked_queries"] = int(reranked[0]) if len(reranked) else 0
        strategies[strategy] = entry
    totals = measured.loc[measured["component"] == COMPONENT_ONLINE_TOTAL].pivot_table(index="repetition", columns="strategy", values="seconds", aggfunc="first")
    totals = totals.dropna()
    savings = (1.0 - totals[STRATEGY_ADAPTIVE] / totals[STRATEGY_FULL]).to_numpy(dtype=np.float64)
    return {
        "strategies": strategies,
        "online_saving": {
            "definition": "1 - adaptive online total / full online total, computed per repetition",
            "per_repetition": savings.tolist(),
            "mean": float(savings.mean()) if savings.size else math.nan,
            "sd": float(savings.std(ddof=1)) if savings.size > 1 else 0.0,
        },
    }


def main(
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Language code."),
    retriever: str = typer.Option(DEFAULT_RETRIEVER, help="First-stage dense retriever."),
    predictor_name: str = typer.Option(DEFAULT_PREDICTOR_NAME, help="Deployed predictor namespace (e.g. lightweight/hit_at_1_no_training_rr)."),
    subset_size: int = typer.Option(DEFAULT_SUBSET_SIZE, help="Number of benchmark queries (stratified by the gate decision)."),
    full_test: bool = typer.Option(False, help="Benchmark the complete test set instead of the stratified subset."),
    seed: Optional[int] = typer.Option(None, help="Subset sampling seed (default: runtime.random_seed from the config)."),
    warmup_repetitions: int = typer.Option(DEFAULT_WARMUP_REPETITIONS, help="Untimed warm-up repetitions of both strategies."),
    repetitions: int = typer.Option(DEFAULT_REPETITIONS, help="Measured repetitions of both strategies."),
    reranker_model: str = typer.Option(DEFAULT_RERANKER_MODEL, help="Jina reranker model."),
    reranker_max_length: int = typer.Option(DEFAULT_RERANKER_MAX_LENGTH, help="Reranker maximum sequence length."),
    reranker_batch_size: int = typer.Option(DEFAULT_RERANKER_BATCH_SIZE, help="Reranker batch size."),
    request_size: Optional[int] = typer.Option(None, help="Pairs per compute_score request (default: thesis setting)."),
    gate_batch_size: int = typer.Option(0, help="Queries per gate batch; 0 processes the whole workload in one batch."),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Reranking depth."),
    metric_k: int = typer.Option(DEFAULT_METRIC_K, help="MRR cutoff."),
    device: str = typer.Option("cpu", help="Reranker device (the thesis setup is CPU with dynamic int8)."),
    full_test_sanity: bool = typer.Option(False, help="Additionally run one non-repeated Full and Adaptive pass over the complete test set."),
    fresh: bool = typer.Option(False, help="Discard previous raw timings of this configuration instead of resuming."),
    output_dir: Optional[str] = typer.Option(None, help="Override the report directory."),
    parent_subset_dir: Optional[str] = typer.Option(None, help="Draw the subset (stratified) from the subset_query_ids.json of this report directory; outputs go to <parent>/subset_<size> by default."),
    write_tables: bool = typer.Option(True, help="Write the LaTeX tables to latex/tables/efficiency."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    console = runtime.console
    log = console.log
    console.section("Online cost benchmark: rerank all vs. adaptive lightweight Hit@1 gate")
    retriever = runtime.retrieval.resolve_retriever_name(retriever)
    spec = reranker_spec(reranker_model, reranker_max_length, reranker_batch_size)
    request_size = int(request_size or default_request_size(reranker_batch_size))
    seed = int(runtime.config.runtime.random_seed if seed is None else seed)
    parent_dir = Path(parent_subset_dir).resolve() if parent_subset_dir else None
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
    elif parent_dir is not None:
        target_dir = parent_dir / f"subset_{int(subset_size)}"
    else:
        target_dir = report_dir(runtime, language, retriever, spec, top_k, predictor_name)
    run_dir = multistep_run_dir(runtime, language, retriever, spec, top_k)
    test_key = f"{language}_test"
    threads_at_start = int(torch.get_num_threads())
    environment = environment_metadata(device)
    log(f"Report directory: {target_dir}")
    log(
        f"Environment: CPU {environment['cpu_model']}, torch {environment['torch_version']} "
        f"(threads={threads_at_start}, interop={environment['torch_num_interop_threads']}), CUDA available={environment['cuda_available']}"
    )

    console.section("Preparation (untimed)")
    gate = load_deployed_gate(runtime, language, retriever, predictor_name, test_key, log)
    full_workload = split_workload(runtime, test_key, retriever, top_k)
    log(f"Computing gate decisions for all {len(full_workload.queries)} test queries (for stratification and effectiveness)")
    full_result = run_gate(gate.state, full_workload.inputs)
    cached_full_scores = cached_top_k_scores(run_dir, full_workload.inputs.query_ids, full_workload.top_document_ids, top_k)
    full_effectiveness = effectiveness(full_workload.top_document_ids, cached_full_scores, full_workload.gold_document_ids, ~full_result.skip, top_k, metric_k)
    log(
        f"Full test (cached rerank-all scores): MRR@{metric_k} rerank all {full_effectiveness['mrr_rerank_all']:.6f}, adaptive "
        f"{full_effectiveness['mrr_adaptive']:.6f}; reranked {full_effectiveness['reranked_queries_adaptive']}/{full_effectiveness['queries']} "
        f"queries = {full_effectiveness['reranked_pairs_adaptive']:,} of {full_effectiveness['reranked_pairs_rerank_all']:,} calls"
    )
    if full_test:
        subset_ids = full_workload.inputs.query_ids.copy()
        subset = {"method": "complete test set", "size": int(len(subset_ids)), "seed": None}
    elif parent_dir is not None:
        parent_ids = np.asarray(read_json(parent_dir / "subset_query_ids.json")["query_ids"], dtype=np.int64)
        parent_skip = full_result.skip[pd.Index(full_workload.inputs.query_ids).get_indexer(parent_ids)]
        log(f"Drawing a nested stratified subset of {subset_size} from the {len(parent_ids)}-query subset in {parent_dir}")
        subset = load_or_create_subset(
            target_dir / "subset_query_ids.json", parent_ids, parent_skip, subset_size, seed, gate.model_signature, log,
            extra={"drawn_from": str(parent_dir / "subset_query_ids.json")},
        )
        subset_ids = np.asarray(subset["query_ids"], dtype=np.int64)
    else:
        subset = load_or_create_subset(target_dir / "subset_query_ids.json", full_workload.inputs.query_ids, full_result.skip, subset_size, seed, gate.model_signature, log)
        subset_ids = np.asarray(subset["query_ids"], dtype=np.int64)
    subset_rows = pd.Index(full_workload.inputs.query_ids).get_indexer(subset_ids)
    expected_skip = full_result.skip[subset_rows]
    workload = split_workload(runtime, test_key, retriever, top_k, query_ids=subset_ids)
    query_count = len(subset_ids)
    log(f"Benchmark workload: {query_count} queries, expected adaptive reranking of {int((~expected_skip).sum())} queries")

    query_text_by_id = full_workload.queries.set_index("query_id")["text"].astype(str).to_dict()
    document_text_by_id = {
        int(document_id): str(text)
        for document_id, text in zip(runtime.retrieval.document_ids.tolist(), runtime.retrieval.document_scope_bundle(retriever)["texts"], strict=True)
    }
    texts = (query_text_by_id, document_text_by_id)
    log(f"Loading {spec.model_name} on {device} (dynamic int8={spec.quantize_dynamic_int8}, batch={reranker_batch_size}, max_length={reranker_max_length}, request={request_size})")
    reranker = load_jina_reranker(spec.model_name, spec.trust_remote_code, spec.use_flash_attn, spec.quantize_dynamic_int8, device)
    score_pairs(reranker, [[query_text_by_id[int(subset_ids[0])], document_text_by_id[int(workload.top_document_ids[0, 0])]]], reranker_batch_size, reranker_max_length, request_size)

    representativeness_path = target_dir / "subset_representativeness.json"
    if not full_test and (fresh or not representativeness_path.exists()):
        representativeness = _representativeness(reranker._tokenizer, full_workload, subset_rows, ~full_result.skip, texts, top_k, reranker_max_length, log)
        representativeness["skip_rate"] = {"full_test": float(full_result.skip.mean()), "subset": float(expected_skip.mean())}
        write_json(representativeness_path, representativeness)

    settings = {
        "predictor_name": predictor_name,
        "model_signature": gate.model_signature,
        "query_count": int(query_count),
        "full_test": bool(full_test),
        "seed": seed,
        "warmup_repetitions": int(warmup_repetitions),
        "repetitions": int(repetitions),
        "batch_size": int(reranker_batch_size),
        "max_length": int(reranker_max_length),
        "request_size": request_size,
        "gate_batch_size": int(gate_batch_size),
        "top_k": int(top_k),
        "device": device,
        "torch_num_threads": threads_at_start,
    }
    raw_path = target_dir / ("raw_timings_full_test.csv" if full_test else "raw_timings.csv")
    settings_path = raw_path.with_suffix(".settings.json")
    score_dir = target_dir / ("fresh_scores_full_test" if full_test else "fresh_scores")
    if fresh:
        for path in (raw_path, settings_path):
            if path.exists():
                path.unlink()
        if score_dir.exists():
            shutil.rmtree(score_dir)
    comparable = {key: value for key, value in settings.items() if key not in ("warmup_repetitions", "repetitions")}
    if settings_path.exists():
        previous = {key: value for key, value in read_json(settings_path).items() if key not in ("warmup_repetitions", "repetitions")}
        if previous != comparable:
            raise RuntimeError(
                f"Existing raw timings at {raw_path} were recorded with different settings {previous}; "
                "rerun with --fresh to discard them"
            )
    write_json(settings_path, settings)

    runner = StrategyRunner(gate, workload, reranker, texts, settings, log)
    plan = [("warmup", index) for index in range(int(warmup_repetitions))] + [("measured", index) for index in range(int(repetitions))]
    raw = _read_raw(raw_path)
    completed = {
        (row.phase, int(row.repetition), row.strategy)
        for row in raw.loc[raw["phase"] == "measured"].itertuples(index=False)
    }
    if completed:
        log(f"Resuming: {len(completed)} measured strategy runs already recorded in {raw_path}; warm-ups are repeated in this process")
    durations: dict[str, list[float]] = {STRATEGY_FULL: [], STRATEGY_ADAPTIVE: []}
    for row in raw.loc[raw["component"] == COMPONENT_ONLINE_TOTAL].itertuples(index=False):
        durations[row.strategy].append(float(row.seconds))
    measured_pending = any(
        ("measured", repetition, strategy) not in completed
        for phase, repetition in plan
        if phase == "measured"
        for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE)
    )
    pending = [
        (phase, repetition, strategy)
        for phase, repetition in plan
        for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE)
        if (phase == "warmup" and measured_pending) or (phase == "measured" and (phase, repetition, strategy) not in completed)
    ]
    log(f"{len(pending)} strategy runs pending ({sum(1 for item in pending if item[0] == 'warmup')} warm-up, {sum(1 for item in pending if item[0] == 'measured')} measured)")

    console.section("Timed runs (ABBA-balanced order)")
    for sequence_index, (phase, repetition) in enumerate(plan):
        order = (STRATEGY_FULL, STRATEGY_ADAPTIVE) if sequence_index % 2 == 0 else (STRATEGY_ADAPTIVE, STRATEGY_FULL)
        for position, strategy in enumerate(order):
            if (phase, repetition, strategy) not in pending:
                continue
            label = f"{phase} {repetition + 1} {strategy}"
            log(f"Starting {label} (sequence {sequence_index + 1}/{len(plan)}, position {position + 1})")
            if strategy == STRATEGY_FULL:
                timings, details = runner.run_full(label)
            else:
                timings, details = runner.run_adaptive(label)
                if not np.array_equal(~details["rerank_mask"], expected_skip):
                    raise RuntimeError("The gate produced different skip decisions than in preparation; aborting")
            rows = [
                {
                    "phase": phase,
                    "repetition": repetition,
                    "sequence_index": sequence_index,
                    "position_in_pair": position + 1,
                    "strategy": strategy,
                    "component": component,
                    "seconds": seconds,
                    "queries": query_count,
                    "reranked_queries": details["reranked_queries"],
                    "reranker_calls": details["reranker_calls"],
                }
                for component, seconds in timings.items()
            ]
            _append_raw(raw_path, rows)
            append_jsonl(
                target_dir / ("run_records_full_test.jsonl" if full_test else "run_records.jsonl"),
                {
                    "experiment": "repeated_fixed_workload",
                    "phase": phase,
                    "repetition": repetition,
                    "sequence_index": sequence_index,
                    "position_in_pair": position + 1,
                    "strategy": strategy,
                    "subset_seed": subset.get("seed"),
                    "subset_file": str(target_dir / "subset_query_ids.json") if not full_test else None,
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
            score_path = score_dir / f"{strategy}.npz"
            if phase == "measured" and not score_path.exists():
                score_dir.mkdir(parents=True, exist_ok=True)
                np.savez(score_path, scores=details["scores"], rerank_mask=details["rerank_mask"], query_ids=subset_ids)
            durations[strategy].append(timings[COMPONENT_ONLINE_TOTAL])
            pending.remove((phase, repetition, strategy))
            remaining_seconds = sum(
                float(np.mean(durations[item[2]])) if durations[item[2]] else float(np.mean([value for values in durations.values() for value in values]))
                for item in pending
            )
            log(
                f"Finished {label}: online total {timings[COMPONENT_ONLINE_TOTAL]:.1f}s"
                + (f", gate {timings[COMPONENT_GATE_TOTAL]:.3f}s" if COMPONENT_GATE_TOTAL in timings else "")
                + f"; {len(pending)} strategy runs pending, ~{remaining_seconds / 3600:.1f} h remaining"
            )

    if int(torch.get_num_threads()) != threads_at_start:
        raise RuntimeError("The torch thread configuration changed during the benchmark")

    console.section("Aggregation")
    raw = _read_raw(raw_path)
    aggregated = _aggregate(raw, query_count)
    cached_subset_scores = cached_full_scores[subset_rows]
    subset_effectiveness = effectiveness(workload.top_document_ids, cached_subset_scores, workload.gold_document_ids, ~expected_skip, top_k, metric_k)
    aggregated["workload"] = {
        "queries": int(query_count),
        "is_full_test": bool(full_test),
        "subset": {key: value for key, value in subset.items() if key != "query_ids"},
        "top_k": int(top_k),
        "metric_k": int(metric_k),
        "measured_repetitions": int(repetitions),
        "warmup_repetitions": int(warmup_repetitions),
        "order": "ABBA-balanced: Full first in even sequence positions, Adaptive first in odd ones",
    }
    aggregated["full_test_effectiveness"] = {key: value for key, value in full_effectiveness.items() if not key.startswith("rr_")}
    aggregated["subset_effectiveness_cached_scores"] = {key: value for key, value in subset_effectiveness.items() if not key.startswith("rr_")}
    for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE):
        entry = aggregated["strategies"][strategy]
        log(
            f"{strategy}: online {entry['online_per_query_mean']:.4f} +/- {entry['online_per_query_sd']:.4f} s/query, "
            f"rerank {entry['rerank_per_query_mean']:.4f} s/query"
            + (f", gate {1000 * entry['gate_per_query_mean']:.3f} +/- {1000 * entry['gate_per_query_sd']:.3f} ms/query" if "gate_per_query_mean" in entry else "")
            + f", reranker calls {entry['reranker_calls']:,}"
        )
    log(f"Online saving: {100 * aggregated['online_saving']['mean']:.2f}% +/- {100 * aggregated['online_saving']['sd']:.2f}% (run-to-run execution variability)")
    aggregated["variability"] = VARIABILITY_LABEL
    measured_raw = raw.loc[raw["phase"] == "measured"]
    per_repetition = measured_raw.pivot_table(index="repetition", columns=["strategy", "component"], values="seconds", aggfunc="first")
    repetition_rows = []
    for repetition_index in per_repetition.index:
        row = per_repetition.loc[repetition_index]
        if (STRATEGY_FULL, COMPONENT_ONLINE_TOTAL) not in row.index or (STRATEGY_ADAPTIVE, COMPONENT_ONLINE_TOTAL) not in row.index:
            continue
        full_online = float(row[(STRATEGY_FULL, COMPONENT_ONLINE_TOTAL)])
        adaptive_online = float(row[(STRATEGY_ADAPTIVE, COMPONENT_ONLINE_TOTAL)])
        if math.isnan(full_online) or math.isnan(adaptive_online):
            continue
        repetition_rows.append(
            {
                "repetition": int(repetition_index),
                "queries": query_count,
                "skipped_queries": int(expected_skip.sum()),
                "reranked_queries": int((~expected_skip).sum()),
                "full_reranker_calls": query_count * top_k,
                "adaptive_reranker_calls": int((~expected_skip).sum()) * top_k,
                "full_online_seconds": full_online,
                "adaptive_gate_seconds": float(row[(STRATEGY_ADAPTIVE, COMPONENT_GATE_TOTAL)]),
                "adaptive_rerank_seconds": float(row[(STRATEGY_ADAPTIVE, COMPONENT_RERANK)]),
                "adaptive_online_seconds": adaptive_online,
                "relative_online_saving": 1.0 - adaptive_online / full_online,
            }
        )
    pd.DataFrame(repetition_rows).to_csv(target_dir / ("repetition_summary_full_test.csv" if full_test else "repetition_summary.csv"), index=False)

    agreement: dict[str, Any] = {}
    full_scores_path = score_dir / f"{STRATEGY_FULL}.npz"
    adaptive_scores_path = score_dir / f"{STRATEGY_ADAPTIVE}.npz"
    if full_scores_path.exists():
        fresh_full = np.load(full_scores_path)["scores"]
        agreement["full_vs_cached"] = _score_agreement(fresh_full, cached_subset_scores, workload.top_document_ids, workload.gold_document_ids, top_k, metric_k)
    if adaptive_scores_path.exists():
        payload = np.load(adaptive_scores_path)
        mask = payload["rerank_mask"]
        agreement["adaptive_vs_cached"] = _score_agreement(
            payload["scores"], cached_subset_scores[mask], workload.top_document_ids[mask], workload.gold_document_ids[mask], top_k, metric_k
        )
        if full_scores_path.exists():
            agreement["adaptive_vs_full_fresh"] = _score_agreement(
                payload["scores"], np.load(full_scores_path)["scores"][mask], workload.top_document_ids[mask], workload.gold_document_ids[mask], top_k, metric_k
            )
            rr_no_rerank = gold_reciprocal_ranks(workload.top_document_ids, workload.gold_document_ids, metric_k)
            rr_full = reranked_reciprocal_ranks(workload.top_document_ids[:, :top_k], np.load(full_scores_path)["scores"], workload.gold_document_ids, metric_k)
            rr_adaptive = rr_no_rerank.copy()
            rr_adaptive[mask] = reranked_reciprocal_ranks(workload.top_document_ids[mask, :top_k], payload["scores"], workload.gold_document_ids[mask], metric_k)
            agreement["subset_mrr_fresh_scores"] = {"rerank_all": float(rr_full.mean()), "adaptive": float(rr_adaptive.mean())}
            agreement["subset_mrr_cached_scores"] = {
                "rerank_all": subset_effectiveness["mrr_rerank_all"],
                "adaptive": subset_effectiveness["mrr_adaptive"],
            }
    for name, values in agreement.items():
        log(f"  score agreement {name}: {values}")
    aggregated["score_agreement"] = agreement
    write_json(target_dir / ("aggregated_full_test.json" if full_test else "aggregated.json"), aggregated)

    metadata = {
        "environment": environment,
        "settings": settings,
        "precision": "dynamic int8 quantization of torch.nn.Linear layers (torch.quantization.quantize_dynamic), float32 otherwise" if spec.quantize_dynamic_int8 else "model default dtype",
        "reranker": spec.summary_dict(),
        "inference": "torch.inference_mode around compute_score; pairs flattened over the workload in query and rank order and scored in requests of request_size pairs with batch_size pairs per forward pass",
        "warmup_procedure": f"{warmup_repetitions} complete untimed repetitions of both strategies over the same workload before the measured repetitions, plus one single-pair reranker call after loading",
        "timing": "time.perf_counter wall clock; synchronize_device before starting and before stopping each timer (no-op on CPU)",
        "gate_timing_boundary": "starts after the Qwen ranking and query embedding exist; includes query tokenisation and BM25 top-1 scoring, training-query RR, perturbation stability, the remaining features, and the Random Forest",
        "progress_logging": "one console line per reranker request inside the timed section",
    }
    write_json(target_dir / ("metadata_full_test.json" if full_test else "metadata.json"), metadata)
    if not full_test:
        setup_dir = parent_dir if parent_dir is not None else report_dir(runtime, language, retriever, spec, top_k, predictor_name)
        write_cost_summary(target_dir, log, parent_dir=setup_dir)
    if write_tables and not full_test:
        write_available_tables(runtime, target_dir, language, retriever, predictor_name, log, parent_dir=parent_dir)

    if full_test_sanity and not full_test:
        console.section("Full-test sanity check (single non-repeated pass)")
        sanity_path = target_dir / "full_test_sanity.json"
        sanity = read_json(sanity_path) if sanity_path.exists() and not fresh else {}
        full_runner = StrategyRunner(gate, full_workload, reranker, texts, settings, log)
        for strategy in (STRATEGY_FULL, STRATEGY_ADAPTIVE):
            if strategy in sanity:
                log(f"Full-test sanity {strategy} already recorded; skipping")
                continue
            label = f"full-test sanity {strategy}"
            log(f"Starting {label} over {len(full_workload.queries)} queries")
            timings, details = full_runner.run_full(label) if strategy == STRATEGY_FULL else full_runner.run_adaptive(label)
            sanity[strategy] = {
                "timings": timings,
                "reranked_queries": details["reranked_queries"],
                "reranker_calls": details["reranker_calls"],
            }
            fresh_scores = details["scores"]
            if strategy == STRATEGY_FULL:
                sanity[strategy]["score_agreement_vs_cached"] = _score_agreement(
                    fresh_scores, cached_full_scores, full_workload.top_document_ids, full_workload.gold_document_ids, top_k, metric_k
                )
            else:
                mask = details["rerank_mask"]
                sanity[strategy]["score_agreement_vs_cached"] = _score_agreement(
                    fresh_scores, cached_full_scores[mask], full_workload.top_document_ids[mask], full_workload.gold_document_ids[mask], top_k, metric_k
                )
            write_json(sanity_path, sanity)
            log(f"Finished {label}: online total {timings[COMPONENT_ONLINE_TOTAL] / 3600:.2f} h")
        if STRATEGY_FULL in sanity and STRATEGY_ADAPTIVE in sanity:
            saving = 1.0 - sanity[STRATEGY_ADAPTIVE]["timings"][COMPONENT_ONLINE_TOTAL] / sanity[STRATEGY_FULL]["timings"][COMPONENT_ONLINE_TOTAL]
            sanity["online_saving"] = saving
            sanity["repeated_subset_online_saving_mean"] = aggregated["online_saving"]["mean"]
            write_json(sanity_path, sanity)
            log(f"Full-test online saving {100 * saving:.2f}% vs repeated subset estimate {100 * aggregated['online_saving']['mean']:.2f}%")

    console.section("Done")
    log(f"Outputs in {target_dir}")


if __name__ == "__main__":
    typer.run(main)
