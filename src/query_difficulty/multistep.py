from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import re
import shutil
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import anthropic
import joblib
import numpy as np
import pandas as pd
import requests
import spacy
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from query_difficulty.analysis import reciprocal_rank_at_k
from query_difficulty.artifacts import ArtifactStore
from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
)
from query_difficulty.modeling import (
    load_or_create_calibrated_ordinal_test_predictions,
    load_or_create_hit_at_1_test_predictions,
)
from query_difficulty.modeling_calibrated_ordinal_lightweight import (
    load_or_create_lightweight_calibrated_ordinal_test_predictions,
)
from query_difficulty.modeling_hit_at_1_lightweight import load_or_create_lightweight_hit_at_1_test_predictions
from query_difficulty.modeling_in_top_k_sweep_lightweight import (
    load_or_create_lightweight_in_top_k_test_predictions,
)
from query_difficulty.modeling_in_top_max_depth_lightweight import (
    load_or_create_lightweight_max_depth_test_predictions,
)
from query_difficulty.modeling_tuned_qpp_lightweight import load_or_create_lightweight_tuned_qpp_test_predictions
from query_difficulty.retrieval import RetrievalManager

LIGHTWEIGHT_HIT_AT_1_MANIFEST_PATH = "config/lightweight_hit_at_1_features.json"
PREDICTOR_TIER_NAMES = {
    "lightweight": "lightweight/hit_at_1",
}

SELECTOR_NAMES = ("hit1", "max_depth", "ordinal", "qpp", "in_top_k", "in_top_k_depth")
SELECTOR_LABELS = {
    "hit1": "Hit@1 (lightweight)",
    "max_depth": "Max depth (lightweight)",
    "ordinal": "Calibrated ordinal (lightweight)",
    "qpp": "Tuned QPP (lightweight)",
    "in_top_k": "In-top-k (lightweight)",
    "in_top_k_depth": "In-top-k depth (lightweight)",
}
DEPTH_SELECTOR_NAMES = ("ordinal", "qpp", "in_top_k_depth")
IN_TOP_K_SUPPORTED_DEPTHS = (5, 10, 20, 30, 50, 60, 100, 120, 200, 1000)

CONSIDERED_SELECTOR_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("hit1",),
    ("max_depth",),
    ("hit1", "max_depth"),
    ("ordinal",),
    ("qpp",),
    ("in_top_k",),
    ("hit1", "ordinal"),
    ("hit1", "qpp"),
    ("hit1", "in_top_k"),
)


IN_TOP_K_DEPTH_SELECTOR_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("in_top_k_depth",),
    ("hit1", "in_top_k_depth"),
)


def _selector_condition_combinations() -> list[tuple[str, ...]]:
    return list(CONSIDERED_SELECTOR_COMBINATIONS)


def _policy_comparison_combinations() -> list[tuple[str, ...]]:
    return list(CONSIDERED_SELECTOR_COMBINATIONS) + list(IN_TOP_K_DEPTH_SELECTOR_COMBINATIONS)


def _condition_name(selectors: tuple[str, ...]) -> str:
    return "_and_".join(selectors)


def _condition_rule_description(selectors: tuple[str, ...]) -> str:
    gate_clause = " AND ".join(f"{SELECTOR_LABELS[name]} selects" for name in selectors)
    depth_selectors = [name for name in selectors if name in DEPTH_SELECTOR_NAMES]
    if not depth_selectors:
        depth_clause = "rerank_depth = top_k"
    elif len(depth_selectors) == 1:
        depth_clause = (
            f"{SELECTOR_LABELS[depth_selectors[0]]} selected_size <= top_k; "
            f"rerank_depth = {SELECTOR_LABELS[depth_selectors[0]]} selected_size"
        )
    else:
        depth_clause = (
            " AND ".join(f"{SELECTOR_LABELS[name]} selected_size <= top_k" for name in depth_selectors)
            + "; rerank_depth = min("
            + ", ".join(f"{SELECTOR_LABELS[name]} selected_size" for name in depth_selectors)
            + ")"
        )
    return f"{gate_clause}; {depth_clause}"


def _combined_selector_depth(
    selectors: tuple[str, ...],
    gate_pass: dict[str, bool],
    depth: dict[str, int | None],
    top_k: int,
) -> tuple[bool, int]:
    selected = all(gate_pass[name] for name in selectors)
    if not selected:
        return False, 0
    depth_selectors = [name for name in selectors if depth.get(name) is not None]
    if not depth_selectors:
        return True, int(top_k)
    if any(int(depth[name]) > int(top_k) for name in depth_selectors):
        return False, 0
    combined_depth = min(int(depth[name]) for name in depth_selectors)
    return True, max(0, combined_depth)


_CROSS_ENCODER_INSTANCE_CACHE: dict[tuple[str, int], CrossEncoder] = {}
_FLAGEMBEDDING_INSTANCE_CACHE: dict[tuple[str, bool, bool], Any] = {}
_TRANSFORMERS_SEQUENCE_CLASSIFICATION_CACHE: dict[str, tuple[Any, AutoModelForSequenceClassification]] = {}
_JINA_COMPUTE_SCORE_MODEL_CACHE: dict[tuple[str, bool, bool, bool], AutoModelForSequenceClassification] = {}
_LLAMA_NEMOTRON_MODEL_CACHE: dict[tuple[str, bool, bool], tuple[Any, AutoModelForSequenceClassification]] = {}
_TRANSFORMERS_LISTWISE_MODEL_CACHE: dict[tuple[str, bool], Any] = {}
_COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER = 32
_JINA_COMPUTE_SCORE_HEARTBEAT_SECONDS = 30.0


LLM_POINTWISE_TITLE_ONLY_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and a candidate scientific paper.

Judge how likely the post is referring to this specific paper.

Output ONLY a single integer score from 0 to 10.

10 = the post is almost certainly about this paper.
0 = the paper is unrelated.

Social media post:
{query}

Candidate paper:
Title: {title}

Score (0-10):"""


LLM_POINTWISE_TITLE_ABSTRACT_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and a candidate scientific paper.

Judge how likely the post is referring to this specific paper.

Output ONLY a single integer score from 0 to 10.

10 = the post is almost certainly about this paper.
0 = the paper is unrelated.

Social media post:
{query}

Candidate paper:
Title: {title}
Abstract: {abstract}

Score (0-10):"""


LLM_PAIRWISE_TITLE_ONLY_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and two candidate scientific papers.

Decide which candidate paper is more likely to be the source paper for
the post.

Output ONLY A or B.

Social media post:
{query}

Candidate paper A:
Title: {title_a}

Candidate paper B:
Title: {title_b}

More likely source paper:"""


LLM_PAIRWISE_TITLE_ABSTRACT_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and two candidate scientific papers.

Decide which candidate paper is more likely to be the source paper for
the post.

Output ONLY A or B.

Social media post:
{query}

Candidate paper A:
Title: {title_a}
Abstract: {abstract_a}

Candidate paper B:
Title: {title_b}
Abstract: {abstract_b}

More likely source paper:"""


LLM_POINTWISE_PROMPT = LLM_POINTWISE_TITLE_ONLY_PROMPT


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(float(seconds), 0.0)
    if total_seconds < 1.0:
        return f"{total_seconds * 1000.0:.0f}ms"
    minutes, seconds_remainder = divmod(total_seconds, 60.0)
    hours, minutes_remainder = divmod(int(minutes), 60)
    if hours > 0:
        return f"{hours}h {minutes_remainder:02d}m {seconds_remainder:04.1f}s"
    if minutes_remainder > 0:
        return f"{minutes_remainder}m {seconds_remainder:04.1f}s"
    return f"{seconds_remainder:.2f}s"


def _console_log_timing(console: Any, label: str, seconds: float, *, details: str | None = None) -> None:
    if hasattr(console, "log_timing"):
        console.log_timing(label, seconds, details=details)
        return
    message = f"{label} in {_format_elapsed_seconds(seconds)}"
    if details:
        message = f"{message} ({details})"
    console.log(message)


_SCORING_PROGRESS_LOG_INTERVAL_SECONDS = _JINA_COMPUTE_SCORE_HEARTBEAT_SECONDS


class _BatchScoringProgress:
    def __init__(
        self,
        console: Any,
        *,
        label: str,
        model_name: str,
        total_pairs: int,
        total_batches: int,
        unit_plural: str = "batches",
        log_interval_seconds: float = _SCORING_PROGRESS_LOG_INTERVAL_SECONDS,
    ) -> None:
        self.console = console
        self.label = label
        self.model_name = model_name
        self.total_pairs = total_pairs
        self.total_batches = max(1, total_batches)
        self.unit_plural = unit_plural
        self.log_interval_seconds = log_interval_seconds
        self.started_at = time.perf_counter()
        self._last_logged_at = self.started_at
        console.log(
            f"Starting {label} for {model_name}: "
            f"{self.total_batches} {unit_plural}, {total_pairs} query-document pairs"
        )

    def update(self, batch_index: int, processed_pairs: int) -> None:
        now = time.perf_counter()
        is_last = batch_index >= self.total_batches
        if not is_last and (now - self._last_logged_at) < self.log_interval_seconds:
            return
        self._last_logged_at = now
        elapsed_seconds = now - self.started_at
        processed_pairs = min(processed_pairs, self.total_pairs)
        percent_complete = 100.0 if self.total_pairs == 0 else 100.0 * processed_pairs / self.total_pairs
        pairs_per_second = processed_pairs / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining_pairs = max(0, self.total_pairs - processed_pairs)
        eta_text = (
            _format_elapsed_seconds(remaining_pairs / pairs_per_second)
            if pairs_per_second > 0
            else "unknown"
        )
        self.console.log(
            f"{self.label} progress for {self.model_name}: "
            f"{self.unit_plural} {batch_index}/{self.total_batches} ({percent_complete:.1f}%), "
            f"pairs {processed_pairs}/{self.total_pairs}, "
            f"elapsed {_format_elapsed_seconds(elapsed_seconds)}, "
            f"rate {pairs_per_second:.2f} pairs/s, eta {eta_text}"
        )


def _parse_llm_score(text: str) -> float:
    match = re.search(r"\b(10|[0-9])\b", text)
    if match is None:
        return -1.0
    return float(int(match.group(1)))


def _ollama_response_text(payload: dict[str, Any]) -> str:
    response_text = str(payload.get("response", "") or "").strip()
    if response_text:
        return response_text
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content", "") or "").strip()
    return ""


def _ollama_thinking_text(payload: dict[str, Any]) -> str:
    thinking_text = str(payload.get("thinking", "") or "").strip()
    if thinking_text:
        return thinking_text
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("thinking", "") or "").strip()
    return ""


def _prefers_ollama_thinking(model_name: str) -> bool:
    return model_name.strip().lower().startswith("qwen3")


def render_llm_pairwise_prompt(
    query: str,
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    *,
    candidate_fields: str = "title",
) -> str:
    if candidate_fields == "title":
        return LLM_PAIRWISE_TITLE_ONLY_PROMPT.format(
            query=query,
            title_a=str(candidate_a.get("title", "")),
            title_b=str(candidate_b.get("title", "")),
        )
    if candidate_fields == "title_abstract":
        return LLM_PAIRWISE_TITLE_ABSTRACT_PROMPT.format(
            query=query,
            title_a=str(candidate_a.get("title", "")),
            abstract_a=str(candidate_a.get("abstract", "")),
            title_b=str(candidate_b.get("title", "")),
            abstract_b=str(candidate_b.get("abstract", "")),
        )
    raise ValueError(f"Unsupported candidate_fields value: {candidate_fields}")


def _parse_llm_pairwise_choice(text: str) -> str | None:
    direct_match = re.search(r"(?i)^\s*(?:candidate|paper|document|passage)?\s*([AB])\b", text)
    if direct_match is not None:
        return direct_match.group(1).upper()
    fallback_match = re.search(r"(?i)\b([AB])\b", text)
    if fallback_match is None:
        return None
    return fallback_match.group(1).upper()


LLM_LISTWISE_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and a list of candidate scientific papers.

Rank the candidate papers from most likely to least likely to be the
source paper for the post.

Return ONLY the candidate numbers in ranked order as a comma-separated
list.

Do not include any explanation or other text.
Use each candidate number exactly once.

Social media post:
{query}

Candidate papers:
{candidates}

Ranked candidate numbers:"""


def render_llm_listwise_prompt(
    query: str,
    candidates: list[dict[str, str]],
    *,
    candidate_fields: str = "title",
) -> str:
    if candidate_fields not in {"title", "title_abstract"}:
        raise ValueError(f"Unsupported candidate_fields value: {candidate_fields}")
    candidate_lines: list[str] = []
    for candidate_number, candidate in enumerate(candidates, start=1):
        title = str(candidate.get("title", ""))
        candidate_lines.append(f"{candidate_number}. Title: {title}")
        if candidate_fields == "title_abstract":
            abstract = str(candidate.get("abstract", ""))
            candidate_lines.append(f"   Abstract: {abstract}")
    return LLM_LISTWISE_PROMPT.format(query=query, candidates="\n".join(candidate_lines))


def _parse_llm_ranked_candidates(text: str, candidate_count: int) -> list[int]:
    if candidate_count <= 0:
        return []
    ranked_candidates: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\b\d+\b", text):
        candidate_number = int(match.group(0))
        if candidate_number < 1 or candidate_number > int(candidate_count):
            continue
        if candidate_number in seen:
            continue
        seen.add(candidate_number)
        ranked_candidates.append(candidate_number)
    return ranked_candidates


@dataclass(slots=True)
class CrossEncoderRerankerSpec:
    model_name: str
    max_length: int
    batch_size: int

    @property
    def method(self) -> str:
        return "cross_encoder"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        return f"maxlen_{self.max_length}"

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max length: {self.max_length}",
            f"- Batch size: {self.batch_size}",
        ]


@dataclass(slots=True)
class FlagEmbeddingRerankerSpec:
    model_name: str
    max_length: int
    batch_size: int
    use_fp16: bool = False
    normalize: bool = False
    candidate_fields: str = "title_abstract"
    implementation_version: int = 2
    quantize_dynamic_int8: bool = False

    @property
    def method(self) -> str:
        return "flagembedding"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        precision = "fp16" if self.use_fp16 else "fp32"
        normalized = "on" if self.normalize else "off"
        quantized = "on" if self.quantize_dynamic_int8 else "off"
        return (
            f"maxlen_{self.max_length}_batch_{self.batch_size}_{precision}_normalize_{normalized}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_quantized_{quantized}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "use_fp16": self.use_fp16,
            "normalize": self.normalize,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
            "quantize_dynamic_int8": self.quantize_dynamic_int8,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max length: {self.max_length}",
            f"- Batch size: {self.batch_size}",
            f"- Use fp16: {self.use_fp16}",
            f"- Normalize scores: {self.normalize}",
            f"- Candidate fields: {self.candidate_fields}",
            f"- Implementation version: {self.implementation_version}",
            f"- Dynamic int8 quantization: {self.quantize_dynamic_int8}",
        ]


@dataclass(slots=True)
class JinaComputeScoreRerankerSpec:
    model_name: str
    max_length: int
    batch_size: int
    trust_remote_code: bool = True
    use_flash_attn: bool = False
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1
    quantize_dynamic_int8: bool = False

    @property
    def method(self) -> str:
        return "jina_compute_score"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        remote_code = "on" if self.trust_remote_code else "off"
        flash_attn = "on" if self.use_flash_attn else "off"
        quantized = "on" if self.quantize_dynamic_int8 else "off"
        return (
            f"maxlen_{self.max_length}_batch_{self.batch_size}"
            f"_remote_code_{remote_code}_flash_attn_{flash_attn}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_quantized_{quantized}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "trust_remote_code": self.trust_remote_code,
            "use_flash_attn": self.use_flash_attn,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
            "quantize_dynamic_int8": self.quantize_dynamic_int8,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max length: {self.max_length}",
            f"- Batch size: {self.batch_size}",
            f"- Trust remote code: {self.trust_remote_code}",
            f"- Use flash attention: {self.use_flash_attn}",
            f"- Candidate fields: {self.candidate_fields}",
            f"- Dynamic int8 quantization: {self.quantize_dynamic_int8}",
            f"- Implementation version: {self.implementation_version}",
        ]


@dataclass(slots=True)
class LlamaNemotronRerankerSpec:
    model_name: str
    max_length: int
    batch_size: int
    trust_remote_code: bool = True
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1
    quantize_dynamic_int8: bool = False

    @property
    def method(self) -> str:
        return "llama_nemotron"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        remote_code = "on" if self.trust_remote_code else "off"
        quantized = "on" if self.quantize_dynamic_int8 else "off"
        return (
            f"maxlen_{self.max_length}_batch_{self.batch_size}"
            f"_remote_code_{remote_code}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_quantized_{quantized}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "trust_remote_code": self.trust_remote_code,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
            "quantize_dynamic_int8": self.quantize_dynamic_int8,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max length: {self.max_length}",
            f"- Batch size: {self.batch_size}",
            f"- Trust remote code: {self.trust_remote_code}",
            f"- Candidate fields: {self.candidate_fields}",
            f"- Dynamic int8 quantization: {self.quantize_dynamic_int8}",
            f"- Implementation version: {self.implementation_version}",
        ]


@dataclass(slots=True)
class TransformersListwiseRerankerSpec:
    model_name: str
    trust_remote_code: bool = True
    max_documents_per_query: int | None = None
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1

    @property
    def method(self) -> str:
        return "transformers_listwise"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        remote_code = "on" if self.trust_remote_code else "off"
        max_documents = "none" if self.max_documents_per_query is None else str(int(self.max_documents_per_query))
        return (
            f"remote_code_{remote_code}_maxdocs_{max_documents}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "trust_remote_code": self.trust_remote_code,
            "max_documents_per_query": self.max_documents_per_query,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Trust remote code: {self.trust_remote_code}",
            (
                "- Max documents per query: none"
                if self.max_documents_per_query is None
                else f"- Max documents per query: {int(self.max_documents_per_query)}"
            ),
            f"- Candidate fields: {self.candidate_fields}",
            f"- Implementation version: {self.implementation_version}",
        ]


@dataclass(slots=True)
class LLMPointwiseRerankerSpec:
    model_name: str
    max_input_tokens: int
    max_new_tokens: int
    batch_size: int
    ollama_url: str
    timeout_seconds: int
    max_retries: int
    temperature: float
    disable_thinking: bool = True
    candidate_fields: str = "title"
    prompt_template: str | None = None

    @property
    def method(self) -> str:
        return "llm"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        prompt_hash = hashlib.sha1(self.resolved_prompt_template().encode("utf-8")).hexdigest()[:10]
        thinking_mode = "off" if self.disable_thinking else "on"
        return (
            f"maxlen_{self.max_input_tokens}_newtokens_{self.max_new_tokens}"
            f"_provider_ollama_temp_{self.temperature:.1f}"
            f"_think_{thinking_mode}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_prompt_{prompt_hash}"
        )

    def render_prompt(self, query: str, title: str, abstract: str) -> str:
        return self.resolved_prompt_template().format(query=query, title=title, abstract=abstract)

    def resolved_prompt_template(self) -> str:
        if self.prompt_template is not None:
            return self.prompt_template
        if self.candidate_fields == "title_abstract":
            return LLM_POINTWISE_TITLE_ABSTRACT_PROMPT
        if self.candidate_fields != "title":
            raise ValueError(f"Unsupported candidate_fields value: {self.candidate_fields}")
        return LLM_POINTWISE_TITLE_ONLY_PROMPT

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "provider": "ollama",
            "ollama_url": self.ollama_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "disable_thinking": self.disable_thinking,
            "candidate_fields": self.candidate_fields,
        }

    def config_dict(self) -> dict[str, Any]:
        payload = self.summary_dict()
        payload["prompt_template"] = self.resolved_prompt_template()
        return payload

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max input tokens: {self.max_input_tokens}",
            f"- Max new tokens: {self.max_new_tokens}",
            f"- Batch size: {self.batch_size}",
            f"- Provider: ollama",
            f"- Ollama URL: {self.ollama_url}",
            f"- Timeout seconds: {self.timeout_seconds}",
            f"- Max retries: {self.max_retries}",
            f"- Temperature: {self.temperature}",
            f"- Disable thinking: {self.disable_thinking}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


@dataclass(slots=True)
class LLMPairwiseRerankerSpec:
    model_name: str
    max_input_tokens: int
    max_new_tokens: int
    batch_size: int
    ollama_url: str
    timeout_seconds: int
    max_retries: int
    temperature: float
    disable_thinking: bool = True
    candidate_fields: str = "title"
    prompt_template: str | None = None

    @property
    def method(self) -> str:
        return "llm_pairwise"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        prompt_hash = hashlib.sha1(self.resolved_prompt_template().encode("utf-8")).hexdigest()[:10]
        thinking_mode = "off" if self.disable_thinking else "on"
        return (
            f"maxlen_{self.max_input_tokens}_newtokens_{self.max_new_tokens}"
            f"_provider_ollama_temp_{self.temperature:.1f}"
            f"_think_{thinking_mode}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_prompt_{prompt_hash}"
        )

    def render_prompt(
        self,
        *,
        query: str,
        candidate_a: dict[str, str],
        candidate_b: dict[str, str],
    ) -> str:
        template = self.resolved_prompt_template()
        return template.format(
            query=query,
            title_a=str(candidate_a.get("title", "")),
            abstract_a=str(candidate_a.get("abstract", "")),
            title_b=str(candidate_b.get("title", "")),
            abstract_b=str(candidate_b.get("abstract", "")),
        )

    def resolved_prompt_template(self) -> str:
        if self.prompt_template is not None:
            return self.prompt_template
        if self.candidate_fields == "title_abstract":
            return LLM_PAIRWISE_TITLE_ABSTRACT_PROMPT
        if self.candidate_fields != "title":
            raise ValueError(f"Unsupported candidate_fields value: {self.candidate_fields}")
        return LLM_PAIRWISE_TITLE_ONLY_PROMPT

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "provider": "ollama",
            "ollama_url": self.ollama_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "disable_thinking": self.disable_thinking,
            "candidate_fields": self.candidate_fields,
        }

    def config_dict(self) -> dict[str, Any]:
        payload = self.summary_dict()
        payload["prompt_template"] = self.resolved_prompt_template()
        return payload

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Max input tokens: {self.max_input_tokens}",
            f"- Max new tokens: {self.max_new_tokens}",
            f"- Batch size: {self.batch_size}",
            f"- Provider: ollama",
            f"- Ollama URL: {self.ollama_url}",
            f"- Timeout seconds: {self.timeout_seconds}",
            f"- Max retries: {self.max_retries}",
            f"- Temperature: {self.temperature}",
            f"- Disable thinking: {self.disable_thinking}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


CLAUDE_LISTWISE_RERANK_SYSTEM_PROMPT = """You are a search relevance ranking assistant for scientific literature retrieval.

You will be given a social media post that paraphrases a scientific finding, and a numbered list of
candidate scientific papers (title and abstract). Rank the candidates by how likely each one is to be
the specific paper the post is referring to, from most to least likely.

Respond with ONLY a JSON array of the candidate numbers in ranked order (most likely first). The array
must contain every number from 1 to N exactly once, with no other text, explanation, or formatting.

Example response for 4 candidates: [3, 1, 4, 2]"""


@dataclass(slots=True)
class ClaudeListwiseRerankerSpec:
    model_name: str
    window_size: int
    window_step: int
    max_output_tokens: int
    temperature: float
    timeout_seconds: int
    max_retries: int
    max_concurrency: int
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1

    @property
    def method(self) -> str:
        return "claude_listwise"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        return (
            f"window_{self.window_size}_step_{self.window_step}"
            f"_maxout_{self.max_output_tokens}_temp_{self.temperature:.1f}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_concurrency": self.max_concurrency,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Window size: {self.window_size}",
            f"- Window step: {self.window_step}",
            f"- Max output tokens: {self.max_output_tokens}",
            f"- Temperature: {self.temperature}",
            f"- Timeout seconds: {self.timeout_seconds}",
            f"- Max retries: {self.max_retries}",
            f"- Max concurrency: {self.max_concurrency}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


def _parse_claude_rank_permutation(text: str, n: int) -> list[int]:
    match = re.search(r"\[[\d,\s]+\]", text)
    if not match:
        raise ValueError(f"No JSON array found in Claude ranking response: {text!r}")
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON array from Claude ranking response: {text!r}") from exc
    if not isinstance(values, list):
        raise ValueError(f"Claude ranking response is not a JSON array: {values!r}")
    try:
        integer_values = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Claude ranking response contains non-integer entries: {values!r}") from exc
    if sorted(integer_values) != list(range(1, n + 1)):
        raise ValueError(f"Claude ranking response is not a permutation of 1..{n}: {integer_values!r}")
    return integer_values


QWEN3_RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
QWEN3_RERANKER_TASK_INSTRUCTION = "Given a social media post, retrieve the scientific paper that the post refers to"


@dataclass(slots=True)
class Qwen3RerankerSpec:
    model_name: str
    instruction: str
    max_length: int
    batch_size: int
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1

    @property
    def method(self) -> str:
        return "qwen3_reranker"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        instruction_hash = hashlib.sha1(self.instruction.encode("utf-8")).hexdigest()[:10]
        return (
            f"maxlen_{self.max_length}_instruction_{instruction_hash}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "model_name": self.model_name,
            "instruction": self.instruction,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Model: {self.model_name}",
            f"- Instruction: {self.instruction}",
            f"- Max length: {self.max_length}",
            f"- Batch size: {self.batch_size}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


LLM_LISTWISE_TOP5_PROMPT = """You are an expert scientific source identifier.

You will be given a social media post that paraphrases a scientific
finding, and a list of candidate scientific papers.

Select the five candidate papers most likely to be the source paper
for the post, and rank them from most likely to least likely.

Return ONLY these five candidate numbers in ranked order as a
comma-separated list.

Do not include any explanation or other text.
Use each candidate number at most once.

Social media post:
{query}

Candidate papers:
{candidates}

Top five ranked candidate numbers:"""

LLM_LISTWISE_TOP5_SELECTION_SIZE = 5


def render_llm_listwise_top5_prompt(
    query: str,
    candidates: list[dict[str, str]],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    if candidate_fields not in {"title", "title_abstract"}:
        raise ValueError(f"Unsupported candidate_fields value: {candidate_fields}")
    candidate_lines: list[str] = []
    for candidate_number, candidate in enumerate(candidates, start=1):
        title = str(candidate.get("title", ""))
        candidate_lines.append(f"{candidate_number}. Title: {title}")
        if candidate_fields == "title_abstract":
            abstract = str(candidate.get("abstract", ""))
            candidate_lines.append(f"   Abstract: {abstract}")
    return LLM_LISTWISE_TOP5_PROMPT.format(query=query, candidates="\n".join(candidate_lines))


@dataclass(slots=True)
class OllamaListwiseTop5RerankerSpec:
    model_name: str
    tokenizer_name: str
    ollama_url: str
    think: bool
    num_predict: int
    temperature: float
    top_p: float
    seed: int
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1

    @property
    def method(self) -> str:
        return "ollama_listwise_top5"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        return (
            f"think_{'on' if self.think else 'off'}_predict_{self.num_predict}"
            f"_temp_{self.temperature:.1f}_topp_{self.top_p:.2f}_seed_{self.seed}"
            f"_candidate_{_slugify(self.candidate_fields)}"
            f"_impl_v{self.implementation_version}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "provider": "ollama",
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "ollama_url": self.ollama_url,
            "think": self.think,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "candidate_fields": self.candidate_fields,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Provider: ollama",
            f"- Model: {self.model_name}",
            f"- Tokenizer: {self.tokenizer_name}",
            f"- Ollama URL: {self.ollama_url}",
            f"- Think: {self.think}",
            f"- Num predict: {self.num_predict}",
            f"- Temperature: {self.temperature}",
            f"- Top p: {self.top_p}",
            f"- Seed: {self.seed}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


def _parse_llm_top5_candidates(text: str, candidate_count: int) -> list[int]:
    return _parse_llm_ranked_candidates(text, candidate_count)[:LLM_LISTWISE_TOP5_SELECTION_SIZE]


CLAIM_DECOMPOSITION_FIELDS = ("claim", "entities", "finding")
CLAIM_JUDGMENT_FIELDS = ("entity", "question", "direct_support")
CLAIM_JUDGMENT_VALUES = (0, 1, 2)
CLAIM_PAIRWISE_ANSWERS = ("A", "B", "TIE")
CLAIM_DECOMPOSITION_PROMPT_VERSION = 3
CLAIM_ABSTRACT_MODES = ("full", "top_sentences")

CLAIM_DECOMPOSITION_PROMPT = """Extract the main scientific finding.
Ignore recommendations, implications, and commentary.
Preserve specific names exactly.

Post:
{post}

{{
  "claim": "",
  "entities": [],
  "finding": ""
}}"""

CLAIM_JUDGMENT_DEFINITIONS = """0 = no
1 = partial/unclear
2 = yes

entity = whether the paper studies the same specific entities; a different specific entity is 0.
question = whether the paper answers the same scientific question, not merely discusses related concepts.
direct_support = whether the paper directly reports the specific finding in the claim, rather than merely discussing the same topic."""

CLAIM_JUDGMENT_PROMPT = """Claim:
{claim}

Entities:
{entities}

Paper:
{paper}

{{
  "entity": 0,
  "question": 0,
  "direct_support": 0
}}

{definitions}"""

CLAIM_BATCH_JUDGMENT_PROMPT = """Target claim:
{claim}

Entities:
{entities}

Papers:

{papers}

Judge each paper independently.

Return:
[
{items}
]

{definitions}"""

CLAIM_PAIRWISE_PROMPT = """Claim:
{claim}

Entities:
{entities}

Finding:
{finding}

A:
{paper_a}

B:
{paper_b}

Which is more likely to be the source?

Answer only:
A
B
TIE"""

CLAIM_DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "finding": {"type": "string"},
    },
    "required": list(CLAIM_DECOMPOSITION_FIELDS),
    "additionalProperties": False,
}

CLAIM_JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        field: {"type": "integer", "enum": list(CLAIM_JUDGMENT_VALUES)} for field in CLAIM_JUDGMENT_FIELDS
    },
    "required": list(CLAIM_JUDGMENT_FIELDS),
    "additionalProperties": False,
}

CLAIM_PAIRWISE_SCHEMA = {"type": "string", "enum": list(CLAIM_PAIRWISE_ANSWERS)}


def claim_batch_judgment_schema(batch_size: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": int(batch_size),
        "maxItems": int(batch_size),
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "enum": list(range(1, int(batch_size) + 1))},
                **{field: {"type": "integer", "enum": list(CLAIM_JUDGMENT_VALUES)} for field in CLAIM_JUDGMENT_FIELDS},
            },
            "required": ["id", *CLAIM_JUDGMENT_FIELDS],
            "additionalProperties": False,
        },
    }

_CODE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9.\-]*\d)(?=[A-Za-z0-9.\-]*[A-Za-z])[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*(?![A-Za-z0-9])"
)


def _claim_paper_text(candidate: dict[str, str], candidate_fields: str) -> str:
    if candidate_fields == "title":
        return str(candidate.get("title", ""))
    if candidate_fields == "title_abstract":
        return f"{candidate.get('title', '')}\n{candidate.get('abstract', '')}"
    raise ValueError(f"Unsupported candidate_fields value: {candidate_fields}")


def _claim_target_values(claim: dict[str, Any]) -> dict[str, str]:
    return {
        "claim": str(claim["claim"]),
        "entities": ", ".join(str(entity) for entity in claim["entities"]),
        "finding": str(claim["finding"]),
    }


def render_claim_decomposition_prompt(post: str) -> str:
    return CLAIM_DECOMPOSITION_PROMPT.format(post=post)


def render_claim_judgment_prompt(
    claim: dict[str, Any],
    candidate: dict[str, str],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    return CLAIM_JUDGMENT_PROMPT.format(
        **_claim_target_values(claim),
        paper=_claim_paper_text(candidate, candidate_fields),
        definitions=CLAIM_JUDGMENT_DEFINITIONS,
    )


def render_claim_batch_judgment_prompt(
    claim: dict[str, Any],
    candidates: list[dict[str, str]],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    papers = "\n\n".join(
        f"[{number}]\n{_claim_paper_text(candidate, candidate_fields)}"
        for number, candidate in enumerate(candidates, start=1)
    )
    items = ",\n".join(
        "  {" + f'"id": {number}, ' + ", ".join(f'"{field}": 0' for field in CLAIM_JUDGMENT_FIELDS) + "}"
        for number in range(1, len(candidates) + 1)
    )
    return CLAIM_BATCH_JUDGMENT_PROMPT.format(
        **_claim_target_values(claim),
        papers=papers,
        items=items,
        definitions=CLAIM_JUDGMENT_DEFINITIONS,
    )


def render_claim_pairwise_prompt(
    claim: dict[str, Any],
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    return CLAIM_PAIRWISE_PROMPT.format(
        **_claim_target_values(claim),
        paper_a=_claim_paper_text(candidate_a, candidate_fields),
        paper_b=_claim_paper_text(candidate_b, candidate_fields),
    )


def _parse_claim_decomposition(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != set(CLAIM_DECOMPOSITION_FIELDS):
        return None
    if not isinstance(payload["claim"], str) or not isinstance(payload["finding"], str):
        return None
    entities = payload["entities"]
    if not isinstance(entities, list) or not all(isinstance(entity, str) for entity in entities):
        return None
    return {field: payload[field] for field in CLAIM_DECOMPOSITION_FIELDS}


def _parse_claim_judgment(text: str) -> dict[str, int] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != set(CLAIM_JUDGMENT_FIELDS):
        return None
    judgment: dict[str, int] = {}
    for field in CLAIM_JUDGMENT_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value not in CLAIM_JUDGMENT_VALUES:
            return None
        judgment[field] = int(value)
    return judgment


def _parse_claim_batch_judgments(text: str, batch_size: int) -> list[dict[str, int]] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or len(payload) != int(batch_size):
        return None
    judgments_by_id: dict[int, dict[str, int]] = {}
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"id", *CLAIM_JUDGMENT_FIELDS}:
            return None
        item_id = item["id"]
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id in judgments_by_id:
            return None
        judgment = _parse_claim_judgment(json.dumps({field: item[field] for field in CLAIM_JUDGMENT_FIELDS}))
        if judgment is None:
            return None
        judgments_by_id[item_id] = judgment
    if set(judgments_by_id) != set(range(1, int(batch_size) + 1)):
        return None
    return [judgments_by_id[number] for number in range(1, int(batch_size) + 1)]


def _parse_claim_pairwise_answer(text: str) -> str | None:
    value = str(text or "").strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = value
    if not isinstance(decoded, str):
        return None
    answer = decoded.strip().upper()
    return answer if answer in CLAIM_PAIRWISE_ANSWERS else None


def _spaced_normalized(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).lower()).split())


def _squashed_normalized(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text).lower())


def _claim_entity_found(entity: str, spaced_text: str, squashed_text: str) -> bool:
    spaced_entity = _spaced_normalized(entity)
    if not spaced_entity:
        return False
    if f" {spaced_entity} " in f" {spaced_text} ":
        return True
    return any(character.isdigit() for character in spaced_entity) and _squashed_normalized(entity) in squashed_text


def claim_entity_anchor(
    entities: list[str],
    target_text: str,
    candidate_text: str,
    *,
    penalize_conflicts: bool = False,
) -> dict[str, Any]:
    usable_entities = [str(entity) for entity in entities if _spaced_normalized(entity)]
    if not usable_entities:
        return {"score": None, "matched": [], "missing": [], "conflicting_identifiers": [], "conflict": False}
    spaced_candidate = _spaced_normalized(candidate_text)
    squashed_candidate = _squashed_normalized(candidate_text)
    matched = [entity for entity in usable_entities if _claim_entity_found(entity, spaced_candidate, squashed_candidate)]
    missing = [entity for entity in usable_entities if entity not in matched]
    squashed_target = _squashed_normalized(" ".join(usable_entities) + " " + str(target_text))
    conflicting_identifiers = sorted(
        {
            identifier
            for identifier in _CODE_IDENTIFIER_PATTERN.findall(str(candidate_text))
            if _squashed_normalized(identifier) not in squashed_target
        }
    )
    conflict = bool(penalize_conflicts) and bool(missing) and bool(conflicting_identifiers)
    score = 0.0 if conflict else len(matched) / len(usable_entities)
    return {
        "score": score,
        "matched": matched,
        "missing": missing,
        "conflicting_identifiers": conflicting_identifiers,
        "conflict": conflict,
    }


def claim_semantic_score(
    judgment: dict[str, int],
    field_weights: dict[str, float],
    *,
    anchor_score: float | None = None,
    anchor_weight: float = 0.0,
) -> float:
    total_weight = sum(float(field_weights[field]) for field in CLAIM_JUDGMENT_FIELDS)
    weighted = sum(
        float(field_weights[field]) * float(judgment[field]) / max(CLAIM_JUDGMENT_VALUES)
        for field in CLAIM_JUDGMENT_FIELDS
    )
    if anchor_score is not None:
        total_weight += float(anchor_weight)
        weighted += float(anchor_weight) * float(anchor_score)
    return weighted / total_weight


def claim_fused_score(
    first_stage_normalized: float,
    semantic_score: float,
    *,
    first_stage_weight: float,
    semantic_weight: float,
) -> float:
    total_weight = float(first_stage_weight) + float(semantic_weight)
    return (
        float(first_stage_weight) * float(first_stage_normalized) + float(semantic_weight) * float(semantic_score)
    ) / total_weight


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(float(value) - low) / (high - low) for value in values]


@dataclass(slots=True)
class OllamaClaimDecompositionRerankerSpec:
    model_name: str
    fallback_model_name: str | None
    ollama_url: str
    think: bool
    temperature: float
    seed: int
    timeout_seconds: int
    max_retries: int
    keep_alive: int
    semantic_top_k: int
    field_weights: tuple[tuple[str, float], ...]
    anchor_weight: float
    first_stage_weight: float
    semantic_weight: float
    pairwise_enabled: bool
    pairwise_max_gap: float
    pairwise_top_n: int
    abstract_mode: str
    abstract_sentence_count: int
    sentence_encoder: str
    judgment_batch_size: int
    anchor_conflicts_enabled: bool
    candidate_fields: str = "title_abstract"
    implementation_version: int = 3

    @property
    def method(self) -> str:
        return "ollama_claim_decomposition"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def field_weight_map(self) -> dict[str, float]:
        return {str(field): float(weight) for field, weight in self.field_weights}

    def _abstract_slug(self) -> str:
        if self.abstract_mode == "top_sentences":
            return f"abstract_top_{self.abstract_sentence_count}_sentences_{self.sentence_encoder}"
        return f"abstract_{self.abstract_mode}"

    @property
    def cache_dir_name(self) -> str:
        weights = "_".join(f"{field}_{self.field_weight_map[field]:g}" for field in CLAIM_JUDGMENT_FIELDS)
        pairwise = (
            f"pair_gap_{self.pairwise_max_gap:g}_top_{self.pairwise_top_n}" if self.pairwise_enabled else "pair_off"
        )
        return _slugify(
            f"think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_semk_{self.semantic_top_k}_w_{weights}_anchor_{self.anchor_weight:g}"
            f"_conflicts_{'on' if self.anchor_conflicts_enabled else 'off'}"
            f"_fuse_{self.first_stage_weight:g}_{self.semantic_weight:g}_{pairwise}"
            f"_batch_{self.judgment_batch_size}"
            f"_{self._abstract_slug()}_candidate_{self.candidate_fields}_impl_v{self.implementation_version}"
        )

    def llm_cache_dir_name(self, model_name: str) -> str:
        return _slugify(
            f"{model_name}_think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_batch_{self.judgment_batch_size}_{self._abstract_slug()}_candidate_{self.candidate_fields}"
            f"_prompt_v{CLAIM_DECOMPOSITION_PROMPT_VERSION}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "provider": "ollama",
            "model_name": self.model_name,
            "fallback_model_name": self.fallback_model_name,
            "ollama_url": self.ollama_url,
            "think": self.think,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "keep_alive": self.keep_alive,
            "semantic_top_k": self.semantic_top_k,
            "field_weights": self.field_weight_map,
            "anchor_weight": self.anchor_weight,
            "first_stage_weight": self.first_stage_weight,
            "semantic_weight": self.semantic_weight,
            "pairwise_enabled": self.pairwise_enabled,
            "pairwise_max_gap": self.pairwise_max_gap,
            "pairwise_top_n": self.pairwise_top_n,
            "abstract_mode": self.abstract_mode,
            "abstract_sentence_count": self.abstract_sentence_count,
            "sentence_encoder": self.sentence_encoder,
            "judgment_batch_size": self.judgment_batch_size,
            "anchor_conflicts_enabled": self.anchor_conflicts_enabled,
            "candidate_fields": self.candidate_fields,
            "prompt_version": CLAIM_DECOMPOSITION_PROMPT_VERSION,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        pairwise = (
            f"adjacent pairs within top {self.pairwise_top_n} with fused gap <= {self.pairwise_max_gap}"
            if self.pairwise_enabled
            else "disabled"
        )
        abstract = (
            f"top {self.abstract_sentence_count} sentences by {self.sentence_encoder} similarity to claim+finding"
            if self.abstract_mode == "top_sentences"
            else self.abstract_mode
        )
        return [
            f"- Reranker kind: {self.method}",
            f"- Provider: ollama",
            f"- Model: {self.model_name}",
            f"- Fallback model: {self.fallback_model_name}",
            f"- Ollama URL: {self.ollama_url}",
            f"- Think: {self.think}",
            f"- Temperature: {self.temperature}",
            f"- Seed: {self.seed}",
            f"- Keep alive: {self.keep_alive}",
            f"- Timeout seconds: {self.timeout_seconds}",
            f"- Max retries: {self.max_retries}",
            f"- Semantic top k: {self.semantic_top_k}",
            f"- Field weights: {self.field_weight_map}, anchor weight: {self.anchor_weight}",
            f"- Anchor conflict penalty: {'on' if self.anchor_conflicts_enabled else 'off'}",
            f"- Fusion: {self.first_stage_weight} * first_stage + {self.semantic_weight} * semantic",
            f"- Pairwise: {pairwise}",
            f"- Judgment batch size: {self.judgment_batch_size}",
            f"- Abstract: {abstract}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


CLAIM_ONLY_PROMPT_VERSION = 3
CLAIM_MATCH_VALUES = (0, 1, 2, 3)
CLAIM_EVIDENCE_VALUES = (0, 1, 2)
CLAIM_CLUE_COUNT_RANGE = (1, 4)
CLAIM_MATCH_CAP_WITH_ENTITY_CONFLICT = 2

CLAIM_CLUE_EXTRACTION_PROMPT = """Extract the scientific finding.

Ignore source/venue, commentary, recommendations, and implications.

entities: only concrete identifying entities where identity matters, such as variant names, drugs, genes/proteins, species, named populations/cohorts, or pathogens.
clues: 1-4 other concrete facts, such as mechanism, comparison, outcome, or direction of the finding. Short factual phrases only.

Post:
{post}

{{
  "claim": "",
  "entities": [],
  "clues": []
}}"""

CLAIM_MATCH_JUDGMENT_PROMPT = """Claim:
{claim}

Entities:
{entities}

Clues:
{clues}

Paper:
{paper}

{{
  "claim_match": 0,
  "entities": [],
  "clues": []
}}

claim_match:
0 = unrelated or contradictory
1 = related topic but different scientific finding/question
2 = same scientific question/finding, partial support
3 = directly supports the claimed finding

entities: one score per entity, in order:
0 = explicitly different entity of the same type
1 = target entity not stated / unclear
2 = target entity matched

clues: one score per clue, in order:
0 = contradicted
1 = not stated / unclear
2 = matched

Rules:
- Absence is not contradiction.
- If the paper is explicitly about a different entity of the same type (for example a different variant, drug, gene, or species than the target), score that entity 0.
- A paper with an explicit entity conflict must not receive claim_match 3.
- Do not infer that a named entity matches merely because the mechanism or topic is similar."""

CLAIM_SOURCE_PAIRWISE_PROMPT = """Claim:
{claim}

A:
{paper_a}

B:
{paper_b}

Which paper is more likely to be the actual source of the claim?

Answer only:
A
B
TIE"""

CLAIM_CLUE_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "clues": {
            "type": "array",
            "minItems": CLAIM_CLUE_COUNT_RANGE[0],
            "maxItems": CLAIM_CLUE_COUNT_RANGE[1],
            "items": {"type": "string"},
        },
    },
    "required": ["claim", "entities", "clues"],
    "additionalProperties": False,
}


def _fixed_score_array(count: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": int(count),
        "maxItems": int(count),
        "items": {"type": "integer", "enum": list(CLAIM_EVIDENCE_VALUES)},
    }


def claim_match_judgment_schema(entity_count: int, clue_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claim_match": {"type": "integer", "enum": list(CLAIM_MATCH_VALUES)},
            "entities": _fixed_score_array(entity_count),
            "clues": _fixed_score_array(clue_count),
        },
        "required": ["claim_match", "entities", "clues"],
        "additionalProperties": False,
    }


def _render_numbered(values: list[str]) -> str:
    if not values:
        return "none"
    return "\n".join(f"{number}. {value}" for number, value in enumerate(values, start=1))


def render_claim_clue_extraction_prompt(post: str) -> str:
    return CLAIM_CLUE_EXTRACTION_PROMPT.format(post=post)


def render_claim_match_judgment_prompt(
    claim: dict[str, Any],
    candidate: dict[str, str],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    return CLAIM_MATCH_JUDGMENT_PROMPT.format(
        claim=str(claim["claim"]),
        entities=_render_numbered(list(claim["entities"])),
        clues=_render_numbered(list(claim["clues"])),
        paper=_claim_paper_text(candidate, candidate_fields),
    )


def render_claim_source_pairwise_prompt(
    claim: dict[str, Any],
    candidate_a: dict[str, str],
    candidate_b: dict[str, str],
    *,
    candidate_fields: str = "title_abstract",
) -> str:
    return CLAIM_SOURCE_PAIRWISE_PROMPT.format(
        claim=str(claim["claim"]),
        paper_a=_claim_paper_text(candidate_a, candidate_fields),
        paper_b=_claim_paper_text(candidate_b, candidate_fields),
    )


def _is_string_list(values: Any) -> bool:
    return isinstance(values, list) and all(isinstance(value, str) and value.strip() for value in values)


def _parse_claim_clue_extraction(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"claim", "entities", "clues"}:
        return None
    if not isinstance(payload["claim"], str) or not payload["claim"].strip():
        return None
    if not _is_string_list(payload["entities"]) or not _is_string_list(payload["clues"]):
        return None
    if not CLAIM_CLUE_COUNT_RANGE[0] <= len(payload["clues"]) <= CLAIM_CLUE_COUNT_RANGE[1]:
        return None
    return {"claim": payload["claim"], "entities": list(payload["entities"]), "clues": list(payload["clues"])}


def _valid_score_list(values: Any, count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == int(count)
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value in CLAIM_EVIDENCE_VALUES
            for value in values
        )
    )


def _parse_claim_match_judgment(text: str, entity_count: int, clue_count: int) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"claim_match", "entities", "clues"}:
        return None
    raw_claim_match = payload["claim_match"]
    if isinstance(raw_claim_match, bool) or not isinstance(raw_claim_match, int) or raw_claim_match not in CLAIM_MATCH_VALUES:
        return None
    if not _valid_score_list(payload["entities"], entity_count) or not _valid_score_list(payload["clues"], clue_count):
        return None
    entities = [int(value) for value in payload["entities"]]
    claim_match = (
        min(int(raw_claim_match), CLAIM_MATCH_CAP_WITH_ENTITY_CONFLICT) if 0 in entities else int(raw_claim_match)
    )
    return {
        "claim_match": claim_match,
        "raw_claim_match": int(raw_claim_match),
        "entities": entities,
        "clues": [int(value) for value in payload["clues"]],
    }


def claim_match_evidence_key(judgment: dict[str, Any]) -> tuple[int, int, int, int, int]:
    entities = [int(value) for value in judgment["entities"]]
    clues = [int(value) for value in judgment["clues"]]
    return (
        -int(judgment["claim_match"]),
        -entities.count(2),
        entities.count(0),
        -clues.count(2),
        clues.count(0),
    )


def claim_match_sort_key(document_id: int, judgment: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    return (*claim_match_evidence_key(judgment), int(document_id))


CLAIM_EXTRACTION_SANITIZER_VERSION = 1
_CLAIM_URL_PATTERN = re.compile(r"(https?://|www\.|\b[\w-]+\.(com|org|net|gov|edu|io|ly)\b)", re.IGNORECASE)
_CLAIM_EDGE_CHARACTERS = "#\"'.,;:!?()[]{}<>*_ \t\n"
_CLAIM_SURFACE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.\-/]*[A-Za-z0-9])?")


def claim_venue_keys(venues: list[str]) -> set[str]:
    keys: set[str] = set()
    for venue in venues:
        key = _squashed_normalized(venue)
        if key:
            keys.add(key)
        if str(venue).strip().lower().startswith("the "):
            stripped = _squashed_normalized(str(venue).strip()[4:])
            if stripped:
                keys.add(stripped)
    return keys


def _sanitize_claim_item(value: str, venue_keys: set[str]) -> tuple[str | None, str | None]:
    text = str(value).strip()
    if "@" in text:
        return None, "handle"
    if _CLAIM_URL_PATTERN.search(text):
        return None, "url"
    text = text.strip(_CLAIM_EDGE_CHARACTERS).lstrip("#").strip(_CLAIM_EDGE_CHARACTERS)
    if not _squashed_normalized(text):
        return None, "empty"
    key = _squashed_normalized(text)
    without_article = _squashed_normalized(text[4:]) if text.lower().startswith("the ") else key
    if key in venue_keys or without_article in venue_keys:
        return None, "venue"
    return text, None


def sanitize_claim_extraction(extraction: dict[str, Any] | None, venue_keys: set[str]) -> dict[str, Any] | None:
    if extraction is None:
        return None
    seen: set[str] = set()
    dropped: list[dict[str, str]] = []
    cleaned: dict[str, list[str]] = {"entities": [], "clues": []}
    for field in ("entities", "clues"):
        for value in extraction[field]:
            text, reason = _sanitize_claim_item(value, venue_keys)
            if text is None:
                dropped.append({"field": field, "value": str(value), "reason": str(reason)})
                continue
            key = _squashed_normalized(text)
            if key in seen:
                dropped.append({"field": field, "value": str(value), "reason": "duplicate"})
                continue
            seen.add(key)
            cleaned[field].append(text)
    if not cleaned["clues"]:
        return None
    return {
        "claim": extraction["claim"],
        "entities": cleaned["entities"],
        "clues": cleaned["clues"][: CLAIM_CLUE_COUNT_RANGE[1]],
        "raw_entities": list(extraction["entities"]),
        "raw_clues": list(extraction["clues"]),
        "dropped": dropped,
    }


def canonicalize_claim_entities(entities: list[str], candidate_texts: list[str]) -> tuple[list[str], dict[str, str]]:
    surfaces: dict[str, Counter] = {}
    for text in candidate_texts:
        for token in _CLAIM_SURFACE_TOKEN_PATTERN.findall(str(text)):
            surfaces.setdefault(_squashed_normalized(token), Counter())[token] += 1
    canonical: list[str] = []
    changes: dict[str, str] = {}
    for entity in entities:
        replacement = entity
        if " " not in entity.strip():
            counts = surfaces.get(_squashed_normalized(entity))
            if counts:
                replacement = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if replacement != entity:
            changes[entity] = replacement
        canonical.append(replacement)
    return canonical, changes


@dataclass(slots=True)
class OllamaClaimJudgmentRerankerSpec:
    model_name: str
    fallback_model_name: str | None
    ollama_url: str
    think: bool
    temperature: float
    seed: int
    timeout_seconds: int
    max_retries: int
    keep_alive: int
    semantic_top_k: int
    pairwise_enabled: bool
    pairwise_max_group_size: int | None
    candidate_fields: str = "title_abstract"
    implementation_version: int = 1

    @property
    def method(self) -> str:
        return "ollama_claim_judgment"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def cache_dir_name(self) -> str:
        pairwise = (
            f"pair_rank1_ties_cap_{self.pairwise_max_group_size if self.pairwise_max_group_size else 'none'}"
            if self.pairwise_enabled
            else "pair_off"
        )
        return _slugify(
            f"think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_semk_{self.semantic_top_k}_{pairwise}_candidate_{self.candidate_fields}"
            f"_impl_v{self.implementation_version}"
        )

    def llm_cache_dir_name(self, model_name: str) -> str:
        return _slugify(
            f"{model_name}_think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_candidate_{self.candidate_fields}_prompt_v{CLAIM_ONLY_PROMPT_VERSION}"
            f"_sanitize_v{CLAIM_EXTRACTION_SANITIZER_VERSION}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "provider": "ollama",
            "model_name": self.model_name,
            "fallback_model_name": self.fallback_model_name,
            "ollama_url": self.ollama_url,
            "think": self.think,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "keep_alive": self.keep_alive,
            "semantic_top_k": self.semantic_top_k,
            "pairwise_enabled": self.pairwise_enabled,
            "pairwise_max_group_size": self.pairwise_max_group_size,
            "candidate_fields": self.candidate_fields,
            "prompt_version": CLAIM_ONLY_PROMPT_VERSION,
            "extraction_sanitizer_version": CLAIM_EXTRACTION_SANITIZER_VERSION,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        pairwise = (
            "knockout among candidates exactly tied with rank 1 on the full evidence key, both orders, "
            f"group cap={self.pairwise_max_group_size}"
            if self.pairwise_enabled
            else "disabled"
        )
        return [
            f"- Reranker kind: {self.method}",
            f"- Provider: ollama",
            f"- Model: {self.model_name}",
            f"- Fallback model: {self.fallback_model_name}",
            f"- Ollama URL: {self.ollama_url}",
            f"- Think: {self.think}",
            f"- Temperature: {self.temperature}",
            f"- Seed: {self.seed}",
            f"- Keep alive: {self.keep_alive}",
            f"- Timeout seconds: {self.timeout_seconds}",
            f"- Max retries: {self.max_retries}",
            f"- Judged candidates: top {self.semantic_top_k} first-stage candidates; first-stage scores and ranks discarded",
            "- Ordering: claim_match desc (capped at 2 with an entity conflict), entities scored 2 desc, "
            "entities scored 0 asc, clues scored 2 desc, clues scored 0 asc, document id asc",
            f"- Pairwise: {pairwise}",
            f"- Candidate fields: {self.candidate_fields}",
        ]


LISTWISE_CLAIM_PROMPT_VERSION = 2

LISTWISE_CLAIM_INSTRUCTION = """Rank the papers by how likely they are to be the source of the post.

Use the claim as a summary, but preserve specific evidence from the post, including named entities/treatments, populations or subgroups, numbers, outcomes, and comparisons."""

LISTWISE_CLAIM_TITLE_PROMPT = """Post:
{post}

Claim:
{claim}

Candidates:
{candidates}

{instruction}

Return only:
{{"ranking": [IDs]}}"""

LISTWISE_CLAIM_ABSTRACT_PROMPT = """Post:
{post}

Claim:
{claim}

Candidates:

{candidates}

{instruction}

Return only:
{{"ranking": [IDs]}}"""


def listwise_claim_ranking_schema(labels: list[int]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "minItems": len(labels),
                "maxItems": len(labels),
                "items": {"type": "integer", "enum": sorted(int(label) for label in labels)},
            }
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }


def render_listwise_claim_title_prompt(post: str, claim: str, labelled_titles: list[tuple[int, str]]) -> str:
    return LISTWISE_CLAIM_TITLE_PROMPT.format(
        post=post,
        claim=claim,
        candidates="\n".join(f"{label}. {title}" for label, title in labelled_titles),
        instruction=LISTWISE_CLAIM_INSTRUCTION,
    )


def render_listwise_claim_abstract_prompt(
    post: str,
    claim: str,
    labelled_papers: list[tuple[int, dict[str, str]]],
) -> str:
    return LISTWISE_CLAIM_ABSTRACT_PROMPT.format(
        post=post,
        claim=claim,
        instruction=LISTWISE_CLAIM_INSTRUCTION,
        candidates="\n\n".join(
            f"{label}\nTitle: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"
            for label, paper in labelled_papers
        ),
    )


LISTWISE_CLAIM_WINDOW_INSTRUCTION = (
    LISTWISE_CLAIM_INSTRUCTION
    + "\n\nSome papers are marked \"Abstract incomplete\". Missing abstract detail is not evidence against a paper; "
    "rank every paper from the evidence that is available."
)
LISTWISE_CLAIM_INCOMPLETE_MARKER = "Abstract incomplete"


def render_listwise_claim_window_prompt(
    post: str,
    claim: str,
    labelled_papers: list[tuple[int, dict[str, str], bool]],
) -> str:
    blocks = []
    for label, paper, complete in labelled_papers:
        lines = [str(label), f"Title: {paper.get('title', '')}", f"Abstract: {paper.get('abstract', '')}"]
        if not complete:
            lines.append(LISTWISE_CLAIM_INCOMPLETE_MARKER)
        blocks.append("\n".join(lines))
    return LISTWISE_CLAIM_ABSTRACT_PROMPT.format(
        post=post,
        claim=claim,
        instruction=LISTWISE_CLAIM_WINDOW_INSTRUCTION,
        candidates="\n\n".join(blocks),
    )


def sliding_window_starts(candidate_count: int, window_size: int, step: int) -> list[int]:
    if candidate_count <= window_size:
        return [0]
    starts = list(range(candidate_count - window_size, -1, -int(step)))
    if starts[-1] != 0:
        starts.append(0)
    return starts


def apply_window_ranking(window_documents: list[int], ranking_labels: list[int]) -> list[int]:
    labels = list(range(1, len(window_documents) + 1))
    positions = listwise_positions(ranking_labels, labels)
    return [
        window_documents[label - 1]
        for label in sorted(labels, key=lambda label: (positions[label], label))
    ]


def protect_incomplete_in_window(
    window_documents: list[int],
    raw_order: list[int],
    incomplete: set[int],
) -> tuple[list[int], dict[str, Any]]:
    incoming = {document_id: position for position, document_id in enumerate(window_documents)}
    raw = {document_id: position for position, document_id in enumerate(raw_order)}
    protected = [document_id for document_id in window_documents if document_id in incomplete]
    slots: list[int | None] = [None] * len(window_documents)
    clamped = [document_id for document_id in protected if raw[document_id] > incoming[document_id]]
    promoted = sorted(
        (document_id for document_id in protected if raw[document_id] <= incoming[document_id]),
        key=lambda document_id: raw[document_id],
    )
    for document_id in clamped:
        slots[incoming[document_id]] = document_id
    for document_id in promoted:
        allowed = [position for position in range(raw[document_id], incoming[document_id] + 1) if slots[position] is None]
        if not allowed:
            allowed = [position for position in range(raw[document_id] - 1, -1, -1) if slots[position] is None]
        slots[allowed[0]] = document_id
    remaining = iter(document_id for document_id in raw_order if document_id not in incomplete)
    order = [slot if slot is not None else next(remaining) for slot in slots]
    final = {document_id: position for position, document_id in enumerate(order)}
    return order, {
        "protected": [
            {
                "document_id": document_id,
                "incoming_position": incoming[document_id] + 1,
                "raw_position": raw[document_id] + 1,
                "protected_position": final[document_id] + 1,
                "clamped": document_id in clamped,
            }
            for document_id in protected
        ],
        "displaced": [
            {
                "document_id": document_id,
                "raw_position": raw[document_id] + 1,
                "protected_position": final[document_id] + 1,
            }
            for document_id in raw_order
            if document_id not in incomplete and final[document_id] > raw[document_id]
        ],
    }


def abstract_length_protection(word_count: int, full_protection_words: int, no_protection_words: int) -> float:
    if word_count <= int(full_protection_words):
        return 1.0
    if word_count >= int(no_protection_words):
        return 0.0
    return (float(no_protection_words) - float(word_count)) / (float(no_protection_words) - float(full_protection_words))


def soft_protect_window(
    window_documents: list[int],
    raw_order: list[int],
    word_counts: dict[int, int],
    *,
    full_protection_words: int,
    no_protection_words: int,
) -> tuple[list[int], dict[str, Any]]:
    incoming = {document_id: position for position, document_id in enumerate(window_documents, start=1)}
    raw = {document_id: position for position, document_id in enumerate(raw_order, start=1)}
    protection = {
        document_id: abstract_length_protection(word_counts[document_id], full_protection_words, no_protection_words)
        for document_id in window_documents
    }
    effective = {
        document_id: protection[document_id] * incoming[document_id] + (1.0 - protection[document_id]) * raw[document_id]
        for document_id in window_documents
    }
    order = sorted(window_documents, key=lambda document_id: (effective[document_id], raw[document_id]))
    final = {document_id: position for position, document_id in enumerate(order, start=1)}
    return order, {
        "protected": [
            {
                "document_id": document_id,
                "words": int(word_counts[document_id]),
                "protection": protection[document_id],
                "incoming_position": incoming[document_id],
                "raw_position": raw[document_id],
                "effective_rank": effective[document_id],
                "protected_position": final[document_id],
                "clamped": False,
            }
            for document_id in window_documents
            if protection[document_id] > 0.0
        ],
        "displaced": [
            {
                "document_id": document_id,
                "raw_position": raw[document_id],
                "protected_position": final[document_id],
            }
            for document_id in raw_order
            if protection[document_id] == 0.0 and final[document_id] > raw[document_id]
        ],
    }


def _parse_listwise_claim_ranking(text: str, labels: list[int]) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"ranking"} or not isinstance(payload["ranking"], list):
        return None
    allowed = {int(label) for label in labels}
    ranking: list[int] = []
    duplicates: list[int] = []
    invalid: list[Any] = []
    for value in payload["ranking"]:
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            invalid.append(value)
            continue
        if value in ranking:
            duplicates.append(int(value))
            continue
        ranking.append(int(value))
    if not ranking:
        return None
    return {
        "ranking": ranking,
        "missing": [label for label in sorted(allowed) if label not in ranking],
        "duplicates": duplicates,
        "invalid": invalid,
    }


_ABSTRACT_WORD_PATTERN = re.compile(r"\w+")


def abstract_word_count(abstract: str) -> int:
    return len(_ABSTRACT_WORD_PATTERN.findall(str(abstract or "")))


def abstract_is_sufficient(
    word_count: int,
    sentence_count: int,
    *,
    min_words: int | None,
    min_sentences: int | None,
) -> bool:
    if word_count == 0:
        return False
    if min_words is not None and word_count < int(min_words):
        return False
    if min_sentences is not None and sentence_count < int(min_sentences):
        return False
    return True


def combine_title_and_abstract_ranks(
    documents: list[int],
    title_mean_rank: dict[int, float],
    abstract_mean_rank: dict[int, float],
    sufficient: dict[int, bool],
    *,
    title_weight: float,
    abstract_weight: float,
) -> tuple[list[int], dict[int, float]]:
    combined: dict[int, float] = {}
    for document_id in documents:
        if sufficient[document_id]:
            combined[document_id] = float(title_weight) * float(title_mean_rank[document_id]) + float(
                abstract_weight
            ) * float(abstract_mean_rank[document_id])
        else:
            combined[document_id] = float(title_mean_rank[document_id])
    order = sorted(documents, key=lambda document_id: (combined[document_id], int(document_id)))
    return order, combined


def listwise_positions(ranking: list[int], labels: list[int]) -> dict[int, float]:
    positions = {int(label): float(position) for position, label in enumerate(ranking, start=1)}
    missing = [int(label) for label in labels if int(label) not in positions]
    if missing:
        shared_position = statistics.fmean(range(len(ranking) + 1, len(labels) + 1))
        for label in missing:
            positions[label] = float(shared_position)
    return positions


def aggregate_listwise_rankings(
    rankings: list[list[int]],
    label_to_document: dict[int, int],
) -> tuple[list[int], dict[int, float]]:
    labels = sorted(label_to_document)
    position_maps = [listwise_positions(ranking, labels) for ranking in rankings]
    mean_rank = {
        label_to_document[label]: statistics.fmean(position_map[label] for position_map in position_maps)
        for label in labels
    }
    order = sorted(mean_rank, key=lambda document_id: (mean_rank[document_id], int(document_id)))
    return order, mean_rank


def spearman_with_display_order(ranking: list[int], display_labels: list[int]) -> float | None:
    labels = list(display_labels)
    if len(labels) < 2:
        return None
    returned = listwise_positions(ranking, labels)
    shown = {label: float(position) for position, label in enumerate(labels, start=1)}
    returned_values = [returned[label] for label in labels]
    shown_values = [shown[label] for label in labels]
    returned_mean = statistics.fmean(returned_values)
    shown_mean = statistics.fmean(shown_values)
    covariance = sum((a - returned_mean) * (b - shown_mean) for a, b in zip(returned_values, shown_values, strict=True))
    returned_spread = math.sqrt(sum((a - returned_mean) ** 2 for a in returned_values))
    shown_spread = math.sqrt(sum((b - shown_mean) ** 2 for b in shown_values))
    if returned_spread == 0 or shown_spread == 0:
        return None
    return covariance / (returned_spread * shown_spread)


@dataclass(slots=True)
class OllamaListwiseClaimRerankerSpec:
    model_name: str
    fallback_model_name: str | None
    tokenizer_name: str
    ollama_url: str
    think: bool
    temperature: float
    seed: int
    timeout_seconds: int
    max_retries: int
    keep_alive: int
    semantic_top_k: int
    abstract_stage_top_n: int
    abstract_min_words: int | None = None
    abstract_min_sentences: int | None = None
    title_weight: float = 0.6
    abstract_weight: float = 0.4
    abstract_stage_mode: str = "sliding_window"
    window_size: int = 10
    window_step: int = 5
    incomplete_protection: str = "soft"
    protection_full_words: int = 24
    protection_none_words: int = 64
    candidate_fields: str = "title_abstract"
    implementation_version: int = 5

    @property
    def method(self) -> str:
        return "ollama_listwise_claim"

    @property
    def key(self) -> str:
        return _slugify(self.model_name)

    @property
    def pairwise_enabled(self) -> bool:
        return False

    @property
    def pairwise_max_group_size(self) -> int | None:
        return None

    @property
    def cache_dir_name(self) -> str:
        return _slugify(
            f"think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_semk_{self.semantic_top_k}_abstract_top_{self.abstract_stage_top_n}"
            f"_min_words_{self.abstract_min_words}_min_sentences_{self.abstract_min_sentences}"
            f"_weights_{self.title_weight:g}_{self.abstract_weight:g}"
            f"_{self.abstract_stage_mode}_w{self.window_size}_s{self.window_step}"
            f"_protect_{self.incomplete_protection}_{self.protection_full_words}_{self.protection_none_words}"
            f"_impl_v{self.implementation_version}"
        )

    def llm_cache_dir_name(self, model_name: str) -> str:
        return _slugify(
            f"{model_name}_think_{'on' if self.think else 'off'}_temp_{self.temperature:.1f}_seed_{self.seed}"
            f"_extract_v{CLAIM_ONLY_PROMPT_VERSION}_sanitize_v{CLAIM_EXTRACTION_SANITIZER_VERSION}"
            f"_listwise_v{LISTWISE_CLAIM_PROMPT_VERSION}"
        )

    def summary_dict(self) -> dict[str, Any]:
        return {
            "kind": self.method,
            "provider": "ollama",
            "model_name": self.model_name,
            "fallback_model_name": self.fallback_model_name,
            "tokenizer_name": self.tokenizer_name,
            "ollama_url": self.ollama_url,
            "think": self.think,
            "temperature": self.temperature,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "keep_alive": self.keep_alive,
            "semantic_top_k": self.semantic_top_k,
            "abstract_stage_top_n": self.abstract_stage_top_n,
            "abstract_min_words": self.abstract_min_words,
            "abstract_min_sentences": self.abstract_min_sentences,
            "title_weight": self.title_weight,
            "abstract_weight": self.abstract_weight,
            "abstract_stage_mode": self.abstract_stage_mode,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "incomplete_protection": self.incomplete_protection,
            "protection_full_words": self.protection_full_words,
            "protection_none_words": self.protection_none_words,
            "extraction_prompt_version": CLAIM_ONLY_PROMPT_VERSION,
            "extraction_sanitizer_version": CLAIM_EXTRACTION_SANITIZER_VERSION,
            "listwise_prompt_version": LISTWISE_CLAIM_PROMPT_VERSION,
            "implementation_version": self.implementation_version,
        }

    def config_dict(self) -> dict[str, Any]:
        return self.summary_dict()

    def report_lines(self) -> list[str]:
        return [
            f"- Reranker kind: {self.method}",
            f"- Provider: ollama",
            f"- Model: {self.model_name} (tokenizer {self.tokenizer_name})",
            f"- Fallback model: {self.fallback_model_name}",
            f"- Think: {self.think}, temperature: {self.temperature}, seed: {self.seed}, keep alive: {self.keep_alive}",
            f"- Candidates: top {self.semantic_top_k} first-stage candidates; first-stage scores and ranks discarded",
            "- Ranking prompts: original post plus extracted claim as a summary",
            "- Stage 1: rank all titles, document-id order and reversed, aggregated by mean rank",
            f"- Stage 2: rank the stage-1 top {self.abstract_stage_top_n} with title and abstract, both orders, mean rank",
            f"- Final top {self.abstract_stage_top_n}: {self.title_weight} * title mean rank + {self.abstract_weight} * "
            "abstract mean rank for sufficient abstracts; title mean rank only for insufficient abstracts",
            f"- Insufficient abstract: empty, fewer than {self.abstract_min_words} words, or fewer than "
            f"{self.abstract_min_sentences} sentences (None = not applied)",
            "- Positions after the top: title-stage order",
            f"- Abstract stage mode: {self.abstract_stage_mode} (sliding window: size {self.window_size}, "
            f"step {self.window_step}, bottom-to-top, one call per window, final order = window output)",
            f"- Incomplete-abstract protection: {self.incomplete_protection} (soft: protection 1 at <= "
            f"{self.protection_full_words} words, 0 at >= {self.protection_none_words} words, linear between; "
            "effective rank = protection * incoming rank + (1 - protection) * raw LLM rank; "
            "strict: insufficient abstracts may not move below their incoming position)",
        ]


RerankerSpec = CrossEncoderRerankerSpec
AnyRerankerSpec = (
    CrossEncoderRerankerSpec
    | FlagEmbeddingRerankerSpec
    | JinaComputeScoreRerankerSpec
    | LlamaNemotronRerankerSpec
    | TransformersListwiseRerankerSpec
    | LLMPointwiseRerankerSpec
    | LLMPairwiseRerankerSpec
    | ClaudeListwiseRerankerSpec
    | Qwen3RerankerSpec
    | OllamaListwiseTop5RerankerSpec
    | OllamaClaimDecompositionRerankerSpec
    | OllamaClaimJudgmentRerankerSpec
    | OllamaListwiseClaimRerankerSpec
)


@dataclass(slots=True)
class ConditionMetrics:
    mrr_at_5: float
    reranked_queries: int
    reranked_pairs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mrr_at_5": self.mrr_at_5,
            "reranked_queries": self.reranked_queries,
            "reranked_pairs": self.reranked_pairs,
        }


class MultiStepPaths:
    def __init__(
        self,
        store: ArtifactStore,
        selection_namespace: str,
        language: str,
        retriever_name: str,
        reranker: AnyRerankerSpec,
        top_k: int,
    ) -> None:
        self.top_k = int(top_k)
        self.selection_namespace = str(selection_namespace)
        self.language = str(language)
        self.retriever_name = str(retriever_name)
        self.store = store
        self.root = (
            store.root_dir
            / "multistep"
            / self.selection_namespace
            / language
            / retriever_name
            / reranker.method
            / reranker.key
            / f"top_{self.top_k}"
            / reranker.cache_dir_name
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def test_score_path(self) -> Path:
        return self.root / f"test_top{self.top_k}_scores.parquet"

    def test_outcome_path(self) -> Path:
        return self.root / "test_query_outcomes.parquet"

    def test_ranking_path(self) -> Path:
        return self.root / "test_rankings.parquet"

    def summary_path(self) -> Path:
        return self.root / "summary.json"

    def report_path(self) -> Path:
        return self.root / "report.md"

    def config_path(self) -> Path:
        return self.root / "config.json"

    def score_checkpoint_dir(self) -> Path:
        checkpoint_dir = self.root / "_score_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def score_checkpoint_path(self, chunk_index: int) -> Path:
        return self.score_checkpoint_dir() / f"chunk_{int(chunk_index):05d}.parquet"

    def score_checkpoint_paths(self) -> list[Path]:
        checkpoint_dir = self.root / "_score_checkpoints"
        if not checkpoint_dir.exists():
            return []
        return sorted(checkpoint_dir.glob("chunk_*.parquet"))

    def next_score_checkpoint_index(self) -> int:
        highest_index = -1
        for path in self.score_checkpoint_paths():
            match = re.fullmatch(r"chunk_(\d+)\.parquet", path.name)
            if match is None:
                continue
            highest_index = max(highest_index, int(match.group(1)))
        return highest_index + 1

    def clear_score_checkpoints(self) -> int:
        checkpoint_paths = self.score_checkpoint_paths()
        checkpoint_dir = self.root / "_score_checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return len(checkpoint_paths)

    def shared_cache_root(self) -> Path:
        root = self.root.parent.parent / "_shared" / self.root.name
        root.mkdir(parents=True, exist_ok=True)
        return root

    def stale_paths(self) -> list[Path]:
        return [
            self.root / "dev_stage5_probabilities.parquet",
            self.root / "test_stage5_probabilities.parquet",
            self.root / "dev_top5_scores.parquet",
            self.root / "dev_base_query_outcomes.parquet",
            self.root / "test_base_query_outcomes.parquet",
            self.root / "dev_query_outcomes.parquet",
            self.root / "dev_rankings.parquet",
            self.root / "threshold_selection.json",
        ]

    def exact_cache_candidates(self, reranker: AnyRerankerSpec, top_k: int) -> list[Path]:
        patterns = [
            (
                f"*/{self.language}/{self.retriever_name}/{reranker.method}/{reranker.key}/"
                f"top_{top_k}/{reranker.cache_dir_name}/test_top{top_k}_scores.parquet"
            ),
            (
                f"*/{self.language}/{self.retriever_name}/{reranker.key}/"
                f"top_{top_k}/{reranker.cache_dir_name}/test_top{top_k}_scores.parquet"
            ),
        ]
        seen: set[str] = set()
        candidates: list[Path] = []
        for pattern in patterns:
            for path in (self.store.root_dir / "multistep").glob(pattern):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
        return candidates

    def superset_cache_candidates(self, reranker: AnyRerankerSpec) -> list[tuple[int, Path]]:
        candidates: list[tuple[int, Path]] = []
        roots = [
            self.store.root_dir / "multistep",
        ]
        seen: set[str] = set()
        for root in roots:
            current_pattern = (
                f"*/{self.language}/{self.retriever_name}/{reranker.method}/{reranker.key}/top_*/"
                f"{reranker.cache_dir_name}/test_top*_scores.parquet"
            )
            legacy_pattern = (
                f"*/{self.language}/{self.retriever_name}/{reranker.key}/top_*/"
                f"{reranker.cache_dir_name}/test_top*_scores.parquet"
            )
            for pattern in [current_pattern, legacy_pattern]:
                for path in root.glob(pattern):
                    key = str(path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    top_dir = path.parent.parent.name
                    try:
                        cached_top_k = int(top_dir.split("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    if cached_top_k >= self.top_k:
                        candidates.append((cached_top_k, path))
        return candidates


def _ordinal_selection_mask(frame: pd.DataFrame, top_k: int) -> pd.Series:
    selected_size = frame["selected_size"].fillna(-1).astype(float)
    return (
        (selected_size > 0.0)
        & (selected_size <= float(top_k))
        & (~frame["predicted_failure"].astype(bool))
    )


def _canonical_cross_encoder_name(model_name: str) -> str:
    normalized = model_name.strip()
    aliases = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "ettin-reranker-17m-v1": "cross-encoder/ettin-reranker-17m-v1",
        "ettin-reranker-32m-v1": "cross-encoder/ettin-reranker-32m-v1",
        "ettin-reranker-150m-v1": "cross-encoder/ettin-reranker-150m-v1",
    }
    return aliases.get(normalized, normalized)


def _is_modular_ettin_reranker(model_name: str) -> bool:
    normalized = _canonical_cross_encoder_name(model_name)
    return normalized.startswith("cross-encoder/ettin-reranker-")


def _sanitize_tokenizer_config(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    if updated.get("tokenizer_class") == "TokenizersBackend":
        updated.pop("tokenizer_class", None)
    return updated


def _requires_cross_encoder_remote_code(model_name: str) -> bool:
    normalized = _canonical_cross_encoder_name(model_name).lower()
    return normalized.startswith("jinaai/jina-reranker-")


def _cross_encoder_tokenizer_kwargs(model_name: str) -> dict[str, Any] | None:
    if _requires_cross_encoder_remote_code(model_name):
        return {"padding_side": "right"}
    return None


def _assign_cross_encoder_pad_token(model: CrossEncoder) -> bool:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return False
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return True
    for token_name in ("eos_token", "sep_token", "cls_token", "unk_token"):
        token_value = getattr(tokenizer, token_name, None)
        if token_value is None:
            continue
        try:
            tokenizer.pad_token = token_value
        except Exception:
            continue
        break
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return False
    model_config = getattr(getattr(model, "model", None), "config", None)
    if model_config is not None and getattr(model_config, "pad_token_id", None) is None:
        try:
            model_config.pad_token_id = int(pad_token_id)
        except Exception:
            pass
    return True


class OrdinalPredictorSelections:
    def __init__(
        self,
        store: ArtifactStore,
        console: Any,
        features: Any,
        retrieval: RetrievalManager,
        language: str,
        retriever_name: str,
        top_k: int,
    ) -> None:
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.language = language
        self.retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        self.top_k = int(top_k)

    def test(self) -> pd.DataFrame:
        frame = self._test_predictions()
        selected = _ordinal_selection_mask(frame, self.top_k)
        return pd.DataFrame(
            {
                "query_id": frame["query_id"].astype(int),
                "selected_for_selective_rerank": selected.astype(bool),
            }
        )

    def _test_predictions(self) -> pd.DataFrame:
        return load_or_create_calibrated_ordinal_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
        )

    @property
    def namespace(self) -> str:
        return "calibrated_ordinal"

    @property
    def source(self) -> str:
        return f"calibrated ordinal predictor output on {self.language}_test"

    @property
    def rule(self) -> str:
        return f"selected_size <= {self.top_k} and predicted_failure == False"


class HitAtOnePredictorSelections:
    def __init__(
        self,
        store: ArtifactStore,
        console: Any,
        features: Any,
        retrieval: RetrievalManager,
        language: str,
        retriever_name: str,
    ) -> None:
        self.store = store
        self.console = console
        self.features = features
        self.retrieval = retrieval
        self.language = language
        self.retriever_name = self.retrieval.resolve_retriever_name(retriever_name)

    def test(self) -> pd.DataFrame:
        frame = self._test_predictions()
        return pd.DataFrame(
            {
                "query_id": frame["query_id"].astype(int),
                "selected_for_selective_rerank": (~frame["predicted_hit_at_1"].astype(bool)).astype(bool),
            }
        )

    def _test_predictions(self) -> pd.DataFrame:
        return load_or_create_hit_at_1_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
        )

    @property
    def namespace(self) -> str:
        return "hit_at_1"

    @property
    def source(self) -> str:
        return f"hit@1 predictor output on {self.language}_test"

    @property
    def rule(self) -> str:
        return "predicted_hit_at_1 == False"


class OrdinalAndHitAtOnePredictorSelections:
    def __init__(
        self,
        store: ArtifactStore,
        console: Any,
        features: Any,
        retrieval: RetrievalManager,
        language: str,
        retriever_name: str,
        top_k: int,
    ) -> None:
        self.store = store
        self.language = language
        self.top_k = int(top_k)
        self.ordinal = OrdinalPredictorSelections(
            store=store,
            console=console,
            features=features,
            retrieval=retrieval,
            language=language,
            retriever_name=retriever_name,
            top_k=top_k,
        )
        self.hit_at_1 = HitAtOnePredictorSelections(
            store=store,
            console=console,
            features=features,
            retrieval=retrieval,
            language=language,
            retriever_name=retriever_name,
        )

    def test(self) -> pd.DataFrame:
        ordinal_frame = self.ordinal._test_predictions()[["query_id", "selected_size", "predicted_failure"]].copy()
        hit_frame = self.hit_at_1._test_predictions()[["query_id", "predicted_hit_at_1"]].copy()
        if hit_frame.duplicated("query_id").any():
            hit_frame = hit_frame.drop_duplicates("query_id")
        merged = ordinal_frame.merge(hit_frame, on="query_id", how="inner")
        selected = _ordinal_selection_mask(merged, self.top_k) & (~merged["predicted_hit_at_1"].astype(bool))
        return pd.DataFrame(
            {
                "query_id": merged["query_id"].astype(int),
                "selected_for_selective_rerank": selected.astype(bool),
            }
        )

    @property
    def namespace(self) -> str:
        return "calibrated_ordinal_and_hit_at_1"

    @property
    def source(self) -> str:
        return f"calibrated ordinal predictor and hit@1 predictor output on {self.language}_test"

    @property
    def rule(self) -> str:
        return (
            f"selected_size <= {self.top_k} and predicted_failure == False and predicted_hit_at_1 == False"
        )


class TopKScoreRunner:
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: AnyRerankerSpec,
        top_k: int,
    ) -> None:
        self.console = console
        self.store = store
        self.retrieval = retrieval
        self.paths = paths
        self.reranker = reranker
        self.top_k = int(top_k)

    def score_test_split(
        self,
        retriever_name: str,
        language: str,
        selected_query_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        total_started_at = time.perf_counter()
        split_key = f"{language}_test"
        resolved_retriever = self.retrieval.resolve_retriever_name(retriever_name)
        cache_path = self.paths.test_score_path()
        query_filter = None if selected_query_ids is None else {int(query_id) for query_id in selected_query_ids}
        retrieval_started_at = time.perf_counter()
        top_frame = self._reference_top_frame(split_key, resolved_retriever, query_filter)
        if cache_path.exists():
            self.console.log(
                "Using cached reranker scores for en_test"
                if query_filter is None
                else "Checking cached reranker scores for the requested query subset"
            )
            read_started_at = time.perf_counter()
            cached = self.store.read_parquet(cache_path)
            if not self._cache_covers_request(cached, split_key, query_filter):
                self.console.log(
                    (
                        f"Ignoring incomplete cached reranker scores at {cache_path.name}; "
                        "the cache does not cover the full test split."
                    )
                    if query_filter is None
                    else (
                        f"Ignoring cached reranker scores at {cache_path.name}; "
                        "the cache does not cover the requested query subset."
                    )
                )
            elif not self._matches_reference(cached, top_frame):
                self.console.log(
                    f"Ignoring cached reranker scores at {cache_path.name}; its (query, rank, document) pairs "
                    f"do not match the current {resolved_retriever} top-{self.top_k} ranking."
                )
            else:
                cached = self._subset_cached_frame(cached, query_filter)
                _console_log_timing(
                    self.console,
                    "Loaded cached reranker scores",
                    time.perf_counter() - read_started_at,
                    details=f"split={split_key}, rows={len(cached)}, path={cache_path.name}",
                )
                _console_log_timing(
                    self.console,
                    "Reranker scoring pipeline complete",
                    time.perf_counter() - total_started_at,
                    details=f"split={split_key}, reused_cache=true, rows={len(cached)}",
                )
                return cached
        exact_started_at = time.perf_counter()
        exact_cache = self._load_exact_cache_from_existing_runs(split_key, query_filter, reference=top_frame)
        if exact_cache is not None:
            self.console.log(
                "Importing cached reranker scores from an existing run"
                if query_filter is None
                else "Importing cached reranker scores for the requested query subset from an existing run"
            )
            write_started_at = time.perf_counter()
            self.store.write_parquet(cache_path, exact_cache)
            _console_log_timing(
                self.console,
                "Imported exact reranker cache",
                time.perf_counter() - exact_started_at,
                details=f"split={split_key}, rows={len(exact_cache)}",
            )
            _console_log_timing(
                self.console,
                "Saved imported reranker cache",
                time.perf_counter() - write_started_at,
                details=f"path={cache_path.name}",
            )
            _console_log_timing(
                self.console,
                "Reranker scoring pipeline complete",
                time.perf_counter() - total_started_at,
                details=f"split={split_key}, imported_exact_cache=true, rows={len(exact_cache)}",
            )
            return exact_cache
        superset_started_at = time.perf_counter()
        superset_cache = self._load_superset_cache_from_existing_runs(split_key, query_filter, reference=top_frame)
        if superset_cache is not None:
            self.console.log(
                "Importing reranker scores from a larger cached run"
                if query_filter is None
                else "Importing reranker scores for the requested query subset from a larger cached run"
            )
            write_started_at = time.perf_counter()
            self.store.write_parquet(cache_path, superset_cache)
            _console_log_timing(
                self.console,
                "Imported superset reranker cache",
                time.perf_counter() - superset_started_at,
                details=f"split={split_key}, rows={len(superset_cache)}, top_k={self.top_k}",
            )
            _console_log_timing(
                self.console,
                "Saved imported reranker cache",
                time.perf_counter() - write_started_at,
                details=f"path={cache_path.name}",
            )
            _console_log_timing(
                self.console,
                "Reranker scoring pipeline complete",
                time.perf_counter() - total_started_at,
                details=f"split={split_key}, imported_superset_cache=true, rows={len(superset_cache)}",
            )
            return superset_cache
        _console_log_timing(
            self.console,
            "Prepared reranker inputs",
            time.perf_counter() - retrieval_started_at,
            details=(
                f"split={split_key}, retriever={resolved_retriever}, "
                f"pairs={len(top_frame)}, queries={top_frame['query_id'].nunique() if len(top_frame) else 0}, top_k={self.top_k}"
            ),
        )
        if top_frame.empty:
            output = top_frame.rename(columns={"score": "retrieval_score"}).copy()
            output["reranker_score"] = np.asarray([], dtype=np.float32)
            write_started_at = time.perf_counter()
            self.store.write_parquet(cache_path, output)
            _console_log_timing(
                self.console,
                "Saved empty reranker score cache",
                time.perf_counter() - write_started_at,
                details=f"path={cache_path.name}",
            )
            _console_log_timing(
                self.console,
                "Reranker scoring pipeline complete",
                time.perf_counter() - total_started_at,
                details=f"split={split_key}, pairs=0",
            )
            return output
        checkpoint_cache = None
        reusable = None
        reusable_size = 0
        missing_frame = top_frame
        reusable_started_at = time.perf_counter()
        checkpoint_cache = self._load_checkpoint_cache(query_filter)
        reusable_cross_top_k = self._load_reusable_cache(split_key, query_filter)
        reusable = self._merge_score_cache_frames(
            frame
            for frame in (checkpoint_cache, reusable_cross_top_k)
            if frame is not None and not frame.empty
        )
        if reusable is not None:
            reusable = self._restrict_to_reference(reusable, top_frame)
        reusable_size = 0 if reusable_cross_top_k is None or reusable_cross_top_k.empty else int(reusable_cross_top_k["rank"].max())
        missing_frame = top_frame if reusable is None else self._remaining_unscored_rows(top_frame, reusable)
        _console_log_timing(
            self.console,
            "Checked reusable reranker caches",
            time.perf_counter() - reusable_started_at,
            details=(
                f"split={split_key}, checkpoint_pairs={0 if checkpoint_cache is None else len(checkpoint_cache)}, "
                f"reusable_pairs={0 if reusable_cross_top_k is None else len(reusable_cross_top_k)}, "
                f"recovered_pairs={0 if reusable is None else len(reusable)}, "
                f"new_pairs={len(missing_frame)}"
            ),
        )
        output_parts: list[pd.DataFrame] = []
        if reusable is not None:
            if checkpoint_cache is not None and not checkpoint_cache.empty:
                self.console.log(
                    f"Recovered {len(checkpoint_cache)} checkpointed reranker pairs from {len(self.paths.score_checkpoint_paths())} files"
                )
            if reusable_cross_top_k is not None and not reusable_cross_top_k.empty:
                self.console.log(f"Reusing cached reranker scores up to top {reusable_size} for en_test")
            output_parts.append(reusable.copy())
        if not missing_frame.empty:
            self.console.log(
                f"Scoring {len(missing_frame)} query-document pairs for {split_key} "
                f"with {self.reranker.model_name}"
            )
            scoring_started_at = time.perf_counter()
            output_parts.append(self._score_pairs(resolved_retriever, split_key, missing_frame))
            _console_log_timing(
                self.console,
                "Scored reranker pairs",
                time.perf_counter() - scoring_started_at,
                details=(
                    f"split={split_key}, new_pairs={len(missing_frame)}, "
                    f"model={self.reranker.model_name}"
                ),
            )
        concat_started_at = time.perf_counter()
        output = pd.concat(output_parts, ignore_index=True).sort_values(["query_id", "rank"]).reset_index(drop=True)
        write_started_at = time.perf_counter()
        self.store.write_parquet(cache_path, output)
        _console_log_timing(
            self.console,
            "Merged reranker outputs",
            time.perf_counter() - concat_started_at,
            details=f"rows={len(output)}",
        )
        _console_log_timing(
            self.console,
            "Saved reranker score cache",
            time.perf_counter() - write_started_at,
            details=f"path={cache_path.name}",
        )
        removed_checkpoints = self.paths.clear_score_checkpoints()
        if removed_checkpoints > 0:
            self.console.log(f"Cleared {removed_checkpoints} reranker checkpoint files after final cache save")
        _console_log_timing(
            self.console,
            "Reranker scoring pipeline complete",
            time.perf_counter() - total_started_at,
            details=(
                f"split={split_key}, total_rows={len(output)}, "
                f"queries={output['query_id'].nunique() if len(output) else 0}"
            ),
        )
        return output

    def _load_exact_cache_from_existing_runs(
        self,
        split_key: str,
        query_filter: set[int] | None,
        reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        for path in self.paths.exact_cache_candidates(self.reranker, self.top_k):
            if path.resolve() == self.paths.test_score_path().resolve():
                continue
            if path.exists():
                frame = self.store.read_parquet(path)
                if not self._cache_covers_request(frame, split_key, query_filter):
                    continue
                if reference is not None and not self._matches_reference(frame, reference):
                    self.console.log(
                        f"Ignoring cached reranker scores at {path}; its (query, rank, document) pairs "
                        f"do not match the current top-{self.top_k} ranking."
                    )
                    continue
                return self._subset_cached_frame(frame, query_filter)
        return None

    def _load_superset_cache_from_existing_runs(
        self,
        split_key: str,
        query_filter: set[int] | None,
        reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        if not self._supports_cross_top_k_cache_reuse():
            return None
        candidates = self.paths.superset_cache_candidates(self.reranker)
        if not candidates:
            return None
        for _, path in sorted(candidates, key=lambda item: item[0]):
            frame = self.store.read_parquet(path)
            frame = frame.loc[frame["rank"] <= self.top_k].copy()
            if not self._cache_covers_request(frame, split_key, query_filter):
                continue
            if reference is not None and not self._matches_reference(frame, reference):
                self.console.log(
                    f"Ignoring larger cached reranker scores at {path}; its (query, rank, document) pairs "
                    f"do not match the current top-{self.top_k} ranking."
                )
                continue
            return self._subset_cached_frame(frame, query_filter)
        return None

    def _load_reusable_cache(
        self,
        split_key: str,
        query_filter: set[int] | None,
    ) -> pd.DataFrame | None:
        if not self._supports_cross_top_k_cache_reuse():
            return None
        if self.top_k <= 1:
            return None
        model_root = self.paths.root.parent.parent
        candidates: list[tuple[int, Path]] = []
        for top_dir in model_root.glob("top_*"):
            if not top_dir.is_dir():
                continue
            try:
                cached_top_k = int(top_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if cached_top_k >= self.top_k:
                continue
            cache_path = top_dir / self.paths.root.name / f"test_top{cached_top_k}_scores.parquet"
            if cache_path.exists():
                candidates.append((cached_top_k, cache_path))
        if not candidates:
            return None
        for cached_top_k, cache_path in sorted(candidates, key=lambda item: item[0], reverse=True):
            cached = self.store.read_parquet(cache_path)
            cached = cached.loc[cached["rank"] <= cached_top_k].copy()
            if not self._cache_covers_request(cached, split_key, query_filter):
                continue
            return self._subset_cached_frame(cached, query_filter)
        return None

    def _load_checkpoint_cache(self, query_filter: set[int] | None) -> pd.DataFrame | None:
        required_columns = {"query_id", "rank", "document_id", "retrieval_score", "reranker_score"}
        checkpoint_frames: list[pd.DataFrame] = []
        for checkpoint_path in self.paths.score_checkpoint_paths():
            try:
                frame = self.store.read_parquet(checkpoint_path)
            except Exception as exc:
                self.console.log(f"Ignoring unreadable reranker checkpoint {checkpoint_path.name}: {exc}")
                continue
            if not required_columns.issubset(frame.columns):
                self.console.log(
                    f"Ignoring reranker checkpoint {checkpoint_path.name}: "
                    f"missing columns {sorted(required_columns - set(frame.columns))}"
                )
                continue
            checkpoint_frames.append(self._subset_cached_frame(frame, query_filter))
        merged = self._merge_score_cache_frames(checkpoint_frames)
        return None if merged is None or merged.empty else merged

    def _reference_top_frame(
        self,
        split_key: str,
        retriever_name: str,
        query_filter: set[int] | None,
    ) -> pd.DataFrame:
        ranking_frame = self.retrieval.ensure_rankings(split_key, retriever_name)
        top_frame = ranking_frame.loc[
            ranking_frame["rank"] <= self.top_k,
            ["query_id", "rank", "document_id", "score"],
        ].copy()
        if query_filter is not None:
            top_frame = top_frame.loc[top_frame["query_id"].astype(int).isin(query_filter)].copy()
        return top_frame.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def _matches_reference(self, frame: pd.DataFrame, reference: pd.DataFrame) -> bool:
        keys = ["query_id", "rank", "document_id"]
        if not set(keys).issubset(frame.columns):
            return False
        expected = reference[keys].astype("int64")
        candidate = frame.loc[frame["rank"] <= self.top_k, keys].astype("int64")
        candidate = candidate.loc[candidate["query_id"].isin(expected["query_id"].unique())]
        if len(candidate) != len(expected) or candidate.duplicated(keys).any():
            return False
        return len(candidate.merge(expected, on=keys, how="inner")) == len(expected)

    def _restrict_to_reference(self, frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        keys = ["query_id", "rank", "document_id"]
        expected = reference[keys].astype("int64").drop_duplicates()
        restricted = frame.astype({key: "int64" for key in keys}).merge(expected, on=keys, how="inner")
        dropped = len(frame) - len(restricted)
        if dropped > 0:
            self.console.log(
                f"Dropped {dropped} reusable reranker pairs whose (query, rank, document) do not match the "
                f"current top-{self.top_k} ranking"
            )
        return restricted.sort_values(["query_id", "rank"]).reset_index(drop=True)

    @staticmethod
    def _merge_score_cache_frames(frames: Any) -> pd.DataFrame | None:
        materialized = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not materialized:
            return None
        merged = pd.concat(materialized, ignore_index=True)
        merged = merged.drop_duplicates(subset=["query_id", "rank", "document_id"], keep="last")
        return merged.sort_values(["query_id", "rank"]).reset_index(drop=True)

    @staticmethod
    def _remaining_unscored_rows(top_frame: pd.DataFrame, scored_frame: pd.DataFrame) -> pd.DataFrame:
        scored_keys = scored_frame[["query_id", "rank", "document_id"]].drop_duplicates().copy()
        scored_keys["_already_scored"] = True
        remaining = top_frame.merge(scored_keys, on=["query_id", "rank", "document_id"], how="left")
        remaining = remaining.loc[~remaining["_already_scored"].eq(True), top_frame.columns].copy()
        return remaining.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def _is_complete_cache(self, frame: pd.DataFrame, split_key: str) -> bool:
        if frame.empty:
            return False
        if "query_id" not in frame.columns:
            return False
        expected_query_ids = {
            int(query_id)
            for query_id in self.retrieval.queries[split_key]["query_id"].astype(int).tolist()
        }
        cached_query_ids = {
            int(query_id)
            for query_id in frame["query_id"].astype(int).unique().tolist()
        }
        return cached_query_ids == expected_query_ids

    def _cache_covers_request(
        self,
        frame: pd.DataFrame,
        split_key: str,
        query_filter: set[int] | None,
    ) -> bool:
        if query_filter is None:
            return self._is_complete_cache(frame, split_key)
        if frame.empty:
            return False
        if "query_id" not in frame.columns:
            return False
        cached_query_ids = {
            int(query_id)
            for query_id in frame["query_id"].astype(int).unique().tolist()
        }
        return set(int(query_id) for query_id in query_filter).issubset(cached_query_ids)

    @staticmethod
    def _subset_cached_frame(frame: pd.DataFrame, query_filter: set[int] | None) -> pd.DataFrame:
        if query_filter is None:
            return frame.sort_values(["query_id", "rank"]).reset_index(drop=True)
        filtered = frame.loc[frame["query_id"].astype(int).isin(query_filter)].copy()
        return filtered.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def _candidate_document_text_by_id(
        self,
        retriever_name: str,
        *,
        candidate_fields: str = "title_abstract",
    ) -> dict[int, str]:
        if candidate_fields == "title":
            titles = self.retrieval.publications["title"].fillna("").astype(str).tolist()
            return {
                int(document_id): title
                for document_id, title in zip(self.retrieval.document_ids.tolist(), titles, strict=True)
            }
        if candidate_fields != "title_abstract":
            raise ValueError(f"Unsupported candidate_fields value: {candidate_fields}")
        bundle = self.retrieval.document_scope_bundle(retriever_name)
        return {
            int(document_id): str(text)
            for document_id, text in zip(self.retrieval.document_ids.tolist(), bundle["texts"], strict=True)
        }

    def _document_text_by_id(self, retriever_name: str) -> dict[int, str]:
        return self._candidate_document_text_by_id(retriever_name, candidate_fields="title_abstract")

    def _supports_cross_top_k_cache_reuse(self) -> bool:
        return True

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class CrossEncoderTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: CrossEncoderRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._model: CrossEncoder | None = None
        self._ettin_model: AutoModel | None = None
        self._ettin_tokenizer = None
        self._ettin_snapshot_path: Path | None = None
        self._ettin_dense_weight: torch.Tensor | None = None
        self._ettin_dense_bias: torch.Tensor | None = None
        self._ettin_norm_weight: torch.Tensor | None = None
        self._ettin_norm_bias: torch.Tensor | None = None
        self._ettin_out_weight: torch.Tensor | None = None
        self._ettin_out_bias: torch.Tensor | None = None

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._document_text_by_id(retriever_name)
        pairs = [
            (str(query_text_by_id[int(row.query_id)]), str(document_text_by_id[int(row.document_id)]))
            for row in frame.itertuples(index=False)
        ]
        checkpoint_start_index = self.paths.next_score_checkpoint_index()

        def _save_checkpoint_chunk(request_index: int, start: int, batch_scores: np.ndarray) -> None:
            checkpoint_frame = frame.iloc[start : start + len(batch_scores)].rename(columns={"score": "retrieval_score"}).copy()
            checkpoint_frame["reranker_score"] = batch_scores.astype(float)
            checkpoint_path = self.paths.score_checkpoint_path(checkpoint_start_index + request_index - 1)
            self.store.write_parquet(checkpoint_path, checkpoint_frame)

        if _is_modular_ettin_reranker(self.reranker.model_name):
            scores = self._predict_with_modular_ettin(pairs, progress_callback=_save_checkpoint_chunk)
        else:
            model = self._model_instance()
            scores = self._predict_with_sentence_transformers(
                model, pairs, progress_callback=_save_checkpoint_chunk
            )
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = scores.astype(float)
        return scored

    def _predict_with_sentence_transformers(
        self,
        model: CrossEncoder,
        pairs: list[tuple[str, str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        predict_batch_size = self._predict_batch_size(model)
        request_size = max(1, predict_batch_size * _COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER)
        total_pairs = len(pairs)
        total_requests = max(1, math.ceil(total_pairs / request_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Cross-encoder scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_requests,
            unit_plural="requests",
        )
        outputs: list[np.ndarray] = []
        for request_index, start in enumerate(range(0, len(pairs), request_size), start=1):
            batch_pairs = pairs[start : start + request_size]
            request_end = min(start + len(batch_pairs), total_pairs)
            batch_scores = np.asarray(
                model.predict(
                    batch_pairs,
                    batch_size=predict_batch_size,
                    show_progress_bar=self.console.verbose,
                ),
                dtype=np.float32,
            )
            outputs.append(batch_scores)
            if progress_callback is not None:
                progress_callback(request_index, start, batch_scores)
            progress.update(request_index, request_end)
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _model_instance(self) -> CrossEncoder:
        if self._model is None:
            cache_key = (self.reranker.model_name, int(self.reranker.max_length))
            cached = _CROSS_ENCODER_INSTANCE_CACHE.get(cache_key)
            if cached is None:
                self.console.log(
                    f"Loading reranker {self.reranker.model_name} "
                    f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size})"
                )
                tokenizer_kwargs = _cross_encoder_tokenizer_kwargs(self.reranker.model_name)
                cached = CrossEncoder(
                    self.reranker.model_name,
                    max_length=self.reranker.max_length,
                    device="cpu",
                    trust_remote_code=_requires_cross_encoder_remote_code(self.reranker.model_name),
                    tokenizer_kwargs=tokenizer_kwargs,
                )
                _CROSS_ENCODER_INSTANCE_CACHE[cache_key] = cached
            else:
                self.console.log(
                    f"Reusing loaded reranker {self.reranker.model_name} "
                    f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size})"
                )
            self._model = cached
            if _assign_cross_encoder_pad_token(self._model):
                tokenizer = getattr(self._model, "tokenizer", None)
                pad_token = None if tokenizer is None else getattr(tokenizer, "pad_token", None)
                if pad_token is not None:
                    self.console.log(
                        f"Configured pad token for {self.reranker.model_name}: {pad_token}"
                    )
        return self._model

    def _predict_batch_size(self, model: CrossEncoder) -> int:
        tokenizer = getattr(model, "tokenizer", None)
        pad_token_id = None if tokenizer is None else getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None and int(self.reranker.batch_size) > 1:
            self.console.log(
                f"Tokenizer for {self.reranker.model_name} has no pad token; "
                "falling back to batch_size=1."
            )
            return 1
        return int(self.reranker.batch_size)

    def _predict_with_modular_ettin(
        self,
        pairs: list[tuple[str, str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        self._ensure_modular_ettin_loaded()
        assert self._ettin_model is not None
        assert self._ettin_tokenizer is not None
        assert self._ettin_dense_weight is not None
        assert self._ettin_norm_weight is not None
        assert self._ettin_norm_bias is not None
        assert self._ettin_out_weight is not None
        assert self._ettin_out_bias is not None
        total_pairs = len(pairs)
        total_batches = max(1, math.ceil(total_pairs / self.reranker.batch_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Modular Ettin scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_batches,
        )
        outputs: list[np.ndarray] = []
        for batch_index, start in enumerate(range(0, len(pairs), self.reranker.batch_size), start=1):
            batch_pairs = pairs[start : start + self.reranker.batch_size]
            queries = [query for query, _ in batch_pairs]
            documents = [document for _, document in batch_pairs]
            encoded = self._ettin_tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.reranker.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                model_output = self._ettin_model(**encoded)
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)
                hidden = pooled @ self._ettin_dense_weight.T
                if self._ettin_dense_bias is not None:
                    hidden = hidden + self._ettin_dense_bias
                hidden = F.layer_norm(
                    hidden,
                    (hidden.shape[-1],),
                    weight=self._ettin_norm_weight,
                    bias=self._ettin_norm_bias,
                )
                scores = hidden @ self._ettin_out_weight.T + self._ettin_out_bias
            batch_array = scores.squeeze(-1).cpu().numpy().astype(np.float32)
            outputs.append(batch_array)
            if progress_callback is not None:
                progress_callback(batch_index, start, batch_array)
            progress.update(batch_index, start + len(batch_pairs))
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _ensure_modular_ettin_loaded(self) -> None:
        if self._ettin_model is not None and self._ettin_tokenizer is not None:
            return
        self.console.log(
            f"Loading modular Ettin reranker {self.reranker.model_name} "
            f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size})"
        )
        snapshot_path = Path(
            snapshot_download(
                self.reranker.model_name,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                    "2_Dense/model.safetensors",
                    "3_LayerNorm/model.safetensors",
                    "4_Dense/model.safetensors",
                ],
            )
        )
        tokenizer_dir = self._prepare_tokenizer_dir(snapshot_path)
        self._ettin_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self._ettin_model = AutoModel.from_pretrained(snapshot_path)
        self._ettin_model.eval()
        dense_payload = load_safetensors_file(str(snapshot_path / "2_Dense" / "model.safetensors"))
        norm_payload = load_safetensors_file(str(snapshot_path / "3_LayerNorm" / "model.safetensors"))
        out_payload = load_safetensors_file(str(snapshot_path / "4_Dense" / "model.safetensors"))
        self._ettin_dense_weight = dense_payload["linear.weight"].to(dtype=torch.float32)
        self._ettin_dense_bias = dense_payload.get("linear.bias")
        if self._ettin_dense_bias is not None:
            self._ettin_dense_bias = self._ettin_dense_bias.to(dtype=torch.float32)
        self._ettin_norm_weight = norm_payload["norm.weight"].to(dtype=torch.float32)
        self._ettin_norm_bias = norm_payload["norm.bias"].to(dtype=torch.float32)
        self._ettin_out_weight = out_payload["linear.weight"].to(dtype=torch.float32)
        self._ettin_out_bias = out_payload["linear.bias"].to(dtype=torch.float32)
        self._ettin_snapshot_path = snapshot_path

    def _prepare_tokenizer_dir(self, snapshot_path: Path) -> Path:
        tokenizer_root = self.store.root_dir / "tmp" / "patched_tokenizers" / _slugify(self.reranker.model_name)
        tokenizer_root.mkdir(parents=True, exist_ok=True)
        for name in ["tokenizer.json", "config.json", "special_tokens_map.json", "vocab.txt"]:
            source = snapshot_path / name
            target = tokenizer_root / name
            if source.exists() and (not target.exists() or source.stat().st_mtime > target.stat().st_mtime):
                shutil.copy2(source, target)
        source_config = snapshot_path / "tokenizer_config.json"
        target_config = tokenizer_root / "tokenizer_config.json"
        payload = json.loads(source_config.read_text(encoding="utf-8"))
        sanitized = _sanitize_tokenizer_config(payload)
        target_config.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
        return tokenizer_root

    def _document_text_by_id(self, retriever_name: str) -> dict[int, str]:
        bundle = self.retrieval.document_scope_bundle(retriever_name)
        return {
            int(document_id): str(text)
            for document_id, text in zip(self.retrieval.document_ids.tolist(), bundle["texts"], strict=True)
        }


class FlagEmbeddingTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: FlagEmbeddingRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._flag_reranker: Any | None = None
        self._tokenizer = None
        self._model: AutoModelForSequenceClassification | None = None
        self._backend: str | None = None

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        pairs = [
            [str(query_text_by_id[int(row.query_id)]), str(document_text_by_id[int(row.document_id)])]
            for row in frame.itertuples(index=False)
        ]
        checkpoint_start_index = self.paths.next_score_checkpoint_index()

        def _save_checkpoint_chunk(request_index: int, start: int, batch_scores: np.ndarray) -> None:
            checkpoint_frame = frame.iloc[start : start + len(batch_scores)].rename(columns={"score": "retrieval_score"}).copy()
            checkpoint_frame["reranker_score"] = batch_scores.astype(float)
            checkpoint_path = self.paths.score_checkpoint_path(checkpoint_start_index + request_index - 1)
            self.store.write_parquet(checkpoint_path, checkpoint_frame)

        if self._backend_name() == "flagembedding":
            scores = self._predict_with_flagembedding(pairs, progress_callback=_save_checkpoint_chunk)
        else:
            scores = self._predict_with_transformers(pairs, progress_callback=_save_checkpoint_chunk)
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = scores.astype(float)
        return scored

    def _backend_name(self) -> str:
        if self._backend is not None:
            return self._backend
        if importlib.util.find_spec("FlagEmbedding") is not None:
            self._backend = "flagembedding"
        else:
            self._backend = "transformers"
        self.console.log(
            f"Loading reranker {self.reranker.model_name} via {self._backend} "
            f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size})"
        )
        return self._backend

    def _predict_with_flagembedding(
        self,
        pairs: list[list[str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        reranker = self._flag_reranker_instance()
        outputs: list[np.ndarray] = []
        total_pairs = len(pairs)
        request_size = max(1, int(self.reranker.batch_size) * _COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER)
        total_requests = max(1, math.ceil(total_pairs / request_size))
        progress = _BatchScoringProgress(
            self.console,
            label="FlagEmbedding scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_requests,
            unit_plural="requests",
        )
        for request_index, start in enumerate(range(0, len(pairs), request_size), start=1):
            batch_pairs = pairs[start : start + request_size]
            batch_scores = reranker.compute_score(
                batch_pairs,
                batch_size=self.reranker.batch_size,
                normalize=self.reranker.normalize,
            )
            batch_array = np.asarray(batch_scores, dtype=np.float32).reshape(-1)
            outputs.append(batch_array)
            if progress_callback is not None:
                progress_callback(request_index, start, batch_array)
            progress.update(request_index, start + len(batch_pairs))
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _flag_reranker_instance(self) -> Any:
        if self._flag_reranker is None:
            cache_key = (
                self.reranker.model_name,
                bool(self.reranker.use_fp16),
                bool(self.reranker.quantize_dynamic_int8),
            )
            cached = _FLAGEMBEDDING_INSTANCE_CACHE.get(cache_key)
            if cached is None:
                from FlagEmbedding import FlagReranker

                cached = FlagReranker(
                    self.reranker.model_name,
                    use_fp16=self.reranker.use_fp16,
                )
                if self.reranker.quantize_dynamic_int8:
                    self.console.log(
                        f"Applying dynamic int8 quantization to {self.reranker.model_name} (torch.nn.Linear layers)"
                    )
                    cached.model = cached.model.float()
                    cached.model = torch.quantization.quantize_dynamic(
                        cached.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                _FLAGEMBEDDING_INSTANCE_CACHE[cache_key] = cached
            else:
                self.console.log(f"Reusing loaded FlagEmbedding reranker {self.reranker.model_name}")
            self._flag_reranker = cached
        return self._flag_reranker

    def _predict_with_transformers(
        self,
        pairs: list[list[str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        tokenizer, model = self._transformers_components()
        outputs: list[np.ndarray] = []
        total_pairs = len(pairs)
        total_batches = max(1, math.ceil(total_pairs / self.reranker.batch_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Transformers reranker scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_batches,
        )
        for batch_index, start in enumerate(range(0, len(pairs), self.reranker.batch_size), start=1):
            batch_pairs = pairs[start : start + self.reranker.batch_size]
            queries = [pair[0] for pair in batch_pairs]
            documents = [pair[1] for pair in batch_pairs]
            encoded = tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.reranker.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(**encoded, return_dict=True).logits.view(-1)
                batch_scores = logits.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
            if self.reranker.normalize:
                batch_scores = 1.0 / (1.0 + np.exp(-batch_scores))
            outputs.append(batch_scores)
            if progress_callback is not None:
                progress_callback(batch_index, start, batch_scores)
            progress.update(batch_index, start + len(batch_pairs))
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _transformers_components(self) -> tuple[Any, AutoModelForSequenceClassification]:
        if self._tokenizer is None or self._model is None:
            cached = _TRANSFORMERS_SEQUENCE_CLASSIFICATION_CACHE.get(self.reranker.model_name)
            if cached is None:
                tokenizer = AutoTokenizer.from_pretrained(self.reranker.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.reranker.model_name)
                model.to("cpu")
                model.eval()
                cached = (tokenizer, model)
                _TRANSFORMERS_SEQUENCE_CLASSIFICATION_CACHE[self.reranker.model_name] = cached
            else:
                self.console.log(f"Reusing loaded transformers reranker {self.reranker.model_name}")
            self._tokenizer, self._model = cached
        return self._tokenizer, self._model

    def _document_text_by_id(self, retriever_name: str) -> dict[int, str]:
        bundle = self.retrieval.document_scope_bundle(retriever_name)
        return {
            int(document_id): str(text)
            for document_id, text in zip(self.retrieval.document_ids.tolist(), bundle["texts"], strict=True)
        }


class JinaComputeScoreTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: JinaComputeScoreRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._model: AutoModelForSequenceClassification | None = None

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        pairs = [
            [str(query_text_by_id[int(row.query_id)]), str(document_text_by_id[int(row.document_id)])]
            for row in frame.itertuples(index=False)
        ]
        checkpoint_start_index = self.paths.next_score_checkpoint_index()

        def _save_checkpoint_chunk(request_index: int, start: int, batch_scores: np.ndarray) -> None:
            checkpoint_frame = frame.iloc[start : start + len(batch_scores)].rename(columns={"score": "retrieval_score"}).copy()
            checkpoint_frame["reranker_score"] = batch_scores.astype(float)
            checkpoint_path = self.paths.score_checkpoint_path(checkpoint_start_index + request_index - 1)
            self.store.write_parquet(checkpoint_path, checkpoint_frame)

        scores = self._compute_scores(pairs, progress_callback=_save_checkpoint_chunk)
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = scores.astype(float)
        return scored

    def _compute_scores(
        self,
        pairs: list[list[str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        model = self._model_instance()
        outputs: list[np.ndarray] = []
        total_pairs = len(pairs)
        request_size = max(1, int(self.reranker.batch_size) * _COMPUTE_SCORE_REQUEST_BATCH_MULTIPLIER)
        total_requests = max(1, math.ceil(len(pairs) / request_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Jina compute_score",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_requests,
            unit_plural="requests",
        )
        for request_index, start in enumerate(range(0, len(pairs), request_size), start=1):
            batch_pairs = pairs[start : start + request_size]
            request_started_at = time.perf_counter()
            request_end = min(start + len(batch_pairs), total_pairs)
            self.console.log(
                f"Starting Jina compute_score request {request_index}/{total_requests}: "
                f"pairs {start + 1}-{request_end}/{total_pairs}"
            )
            heartbeat_stop = threading.Event()

            def _heartbeat() -> None:
                while not heartbeat_stop.wait(_JINA_COMPUTE_SCORE_HEARTBEAT_SECONDS):
                    elapsed = time.perf_counter() - request_started_at
                    self.console.log(
                        f"Jina compute_score request {request_index}/{total_requests} still running after "
                        f"{_format_elapsed_seconds(elapsed)}: pairs {start + 1}-{request_end}/{total_pairs}"
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                batch_scores = model.compute_score(
                    batch_pairs,
                    batch_size=self.reranker.batch_size,
                    max_length=self.reranker.max_length,
                )
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=0.1)
            batch_array = np.asarray(batch_scores, dtype=np.float32).reshape(-1)
            outputs.append(batch_array)
            if progress_callback is not None:
                progress_callback(request_index, start, batch_array)
            progress.update(request_index, start + len(batch_pairs))
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _model_instance(self) -> AutoModelForSequenceClassification:
        if self._model is None:
            cache_key = (
                self.reranker.model_name,
                bool(self.reranker.trust_remote_code),
                bool(self.reranker.use_flash_attn),
                bool(self.reranker.quantize_dynamic_int8),
            )
            cached = _JINA_COMPUTE_SCORE_MODEL_CACHE.get(cache_key)
            if cached is None:
                self.console.log(
                    f"Loading Jina reranker {self.reranker.model_name} "
                    f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size}, "
                    f"trust_remote_code={self.reranker.trust_remote_code}, use_flash_attn={self.reranker.use_flash_attn}, "
                    f"quantize_dynamic_int8={self.reranker.quantize_dynamic_int8})"
                )
                cached = AutoModelForSequenceClassification.from_pretrained(
                    self.reranker.model_name,
                    dtype="auto",
                    trust_remote_code=self.reranker.trust_remote_code,
                    use_flash_attn=self.reranker.use_flash_attn,
                )
                cached.to("cpu")
                cached.eval()
                if self.reranker.quantize_dynamic_int8:
                    self.console.log(
                        f"Applying dynamic int8 quantization to {self.reranker.model_name} (torch.nn.Linear layers)"
                    )
                    cached = cached.float()
                    cached = torch.quantization.quantize_dynamic(cached, {torch.nn.Linear}, dtype=torch.qint8)
                _JINA_COMPUTE_SCORE_MODEL_CACHE[cache_key] = cached
            else:
                self.console.log(f"Reusing loaded Jina reranker {self.reranker.model_name}")
            self._model = cached
            if not hasattr(self._model, "compute_score"):
                raise AttributeError(
                    f"Model {self.reranker.model_name} does not expose compute_score(sentence_pairs, ...)."
                )
        return self._model


class LlamaNemotronTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: LlamaNemotronRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._tokenizer: Any | None = None
        self._model: AutoModelForSequenceClassification | None = None

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        pairs = [
            [str(query_text_by_id[int(row.query_id)]), str(document_text_by_id[int(row.document_id)])]
            for row in frame.itertuples(index=False)
        ]
        checkpoint_start_index = self.paths.next_score_checkpoint_index()

        def _save_checkpoint_chunk(request_index: int, start: int, batch_scores: np.ndarray) -> None:
            checkpoint_frame = frame.iloc[start : start + len(batch_scores)].rename(columns={"score": "retrieval_score"}).copy()
            checkpoint_frame["reranker_score"] = batch_scores.astype(float)
            checkpoint_path = self.paths.score_checkpoint_path(checkpoint_start_index + request_index - 1)
            self.store.write_parquet(checkpoint_path, checkpoint_frame)

        scores = self._predict_with_llama_nemotron(pairs, progress_callback=_save_checkpoint_chunk)
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = scores.astype(float)
        return scored

    @staticmethod
    def _prompt_template(query: str, passage: str) -> str:
        return f"question:{query} \n \n passage:{passage}"

    def _predict_with_llama_nemotron(
        self,
        pairs: list[list[str]],
        *,
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> np.ndarray:
        tokenizer, model = self._transformers_components()
        outputs: list[np.ndarray] = []
        total_pairs = len(pairs)
        total_batches = max(1, math.ceil(total_pairs / self.reranker.batch_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Llama Nemotron reranker scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_batches,
        )
        for batch_index, start in enumerate(range(0, len(pairs), self.reranker.batch_size), start=1):
            batch_pairs = pairs[start : start + self.reranker.batch_size]
            texts = [self._prompt_template(query, passage) for query, passage in batch_pairs]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.reranker.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                logits = model(**encoded, return_dict=True).logits.view(-1)
                batch_scores = logits.to(dtype=torch.float32).cpu().numpy().astype(np.float32)
            outputs.append(batch_scores)
            if progress_callback is not None:
                progress_callback(batch_index, start, batch_scores)
            progress.update(batch_index, start + len(batch_pairs))
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def _transformers_components(self) -> tuple[Any, AutoModelForSequenceClassification]:
        if self._tokenizer is None or self._model is None:
            cache_key = (
                self.reranker.model_name,
                bool(self.reranker.trust_remote_code),
                bool(self.reranker.quantize_dynamic_int8),
            )
            cached = _LLAMA_NEMOTRON_MODEL_CACHE.get(cache_key)
            if cached is None:
                self.console.log(
                    f"Loading Llama Nemotron reranker {self.reranker.model_name} "
                    f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size}, "
                    f"trust_remote_code={self.reranker.trust_remote_code}, "
                    f"quantize_dynamic_int8={self.reranker.quantize_dynamic_int8})"
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    self.reranker.model_name,
                    trust_remote_code=self.reranker.trust_remote_code,
                    padding_side="left",
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.reranker.model_name,
                    trust_remote_code=self.reranker.trust_remote_code,
                    dtype="auto",
                )
                if model.config.pad_token_id is None:
                    model.config.pad_token_id = tokenizer.eos_token_id
                model.to("cpu")
                model.eval()
                if self.reranker.quantize_dynamic_int8:
                    self.console.log(
                        f"Applying dynamic int8 quantization to {self.reranker.model_name} (torch.nn.Linear layers)"
                    )
                    model = model.float()
                    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                cached = (tokenizer, model)
                _LLAMA_NEMOTRON_MODEL_CACHE[cache_key] = cached
            else:
                self.console.log(f"Reusing loaded Llama Nemotron reranker {self.reranker.model_name}")
            self._tokenizer, self._model = cached
        return self._tokenizer, self._model


class TransformersListwiseTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: TransformersListwiseRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._model: Any | None = None

    def _supports_cross_top_k_cache_reuse(self) -> bool:
        return False

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        model = self._model_instance()
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.nan
        total_queries = int(frame["query_id"].nunique()) if len(frame) else 0
        processed_queries = 0
        started_at = time.perf_counter()
        progress_interval = max(1, math.ceil(total_queries / 25)) if total_queries > 0 else 1
        for query_id, query_group in frame.groupby("query_id", sort=False):
            query_started_at = time.perf_counter()
            ordered = query_group.sort_values("rank").reset_index(drop=True)
            rows = list(ordered.itertuples(index=False))
            max_documents = self.reranker.max_documents_per_query
            if max_documents is not None and len(rows) > int(max_documents):
                raise ValueError(
                    f"{self.reranker.model_name} supports up to {int(max_documents)} documents per query, "
                    f"but received {len(rows)} for query_id={int(query_id)}."
                )
            query_text = str(query_text_by_id[int(query_id)])
            documents = [str(document_text_by_id[int(row.document_id)]) for row in rows]
            try:
                results = model.rerank(query_text, documents)
            except Exception as exc:
                total_elapsed = time.perf_counter() - started_at
                raise RuntimeError(
                    f"Transformers listwise reranking failed for query_id={int(query_id)}, "
                    f"candidate_count={len(rows)}, processed_queries={processed_queries}/{total_queries}, "
                    f"elapsed_seconds={total_elapsed:.2f}: {exc}"
                ) from exc
            if not isinstance(results, list):
                raise ValueError(
                    f"Expected {self.reranker.model_name} to return a list of rerank results, got {type(results).__name__}."
                )
            score_by_index: dict[int, float] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                relevance_score = item.get("relevance_score")
                if index is None or relevance_score is None:
                    continue
                try:
                    result_index = int(index)
                    result_score = float(relevance_score)
                except (TypeError, ValueError):
                    continue
                if result_index < 0 or result_index >= len(rows):
                    continue
                score_by_index[result_index] = result_score
            missing_indices = [index for index in range(len(rows)) if index not in score_by_index]
            if missing_indices:
                raise ValueError(
                    f"{self.reranker.model_name} did not return scores for all candidates in query_id={int(query_id)}; "
                    f"missing_indices={missing_indices[:5]}"
                )
            mask = scored["query_id"] == int(query_id)
            scored.loc[mask, "reranker_score"] = np.asarray(
                [score_by_index[index] for index in range(len(rows))],
                dtype=np.float32,
            )
            processed_queries += 1
            if processed_queries % progress_interval == 0 or processed_queries == total_queries:
                _console_log_timing(
                    self.console,
                    f"Processed listwise reranker queries up to query_id={int(query_id)}",
                    time.perf_counter() - query_started_at,
                    details=(
                        f"processed_queries={processed_queries}/{total_queries}, "
                        f"last_query_candidates={len(rows)}"
                    ),
                )
        _console_log_timing(
            self.console,
            "Completed transformers listwise scoring",
            time.perf_counter() - started_at,
            details=(
                f"split={split_key}, queries={frame['query_id'].nunique() if len(frame) else 0}, "
                f"pairs={len(frame)}"
            ),
        )
        scored["reranker_score"] = scored["reranker_score"].astype(np.float32)
        return scored.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def _model_instance(self) -> Any:
        if self._model is None:
            cache_key = (self.reranker.model_name, bool(self.reranker.trust_remote_code))
            cached = _TRANSFORMERS_LISTWISE_MODEL_CACHE.get(cache_key)
            if cached is None:
                self.console.log(
                    f"Loading listwise reranker {self.reranker.model_name} "
                    f"(trust_remote_code={self.reranker.trust_remote_code})"
                )
                cached = AutoModel.from_pretrained(
                    self.reranker.model_name,
                    trust_remote_code=self.reranker.trust_remote_code,
                )
                if hasattr(cached, "eval"):
                    cached.eval()
                _TRANSFORMERS_LISTWISE_MODEL_CACHE[cache_key] = cached
            else:
                self.console.log(f"Reusing loaded listwise reranker {self.reranker.model_name}")
            self._model = cached
            if not hasattr(self._model, "rerank"):
                raise AttributeError(
                    f"Model {self.reranker.model_name} does not expose a rerank(query, documents) method."
                )
        return self._model


_CLAUDE_LISTWISE_LOG_INTERVAL_SECONDS = 10.0


class ClaudeListwiseTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: ClaudeListwiseRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._client: anthropic.Anthropic | None = None
        self._usage_lock = threading.Lock()
        self._usage_totals = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def usage_summary(self) -> dict[str, Any]:
        with self._usage_lock:
            return dict(self._usage_totals)

    def _supports_cross_top_k_cache_reuse(self) -> bool:
        return False

    def _client_instance(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(
                timeout=float(self.reranker.timeout_seconds),
                max_retries=int(self.reranker.max_retries),
            )
        return self._client

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.nan

        query_groups = list(frame.groupby("query_id", sort=False))
        total_queries = len(query_groups)
        started_at = time.perf_counter()
        last_logged_at = started_at
        completed_queries = 0

        def _rank_one(query_id: int, group: pd.DataFrame) -> tuple[int, dict[int, float]]:
            ordered = group.sort_values("rank")
            document_ids = [int(document_id) for document_id in ordered["document_id"].tolist()]
            query_text = str(query_text_by_id[query_id])
            ranked_document_ids = self._cached_or_ranked_order(
                query_id, document_ids, query_text, document_text_by_id
            )
            total = len(ranked_document_ids)
            score_by_document = {
                document_id: float(total - position)
                for position, document_id in enumerate(ranked_document_ids)
            }
            return query_id, score_by_document

        results_by_query: dict[int, dict[int, float]] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(self.reranker.max_concurrency))) as executor:
            futures = {
                executor.submit(_rank_one, int(query_id), group): int(query_id)
                for query_id, group in query_groups
            }
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    _, score_by_document = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Claude listwise reranking failed for query_id={query_id}, "
                        f"completed_queries={completed_queries}/{total_queries}: {exc}"
                    ) from exc
                results_by_query[query_id] = score_by_document
                completed_queries += 1
                now = time.perf_counter()
                if completed_queries == total_queries or (now - last_logged_at) >= _CLAUDE_LISTWISE_LOG_INTERVAL_SECONDS:
                    last_logged_at = now
                    elapsed = now - started_at
                    rate = completed_queries / elapsed if elapsed > 0 else 0.0
                    remaining = total_queries - completed_queries
                    eta_minutes = (remaining / rate / 60.0) if rate > 0 else 0.0
                    self.console.log(
                        f"Claude listwise reranking progress for {split_key}: "
                        f"{completed_queries}/{total_queries} queries "
                        f"({100.0 * completed_queries / total_queries:.1f}%), "
                        f"elapsed={elapsed / 60.0:.1f}m, eta_minutes={eta_minutes:.1f}"
                    )

        for query_id, score_by_document in results_by_query.items():
            mask = scored["query_id"] == query_id
            scored.loc[mask, "reranker_score"] = (
                scored.loc[mask, "document_id"].map(score_by_document).astype(float)
            )
        return scored

    def _query_cache_path(self, query_id: int) -> Path:
        return self.paths.root / "claude_listwise_cache" / f"query_{int(query_id)}.json"

    def _cached_or_ranked_order(
        self,
        query_id: int,
        document_ids: list[int],
        query_text: str,
        document_text_by_id: dict[int, str],
    ) -> list[int]:
        cache_path = self._query_cache_path(query_id)
        cache_key = {
            "model_name": self.reranker.model_name,
            "window_size": self.reranker.window_size,
            "window_step": self.reranker.window_step,
            "candidate_fields": self.reranker.candidate_fields,
            "document_ids": document_ids,
        }
        if cache_path.exists():
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached_payload.get("cache_key") == cache_key:
                return [int(document_id) for document_id in cached_payload["ranked_document_ids"]]
        ranked_document_ids = self._sliding_window_rank(query_text, document_ids, document_text_by_id)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"cache_key": cache_key, "ranked_document_ids": ranked_document_ids}),
            encoding="utf-8",
        )
        return ranked_document_ids

    def _sliding_window_rank(
        self,
        query_text: str,
        document_ids: list[int],
        document_text_by_id: dict[int, str],
    ) -> list[int]:
        ranking = list(document_ids)
        window_size = int(self.reranker.window_size)
        step = max(1, int(self.reranker.window_step))
        end = len(ranking)
        start = max(0, end - window_size)
        while True:
            ranking[start:end] = self._rank_window(query_text, ranking[start:end], document_text_by_id)
            if start == 0:
                break
            end = max(0, end - step)
            start = max(0, end - window_size)
        return ranking

    def _rank_window(
        self,
        query_text: str,
        window_document_ids: list[int],
        document_text_by_id: dict[int, str],
    ) -> list[int]:
        total = len(window_document_ids)
        if total <= 1:
            return list(window_document_ids)
        passages = "\n".join(
            f"[{index + 1}] {document_text_by_id[document_id]}"
            for index, document_id in enumerate(window_document_ids)
        )
        user_prompt = (
            f"Query (social media post): {query_text}\n\n"
            f"Candidates:\n{passages}\n\n"
            f"Rank all {total} candidates from most to least likely, as a JSON array of numbers 1-{total}."
        )
        client = self._client_instance()
        last_error: Exception | None = None
        attempts = max(1, int(self.reranker.max_retries))
        for attempt in range(1, attempts + 1):
            try:
                response = client.messages.create(
                    model=self.reranker.model_name,
                    max_tokens=int(self.reranker.max_output_tokens),
                    temperature=float(self.reranker.temperature),
                    system=[
                        {
                            "type": "text",
                            "text": CLAUDE_LISTWISE_RERANK_SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                usage = response.usage
                with self._usage_lock:
                    self._usage_totals["requests"] += 1
                    self._usage_totals["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                    self._usage_totals["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
                    self._usage_totals["cache_read_input_tokens"] += int(
                        getattr(usage, "cache_read_input_tokens", 0) or 0
                    )
                    self._usage_totals["cache_creation_input_tokens"] += int(
                        getattr(usage, "cache_creation_input_tokens", 0) or 0
                    )
                response_text = "".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                )
                permutation = _parse_claude_rank_permutation(response_text, total)
                return [window_document_ids[index - 1] for index in permutation]
            except Exception as exc:
                last_error = exc
                self.console.log(
                    f"Claude listwise rank call failed on attempt {attempt}/{attempts} "
                    f"for {self.reranker.model_name}: {exc}"
                )
        assert last_error is not None
        raise last_error


class Qwen3RerankerTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: Qwen3RerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._tokenizer = None
        self._model: AutoModelForCausalLM | None = None

    def _components(self) -> tuple[Any, AutoModelForCausalLM]:
        if self._tokenizer is None or self._model is None:
            self.console.log(
                f"Loading Qwen3 reranker {self.reranker.model_name} as a causal LM "
                f"(max_length={self.reranker.max_length}, batch_size={self.reranker.batch_size})"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.reranker.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(self.reranker.model_name)
            self._model.to("cpu")
            self._model.eval()
        return self._tokenizer, self._model

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_text_by_id = self._candidate_document_text_by_id(
            retriever_name,
            candidate_fields=self.reranker.candidate_fields,
        )
        texts = [
            (
                f"<Instruct>: {self.reranker.instruction}\n"
                f"<Query>: {query_text_by_id[int(row.query_id)]}\n"
                f"<Document>: {document_text_by_id[int(row.document_id)]}"
            )
            for row in frame.itertuples(index=False)
        ]
        tokenizer, model = self._components()
        prefix_ids = tokenizer.encode(
            f"<|im_start|>system\n{QWEN3_RERANKER_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        suffix_ids = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        true_token_id = tokenizer.convert_tokens_to_ids("yes")
        false_token_id = tokenizer.convert_tokens_to_ids("no")
        checkpoint_start_index = self.paths.next_score_checkpoint_index()
        batch_size = max(1, int(self.reranker.batch_size))
        total_pairs = len(texts)
        total_batches = max(1, math.ceil(total_pairs / batch_size))
        progress = _BatchScoringProgress(
            self.console,
            label="Qwen3 reranker scoring",
            model_name=self.reranker.model_name,
            total_pairs=total_pairs,
            total_batches=total_batches,
        )
        outputs: list[np.ndarray] = []
        for batch_index, start in enumerate(range(0, total_pairs, batch_size), start=1):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=int(self.reranker.max_length) - len(prefix_ids) - len(suffix_ids),
            )
            encoded["input_ids"] = [prefix_ids + input_ids + suffix_ids for input_ids in encoded["input_ids"]]
            padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            with torch.inference_mode():
                last_logits = model(**padded, logits_to_keep=1).logits[:, -1, :]
                pair_logits = torch.stack([last_logits[:, false_token_id], last_logits[:, true_token_id]], dim=1)
                batch_scores = (
                    F.log_softmax(pair_logits.to(dtype=torch.float32), dim=1)[:, 1].exp().cpu().numpy().astype(np.float32)
                )
            outputs.append(batch_scores)
            checkpoint_frame = (
                frame.iloc[start : start + len(batch_scores)].rename(columns={"score": "retrieval_score"}).copy()
            )
            checkpoint_frame["reranker_score"] = batch_scores.astype(float)
            self.store.write_parquet(
                self.paths.score_checkpoint_path(checkpoint_start_index + batch_index - 1),
                checkpoint_frame,
            )
            progress.update(batch_index, start + len(batch_scores))
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = (
            np.concatenate(outputs, axis=0).astype(float) if outputs else np.asarray([], dtype=float)
        )
        return scored


class OllamaListwiseTop5TopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: OllamaListwiseTop5RerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._tokenizer = None

    def _supports_cross_top_k_cache_reuse(self) -> bool:
        return False

    def _tokenizer_instance(self) -> Any:
        if self._tokenizer is None:
            self.console.log(f"Loading tokenizer {self.reranker.tokenizer_name} to size the Ollama context window")
            self._tokenizer = AutoTokenizer.from_pretrained(self.reranker.tokenizer_name)
        return self._tokenizer

    def _query_cache_path(self, query_id: int) -> Path:
        return self.paths.root / "ollama_listwise_top5_cache" / f"query_{int(query_id)}.json"

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        publications = self.retrieval.publications
        candidate_by_id = {
            int(document_id): {"title": str(title), "abstract": str(abstract)}
            for document_id, title, abstract in zip(
                self.retrieval.document_ids.tolist(),
                publications["title"].fillna("").astype(str).tolist(),
                publications["abstract"].fillna("").astype(str).tolist(),
                strict=True,
            )
        }
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.nan
        query_groups = list(frame.groupby("query_id", sort=False))
        total_queries = len(query_groups)
        started_at = time.perf_counter()
        self.console.log(
            f"Ollama listwise top-5 reranking for {split_key}: {total_queries} queries with {self.reranker.model_name} "
            f"(think={self.reranker.think}, num_predict={self.reranker.num_predict}, "
            f"temperature={self.reranker.temperature}, top_p={self.reranker.top_p}, seed={self.reranker.seed})"
        )
        for query_index, (query_id, group) in enumerate(query_groups, start=1):
            ordered = group.sort_values("rank")
            document_ids = [int(document_id) for document_id in ordered["document_id"].tolist()]
            prompt = render_llm_listwise_top5_prompt(
                str(query_text_by_id[int(query_id)]),
                [candidate_by_id[document_id] for document_id in document_ids],
                candidate_fields=self.reranker.candidate_fields,
            )
            selected_numbers = self._cached_or_selected_numbers(int(query_id), document_ids, prompt)
            selected_document_ids = [document_ids[number - 1] for number in selected_numbers]
            ranked_document_ids = selected_document_ids + [
                document_id for document_id in document_ids if document_id not in set(selected_document_ids)
            ]
            total = len(ranked_document_ids)
            score_by_document = {
                document_id: float(total - position) for position, document_id in enumerate(ranked_document_ids)
            }
            mask = scored["query_id"] == query_id
            scored.loc[mask, "reranker_score"] = scored.loc[mask, "document_id"].map(score_by_document).astype(float)
            elapsed = time.perf_counter() - started_at
            rate = query_index / elapsed if elapsed > 0 else 0.0
            eta_minutes = ((total_queries - query_index) / rate / 60.0) if rate > 0 else 0.0
            self.console.log(
                f"Ollama listwise top-5 progress for {split_key}: {query_index}/{total_queries} queries "
                f"({100.0 * query_index / total_queries:.1f}%), query_id={int(query_id)}, "
                f"selected_candidates={selected_numbers}, elapsed={elapsed / 60.0:.1f}m, eta_minutes={eta_minutes:.1f}"
            )
        return scored

    def _cached_or_selected_numbers(self, query_id: int, document_ids: list[int], prompt: str) -> list[int]:
        cache_path = self._query_cache_path(query_id)
        cache_key = {**self.reranker.summary_dict(), "document_ids": document_ids}
        cache_key.pop("ollama_url", None)
        if cache_path.exists():
            cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached_payload.get("cache_key") == cache_key:
                self.console.log(f"Using cached Ollama top-5 selection for query_id={query_id}")
                return [int(number) for number in cached_payload["selected_candidate_numbers"]]
        result = self._generate(prompt)
        selected_numbers = _parse_llm_top5_candidates(result["response"], len(document_ids))
        if len(selected_numbers) < LLM_LISTWISE_TOP5_SELECTION_SIZE:
            self.console.log(
                f"Ollama top-5 answer for query_id={query_id} contained {len(selected_numbers)} valid candidate numbers "
                f"(done_reason={result['done_reason']}, response={result['response']!r}); "
                "unselected candidates keep their retriever order"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "selected_candidate_numbers": selected_numbers,
                    "selected_document_ids": [document_ids[number - 1] for number in selected_numbers],
                    **result,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return selected_numbers

    def _generate(self, prompt: str) -> dict[str, Any]:
        tokenizer = self._tokenizer_instance()
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        num_ctx = prompt_tokens + int(self.reranker.num_predict)
        self.console.log(
            f"Sending Ollama request: prompt_tokens={prompt_tokens}, num_ctx={num_ctx}; "
            "no output is streamed while the prompt is evaluated"
        )
        started_at = time.perf_counter()
        last_logged_at = started_at
        first_token_seconds: float | None = None
        thinking_parts: list[str] = []
        response_parts: list[str] = []
        final_chunk: dict[str, Any] = {}
        with requests.post(
            self.reranker.ollama_url,
            json={
                "model": self.reranker.model_name,
                "prompt": prompt,
                "stream": True,
                "think": bool(self.reranker.think),
                "options": {
                    "temperature": float(self.reranker.temperature),
                    "top_p": float(self.reranker.top_p),
                    "seed": int(self.reranker.seed),
                    "num_predict": int(self.reranker.num_predict),
                    "num_ctx": int(num_ctx),
                },
            },
            stream=True,
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"Ollama request failed with status {response.status_code}: {response.text}")
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece_thinking = str(chunk.get("thinking", "") or "")
                piece_response = str(chunk.get("response", "") or "")
                if (piece_thinking or piece_response) and first_token_seconds is None:
                    first_token_seconds = time.perf_counter() - started_at
                    self.console.log(f"Ollama prompt evaluated; first token after {first_token_seconds:.1f}s")
                if piece_thinking:
                    thinking_parts.append(piece_thinking)
                if piece_response:
                    response_parts.append(piece_response)
                now = time.perf_counter()
                if (now - last_logged_at) >= _SCORING_PROGRESS_LOG_INTERVAL_SECONDS:
                    last_logged_at = now
                    self.console.log(
                        f"Ollama generation progress: thinking_chunks={len(thinking_parts)}, "
                        f"answer_chunks={len(response_parts)}, elapsed={(now - started_at) / 60.0:.1f}m"
                    )
                if chunk.get("done"):
                    final_chunk = chunk
        elapsed_seconds = time.perf_counter() - started_at
        result = {
            "prompt_tokens": prompt_tokens,
            "num_ctx": num_ctx,
            "prompt_eval_count": final_chunk.get("prompt_eval_count"),
            "eval_count": final_chunk.get("eval_count"),
            "done_reason": final_chunk.get("done_reason"),
            "first_token_seconds": first_token_seconds,
            "elapsed_seconds": elapsed_seconds,
            "thinking": "".join(thinking_parts),
            "response": "".join(response_parts),
        }
        self.console.log(
            f"Ollama request finished in {elapsed_seconds / 60.0:.1f}m: eval_count={result['eval_count']}, "
            f"done_reason={result['done_reason']}, answer={result['response'].strip()!r}"
        )
        return result


class OllamaClaimDecompositionTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: OllamaClaimDecompositionRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._model_resolved = False
        self._active_model: str | None = None
        self._sentence_splitter: Any = None
        self._logged_think_check = False
        self.warmup: dict[str, Any] | None = None
        self.query_details: dict[int, dict[str, Any]] = {}

    def _supports_cross_top_k_cache_reuse(self) -> bool:
        return False

    def _tags_url(self) -> str:
        value = self.reranker.ollama_url.rstrip("/")
        if value.endswith("/api/generate"):
            return value[: -len("/api/generate")] + "/api/tags"
        return value + "/api/tags"

    def resolve_active_model(self) -> str | None:
        if self._model_resolved:
            return self._active_model
        self._model_resolved = True
        tags_url = self._tags_url()
        try:
            response = requests.get(tags_url, timeout=min(int(self.reranker.timeout_seconds), 30))
            response.raise_for_status()
            raw_models = response.json().get("models", [])
        except (requests.exceptions.RequestException, ValueError) as exc:
            self.console.log(
                f"Ollama is unavailable at {tags_url} ({exc}); using the deterministic first-stage ranking"
            )
            return None
        installed = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in raw_models
            if isinstance(item, dict)
        }
        if self.reranker.model_name in installed:
            self._active_model = self.reranker.model_name
        elif self.reranker.fallback_model_name and self.reranker.fallback_model_name in installed:
            self._active_model = self.reranker.fallback_model_name
            self.console.log(
                f"Ollama model {self.reranker.model_name} is not installed; using fallback model "
                f"{self.reranker.fallback_model_name} (run `ollama pull {self.reranker.model_name}` to use the primary model)"
            )
        else:
            self.console.log(
                f"Neither {self.reranker.model_name} nor {self.reranker.fallback_model_name} is installed in Ollama "
                f"(installed: {sorted(installed)}); using the deterministic first-stage ranking"
            )
            return None
        self.console.log(f"Active Ollama model for claim decomposition reranking: {self._active_model}")
        self._warm_up(self._active_model)
        return self._active_model

    def _warm_up(self, model_name: str) -> None:
        started_at = time.perf_counter()
        try:
            response = requests.post(
                self.reranker.ollama_url,
                json={"model": model_name, "prompt": "", "stream": False, "keep_alive": self.reranker.keep_alive},
                timeout=int(self.reranker.timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self.console.log(f"Ollama warm-up for {model_name} failed ({exc}); the first call will load the model")
            return
        load_seconds = float(payload.get("load_duration") or 0) / 1e9
        self.warmup = {
            "model_name": model_name,
            "elapsed_seconds": float(time.perf_counter() - started_at),
            "load_seconds": load_seconds,
            "keep_alive": self.reranker.keep_alive,
        }
        self.console.log(
            f"Ollama warm-up for {model_name}: load_duration={load_seconds:.1f}s, "
            f"elapsed={self.warmup['elapsed_seconds']:.1f}s, keep_alive={self.reranker.keep_alive}"
        )

    def _llm_cache_path(self, query_id: int, model_name: str) -> Path:
        return (
            self.paths.root.parent.parent
            / "_llm_cache"
            / self.reranker.llm_cache_dir_name(model_name)
            / f"query_{int(query_id)}.json"
        )

    @staticmethod
    def _load_llm_cache(path: Path, post: str) -> dict[str, Any]:
        empty = {"post": post, "decomposition": None, "judgments": {}, "pairwise": {}}
        if not path.exists():
            return empty
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return empty
        if not isinstance(payload, dict) or payload.get("post") != post:
            return empty
        return {
            "post": post,
            "decomposition": payload.get("decomposition"),
            "judgments": dict(payload.get("judgments") or {}),
            "pairwise": dict(payload.get("pairwise") or {}),
        }

    @staticmethod
    def _save_llm_cache(path: Path, cache: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _post_ollama(self, prompt: str, schema: dict[str, Any], model_name: str) -> dict[str, Any]:
        response = requests.post(
            self.reranker.ollama_url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "think": bool(self.reranker.think),
                "format": schema,
                "keep_alive": self.reranker.keep_alive,
                "options": {
                    "temperature": float(self.reranker.temperature),
                    "seed": int(self.reranker.seed),
                },
            },
            timeout=int(self.reranker.timeout_seconds),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama request failed with status {response.status_code}: {response.text}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama response is not a JSON object")
        return payload

    def _call_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        parser: Callable[[str], Any],
        model_name: str,
        label: str,
    ) -> dict[str, Any] | None:
        attempts = max(int(self.reranker.max_retries), 1)
        started_at = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                payload = self._post_ollama(prompt, schema, model_name)
            except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
                self.console.log(f"Ollama {label} call failed on attempt {attempt}/{attempts}: {exc}")
                continue
            thinking_text = _ollama_thinking_text(payload)
            if thinking_text:
                self.console.log(
                    f"Ollama {label} returned thinking text although think={self.reranker.think} was sent "
                    f"({len(thinking_text)} characters)"
                )
            elif not self._logged_think_check:
                self._logged_think_check = True
                self.console.log(
                    f"Verified Ollama request sends think={self.reranker.think}; the response contains no thinking text"
                )
            response_text = _ollama_response_text(payload)
            parsed = parser(response_text)
            if parsed is None:
                self.console.log(
                    f"Ollama {label} returned invalid output on attempt {attempt}/{attempts}: {response_text!r}"
                )
                continue
            return {
                "value": parsed,
                "response": response_text,
                "attempts": attempt,
                "elapsed_seconds": float(time.perf_counter() - started_at),
                "load_seconds": float(payload.get("load_duration") or 0) / 1e9,
                "prompt_eval_count": payload.get("prompt_eval_count"),
                "eval_count": payload.get("eval_count"),
            }
        self.console.log(f"Ollama {label} failed after {attempts} attempts; using the deterministic fallback")
        return None

    def _split_sentences(self, text: str) -> list[str]:
        if self._sentence_splitter is None:
            self._sentence_splitter = spacy.blank("en")
            self._sentence_splitter.add_pipe("sentencizer")
        return [sentence.text.strip() for sentence in self._sentence_splitter(str(text)).sents if sentence.text.strip()]

    def _select_abstract_sentences(
        self,
        claim: dict[str, Any],
        document_ids: list[int],
        candidate_by_id: dict[int, dict[str, str]],
    ) -> dict[int, dict[str, Any]]:
        sentence_count = int(self.reranker.abstract_sentence_count)
        sentences_by_document = {
            document_id: self._split_sentences(candidate_by_id[document_id]["abstract"]) for document_id in document_ids
        }
        selections: dict[int, dict[str, Any]] = {
            document_id: {
                "abstract": candidate_by_id[document_id]["abstract"],
                "sentences_total": len(sentences),
                "selected_indices": list(range(len(sentences))),
            }
            for document_id, sentences in sentences_by_document.items()
        }
        to_encode = [
            (document_id, index, sentence)
            for document_id, sentences in sentences_by_document.items()
            if len(sentences) > sentence_count
            for index, sentence in enumerate(sentences)
        ]
        if not to_encode:
            return selections
        spec = self.retrieval.semantic_spec(self.reranker.sentence_encoder)
        query_text = (spec.query_prefix + f"{claim['claim']} {claim['finding']}").strip()
        texts = [query_text] + [(spec.document_prefix + sentence).strip() for _, _, sentence in to_encode]
        embeddings = self.retrieval._encode_texts_in_memory(spec, texts, "claim and abstract sentences")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        similarities = embeddings[1:] @ embeddings[0]
        scored_by_document: dict[int, list[tuple[float, int]]] = {}
        for (document_id, index, _), similarity in zip(to_encode, similarities.tolist(), strict=True):
            scored_by_document.setdefault(document_id, []).append((float(similarity), index))
        for document_id, scored in scored_by_document.items():
            kept = sorted(index for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:sentence_count])
            sentences = sentences_by_document[document_id]
            selections[document_id] = {
                "abstract": " ".join(sentences[index] for index in kept),
                "sentences_total": len(sentences),
                "selected_indices": kept,
            }
        return selections

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        model_name = self.resolve_active_model()
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        publications = self.retrieval.publications
        candidate_by_id = {
            int(document_id): {"title": str(title), "abstract": str(abstract)}
            for document_id, title, abstract in zip(
                self.retrieval.document_ids.tolist(),
                publications["title"].fillna("").astype(str).tolist(),
                publications["abstract"].fillna("").astype(str).tolist(),
                strict=True,
            )
        }
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        for column in ("reranker_score", "first_stage_normalized", "semantic_score", "fused_score"):
            scored[column] = np.nan
        query_groups = list(frame.groupby("query_id", sort=False))
        total_queries = len(query_groups)
        started_at = time.perf_counter()
        self.console.log(
            f"Claim decomposition reranking for {split_key}: {total_queries} queries with model={model_name}, "
            f"semantic_top_k={self.reranker.semantic_top_k}, field_weights={self.reranker.field_weight_map}, "
            f"anchor_weight={self.reranker.anchor_weight}, "
            f"fusion={self.reranker.first_stage_weight}*first_stage+{self.reranker.semantic_weight}*semantic, "
            f"pairwise_enabled={self.reranker.pairwise_enabled}, abstract_mode={self.reranker.abstract_mode}"
        )
        for query_index, (query_id, group) in enumerate(query_groups, start=1):
            details = self._rerank_query(
                int(query_id),
                str(query_text_by_id[int(query_id)]),
                group.sort_values("rank"),
                candidate_by_id,
                model_name,
            )
            self.query_details[int(query_id)] = details
            row_by_document = {int(row["document_id"]): row for row in details["rows"]}
            mask = scored["query_id"] == query_id
            for column in ("reranker_score", "first_stage_normalized", "semantic_score", "fused_score"):
                scored.loc[mask, column] = (
                    scored.loc[mask, "document_id"]
                    .map(lambda document_id, column=column: row_by_document[int(document_id)][column])
                    .astype(float)
                )
            elapsed = time.perf_counter() - started_at
            rate = query_index / elapsed if elapsed > 0 else 0.0
            eta_minutes = ((total_queries - query_index) / rate / 60.0) if rate > 0 else 0.0
            self.console.log(
                f"Claim decomposition progress for {split_key}: {query_index}/{total_queries} queries "
                f"({100.0 * query_index / total_queries:.1f}%), query_id={int(query_id)}, "
                f"llm_calls={details['timings']['llm_calls']}, cached_calls={details['timings']['cached_calls']}, "
                f"pairwise_calls={len(details['pairwise'])}, fallback={details['fallback']}, "
                f"elapsed={elapsed / 60.0:.1f}m, eta_minutes={eta_minutes:.1f}"
            )
        return scored

    def _cached_or_called(
        self,
        cache_section: dict[str, Any],
        cache_key: str,
        prompt: str,
        schema: dict[str, Any],
        parser: Callable[[str], Any],
        model_name: str,
        label: str,
        counters: dict[str, Any],
        timing_key: str,
    ) -> tuple[Any, bool]:
        cached = cache_section.get(cache_key)
        if isinstance(cached, dict) and parser(cached.get("response", "")) is not None:
            counters["cached_calls"] += 1
            counters[f"cached_{timing_key}"] += float(cached.get("elapsed_seconds") or 0.0)
            return cached["value"], True
        result = self._call_structured(prompt, schema, parser, model_name, label)
        counters["llm_calls"] += 1
        if result is None:
            return None, False
        counters[timing_key] += result["elapsed_seconds"]
        counters["load_seconds"] += result["load_seconds"]
        cache_section[cache_key] = result
        return result["value"], False

    def _judge_candidates(
        self,
        query_id: int,
        claim: dict[str, Any],
        semantic_ids: list[int],
        prompt_candidates: dict[int, dict[str, str]],
        retriever_rank_by_document: dict[int, int],
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> tuple[dict[int, dict[str, int]], list[int]]:
        judgments: dict[int, dict[str, int]] = {}
        failed: list[int] = []
        pending: list[int] = []
        for document_id in semantic_ids:
            cached = cache["judgments"].get(str(document_id))
            if isinstance(cached, dict) and _parse_claim_judgment(cached.get("response", "")) is not None:
                counters["cached_judgments"] += 1
                counters["cached_judgment_seconds"] += float(cached.get("elapsed_seconds") or 0.0)
                judgments[document_id] = cached["value"]
            else:
                pending.append(document_id)
        if counters["cached_judgments"]:
            self.console.log(f"Query {query_id}: reusing {counters['cached_judgments']} cached candidate judgments")
        batch_size = max(int(self.reranker.judgment_batch_size), 1)
        batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
        for batch_index, batch in enumerate(batches, start=1):
            batch_started_at = time.perf_counter()
            batch_judgments: list[dict[str, int]] | None = None
            if len(batch) > 1:
                result = self._call_structured(
                    render_claim_batch_judgment_prompt(
                        claim,
                        [prompt_candidates[document_id] for document_id in batch],
                        candidate_fields=self.reranker.candidate_fields,
                    ),
                    claim_batch_judgment_schema(len(batch)),
                    lambda text, size=len(batch): _parse_claim_batch_judgments(text, size),
                    model_name,
                    f"judgment batch {batch_index}/{len(batches)} for query_id={query_id}, document_ids={batch}",
                )
                counters["llm_calls"] += 1
                counters["batch_calls"] += 1
                if result is not None:
                    counters["judgment_seconds"] += result["elapsed_seconds"]
                    counters["load_seconds"] += result["load_seconds"]
                    batch_judgments = result["value"]
                    share_seconds = result["elapsed_seconds"] / len(batch)
                    for document_id, judgment in zip(batch, batch_judgments, strict=True):
                        judgments[document_id] = judgment
                        cache["judgments"][str(document_id)] = {
                            "value": judgment,
                            "response": json.dumps(judgment),
                            "elapsed_seconds": share_seconds,
                            "attempts": result["attempts"],
                            "batch_document_ids": batch,
                            "batch_response": result["response"],
                        }
                    self._save_llm_cache(cache_path, cache)
                else:
                    self.console.log(
                        f"Query {query_id}: judgment batch {batch_index}/{len(batches)} failed; "
                        f"judging its {len(batch)} candidates one by one"
                    )
            if batch_judgments is None:
                for document_id in batch:
                    if len(batch) > 1:
                        counters["single_fallback_calls"] += 1
                    judgment, _ = self._cached_or_called(
                        cache["judgments"],
                        str(document_id),
                        render_claim_judgment_prompt(
                            claim,
                            prompt_candidates[document_id],
                            candidate_fields=self.reranker.candidate_fields,
                        ),
                        CLAIM_JUDGMENT_SCHEMA,
                        _parse_claim_judgment,
                        model_name,
                        f"judgment for query_id={query_id}, document_id={document_id}",
                        counters,
                        "judgment_seconds",
                    )
                    if judgment is None:
                        failed.append(document_id)
                    else:
                        judgments[document_id] = judgment
                        self._save_llm_cache(cache_path, cache)
            self.console.log(
                f"Query {query_id} judgment batch {batch_index}/{len(batches)} "
                f"({time.perf_counter() - batch_started_at:.1f}s): "
                + "; ".join(
                    f"doc {document_id} (retriever rank {retriever_rank_by_document[document_id]}) "
                    f"{judgments.get(document_id)}"
                    for document_id in batch
                )
            )
        return judgments, failed

    def _rerank_query(
        self,
        query_id: int,
        post: str,
        ordered: pd.DataFrame,
        candidate_by_id: dict[int, dict[str, str]],
        model_name: str | None,
    ) -> dict[str, Any]:
        query_started_at = time.perf_counter()
        document_ids = [int(document_id) for document_id in ordered["document_id"].tolist()]
        retriever_rank_by_document = dict(zip(document_ids, [int(rank) for rank in ordered["rank"].tolist()]))
        retrieval_score_by_document = dict(zip(document_ids, [float(score) for score in ordered["score"].tolist()]))
        semantic_ids = document_ids[: int(self.reranker.semantic_top_k)]
        remaining_ids = document_ids[int(self.reranker.semantic_top_k) :]
        first_stage_normalized = dict(
            zip(semantic_ids, _min_max_normalize([retrieval_score_by_document[d] for d in semantic_ids]))
        )
        counters: dict[str, Any] = {
            "llm_calls": 0,
            "cached_calls": 0,
            "load_seconds": 0.0,
            "decomposition_seconds": 0.0,
            "selection_seconds": 0.0,
            "judgment_seconds": 0.0,
            "pairwise_seconds": 0.0,
            "cached_decomposition_seconds": 0.0,
            "cached_judgment_seconds": 0.0,
            "cached_pairwise_seconds": 0.0,
            "batch_calls": 0,
            "single_fallback_calls": 0,
            "cached_judgments": 0,
        }
        claim: dict[str, Any] | None = None
        judgments: dict[int, dict[str, int]] = {}
        failed_judgments: list[int] = []
        pairwise_records: list[dict[str, Any]] = []
        prompt_candidates = {document_id: candidate_by_id[document_id] for document_id in semantic_ids}
        selections: dict[int, dict[str, Any]] = {}
        cache_path = None if model_name is None else self._llm_cache_path(query_id, model_name)
        cache = None if cache_path is None else self._load_llm_cache(cache_path, post)
        if model_name is not None and cache is not None and cache_path is not None:
            decomposition_section = {"decomposition": cache["decomposition"]} if cache["decomposition"] else {}
            claim, from_cache = self._cached_or_called(
                decomposition_section,
                "decomposition",
                render_claim_decomposition_prompt(post),
                CLAIM_DECOMPOSITION_SCHEMA,
                _parse_claim_decomposition,
                model_name,
                f"decomposition for query_id={query_id}",
                counters,
                "decomposition_seconds",
            )
            if claim is not None:
                cache["decomposition"] = decomposition_section["decomposition"]
                self._save_llm_cache(cache_path, cache)
                self.console.log(
                    f"Query {query_id} claim ({'cached' if from_cache else 'new'}): {json.dumps(claim, ensure_ascii=False)}"
                )
            else:
                self.console.log(f"Query {query_id}: claim decomposition failed; keeping the first-stage order")
        if claim is not None and self.reranker.abstract_mode == "top_sentences":
            selection_started_at = time.perf_counter()
            try:
                selections = self._select_abstract_sentences(claim, semantic_ids, candidate_by_id)
            except Exception as exc:
                self.console.log(f"Query {query_id}: abstract sentence selection failed ({exc}); using full abstracts")
                selections = {}
            counters["selection_seconds"] = float(time.perf_counter() - selection_started_at)
            prompt_candidates = {
                document_id: {
                    "title": candidate_by_id[document_id]["title"],
                    "abstract": selections[document_id]["abstract"]
                    if document_id in selections
                    else candidate_by_id[document_id]["abstract"],
                }
                for document_id in semantic_ids
            }
            self.console.log(
                f"Query {query_id}: selected up to {self.reranker.abstract_sentence_count} abstract sentences for "
                f"{len(selections)} candidates in {counters['selection_seconds']:.1f}s"
            )
        if claim is not None and cache is not None and cache_path is not None and model_name is not None:
            judgments, failed_judgments = self._judge_candidates(
                query_id,
                claim,
                semantic_ids,
                prompt_candidates,
                retriever_rank_by_document,
                cache,
                cache_path,
                model_name,
                counters,
            )
        anchors: dict[int, dict[str, Any]] = {}
        if claim is not None:
            for document_id in semantic_ids:
                anchors[document_id] = claim_entity_anchor(
                    list(claim["entities"]),
                    post,
                    _claim_paper_text(candidate_by_id[document_id], "title_abstract"),
                    penalize_conflicts=self.reranker.anchor_conflicts_enabled,
                )
        field_weights = self.reranker.field_weight_map
        semantic_score_by_document: dict[int, float | None] = {}
        fused_score_by_document: dict[int, float] = {}
        for document_id in semantic_ids:
            judgment = judgments.get(document_id)
            if judgment is None:
                semantic_score_by_document[document_id] = None
                fused_score_by_document[document_id] = first_stage_normalized[document_id]
                continue
            semantic = claim_semantic_score(
                judgment,
                field_weights,
                anchor_score=anchors[document_id]["score"],
                anchor_weight=self.reranker.anchor_weight,
            )
            semantic_score_by_document[document_id] = semantic
            fused_score_by_document[document_id] = claim_fused_score(
                first_stage_normalized[document_id],
                semantic,
                first_stage_weight=self.reranker.first_stage_weight,
                semantic_weight=self.reranker.semantic_weight,
            )
        order = sorted(
            semantic_ids,
            key=lambda document_id: (-fused_score_by_document[document_id], retriever_rank_by_document[document_id]),
        )
        if (
            self.reranker.pairwise_enabled
            and claim is not None
            and cache is not None
            and cache_path is not None
            and model_name is not None
        ):
            pairwise_limit = min(int(self.reranker.pairwise_top_n), len(order))
            for position in range(pairwise_limit - 1):
                document_a = order[position]
                document_b = order[position + 1]
                gap = abs(fused_score_by_document[document_a] - fused_score_by_document[document_b])
                if gap > float(self.reranker.pairwise_max_gap):
                    continue
                answer, from_cache = self._cached_or_called(
                    cache["pairwise"],
                    f"{document_a}_{document_b}",
                    render_claim_pairwise_prompt(
                        claim,
                        prompt_candidates[document_a],
                        prompt_candidates[document_b],
                        candidate_fields=self.reranker.candidate_fields,
                    ),
                    CLAIM_PAIRWISE_SCHEMA,
                    _parse_claim_pairwise_answer,
                    model_name,
                    f"pairwise for query_id={query_id}, A={document_a}, B={document_b}",
                    counters,
                    "pairwise_seconds",
                )
                if answer is not None and not from_cache:
                    self._save_llm_cache(cache_path, cache)
                swapped = answer == "B"
                if swapped:
                    order[position], order[position + 1] = document_b, document_a
                pairwise_records.append(
                    {
                        "positions": [position + 1, position + 2],
                        "document_a": document_a,
                        "document_b": document_b,
                        "fused_gap": gap,
                        "answer": answer,
                        "cached": from_cache,
                        "swapped": swapped,
                    }
                )
                self.console.log(
                    f"Query {query_id} pairwise at positions {position + 1}/{position + 2}: A={document_a}, "
                    f"B={document_b}, fused_gap={gap:.4f}, answer={answer}, swapped={swapped}"
                )
        final_order = order + remaining_ids
        total = len(final_order)
        rows = []
        for position, document_id in enumerate(final_order):
            in_semantic = document_id in first_stage_normalized
            semantic_value = semantic_score_by_document.get(document_id)
            anchor = anchors.get(document_id)
            selection = selections.get(document_id)
            rows.append(
                {
                    "document_id": document_id,
                    "retriever_rank": retriever_rank_by_document[document_id],
                    "retrieval_score": retrieval_score_by_document[document_id],
                    "first_stage_normalized": first_stage_normalized[document_id] if in_semantic else np.nan,
                    "judgment": judgments.get(document_id),
                    "anchor": anchor,
                    "abstract_selection": None
                    if selection is None
                    else {
                        "sentences_total": selection["sentences_total"],
                        "selected_indices": selection["selected_indices"],
                    },
                    "semantic_score": np.nan if semantic_value is None else semantic_value,
                    "fused_score": fused_score_by_document[document_id] if in_semantic else np.nan,
                    "final_rank": position + 1,
                    "reranker_score": float(total - position),
                }
            )
        if model_name is None:
            fallback = "ollama_unavailable"
        elif claim is None:
            fallback = "decomposition_failed"
        elif failed_judgments:
            fallback = "partial_judgments"
        else:
            fallback = None
        counters["total_seconds"] = float(time.perf_counter() - query_started_at)
        return {
            "query_id": query_id,
            "model_name": model_name,
            "claim": claim,
            "failed_judgments": failed_judgments,
            "fallback": fallback,
            "pairwise": pairwise_records,
            "rows": rows,
            "timings": counters,
        }


class OllamaClaimJudgmentTopKReranker(OllamaClaimDecompositionTopKReranker):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: OllamaClaimJudgmentRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._venue_key_set: set[str] | None = None

    def _venue_keys(self) -> set[str]:
        if self._venue_key_set is None:
            publications = self.retrieval.publications
            venues = publications["venue"].fillna("").astype(str).tolist() if "venue" in publications else []
            self._venue_key_set = claim_venue_keys(venues)
        return self._venue_key_set

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        model_name = self.resolve_active_model()
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        publications = self.retrieval.publications
        candidate_by_id = {
            int(document_id): {"title": str(title), "abstract": str(abstract)}
            for document_id, title, abstract in zip(
                self.retrieval.document_ids.tolist(),
                publications["title"].fillna("").astype(str).tolist(),
                publications["abstract"].fillna("").astype(str).tolist(),
                strict=True,
            )
        }
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.nan
        query_groups = list(frame.groupby("query_id", sort=False))
        total_queries = len(query_groups)
        started_at = time.perf_counter()
        self.console.log(
            f"Claim judgment reranking for {split_key}: {total_queries} queries with model={model_name}, "
            f"judged_top_k={self.reranker.semantic_top_k}, pairwise_enabled={self.reranker.pairwise_enabled}, "
            f"pairwise_max_group_size={self.reranker.pairwise_max_group_size}"
        )
        for query_index, (query_id, group) in enumerate(query_groups, start=1):
            details = self._rerank_query(
                int(query_id),
                str(query_text_by_id[int(query_id)]),
                group.sort_values("rank"),
                candidate_by_id,
                model_name,
            )
            self.query_details[int(query_id)] = details
            score_by_document = {int(row["document_id"]): row["reranker_score"] for row in details["rows"]}
            mask = scored["query_id"] == query_id
            scored.loc[mask, "reranker_score"] = scored.loc[mask, "document_id"].map(score_by_document).astype(float)
            elapsed = time.perf_counter() - started_at
            rate = query_index / elapsed if elapsed > 0 else 0.0
            eta_minutes = ((total_queries - query_index) / rate / 60.0) if rate > 0 else 0.0
            self.console.log(
                f"Claim judgment progress for {split_key}: {query_index}/{total_queries} queries "
                f"({100.0 * query_index / total_queries:.1f}%), query_id={int(query_id)}, "
                f"llm_calls={details['timings']['llm_calls']}, cached_calls={details['timings']['cached_calls']}, "
                f"pairwise_calls={details['timings']['pairwise_calls']}, fallback={details['fallback']}, "
                f"elapsed={elapsed / 60.0:.1f}m, eta_minutes={eta_minutes:.1f}"
            )
        return scored

    def _pairwise_answer(
        self,
        query_id: int,
        claim: dict[str, Any],
        document_a: int,
        document_b: int,
        candidate_by_id: dict[int, dict[str, str]],
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> str | None:
        answer, from_cache = self._cached_or_called(
            cache["pairwise"],
            f"{document_a}_{document_b}",
            render_claim_source_pairwise_prompt(
                claim,
                candidate_by_id[document_a],
                candidate_by_id[document_b],
                candidate_fields=self.reranker.candidate_fields,
            ),
            CLAIM_PAIRWISE_SCHEMA,
            _parse_claim_pairwise_answer,
            model_name,
            f"pairwise for query_id={query_id}, A={document_a}, B={document_b}",
            counters,
            "pairwise_seconds",
        )
        counters["pairwise_calls"] += 1
        if answer is not None and not from_cache:
            self._save_llm_cache(cache_path, cache)
        return answer

    def _resolve_rank_one_ties(
        self,
        query_id: int,
        claim: dict[str, Any],
        order: list[int],
        judgments: dict[int, dict[str, Any]],
        candidate_by_id: dict[int, dict[str, str]],
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> tuple[list[int], dict[str, Any] | None]:
        top_key = claim_match_evidence_key(judgments[order[0]])
        tied = [document_id for document_id in order if claim_match_evidence_key(judgments[document_id]) == top_key]
        if len(tied) < 2:
            return order, None
        cap = self.reranker.pairwise_max_group_size
        contenders = tied if cap is None else tied[: int(cap)]
        self.console.log(
            f"Query {query_id}: {len(tied)} candidates tied for rank 1 on evidence key {top_key}; pairwise knockout over {len(contenders)} candidates "
            f"{'(capped)' if len(contenders) < len(tied) else ''}"
        )
        champion = contenders[0]
        decisions = []
        for challenger in contenders[1:]:
            forward = self._pairwise_answer(
                query_id, claim, champion, challenger, candidate_by_id, cache, cache_path, model_name, counters
            )
            reverse = self._pairwise_answer(
                query_id, claim, challenger, champion, candidate_by_id, cache, cache_path, model_name, counters
            )
            if forward == "B" and reverse == "A":
                winner = challenger
                outcome = "challenger preferred in both orders"
            elif forward == "A" and reverse == "B":
                winner = champion
                outcome = "champion preferred in both orders"
            else:
                winner = champion
                outcome = "inconsistent or tie; champion kept"
            decisions.append(
                {
                    "champion": champion,
                    "challenger": challenger,
                    "champion_first_answer": forward,
                    "challenger_first_answer": reverse,
                    "winner": winner,
                    "outcome": outcome,
                }
            )
            self.console.log(
                f"Query {query_id} pairwise: champion={champion} vs challenger={challenger}: "
                f"A=champion -> {forward}, A=challenger -> {reverse}; winner={winner} ({outcome})"
            )
            champion = winner
        resolved = [champion] + [document_id for document_id in order if document_id != champion]
        return resolved, {
            "tied_group": tied,
            "contenders": contenders,
            "capped": len(contenders) < len(tied),
            "decisions": decisions,
            "rank_one": champion,
        }

    def _rerank_query(
        self,
        query_id: int,
        post: str,
        ordered: pd.DataFrame,
        candidate_by_id: dict[int, dict[str, str]],
        model_name: str | None,
    ) -> dict[str, Any]:
        query_started_at = time.perf_counter()
        document_ids = [int(document_id) for document_id in ordered["document_id"].tolist()]
        retriever_rank_by_document = dict(zip(document_ids, [int(rank) for rank in ordered["rank"].tolist()]))
        judged_ids = document_ids[: int(self.reranker.semantic_top_k)]
        remaining_ids = document_ids[int(self.reranker.semantic_top_k) :]
        counters: dict[str, Any] = {
            "llm_calls": 0,
            "cached_calls": 0,
            "pairwise_calls": 0,
            "load_seconds": 0.0,
            "decomposition_seconds": 0.0,
            "judgment_seconds": 0.0,
            "pairwise_seconds": 0.0,
            "cached_decomposition_seconds": 0.0,
            "cached_judgment_seconds": 0.0,
            "cached_pairwise_seconds": 0.0,
        }
        claim: dict[str, Any] | None = None
        judgments: dict[int, dict[str, Any]] = {}
        failed_judgments: list[int] = []
        pairwise: dict[str, Any] | None = None
        cache_path = None if model_name is None else self._llm_cache_path(query_id, model_name)
        cache = None if cache_path is None else self._load_llm_cache(cache_path, post)
        if model_name is not None and cache is not None and cache_path is not None:
            extraction_section = {"decomposition": cache["decomposition"]} if cache["decomposition"] else {}
            claim, from_cache = self._cached_or_called(
                extraction_section,
                "decomposition",
                render_claim_clue_extraction_prompt(post),
                CLAIM_CLUE_EXTRACTION_SCHEMA,
                lambda text: sanitize_claim_extraction(_parse_claim_clue_extraction(text), self._venue_keys()),
                model_name,
                f"claim extraction for query_id={query_id}",
                counters,
                "decomposition_seconds",
            )
            if claim is not None:
                cache["decomposition"] = extraction_section["decomposition"]
                self._save_llm_cache(cache_path, cache)
                canonical_entities, canonicalized = canonicalize_claim_entities(
                    list(claim["entities"]),
                    [_claim_paper_text(candidate_by_id[document_id], "title_abstract") for document_id in judged_ids],
                )
                claim = {**claim, "entities": canonical_entities, "canonicalized": canonicalized}
                self.console.log(
                    f"Query {query_id} claim ({'cached' if from_cache else 'new'}): raw entities={claim['raw_entities']}, "
                    f"raw clues={claim['raw_clues']}, dropped={claim['dropped']}, canonicalized={canonicalized}"
                )
                self.console.log(
                    f"Query {query_id} sanitized claim: {json.dumps({key: claim[key] for key in ('claim', 'entities', 'clues')}, ensure_ascii=False)}"
                )
            else:
                self.console.log(f"Query {query_id}: claim extraction failed; falling back to the first-stage order")
        if claim is not None and cache is not None and cache_path is not None and model_name is not None:
            entity_count = len(claim["entities"])
            clue_count = len(claim["clues"])
            for candidate_index, document_id in enumerate(judged_ids, start=1):
                judgment_started_at = time.perf_counter()
                judgment, from_cache = self._cached_or_called(
                    cache["judgments"],
                    str(document_id),
                    render_claim_match_judgment_prompt(
                        claim,
                        candidate_by_id[document_id],
                        candidate_fields=self.reranker.candidate_fields,
                    ),
                    claim_match_judgment_schema(entity_count, clue_count),
                    lambda text, entities=entity_count, clues=clue_count: _parse_claim_match_judgment(
                        text, entities, clues
                    ),
                    model_name,
                    f"judgment for query_id={query_id}, document_id={document_id}",
                    counters,
                    "judgment_seconds",
                )
                if judgment is None:
                    failed_judgments.append(document_id)
                else:
                    judgments[document_id] = judgment
                    if not from_cache:
                        self._save_llm_cache(cache_path, cache)
                self.console.log(
                    f"Query {query_id} judgment {candidate_index}/{len(judged_ids)}: document_id={document_id}, "
                    f"qwen_rank={retriever_rank_by_document[document_id]}, judgment={judgment}, "
                    f"{'cached' if from_cache else f'{time.perf_counter() - judgment_started_at:.1f}s'}"
                )
        if model_name is None:
            fallback = "ollama_unavailable"
        elif claim is None:
            fallback = "claim_extraction_failed"
        elif failed_judgments:
            fallback = "failed_judgments_ranked_last"
        else:
            fallback = None
        if claim is None:
            final_order = list(document_ids)
        else:
            order = sorted(judgments, key=lambda document_id: claim_match_sort_key(document_id, judgments[document_id]))
            if (
                self.reranker.pairwise_enabled
                and order
                and cache is not None
                and cache_path is not None
                and model_name is not None
            ):
                order, pairwise = self._resolve_rank_one_ties(
                    query_id, claim, order, judgments, candidate_by_id, cache, cache_path, model_name, counters
                )
            final_order = order + sorted(failed_judgments) + remaining_ids
        total = len(final_order)
        rows = [
            {
                "document_id": document_id,
                "retriever_rank": retriever_rank_by_document[document_id],
                "judgment": judgments.get(document_id),
                "judgment_failed": document_id in failed_judgments,
                "final_rank": position + 1,
                "reranker_score": float(total - position),
            }
            for position, document_id in enumerate(final_order)
        ]
        counters["total_seconds"] = float(time.perf_counter() - query_started_at)
        return {
            "query_id": query_id,
            "model_name": model_name,
            "claim": claim,
            "failed_judgments": failed_judgments,
            "fallback": fallback,
            "pairwise": pairwise,
            "rows": rows,
            "timings": counters,
        }


class OllamaListwiseClaimTopKReranker(OllamaClaimJudgmentTopKReranker):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: OllamaListwiseClaimRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._tokenizer: Any = None
        self._num_ctx_by_prompt: dict[str, int] = {}

    def _tokenizer_instance(self) -> Any:
        if self._tokenizer is None:
            self.console.log(f"Loading tokenizer {self.reranker.tokenizer_name} to size the Ollama context window")
            self._tokenizer = AutoTokenizer.from_pretrained(self.reranker.tokenizer_name)
        return self._tokenizer

    def _context_size(self, prompt: str, labels: list[int]) -> tuple[int, int, int]:
        tokenizer = self._tokenizer_instance()
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        longest_answer = json.dumps({"ranking": sorted((int(label) for label in labels), reverse=True)}, indent=2)
        answer_tokens = len(tokenizer.encode(longest_answer, add_special_tokens=False))
        return prompt_tokens + answer_tokens, prompt_tokens, answer_tokens

    def _post_ollama(self, prompt: str, schema: dict[str, Any], model_name: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": float(self.reranker.temperature),
            "seed": int(self.reranker.seed),
        }
        if prompt in self._num_ctx_by_prompt:
            options["num_ctx"] = int(self._num_ctx_by_prompt[prompt])
        response = requests.post(
            self.reranker.ollama_url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "think": bool(self.reranker.think),
                "format": schema,
                "keep_alive": self.reranker.keep_alive,
                "options": options,
            },
            timeout=int(self.reranker.timeout_seconds),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama request failed with status {response.status_code}: {response.text}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama response is not a JSON object")
        if payload.get("done_reason") == "length":
            self.console.log("Ollama stopped because the output hit the context limit; the answer may be truncated")
        return payload

    def _ranking_pass(
        self,
        query_id: int,
        stage: str,
        direction: str,
        display_documents: list[int],
        label_of: dict[int, int],
        prompt: str,
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> dict[str, Any] | None:
        labels = [label_of[document_id] for document_id in display_documents]
        num_ctx, prompt_tokens, answer_tokens = self._context_size(prompt, labels)
        self._num_ctx_by_prompt[prompt] = num_ctx
        started_at = time.perf_counter()
        result, from_cache = self._cached_or_called(
            cache["judgments"],
            f"{stage}_{direction}_{'-'.join(str(document_id) for document_id in display_documents)}",
            prompt,
            listwise_claim_ranking_schema(labels),
            lambda text, allowed=labels: _parse_listwise_claim_ranking(text, allowed),
            model_name,
            f"{stage} ranking ({direction}) for query_id={query_id}",
            counters,
            f"{stage}_seconds",
        )
        self._num_ctx_by_prompt.pop(prompt, None)
        if result is None:
            self.console.log(f"Query {query_id}: {stage} ranking ({direction}) failed")
            return None
        if not from_cache:
            self._save_llm_cache(cache_path, cache)
        document_of = {label: document_id for document_id, label in label_of.items()}
        display_spearman = spearman_with_display_order(result["ranking"], labels)
        self.console.log(
            f"Query {query_id} {stage} ranking ({direction}, {len(labels)} candidates, prompt_tokens={prompt_tokens}, "
            f"num_ctx={num_ctx}, {'cached' if from_cache else f'{time.perf_counter() - started_at:.1f}s'}): "
            f"labels={result['ranking']}, missing={result['missing']}, duplicates={result['duplicates']}, "
            f"spearman_with_display_order={None if display_spearman is None else round(display_spearman, 3)}"
        )
        return {
            "direction": direction,
            "ranking_labels": result["ranking"],
            "ranking_documents": [document_of[label] for label in result["ranking"]],
            "missing_documents": [document_of[label] for label in result["missing"]],
            "duplicates": result["duplicates"],
            "invalid": result["invalid"],
            "spearman_with_display_order": display_spearman,
            "prompt_tokens": prompt_tokens,
            "answer_token_budget": answer_tokens,
            "num_ctx": num_ctx,
            "cached": from_cache,
        }

    def _stage(
        self,
        query_id: int,
        stage: str,
        documents: list[int],
        render: Callable[[list[int], dict[int, int]], str],
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> dict[str, Any]:
        base_order = sorted(int(document_id) for document_id in documents)
        label_of = {document_id: position for position, document_id in enumerate(base_order, start=1)}
        passes = []
        for direction, display in (("forward", base_order), ("reversed", list(reversed(base_order)))):
            outcome = self._ranking_pass(
                query_id,
                stage,
                direction,
                display,
                label_of,
                render(display, label_of),
                cache,
                cache_path,
                model_name,
                counters,
            )
            if outcome is not None:
                passes.append(outcome)
        if not passes:
            return {"passes": [], "order": None, "mean_rank": None, "label_of": label_of}
        order, mean_rank_by_label_document = aggregate_listwise_rankings(
            [outcome["ranking_labels"] for outcome in passes],
            {label: document_id for document_id, label in label_of.items()},
        )
        return {"passes": passes, "order": order, "mean_rank": mean_rank_by_label_document, "label_of": label_of}

    def _abstract_sufficiency(self, document_ids: list[int], candidate_by_id: dict[int, dict[str, str]]) -> dict[int, dict[str, Any]]:
        sufficiency = {}
        for document_id in document_ids:
            abstract = candidate_by_id[document_id]["abstract"]
            words = abstract_word_count(abstract)
            sentences = len(self._split_sentences(abstract)) if abstract.strip() else 0
            sufficiency[document_id] = {
                "words": words,
                "sentences": sentences,
                "sufficient": abstract_is_sufficient(
                    words,
                    sentences,
                    min_words=self.reranker.abstract_min_words,
                    min_sentences=self.reranker.abstract_min_sentences,
                ),
            }
        return sufficiency

    def _sliding_abstract_stage(
        self,
        query_id: int,
        post: str,
        claim_text: str,
        initial_order: list[int],
        candidate_by_id: dict[int, dict[str, str]],
        cache: dict[str, Any],
        cache_path: Path,
        model_name: str,
        counters: dict[str, Any],
    ) -> dict[str, Any]:
        order = list(initial_order)
        sufficiency = self._abstract_sufficiency(order, candidate_by_id)
        starts = sliding_window_starts(len(order), int(self.reranker.window_size), int(self.reranker.window_step))
        windows = []
        for window_index, start in enumerate(starts, start=1):
            end = min(start + int(self.reranker.window_size), len(order))
            window_documents = order[start:end]
            label_of = {document_id: position for position, document_id in enumerate(window_documents, start=1)}
            prompt = render_listwise_claim_window_prompt(
                post,
                claim_text,
                [
                    (label_of[document_id], candidate_by_id[document_id], sufficiency[document_id]["sufficient"])
                    for document_id in window_documents
                ],
            )
            outcome = self._ranking_pass(
                query_id,
                "abstract",
                f"window_{window_index}_positions_{start + 1}_{end}",
                window_documents,
                label_of,
                prompt,
                cache,
                cache_path,
                model_name,
                counters,
            )
            if outcome is None:
                raw_reranked = list(window_documents)
            else:
                raw_reranked = apply_window_ranking(window_documents, outcome["ranking_labels"])
            if self.reranker.incomplete_protection == "soft":
                reranked, protection = soft_protect_window(
                    window_documents,
                    raw_reranked,
                    {document_id: int(sufficiency[document_id]["words"]) for document_id in window_documents},
                    full_protection_words=self.reranker.protection_full_words,
                    no_protection_words=self.reranker.protection_none_words,
                )
            elif self.reranker.incomplete_protection == "strict":
                reranked, protection = protect_incomplete_in_window(
                    window_documents,
                    raw_reranked,
                    {document_id for document_id in window_documents if not sufficiency[document_id]["sufficient"]},
                )
            else:
                reranked, protection = raw_reranked, {"protected": [], "displaced": []}
            order[start:end] = reranked
            windows.append(
                {
                    "window": window_index,
                    "positions": [start + 1, end],
                    "documents_in": window_documents,
                    "documents_raw": raw_reranked,
                    "documents_out": reranked,
                    "protection": protection,
                    "failed": outcome is None,
                    "pass": outcome,
                    "order_after": list(order),
                }
            )
            self.console.log(
                f"Query {query_id} window {window_index}/{len(starts)} (positions {start + 1}-{end}): "
                f"in={window_documents} raw={raw_reranked} out={reranked}{' FAILED, order kept' if outcome is None else ''}"
                f"{f' protection={protection}' if protection['protected'] else ''}"
            )
        return {"order": order, "windows": windows, "sufficiency": sufficiency}

    def _rerank_query(
        self,
        query_id: int,
        post: str,
        ordered: pd.DataFrame,
        candidate_by_id: dict[int, dict[str, str]],
        model_name: str | None,
    ) -> dict[str, Any]:
        query_started_at = time.perf_counter()
        document_ids = [int(document_id) for document_id in ordered["document_id"].tolist()]
        retriever_rank_by_document = dict(zip(document_ids, [int(rank) for rank in ordered["rank"].tolist()]))
        judged_ids = document_ids[: int(self.reranker.semantic_top_k)]
        remaining_ids = document_ids[int(self.reranker.semantic_top_k) :]
        counters: dict[str, Any] = {
            "llm_calls": 0,
            "cached_calls": 0,
            "pairwise_calls": 0,
            "load_seconds": 0.0,
            "decomposition_seconds": 0.0,
            "cached_decomposition_seconds": 0.0,
            "title_seconds": 0.0,
            "cached_title_seconds": 0.0,
            "abstract_seconds": 0.0,
            "cached_abstract_seconds": 0.0,
        }
        claim: dict[str, Any] | None = None
        title_stage: dict[str, Any] | None = None
        abstract_stage: dict[str, Any] | None = None
        sufficiency: dict[int, dict[str, Any]] = {}
        combined_score: dict[int, float] = {}
        sliding: dict[str, Any] | None = None
        cache_path = None if model_name is None else self._llm_cache_path(query_id, model_name)
        cache = None if cache_path is None else self._load_llm_cache(cache_path, post)
        if model_name is not None and cache is not None and cache_path is not None:
            extraction_section = {"decomposition": cache["decomposition"]} if cache["decomposition"] else {}
            claim, from_cache = self._cached_or_called(
                extraction_section,
                "decomposition",
                render_claim_clue_extraction_prompt(post),
                CLAIM_CLUE_EXTRACTION_SCHEMA,
                lambda text: sanitize_claim_extraction(_parse_claim_clue_extraction(text), self._venue_keys()),
                model_name,
                f"claim extraction for query_id={query_id}",
                counters,
                "decomposition_seconds",
            )
            if claim is not None:
                cache["decomposition"] = extraction_section["decomposition"]
                self._save_llm_cache(cache_path, cache)
                self.console.log(f"Query {query_id} claim ({'cached' if from_cache else 'new'}): {claim['claim']}")
            else:
                self.console.log(f"Query {query_id}: claim extraction failed; falling back to the first-stage order")
        if claim is not None and cache is not None and cache_path is not None and model_name is not None:
            claim_text = str(claim["claim"])
            title_stage = self._stage(
                query_id,
                "title",
                judged_ids,
                lambda display, label_of: render_listwise_claim_title_prompt(
                    post,
                    claim_text,
                    [(label_of[document_id], candidate_by_id[document_id]["title"]) for document_id in display],
                ),
                cache,
                cache_path,
                model_name,
                counters,
            )
            if title_stage["order"] is not None and self.reranker.abstract_stage_mode == "sliding_window":
                sliding = self._sliding_abstract_stage(
                    query_id,
                    post,
                    claim_text,
                    title_stage["order"],
                    candidate_by_id,
                    cache,
                    cache_path,
                    model_name,
                    counters,
                )
            elif title_stage["order"] is not None:
                top_documents = title_stage["order"][: int(self.reranker.abstract_stage_top_n)]
                abstract_stage = self._stage(
                    query_id,
                    "abstract",
                    top_documents,
                    lambda display, label_of: render_listwise_claim_abstract_prompt(
                        post,
                        claim_text,
                        [(label_of[document_id], candidate_by_id[document_id]) for document_id in display],
                    ),
                    cache,
                    cache_path,
                    model_name,
                    counters,
                )
        if model_name is None:
            fallback = "ollama_unavailable"
        elif claim is None:
            fallback = "claim_extraction_failed"
        elif title_stage is None or title_stage["order"] is None:
            fallback = "title_ranking_failed"
        elif sliding is not None and any(window["failed"] for window in sliding["windows"]):
            fallback = "abstract_window_failed_order_kept"
        elif sliding is None and (abstract_stage is None or abstract_stage["order"] is None):
            fallback = "abstract_ranking_failed_title_order_kept"
        elif len(title_stage["passes"]) < 2 or (sliding is None and len(abstract_stage["passes"]) < 2):
            fallback = "single_pass_used"
        else:
            fallback = None
        if title_stage is None or title_stage["order"] is None:
            final_order = list(document_ids)
        elif sliding is not None:
            sufficiency = sliding["sufficiency"]
            final_order = list(sliding["order"]) + remaining_ids
        else:
            top_n = int(self.reranker.abstract_stage_top_n)
            top_documents = title_stage["order"][:top_n]
            if abstract_stage is not None and abstract_stage["order"] is not None:
                sufficiency = {}
                for document_id in top_documents:
                    abstract = candidate_by_id[document_id]["abstract"]
                    words = abstract_word_count(abstract)
                    sentences = len(self._split_sentences(abstract)) if abstract.strip() else 0
                    sufficiency[document_id] = {
                        "words": words,
                        "sentences": sentences,
                        "sufficient": abstract_is_sufficient(
                            words,
                            sentences,
                            min_words=self.reranker.abstract_min_words,
                            min_sentences=self.reranker.abstract_min_sentences,
                        ),
                    }
                head, combined_score = combine_title_and_abstract_ranks(
                    top_documents,
                    title_stage["mean_rank"],
                    abstract_stage["mean_rank"],
                    {document_id: values["sufficient"] for document_id, values in sufficiency.items()},
                    title_weight=self.reranker.title_weight,
                    abstract_weight=self.reranker.abstract_weight,
                )
                insufficient = [document_id for document_id, values in sufficiency.items() if not values["sufficient"]]
                self.console.log(
                    f"Query {query_id}: combined title/abstract evidence for the top {len(top_documents)}; "
                    f"insufficient abstracts kept at title-stage evidence: {insufficient}"
                )
            else:
                head = top_documents
            final_order = head + [document_id for document_id in title_stage["order"] if document_id not in set(head)]
            final_order += remaining_ids
        title_rank = (
            {}
            if title_stage is None or title_stage["order"] is None
            else {document_id: position for position, document_id in enumerate(title_stage["order"], start=1)}
        )
        abstract_rank = (
            {}
            if abstract_stage is None or abstract_stage["order"] is None
            else {document_id: position for position, document_id in enumerate(abstract_stage["order"], start=1)}
        )
        total = len(final_order)
        rows = [
            {
                "document_id": document_id,
                "retriever_rank": retriever_rank_by_document[document_id],
                "title_stage_rank": title_rank.get(document_id),
                "title_stage_mean_rank": None if title_stage is None or title_stage["mean_rank"] is None else title_stage["mean_rank"].get(document_id),
                "abstract_stage_rank": abstract_rank.get(document_id),
                "abstract_stage_mean_rank": None
                if abstract_stage is None or abstract_stage["mean_rank"] is None
                else abstract_stage["mean_rank"].get(document_id),
                "abstract_words": sufficiency.get(document_id, {}).get("words"),
                "abstract_sentences": sufficiency.get(document_id, {}).get("sentences"),
                "abstract_sufficient": sufficiency.get(document_id, {}).get("sufficient"),
                "combined_score": combined_score.get(document_id),
                "final_rank": position + 1,
                "reranker_score": float(total - position),
            }
            for position, document_id in enumerate(final_order)
        ]
        counters["total_seconds"] = float(time.perf_counter() - query_started_at)
        return {
            "query_id": query_id,
            "model_name": model_name,
            "claim": claim,
            "fallback": fallback,
            "failed_judgments": [],
            "pairwise": None,
            "title_stage": None
            if title_stage is None
            else {key: value for key, value in title_stage.items() if key != "label_of"},
            "abstract_stage": None
            if abstract_stage is None
            else {key: value for key, value in abstract_stage.items() if key != "label_of"},
            "sliding_windows": None if sliding is None else sliding["windows"],
            "rows": rows,
            "timings": counters,
        }


class LLMPointwiseTopKReranker(TopKScoreRunner):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: LLMPointwiseRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker
        self._checked_model_availability = False
        self._checked_runtime = False

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        self._ensure_model_available()
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_fields_by_id = self._document_fields_by_id()
        scores: list[float] = []
        responses: list[str] = []
        prompt_tokens: list[int] = []
        response_tokens: list[int] = []
        cache_hits = 0
        started_at = time.perf_counter()
        total_pairs = int(len(frame))
        progress_interval = max(25, math.ceil(total_pairs / 25)) if total_pairs > 0 else 25
        completed_pairs = 0
        for row in frame.itertuples(index=False):
            pair_started_at = time.perf_counter()
            query_id = int(row.query_id)
            document_id = int(row.document_id)
            cached = self._load_pair_cache(query_id, document_id)
            if cached is not None:
                cache_hits += 1
                score, response_text, prompt_eval_count, eval_count = cached
            else:
                prompt = self.reranker.render_prompt(
                    query=str(query_text_by_id[query_id]),
                    title=document_fields_by_id[document_id]["title"],
                    abstract=document_fields_by_id[document_id]["abstract"],
                )
                try:
                    score, response_text, prompt_eval_count, eval_count = self._score_prompt(prompt)
                except Exception as exc:
                    pair_elapsed = time.perf_counter() - pair_started_at
                    total_elapsed = time.perf_counter() - started_at
                    raise RuntimeError(
                        f"LLM reranking failed for query_id={query_id}, document_id={document_id}, "
                        f"bm25_rank={int(row.rank)}, completed_pairs={completed_pairs}/{total_pairs}, "
                        f"pair_elapsed_seconds={pair_elapsed:.2f}, total_elapsed_seconds={total_elapsed:.2f}: {exc}"
                    ) from exc
                self._save_pair_cache(
                    query_id=query_id,
                    document_id=document_id,
                    prompt=prompt,
                    score=score,
                    response_text=response_text,
                    prompt_eval_count=prompt_eval_count,
                    eval_count=eval_count,
                )
            scores.append(score)
            responses.append(response_text)
            prompt_tokens.append(prompt_eval_count)
            response_tokens.append(eval_count)
            completed_pairs += 1
            if completed_pairs % progress_interval == 0 or completed_pairs == total_pairs:
                elapsed = time.perf_counter() - started_at
                avg_seconds = elapsed / float(completed_pairs)
                remaining_pairs = total_pairs - completed_pairs
                eta_seconds = avg_seconds * float(remaining_pairs)
                self.console.log(
                    f"LLM reranking progress for {split_key}: {completed_pairs}/{total_pairs} pairs, "
                    f"elapsed={elapsed/60.0:.2f}m, avg_seconds_per_pair={avg_seconds:.2f}, "
                    f"eta_hours={eta_seconds/3600.0:.2f}"
                )
        if cache_hits:
            self.console.log(f"Reused {cache_hits} cached Ollama pair scores for en_test")
        _console_log_timing(
            self.console,
            "Completed LLM pointwise scoring",
            time.perf_counter() - started_at,
            details=f"split={split_key}, pairs={total_pairs}, cache_hits={cache_hits}",
        )
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.asarray(scores, dtype=np.float32)
        scored["reranker_response"] = responses
        scored["prompt_tokens"] = prompt_tokens
        scored["response_tokens"] = response_tokens
        return scored

    def _ensure_model_available(self) -> None:
        if not self._checked_model_availability:
            models = self._available_ollama_models()
            self._checked_model_availability = True
            if self.reranker.model_name not in models:
                available = ", ".join(models) if models else "(none)"
                raise RuntimeError(
                    f"Ollama model '{self.reranker.model_name}' is not installed at {self.reranker.ollama_url}. "
                    f"Available models: {available}. Run `ollama pull {self.reranker.model_name}` first "
                    f"or pass --reranker with one of the installed tags."
                )
        if self._checked_runtime:
            return
        self._checked_runtime = True
        try:
            self._post_ollama("0", max_new_tokens=1)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Ollama model '{self.reranker.model_name}' is installed but cannot run on this machine: {exc}"
            ) from exc

    def _available_ollama_models(self) -> list[str]:
        tags_url = self._tags_url()
        try:
            response = requests.get(
                tags_url,
                timeout=min(int(self.reranker.timeout_seconds), 30),
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Unable to query Ollama tags at {tags_url}: {exc}") from exc
        payload = response.json()
        raw_models = payload.get("models", [])
        names: list[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("model") or "").strip()
                if name and name not in names:
                    names.append(name)
        return names

    def _tags_url(self) -> str:
        value = self.reranker.ollama_url.rstrip("/")
        if value.endswith("/api/generate"):
            return value[: -len("/api/generate")] + "/api/tags"
        if value.endswith("/api/chat"):
            return value[: -len("/api/chat")] + "/api/tags"
        return value + "/api/tags"

    def _score_prompt(self, prompt: str) -> tuple[float, str, int, int]:
        payload = self._call_ollama(prompt)
        response_text = _ollama_response_text(payload)
        if not response_text:
            raise ValueError(f"Ollama returned an empty reranker response for model {self.reranker.model_name}")
        score = _parse_llm_score(response_text)
        prompt_eval_count = int(payload.get("prompt_eval_count", 0) or 0)
        eval_count = int(payload.get("eval_count", 0) or 0)
        return score, response_text, prompt_eval_count, eval_count

    def _call_ollama(self, prompt: str) -> dict[str, Any]:
        attempts = max(int(self.reranker.max_retries), 1)
        last_error: Exception | None = None
        started_at = time.perf_counter()
        for attempt in range(1, attempts + 1):
            attempt_started_at = time.perf_counter()
            try:
                payload = self._post_ollama(prompt, max_new_tokens=self.reranker.max_new_tokens)
                if not isinstance(payload, dict):
                    raise ValueError("Ollama reranker response is not a JSON object")
                payload["_client_elapsed_seconds"] = float(time.perf_counter() - attempt_started_at)
                return payload
            except (requests.exceptions.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                attempt_elapsed = time.perf_counter() - attempt_started_at
                total_elapsed = time.perf_counter() - started_at
                self.console.log(
                    f"Ollama reranker call failed on attempt {attempt}/{attempts} "
                    f"for {self.reranker.model_name}: {exc} "
                    f"(attempt_seconds={attempt_elapsed:.2f}, total_seconds={total_elapsed:.2f})"
                )
        assert last_error is not None
        raise last_error

    def _post_ollama(self, prompt: str, *, max_new_tokens: int) -> dict[str, Any]:
        response = requests.post(
            self.reranker.ollama_url,
            json={
                "model": self.reranker.model_name,
                "prompt": prompt,
                "stream": False,
                "think": not self.reranker.disable_thinking,
                "options": {
                    "temperature": float(self.reranker.temperature),
                    "num_predict": int(max_new_tokens),
                    "num_ctx": int(self.reranker.max_input_tokens),
                },
            },
            timeout=int(self.reranker.timeout_seconds),
        )
        if response.status_code >= 400:
            raise RuntimeError(self._ollama_error_message(response))
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama reranker response is not a JSON object")
        return payload

    @staticmethod
    def _ollama_error_message(response: requests.Response) -> str:
        default = f"HTTP {response.status_code} from Ollama"
        try:
            payload = response.json()
        except Exception:
            text = response.text.strip()
            return text or default
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
        text = response.text.strip()
        return text or default

    def _pair_cache_path(self, query_id: int, document_id: int) -> Path:
        return self.paths.root / "pair_cache" / f"query_{query_id}" / f"doc_{document_id}.json"

    def _load_pair_cache(self, query_id: int, document_id: int) -> tuple[float, str, int, int] | None:
        path = self._pair_cache_path(query_id, document_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            return (
                float(payload.get("score", -1.0)),
                str(payload.get("response_text", "") or ""),
                int(payload.get("prompt_eval_count", 0) or 0),
                int(payload.get("eval_count", 0) or 0),
            )
        except Exception:
            return None

    def _save_pair_cache(
        self,
        *,
        query_id: int,
        document_id: int,
        prompt: str,
        score: float,
        response_text: str,
        prompt_eval_count: int,
        eval_count: int,
    ) -> None:
        path = self._pair_cache_path(query_id, document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query_id": int(query_id),
            "document_id": int(document_id),
            "model": self.reranker.model_name,
            "provider": "ollama",
            "prompt": prompt,
            "score": float(score),
            "response_text": response_text,
            "prompt_eval_count": int(prompt_eval_count),
            "eval_count": int(eval_count),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _document_fields_by_id(self) -> dict[int, dict[str, str]]:
        frame = self.retrieval.publications[["document_id", "title", "abstract"]].fillna("")
        return {
            int(row.document_id): {
                "title": str(row.title),
                "abstract": str(row.abstract),
            }
            for row in frame.itertuples(index=False)
        }


class LLMPairwiseTopKReranker(LLMPointwiseTopKReranker):
    def __init__(
        self,
        console: Any,
        store: ArtifactStore,
        retrieval: RetrievalManager,
        paths: MultiStepPaths,
        reranker: LLMPairwiseRerankerSpec,
        top_k: int,
    ) -> None:
        super().__init__(console, store, retrieval, paths, reranker, top_k)
        self.reranker = reranker

    def score_test_split(
        self,
        retriever_name: str,
        language: str,
        selected_query_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        cache_path = self.paths.test_score_path()
        if cache_path.exists() and selected_query_ids is None:
            self.console.log("Cached reranker scores found; recomputing per-query timings for this run")
        split_key = f"{language}_test"
        resolved_retriever = self.retrieval.resolve_retriever_name(retriever_name)
        ranking_frame = self.retrieval.ensure_rankings(split_key, resolved_retriever)
        top_frame = ranking_frame.loc[
            ranking_frame["rank"] <= self.top_k,
            ["query_id", "rank", "document_id", "score"],
        ].copy()
        if selected_query_ids is not None:
            query_filter = {int(query_id) for query_id in selected_query_ids}
            top_frame = top_frame.loc[top_frame["query_id"].astype(int).isin(query_filter)].copy()
        top_frame = top_frame.sort_values(["query_id", "rank"]).reset_index(drop=True)
        if top_frame.empty:
            output = top_frame.rename(columns={"score": "retrieval_score"}).copy()
            output["reranker_score"] = np.asarray([], dtype=np.float32)
            self.store.write_parquet(cache_path, output)
            return output
        total_comparisons = sum(
            int(len(group) * max(len(group) - 1, 0) // 2)
            for _, group in top_frame.groupby("query_id", sort=False)
        )
        self.console.log(
            f"Scoring {total_comparisons} query-level pairwise comparisons for {split_key} "
            f"with {self.reranker.model_name}"
        )
        output = self._score_pairs(resolved_retriever, split_key, top_frame)
        self.store.write_parquet(cache_path, output)
        return output

    def _score_pairs(self, retriever_name: str, split_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        self._ensure_model_available()
        query_text_by_id = self.retrieval.queries[split_key].set_index("query_id")["text"].to_dict()
        document_fields_by_id = self._document_fields_by_id()
        scored = frame.rename(columns={"score": "retrieval_score"}).copy()
        scored["reranker_score"] = np.nan
        scored["pairwise_wins"] = 0
        scored["pairwise_losses"] = 0
        scored["pairwise_ties"] = 0
        scored["query_elapsed_seconds"] = np.nan
        scored["query_pairwise_comparisons"] = 0
        scored["query_pairwise_cache_hits"] = 0
        scored["query_pairwise_cache_misses"] = 0
        scored["query_avg_seconds_per_comparison"] = np.nan
        total_comparisons = sum(
            int(len(group) * max(len(group) - 1, 0) // 2)
            for _, group in frame.groupby("query_id", sort=False)
        )
        cache_hits = 0
        completed_comparisons = 0
        started_at = time.perf_counter()
        progress_interval = max(10, math.ceil(total_comparisons / 25)) if total_comparisons > 0 else 10
        total_queries = int(frame["query_id"].nunique()) if len(frame) else 0
        query_progress_interval = max(1, math.ceil(total_queries / 25)) if total_queries > 0 else 1
        processed_queries = 0
        for query_id, query_group in frame.groupby("query_id", sort=False):
            query_started_at = time.perf_counter()
            query_cache_hits_before = cache_hits
            query_completed_comparisons_before = completed_comparisons
            query_cache_misses = 0
            ordered = query_group.sort_values("rank").reset_index(drop=True)
            rows = list(ordered.itertuples(index=False))
            query_comparisons = int(len(rows) * max(len(rows) - 1, 0) // 2)
            stats_by_document = {
                int(row.document_id): {"wins": 0, "losses": 0, "ties": 0}
                for row in rows
            }
            query_text = str(query_text_by_id[int(query_id)])
            for left_index, left_row in enumerate(rows):
                left_document_id = int(left_row.document_id)
                left_payload = document_fields_by_id[left_document_id]
                for right_index in range(left_index + 1, len(rows)):
                    right_row = rows[right_index]
                    right_document_id = int(right_row.document_id)
                    cached = self._load_comparison_cache(int(query_id), left_document_id, right_document_id)
                    if cached is None:
                        query_cache_misses += 1
                        right_payload = document_fields_by_id[right_document_id]
                        prompt_ab = self.reranker.render_prompt(
                            query=query_text,
                            candidate_a=left_payload,
                            candidate_b=right_payload,
                        )
                        prompt_ba = self.reranker.render_prompt(
                            query=query_text,
                            candidate_a=right_payload,
                            candidate_b=left_payload,
                        )
                        try:
                            forward_payload = self._call_ollama(prompt_ab)
                            reverse_payload = self._call_ollama(prompt_ba)
                        except Exception as exc:
                            total_elapsed = time.perf_counter() - started_at
                            raise RuntimeError(
                                f"LLM pairwise reranking failed for query_id={int(query_id)}, "
                                f"document_pair=({left_document_id}, {right_document_id}), "
                                f"completed_comparisons={completed_comparisons}/{total_comparisons}, "
                                f"elapsed_seconds={total_elapsed:.2f}: {exc}"
                            ) from exc
                        choice_ab = _parse_llm_pairwise_choice(_ollama_response_text(forward_payload))
                        choice_ba = _parse_llm_pairwise_choice(_ollama_response_text(reverse_payload))
                        winner_document_id: int | None = None
                        final_preference = "tie"
                        if choice_ab == "A" and choice_ba == "B":
                            winner_document_id = left_document_id
                            final_preference = "candidate_a"
                        elif choice_ab == "B" and choice_ba == "A":
                            winner_document_id = right_document_id
                            final_preference = "candidate_b"
                        cached = {
                            "query_id": int(query_id),
                            "left_document_id": left_document_id,
                            "right_document_id": right_document_id,
                            "left_rank": int(left_row.rank),
                            "right_rank": int(right_row.rank),
                            "model": self.reranker.model_name,
                            "provider": "ollama",
                            "prompt_ab": prompt_ab,
                            "prompt_ba": prompt_ba,
                            "choice_ab": choice_ab,
                            "choice_ba": choice_ba,
                            "forward_response": _ollama_response_text(forward_payload),
                            "reverse_response": _ollama_response_text(reverse_payload),
                            "forward_elapsed_seconds": float(forward_payload.get("_client_elapsed_seconds", 0.0) or 0.0),
                            "reverse_elapsed_seconds": float(reverse_payload.get("_client_elapsed_seconds", 0.0) or 0.0),
                            "forward_prompt_tokens": int(forward_payload.get("prompt_eval_count", 0) or 0),
                            "reverse_prompt_tokens": int(reverse_payload.get("prompt_eval_count", 0) or 0),
                            "forward_response_tokens": int(forward_payload.get("eval_count", 0) or 0),
                            "reverse_response_tokens": int(reverse_payload.get("eval_count", 0) or 0),
                            "final_preference": final_preference,
                            "winner_document_id": winner_document_id,
                        }
                        self._save_comparison_cache(int(query_id), left_document_id, right_document_id, cached)
                    else:
                        cache_hits += 1
                    winner_document_id = cached.get("winner_document_id")
                    if winner_document_id is None:
                        stats_by_document[left_document_id]["ties"] += 1
                        stats_by_document[right_document_id]["ties"] += 1
                    else:
                        winner_document_id = int(winner_document_id)
                        loser_document_id = right_document_id if winner_document_id == left_document_id else left_document_id
                        stats_by_document[winner_document_id]["wins"] += 1
                        stats_by_document[loser_document_id]["losses"] += 1
                    completed_comparisons += 1
                    if completed_comparisons % progress_interval == 0 or completed_comparisons == total_comparisons:
                        elapsed = time.perf_counter() - started_at
                        avg_seconds = elapsed / float(max(completed_comparisons, 1))
                        remaining = total_comparisons - completed_comparisons
                        eta_hours = (avg_seconds * float(remaining)) / 3600.0
                        self.console.log(
                            f"LLM pairwise reranking progress for {split_key}: "
                            f"{completed_comparisons}/{total_comparisons} comparisons, "
                            f"elapsed={elapsed/60.0:.2f}m, avg_seconds_per_comparison={avg_seconds:.2f}, "
                            f"eta_hours={eta_hours:.2f}"
                        )
            score_by_document = {
                document_id: float(stats["wins"]) + 0.5 * float(stats["ties"])
                for document_id, stats in stats_by_document.items()
            }
            mask = scored["query_id"] == int(query_id)
            scored.loc[mask, "reranker_score"] = scored.loc[mask, "document_id"].map(score_by_document)
            scored.loc[mask, "pairwise_wins"] = scored.loc[mask, "document_id"].map(
                {document_id: int(stats["wins"]) for document_id, stats in stats_by_document.items()}
            )
            scored.loc[mask, "pairwise_losses"] = scored.loc[mask, "document_id"].map(
                {document_id: int(stats["losses"]) for document_id, stats in stats_by_document.items()}
            )
            scored.loc[mask, "pairwise_ties"] = scored.loc[mask, "document_id"].map(
                {document_id: int(stats["ties"]) for document_id, stats in stats_by_document.items()}
            )
            query_elapsed_seconds = time.perf_counter() - query_started_at
            query_completed_comparisons = completed_comparisons - query_completed_comparisons_before
            query_cache_hits = cache_hits - query_cache_hits_before
            scored.loc[mask, "query_elapsed_seconds"] = float(query_elapsed_seconds)
            scored.loc[mask, "query_pairwise_comparisons"] = int(query_completed_comparisons)
            scored.loc[mask, "query_pairwise_cache_hits"] = int(query_cache_hits)
            scored.loc[mask, "query_pairwise_cache_misses"] = int(query_cache_misses)
            scored.loc[mask, "query_avg_seconds_per_comparison"] = (
                float(query_elapsed_seconds / float(query_completed_comparisons))
                if query_completed_comparisons > 0
                else 0.0
            )
            processed_queries += 1
            if processed_queries % query_progress_interval == 0 or processed_queries == total_queries:
                _console_log_timing(
                    self.console,
                    f"Processed LLM pairwise queries up to query_id={int(query_id)}",
                    time.perf_counter() - query_started_at,
                    details=(
                        f"processed_queries={processed_queries}/{total_queries}, "
                        f"last_query_candidates={len(rows)}, last_query_comparisons={query_comparisons}, "
                        f"last_query_cache_hits={cache_hits - query_cache_hits_before}"
                    ),
                )
        if cache_hits:
            self.console.log(f"Reused {cache_hits} cached Ollama pairwise comparisons for en_test")
        _console_log_timing(
            self.console,
            "Completed LLM pairwise scoring",
            time.perf_counter() - started_at,
            details=(
                f"split={split_key}, queries={frame['query_id'].nunique() if len(frame) else 0}, "
                f"comparisons={total_comparisons}, cache_hits={cache_hits}"
            ),
        )
        scored["reranker_score"] = scored["reranker_score"].astype(np.float32)
        scored["pairwise_wins"] = scored["pairwise_wins"].astype(int)
        scored["pairwise_losses"] = scored["pairwise_losses"].astype(int)
        scored["pairwise_ties"] = scored["pairwise_ties"].astype(int)
        scored["query_elapsed_seconds"] = scored["query_elapsed_seconds"].astype(np.float32)
        scored["query_pairwise_comparisons"] = scored["query_pairwise_comparisons"].astype(int)
        scored["query_pairwise_cache_hits"] = scored["query_pairwise_cache_hits"].astype(int)
        scored["query_pairwise_cache_misses"] = scored["query_pairwise_cache_misses"].astype(int)
        scored["query_avg_seconds_per_comparison"] = scored["query_avg_seconds_per_comparison"].astype(np.float32)
        return scored.sort_values(["query_id", "rank"]).reset_index(drop=True)

    def _comparison_cache_path(self, query_id: int, left_document_id: int, right_document_id: int) -> Path:
        return (
            self.paths.shared_cache_root()
            / "pairwise_comparisons"
            / f"query_{query_id}"
            / f"doc_{left_document_id}__doc_{right_document_id}.json"
        )

    def _load_comparison_cache(
        self,
        query_id: int,
        left_document_id: int,
        right_document_id: int,
    ) -> dict[str, Any] | None:
        path = self._comparison_cache_path(query_id, left_document_id, right_document_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_comparison_cache(
        self,
        query_id: int,
        left_document_id: int,
        right_document_id: int,
        payload: dict[str, Any],
    ) -> None:
        path = self._comparison_cache_path(query_id, left_document_id, right_document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_top_k_reranker(
    console: Any,
    store: ArtifactStore,
    retrieval: RetrievalManager,
    paths: MultiStepPaths,
    reranker: AnyRerankerSpec,
    top_k: int,
) -> TopKScoreRunner:
    if isinstance(reranker, LLMPairwiseRerankerSpec):
        return LLMPairwiseTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, LLMPointwiseRerankerSpec):
        return LLMPointwiseTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, FlagEmbeddingRerankerSpec):
        return FlagEmbeddingTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, JinaComputeScoreRerankerSpec):
        return JinaComputeScoreTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, LlamaNemotronRerankerSpec):
        return LlamaNemotronTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, TransformersListwiseRerankerSpec):
        return TransformersListwiseTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, ClaudeListwiseRerankerSpec):
        return ClaudeListwiseTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, Qwen3RerankerSpec):
        return Qwen3RerankerTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, OllamaListwiseTop5RerankerSpec):
        return OllamaListwiseTop5TopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, OllamaListwiseClaimRerankerSpec):
        return OllamaListwiseClaimTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, OllamaClaimJudgmentRerankerSpec):
        return OllamaClaimJudgmentTopKReranker(console, store, retrieval, paths, reranker, top_k)
    if isinstance(reranker, OllamaClaimDecompositionRerankerSpec):
        return OllamaClaimDecompositionTopKReranker(console, store, retrieval, paths, reranker, top_k)
    return CrossEncoderTopKReranker(console, store, retrieval, paths, reranker, top_k)


class BM25MultiStepRetrievalExperiment:
    def __init__(
        self,
        store: ArtifactStore,
        console: Any,
        retrieval: RetrievalManager,
        features: Any,
        reranker: AnyRerankerSpec,
        language: str = "en",
        retriever_name: str = "bm25",
        top_k: int = 5,
        selection_source: str = "ordinal_and_hit_at_1",
    ) -> None:
        self.store = store
        self.console = console
        self.retrieval = retrieval
        self.features = features
        self.reranker = reranker
        self.language = language
        self.retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        self.top_k = int(top_k)
        self.metric_k = 5
        self.selection_source = str(selection_source).strip().lower()
        self.selections = self._make_selections()
        self.paths = MultiStepPaths(
            store,
            self.selections.namespace,
            language,
            self.retriever_name,
            reranker,
            self.top_k,
        )
        self.reranker_runner = build_top_k_reranker(console, store, retrieval, self.paths, reranker, self.top_k)
        self.paths.config_path().write_text(
            json.dumps(
                {
                    "language": self.language,
                    "retriever_name": self.retriever_name,
                    "reranker": self.reranker.config_dict(),
                    "top_k": self.top_k,
                    "metric_k": self.metric_k,
                    "selection_source": self.selection_source,
                    "selection_rule": self.selections.rule,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _make_selections(self):
        if self.selection_source == "ordinal_and_hit_at_1":
            return OrdinalAndHitAtOnePredictorSelections(
                self.store,
                self.console,
                self.features,
                self.retrieval,
                self.language,
                self.retriever_name,
                self.top_k,
            )
        if self.selection_source == "hit_at_1":
            return HitAtOnePredictorSelections(
                self.store,
                self.console,
                self.features,
                self.retrieval,
                self.language,
                self.retriever_name,
            )
        if self.selection_source != "ordinal":
            raise ValueError(f"Unsupported selection source: {self.selection_source}")
        return OrdinalPredictorSelections(
            self.store,
            self.console,
            self.features,
            self.retrieval,
            self.language,
            self.retriever_name,
            self.top_k,
        )

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        reranker: AnyRerankerSpec,
        language: str = "en",
        retriever_name: str = "bm25",
        top_k: int = 5,
        selection_source: str = "ordinal_and_hit_at_1",
    ) -> BM25MultiStepRetrievalExperiment:
        return cls(
            store=runtime.store,
            console=runtime.console,
            retrieval=runtime.retrieval,
            features=runtime.features,
            reranker=reranker,
            language=language,
            retriever_name=retriever_name,
            top_k=top_k,
            selection_source=selection_source,
        )

    def run(self) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        self.console.section(f"BM25 multi-step retrieval with {self.reranker.model_name}")
        selection_started_at = time.perf_counter()
        selection_frame = self.selections.test()
        _console_log_timing(
            self.console,
            "Prepared selective reranking decisions",
            time.perf_counter() - selection_started_at,
            details=(
                f"queries={len(selection_frame)}, "
                f"selected_queries={int(selection_frame['selected_for_selective_rerank'].sum()) if len(selection_frame) else 0}, "
                f"selection_source={self.selection_source}"
            ),
        )
        output_started_at = time.perf_counter()
        outcomes, rankings = self._build_test_outputs(selection_frame)
        _console_log_timing(
            self.console,
            "Built multi-step evaluation outputs",
            time.perf_counter() - output_started_at,
            details=f"outcomes={len(outcomes)}, ranking_rows={len(rankings)}",
        )
        persist_started_at = time.perf_counter()
        self.store.write_parquet(self.paths.test_outcome_path(), outcomes)
        self.store.write_parquet(self.paths.test_ranking_path(), rankings)
        _console_log_timing(
            self.console,
            "Saved multi-step parquet artifacts",
            time.perf_counter() - persist_started_at,
            details=(
                f"outcomes_path={self.paths.test_outcome_path().name}, "
                f"rankings_path={self.paths.test_ranking_path().name}"
            ),
        )
        summary_started_at = time.perf_counter()
        summary = self._summary(outcomes)
        self.store.write_json(self.paths.summary_path(), summary)
        self.paths.report_path().write_text(self._report(summary), encoding="utf-8")
        _console_log_timing(
            self.console,
            "Saved multi-step summary artifacts",
            time.perf_counter() - summary_started_at,
            details=(
                f"summary_path={self.paths.summary_path().name}, "
                f"report_path={self.paths.report_path().name}"
            ),
        )
        self.console.log(f"Saved multi-step summary to {self.paths.summary_path()}")
        clean_started_at = time.perf_counter()
        removed_count = self._clean_stale_outputs()
        _console_log_timing(
            self.console,
            "Cleaned stale multi-step outputs",
            time.perf_counter() - clean_started_at,
            details=f"removed_files={removed_count}",
        )
        _console_log_timing(
            self.console,
            "BM25 multi-step pipeline complete",
            time.perf_counter() - run_started_at,
            details=f"reranker_kind={self.reranker.method}, top_k={self.top_k}",
        )
        return summary

    def _clean_stale_outputs(self) -> int:
        removed_count = 0
        for path in self.paths.stale_paths():
            if path.exists():
                path.unlink()
                removed_count += 1
        return removed_count

    def _build_test_outputs(self, selection_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected_query_ids = {
            int(query_id)
            for query_id in selection_frame.loc[
                selection_frame["selected_for_selective_rerank"].astype(bool),
                "query_id",
            ].astype(int).tolist()
        }
        score_frame = self.reranker_runner.score_test_split(
            self.retriever_name,
            self.language,
            selected_query_ids=selected_query_ids,
        )
        ranking_frame = self.retrieval.ensure_rankings(f"{self.language}_test", self.retriever_name)
        query_frame = self.retrieval.queries[f"{self.language}_test"]
        score_top_frame = score_frame.sort_values(["query_id", "rank"]).reset_index(drop=True)
        evaluation_depth = max(self.top_k, self.metric_k)
        ranking_top_frame = ranking_frame.loc[
            ranking_frame["rank"] <= evaluation_depth,
            ["query_id", "rank", "document_id", "score"],
        ].sort_values(["query_id", "rank"]).reset_index(drop=True)
        gold_by_query = query_frame.set_index("query_id")["document_id"].astype(int).to_dict()
        selection_by_query = selection_frame.set_index("query_id")["selected_for_selective_rerank"].astype(bool).to_dict()
        outcome_rows: list[dict[str, Any]] = []
        ranking_rows: list[dict[str, Any]] = []
        score_groups = {
            int(query_id): group.copy()
            for query_id, group in score_top_frame.groupby("query_id", sort=False)
        }
        for query_id, ranking_group in ranking_top_frame.groupby("query_id", sort=False):
            gold_id = int(gold_by_query[int(query_id)])
            selected = bool(selection_by_query.get(int(query_id), False))
            score_group = score_groups.get(int(query_id))
            if score_group is None and selected:
                raise KeyError(f"Missing reranker scores for query_id={int(query_id)}")
            no_rerank_docs = ranking_group.sort_values("rank")["document_id"].astype(int).tolist()
            if score_group is None:
                reranked_head = no_rerank_docs[: self.top_k]
            else:
                reranked_head = (
                    score_group.sort_values(["reranker_score", "rank"], ascending=[False, True])["document_id"]
                    .astype(int)
                    .tolist()
                )
            untouched_tail = (
                ranking_group.loc[ranking_group["rank"] > self.top_k, "document_id"]
                .astype(int)
                .tolist()
            )
            rerank_all_docs = reranked_head + untouched_tail
            selective_docs = rerank_all_docs if selected else no_rerank_docs
            rr_no = reciprocal_rank_at_k(no_rerank_docs, gold_id, self.metric_k)
            rr_all = reciprocal_rank_at_k(rerank_all_docs, gold_id, self.metric_k) if score_group is not None else math.nan
            rr_selective = reciprocal_rank_at_k(selective_docs, gold_id, self.metric_k)
            outcome_rows.append(
                {
                    "query_id": int(query_id),
                    "document_id": gold_id,
                    "selected_for_selective_rerank": selected,
                    "rr_no_rerank": rr_no,
                    "rr_rerank_all": rr_all,
                    "rr_selective": rr_selective,
                    "query_elapsed_seconds": (
                        float(score_group["query_elapsed_seconds"].iloc[0])
                        if score_group is not None and "query_elapsed_seconds" in score_group.columns
                        else math.nan
                    ),
                    "query_pairwise_comparisons": (
                        int(score_group["query_pairwise_comparisons"].iloc[0])
                        if score_group is not None and "query_pairwise_comparisons" in score_group.columns
                        else 0
                    ),
                    "query_pairwise_cache_hits": (
                        int(score_group["query_pairwise_cache_hits"].iloc[0])
                        if score_group is not None and "query_pairwise_cache_hits" in score_group.columns
                        else 0
                    ),
                    "query_pairwise_cache_misses": (
                        int(score_group["query_pairwise_cache_misses"].iloc[0])
                        if score_group is not None and "query_pairwise_cache_misses" in score_group.columns
                        else 0
                    ),
                    "query_avg_seconds_per_comparison": (
                        float(score_group["query_avg_seconds_per_comparison"].iloc[0])
                        if score_group is not None and "query_avg_seconds_per_comparison" in score_group.columns
                        else math.nan
                    ),
                }
            )
            reranker_score_by_doc = (
                {
                    int(row.document_id): float(row.reranker_score)
                    for row in score_group.itertuples(index=False)
                }
                if score_group is not None
                else {}
            )
            query_elapsed_seconds = (
                float(score_group["query_elapsed_seconds"].iloc[0])
                if score_group is not None and "query_elapsed_seconds" in score_group.columns
                else math.nan
            )
            query_pairwise_comparisons = (
                int(score_group["query_pairwise_comparisons"].iloc[0])
                if score_group is not None and "query_pairwise_comparisons" in score_group.columns
                else 0
            )
            query_pairwise_cache_hits = (
                int(score_group["query_pairwise_cache_hits"].iloc[0])
                if score_group is not None and "query_pairwise_cache_hits" in score_group.columns
                else 0
            )
            query_pairwise_cache_misses = (
                int(score_group["query_pairwise_cache_misses"].iloc[0])
                if score_group is not None and "query_pairwise_cache_misses" in score_group.columns
                else 0
            )
            query_avg_seconds_per_comparison = (
                float(score_group["query_avg_seconds_per_comparison"].iloc[0])
                if score_group is not None and "query_avg_seconds_per_comparison" in score_group.columns
                else math.nan
            )
            retrieval_score_by_doc = {
                int(row.document_id): float(row.score)
                for row in ranking_group.itertuples(index=False)
            }
            for condition, documents, applied_rerank in (
                ("no_rerank", no_rerank_docs, False),
                ("rerank_all", rerank_all_docs, score_group is not None),
                ("selective_rerank", selective_docs, selected),
            ):
                for rank, document_id in enumerate(documents, start=1):
                    ranking_rows.append(
                        {
                            "query_id": int(query_id),
                            "condition": condition,
                            "rank": rank,
                            "document_id": int(document_id),
                            "retrieval_score": retrieval_score_by_doc[int(document_id)],
                            "reranker_score": reranker_score_by_doc.get(int(document_id), np.nan),
                            "selected_for_selective_rerank": selected,
                            "applied_rerank": applied_rerank,
                            "query_elapsed_seconds": query_elapsed_seconds,
                            "query_pairwise_comparisons": query_pairwise_comparisons,
                            "query_pairwise_cache_hits": query_pairwise_cache_hits,
                            "query_pairwise_cache_misses": query_pairwise_cache_misses,
                            "query_avg_seconds_per_comparison": query_avg_seconds_per_comparison,
                        }
                    )
        outcomes = pd.DataFrame(outcome_rows).sort_values("query_id").reset_index(drop=True)
        rankings = pd.DataFrame(ranking_rows).sort_values(["condition", "query_id", "rank"]).reset_index(drop=True)
        return outcomes, rankings

    def _summary(self, outcomes: pd.DataFrame) -> dict[str, Any]:
        rerank_all_mask = outcomes["rr_rerank_all"].notna() if len(outcomes) else pd.Series(dtype=bool)
        rerank_all_queries = int(rerank_all_mask.sum()) if len(outcomes) else 0
        selective_queries = int(outcomes["selected_for_selective_rerank"].sum())
        return {
            "language": self.language,
            "retriever_name": self.retriever_name,
            "reranker": self.reranker.summary_dict(),
            "top_k": self.top_k,
            "metric_k": self.metric_k,
            "selection": {
                "source": self.selections.source,
                "rule": self.selections.rule,
                "selected_queries": selective_queries,
            },
            "test": {
                "no_rerank": ConditionMetrics(
                    mrr_at_5=float(outcomes["rr_no_rerank"].mean()) if len(outcomes) else 0.0,
                    reranked_queries=0,
                    reranked_pairs=0,
                ).to_dict(),
                f"rerank_all_top{self.top_k}": ConditionMetrics(
                    mrr_at_5=float(outcomes["rr_rerank_all"].mean()) if rerank_all_queries else 0.0,
                    reranked_queries=rerank_all_queries,
                    reranked_pairs=rerank_all_queries * self.top_k,
                ).to_dict(),
                "selective_rerank": ConditionMetrics(
                    mrr_at_5=float(outcomes["rr_selective"].mean()) if len(outcomes) else 0.0,
                    reranked_queries=selective_queries,
                    reranked_pairs=selective_queries * self.top_k,
                ).to_dict(),
            },
        }

    def _report(self, summary: dict[str, Any]) -> str:
        test = summary["test"]
        selection = summary["selection"]
        rerank_all_key = f"rerank_all_top{summary['top_k']}"
        lines = [
            f"# Multi-step retrieval for {summary['retriever_name']}",
            "",
            *self.reranker.report_lines(),
            f"- Top-k reranked: {summary['top_k']}",
            f"- Selection source: {selection['source']}",
            f"- Selection rule: {selection['rule']}",
            f"- Selected queries: {selection['selected_queries']}",
            "",
            "## Test",
            "",
            f"- No reranking MRR@{summary['metric_k']}: {test['no_rerank']['mrr_at_5']:.6f}",
            f"- Rerank scored top {summary['top_k']} MRR@{summary['metric_k']}: {test[rerank_all_key]['mrr_at_5']:.6f}",
            f"- Selective rerank MRR@{summary['metric_k']}: {test['selective_rerank']['mrr_at_5']:.6f}",
            f"- Selective reranked queries: {test['selective_rerank']['reranked_queries']}",
        ]
        return "\n".join(lines)


class RerankerPolicyComparisonExperiment:
    def __init__(
        self,
        store: ArtifactStore,
        console: Any,
        retrieval: RetrievalManager,
        features: Any,
        reranker: AnyRerankerSpec,
        language: str = "en",
        retriever_name: str = "bm25",
        top_k: int = 10,
        selected_query_ids: set[int] | None = None,
        evaluation_name: str | None = None,
    ) -> None:
        self.store = store
        self.console = console
        self.retrieval = retrieval
        self.features = features
        self.reranker = reranker
        self.language = language
        self.retriever_name = self.retrieval.resolve_retriever_name(retriever_name)
        self.top_k = int(top_k)
        self.metric_k = 5
        self.selected_query_ids = (
            None if selected_query_ids is None else {int(query_id) for query_id in selected_query_ids}
        )
        if self.selected_query_ids is None:
            self.evaluation_name = "full_test"
            selection_namespace = "comparison_lightweight_selectors"
        else:
            query_id_text = ",".join(str(query_id) for query_id in sorted(self.selected_query_ids))
            query_id_digest = hashlib.sha1(query_id_text.encode("utf-8")).hexdigest()[:10]
            normalized_name = _slugify(
                evaluation_name or f"selected_{len(self.selected_query_ids)}q_{query_id_digest}"
            )
            self.evaluation_name = normalized_name
            selection_namespace = f"comparison_lightweight_selectors_{normalized_name}"
        self.paths = MultiStepPaths(
            store,
            selection_namespace,
            language,
            self.retriever_name,
            reranker,
            self.top_k,
        )
        self.reranker_runner = build_top_k_reranker(console, store, retrieval, self.paths, reranker, self.top_k)
        combinations = _policy_comparison_combinations()
        self.paths.config_path().write_text(
            json.dumps(
                {
                    "language": self.language,
                    "retriever_name": self.retriever_name,
                    "reranker": self.reranker.config_dict(),
                    "top_k": self.top_k,
                    "metric_k": self.metric_k,
                    "predictor_tier": "lightweight",
                    "selectors": list(SELECTOR_NAMES),
                    "evaluation": {
                        "scope": "full_test" if self.selected_query_ids is None else "selected_query_ids",
                        "name": self.evaluation_name,
                        "selected_query_count": (
                            None if self.selected_query_ids is None else len(self.selected_query_ids)
                        ),
                        "selected_query_ids": (
                            None
                            if self.selected_query_ids is None
                            else sorted(int(query_id) for query_id in self.selected_query_ids)
                        ),
                    },
                    "policies": {
                        "no_rerank": f"Use the original {self.retriever_name} ranking.",
                        f"rerank_all_top{self.top_k}": (
                            f"Rerank the full {self.retriever_name} top {self.top_k} for every query."
                        ),
                        **{
                            _condition_name(combo): _condition_rule_description(combo)
                            for combo in combinations
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        reranker: AnyRerankerSpec,
        language: str = "en",
        retriever_name: str = "bm25",
        top_k: int = 10,
        selected_query_ids: set[int] | None = None,
        evaluation_name: str | None = None,
    ) -> RerankerPolicyComparisonExperiment:
        return cls(
            store=runtime.store,
            console=runtime.console,
            retrieval=runtime.retrieval,
            features=runtime.features,
            reranker=reranker,
            language=language,
            retriever_name=retriever_name,
            top_k=top_k,
            selected_query_ids=selected_query_ids,
            evaluation_name=evaluation_name,
        )

    def run(self) -> dict[str, Any]:
        run_started_at = time.perf_counter()
        self.console.section(f"{self.retriever_name} reranker policy comparison with {self.reranker.model_name}")
        build_started_at = time.perf_counter()
        outcomes, rankings = self._build_test_outputs()
        _console_log_timing(
            self.console,
            "Built policy comparison outputs",
            time.perf_counter() - build_started_at,
            details=f"outcomes={len(outcomes)}, ranking_rows={len(rankings)}",
        )
        persist_started_at = time.perf_counter()
        self.store.write_parquet(self.paths.test_outcome_path(), outcomes)
        self.store.write_parquet(self.paths.test_ranking_path(), rankings)
        _console_log_timing(
            self.console,
            "Saved policy comparison parquet artifacts",
            time.perf_counter() - persist_started_at,
            details=(
                f"outcomes_path={self.paths.test_outcome_path().name}, "
                f"rankings_path={self.paths.test_ranking_path().name}"
            ),
        )
        summary_started_at = time.perf_counter()
        summary = self._summary(outcomes)
        self.store.write_json(self.paths.summary_path(), summary)
        self.paths.report_path().write_text(self._report(summary), encoding="utf-8")
        _console_log_timing(
            self.console,
            "Saved policy comparison summary artifacts",
            time.perf_counter() - summary_started_at,
            details=(
                f"summary_path={self.paths.summary_path().name}, "
                f"report_path={self.paths.report_path().name}"
            ),
        )
        self.console.log(f"Saved policy comparison summary to {self.paths.summary_path()}")
        clean_started_at = time.perf_counter()
        removed_count = self._clean_stale_outputs()
        _console_log_timing(
            self.console,
            "Cleaned stale policy comparison outputs",
            time.perf_counter() - clean_started_at,
            details=f"removed_files={removed_count}",
        )
        _console_log_timing(
            self.console,
            f"{self.retriever_name} policy comparison complete",
            time.perf_counter() - run_started_at,
            details=f"reranker_kind={self.reranker.method}, top_k={self.top_k}",
        )
        return summary

    def _clean_stale_outputs(self) -> int:
        removed_count = 0
        for path in self.paths.stale_paths():
            if path.exists():
                path.unlink()
                removed_count += 1
        return removed_count

    def _load_selector_frame(self) -> pd.DataFrame:
        lightweight_hit_at_1_features = LightweightHitAtOnePredictorFeatureBuilder(
            self.features.config,
            self.store,
            self.console,
            self.features.bundle,
            self.retrieval,
            self.features.clusters,
            self.features.text_services,
            manifest_path=LIGHTWEIGHT_HIT_AT_1_MANIFEST_PATH,
            consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
        )
        hit_at_1_frame = load_or_create_lightweight_hit_at_1_test_predictions(
            config=self.features.config,
            store=self.store,
            console=self.console,
            features=lightweight_hit_at_1_features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
            predictor_name=PREDICTOR_TIER_NAMES["lightweight"],
        )[["query_id", "predicted_hit_at_1"]].copy()
        if hit_at_1_frame.duplicated("query_id").any():
            hit_at_1_frame = hit_at_1_frame.drop_duplicates("query_id")

        ordinal_frame = load_or_create_lightweight_calibrated_ordinal_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
        )[["query_id", "selected_size", "predicted_failure"]].rename(
            columns={"selected_size": "ordinal_selected_size", "predicted_failure": "ordinal_predicted_failure"}
        )

        qpp_frame = load_or_create_lightweight_tuned_qpp_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
        )[["query_id", "selected_size", "predicted_failure"]].rename(
            columns={"selected_size": "qpp_selected_size", "predicted_failure": "qpp_predicted_failure"}
        )

        if self.top_k not in IN_TOP_K_SUPPORTED_DEPTHS:
            raise ValueError(
                f"No in-top-k predictor is trained for top_k={self.top_k}. "
                f"Supported depths are {IN_TOP_K_SUPPORTED_DEPTHS}; run "
                "scripts/run_in_top_max_depth_sweep_predictor.py for this depth first."
            )
        in_top_k_frame = load_or_create_lightweight_in_top_k_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
            target_depth=self.top_k,
        )[["query_id", "predicted_in_top_k"]].copy()
        if in_top_k_frame.duplicated("query_id").any():
            in_top_k_frame = in_top_k_frame.drop_duplicates("query_id")
        in_top_k_depth_frame = self._load_in_top_k_depth_frame()

        max_depth_frame = load_or_create_lightweight_max_depth_test_predictions(
            store=self.store,
            console=self.console,
            features=self.features,
            retrieval=self.retrieval,
            language=self.language,
            retriever_name=self.retriever_name,
        )[["query_id", "predicted_in_top_max_depth"]].copy()
        if max_depth_frame.duplicated("query_id").any():
            max_depth_frame = max_depth_frame.drop_duplicates("query_id")

        merged = (
            hit_at_1_frame.merge(ordinal_frame, on="query_id", how="inner")
            .merge(qpp_frame, on="query_id", how="inner")
            .merge(in_top_k_frame, on="query_id", how="inner")
            .merge(in_top_k_depth_frame, on="query_id", how="inner")
            .merge(max_depth_frame, on="query_id", how="inner")
        )
        merged["hit1_gate_pass"] = ~merged["predicted_hit_at_1"].astype(bool)
        merged["ordinal_gate_pass"] = ~merged["ordinal_predicted_failure"].astype(bool)
        merged["qpp_gate_pass"] = ~merged["qpp_predicted_failure"].astype(bool)
        merged["in_top_k_gate_pass"] = merged["predicted_in_top_k"].astype(bool)
        merged["in_top_k_depth_gate_pass"] = merged["in_top_k_depth_selected_size"].notna()
        merged["max_depth_gate_pass"] = merged["predicted_in_top_max_depth"].astype(bool)
        return merged

    def _load_in_top_k_depth_frame(self) -> pd.DataFrame:
        depths = [depth for depth in IN_TOP_K_SUPPORTED_DEPTHS if depth <= self.top_k]
        self.console.log(
            f"Loading lightweight in-top-k predictions for depths {depths} to derive the per-query "
            "in-top-k depth (smallest depth predicted to contain the gold document)"
        )
        depth_frame: pd.DataFrame | None = None
        for depth in depths:
            frame = load_or_create_lightweight_in_top_k_test_predictions(
                store=self.store,
                console=self.console,
                features=self.features,
                retrieval=self.retrieval,
                language=self.language,
                retriever_name=self.retriever_name,
                target_depth=depth,
            )[["query_id", "predicted_in_top_k"]].drop_duplicates("query_id")
            frame = frame.rename(columns={"predicted_in_top_k": f"predicted_in_top_{depth}"})
            depth_frame = frame if depth_frame is None else depth_frame.merge(frame, on="query_id", how="inner")
        predicted = np.column_stack(
            [depth_frame[f"predicted_in_top_{depth}"].astype(bool).to_numpy() for depth in depths]
        )
        first_positive = np.argmax(predicted, axis=1)
        any_positive = predicted.any(axis=1)
        depth_frame["in_top_k_depth_selected_size"] = np.where(
            any_positive, np.asarray(depths, dtype=np.float64)[first_positive], np.nan
        )
        distribution = depth_frame["in_top_k_depth_selected_size"].value_counts(dropna=False).sort_index()
        self.console.log(
            "In-top-k depth distribution: "
            + ", ".join(
                f"{'none' if pd.isna(size) else int(size)}={int(count)}" for size, count in distribution.items()
            )
        )
        return depth_frame[["query_id", "in_top_k_depth_selected_size"]]

    def _build_test_outputs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        selected_query_ids = getattr(self, "selected_query_ids", None)
        query_filter = None if selected_query_ids is None else {int(query_id) for query_id in selected_query_ids}
        score_frame = self.reranker_runner.score_test_split(
            self.retriever_name,
            self.language,
            selected_query_ids=query_filter,
        )
        ranking_frame = self.retrieval.ensure_rankings(f"{self.language}_test", self.retriever_name)
        query_frame = self.retrieval.queries[f"{self.language}_test"]
        selector_frame = self._load_selector_frame()
        if query_filter is not None:
            available_query_ids = {
                int(query_id) for query_id in query_frame["query_id"].astype(int).tolist()
            }
            missing_query_ids = sorted(query_filter - available_query_ids)
            if missing_query_ids:
                raise KeyError(
                    "Selected query IDs are missing from the evaluation split: "
                    f"{missing_query_ids[:10]}"
                )
            ranking_frame = ranking_frame.loc[ranking_frame["query_id"].astype(int).isin(query_filter)].copy()
            query_frame = query_frame.loc[query_frame["query_id"].astype(int).isin(query_filter)].copy()
            selector_frame = selector_frame.loc[selector_frame["query_id"].astype(int).isin(query_filter)].copy()
        selector_by_query = {
            int(row.query_id): {
                "gate_pass": {
                    "hit1": bool(row.hit1_gate_pass),
                    "max_depth": bool(row.max_depth_gate_pass),
                    "ordinal": bool(row.ordinal_gate_pass),
                    "qpp": bool(row.qpp_gate_pass),
                    "in_top_k": bool(row.in_top_k_gate_pass),
                    "in_top_k_depth": bool(row.in_top_k_depth_gate_pass),
                },
                "depth": {
                    "hit1": None,
                    "max_depth": None,
                    "ordinal": None if pd.isna(row.ordinal_selected_size) else int(float(row.ordinal_selected_size)),
                    "qpp": None if pd.isna(row.qpp_selected_size) else int(float(row.qpp_selected_size)),
                    "in_top_k": None,
                    "in_top_k_depth": (
                        None
                        if pd.isna(row.in_top_k_depth_selected_size)
                        else int(float(row.in_top_k_depth_selected_size))
                    ),
                },
            }
            for row in selector_frame.itertuples(index=False)
        }
        score_top_frame = score_frame.sort_values(["query_id", "rank"]).reset_index(drop=True)
        evaluation_depth = max(self.top_k, self.metric_k)
        ranking_top_frame = ranking_frame.loc[
            ranking_frame["rank"] <= evaluation_depth,
            ["query_id", "rank", "document_id", "score"],
        ].sort_values(["query_id", "rank"]).reset_index(drop=True)
        gold_by_query = query_frame.set_index("query_id")["document_id"].astype(int).to_dict()
        combinations = _policy_comparison_combinations()
        outcome_rows: list[dict[str, Any]] = []
        ranking_rows: list[dict[str, Any]] = []
        score_groups = {
            int(query_id): group.copy()
            for query_id, group in score_top_frame.groupby("query_id", sort=False)
        }
        for query_id, ranking_group in ranking_top_frame.groupby("query_id", sort=False):
            gold_id = int(gold_by_query[int(query_id)])
            score_group = score_groups.get(int(query_id))
            if score_group is None:
                raise KeyError(f"Missing reranker scores for query_id={int(query_id)}")
            selector = selector_by_query.get(int(query_id))
            if selector is None:
                raise KeyError(f"Missing selector predictions for query_id={int(query_id)}")
            no_rerank_docs = ranking_group.sort_values("rank")["document_id"].astype(int).tolist()
            rerank_all_docs = self._rerank_prefix(ranking_group, score_group, self.top_k)
            reranker_score_by_doc = {
                int(row.document_id): float(row.reranker_score)
                for row in score_group.itertuples(index=False)
            }
            retrieval_score_by_doc = {
                int(row.document_id): float(row.score)
                for row in ranking_group.itertuples(index=False)
            }
            row_record: dict[str, Any] = {
                "query_id": int(query_id),
                "document_id": gold_id,
                "hit1_gate_pass": selector["gate_pass"]["hit1"],
                "max_depth_gate_pass": selector["gate_pass"]["max_depth"],
                "ordinal_gate_pass": selector["gate_pass"]["ordinal"],
                "ordinal_selected_size": selector["depth"]["ordinal"],
                "qpp_gate_pass": selector["gate_pass"]["qpp"],
                "qpp_selected_size": selector["depth"]["qpp"],
                "in_top_k_gate_pass": selector["gate_pass"]["in_top_k"],
                "in_top_k_depth_gate_pass": selector["gate_pass"]["in_top_k_depth"],
                "in_top_k_depth_selected_size": selector["depth"]["in_top_k_depth"],
                "rr_no_rerank": reciprocal_rank_at_k(no_rerank_docs, gold_id, self.metric_k),
                "rr_rerank_all": reciprocal_rank_at_k(rerank_all_docs, gold_id, self.metric_k),
            }
            for condition, documents, rerank_depth in (
                ("no_rerank", no_rerank_docs, 0),
                (f"rerank_all_top{self.top_k}", rerank_all_docs, self.top_k),
            ):
                for rank, document_id in enumerate(documents, start=1):
                    ranking_rows.append(
                        {
                            "query_id": int(query_id),
                            "condition": condition,
                            "rank": rank,
                            "document_id": int(document_id),
                            "retrieval_score": retrieval_score_by_doc[int(document_id)],
                            "reranker_score": reranker_score_by_doc.get(int(document_id), np.nan),
                            "rerank_depth": int(rerank_depth),
                            "applied_rerank": bool(rerank_depth > 0),
                        }
                    )
            for combo in combinations:
                name = _condition_name(combo)
                selected, depth = _combined_selector_depth(
                    combo, selector["gate_pass"], selector["depth"], self.top_k
                )
                documents = self._rerank_prefix(ranking_group, score_group, depth) if selected else no_rerank_docs
                row_record[f"rr_{name}"] = reciprocal_rank_at_k(documents, gold_id, self.metric_k)
                row_record[f"depth_{name}"] = depth if selected else 0
                for rank, document_id in enumerate(documents, start=1):
                    ranking_rows.append(
                        {
                            "query_id": int(query_id),
                            "condition": name,
                            "rank": rank,
                            "document_id": int(document_id),
                            "retrieval_score": retrieval_score_by_doc[int(document_id)],
                            "reranker_score": reranker_score_by_doc.get(int(document_id), np.nan),
                            "rerank_depth": int(depth if selected else 0),
                            "applied_rerank": bool(selected and depth > 0),
                        }
                    )
            outcome_rows.append(row_record)
        outcomes = pd.DataFrame(outcome_rows).sort_values("query_id").reset_index(drop=True)
        rankings = pd.DataFrame(ranking_rows).sort_values(["condition", "query_id", "rank"]).reset_index(drop=True)
        return outcomes, rankings

    def _rerank_prefix(
        self,
        ranking_group: pd.DataFrame,
        score_group: pd.DataFrame,
        rerank_depth: int,
    ) -> list[int]:
        base_docs = ranking_group.sort_values("rank")["document_id"].astype(int).tolist()
        depth = max(0, min(int(rerank_depth), self.top_k, len(base_docs)))
        if depth == 0:
            return base_docs
        reranked_head = (
            score_group.loc[score_group["rank"] <= depth]
            .sort_values(["reranker_score", "rank"], ascending=[False, True])["document_id"]
            .astype(int)
            .tolist()
        )
        tail_docs = base_docs[depth:]
        return reranked_head + tail_docs

    def _summary(self, outcomes: pd.DataFrame) -> dict[str, Any]:
        combinations = _policy_comparison_combinations()
        rerank_all_key = f"rerank_all_top{self.top_k}"
        policy_metrics: dict[str, Any] = {
            "no_rerank": ConditionMetrics(
                mrr_at_5=float(outcomes["rr_no_rerank"].mean()) if len(outcomes) else 0.0,
                reranked_queries=0,
                reranked_pairs=0,
            ).to_dict(),
            rerank_all_key: ConditionMetrics(
                mrr_at_5=float(outcomes["rr_rerank_all"].mean()) if len(outcomes) else 0.0,
                reranked_queries=int(len(outcomes)),
                reranked_pairs=int(len(outcomes)) * self.top_k,
            ).to_dict(),
        }
        policies: dict[str, Any] = {}
        for combo in combinations:
            name = _condition_name(combo)
            depth_column = f"depth_{name}"
            policy_metrics[name] = ConditionMetrics(
                mrr_at_5=float(outcomes[f"rr_{name}"].mean()) if len(outcomes) else 0.0,
                reranked_queries=int((outcomes[depth_column] > 0).sum()) if len(outcomes) else 0,
                reranked_pairs=int(outcomes[depth_column].sum()) if len(outcomes) else 0,
            ).to_dict()
            policies[name] = {
                "selectors": list(combo),
                "rule": _condition_rule_description(combo),
            }
        return {
            "language": self.language,
            "retriever_name": self.retriever_name,
            "reranker": self.reranker.summary_dict(),
            "top_k": self.top_k,
            "metric_k": self.metric_k,
            "evaluation": {
                "scope": "full_test" if getattr(self, "selected_query_ids", None) is None else "selected_query_ids",
                "name": getattr(self, "evaluation_name", "full_test"),
                "evaluated_queries": int(len(outcomes)),
            },
            "policies": policies,
            "test": policy_metrics,
        }

    def _report(self, summary: dict[str, Any]) -> str:
        test = summary["test"]
        rerank_all_key = f"rerank_all_top{summary['top_k']}"
        evaluation = summary.get("evaluation", {})
        evaluation_scope = str(evaluation.get("scope", "full_test"))
        evaluation_name = str(evaluation.get("name", "full_test"))
        evaluated_queries = int(evaluation.get("evaluated_queries", 0))
        lines = [
            f"# {self.retriever_name} reranker policy comparison",
            "",
            *self.reranker.report_lines(),
            f"- Top-k reranked: {summary['top_k']}",
            f"- Evaluation scope: {evaluation_scope}",
            f"- Evaluation name: {evaluation_name}",
            f"- Evaluated queries: {evaluated_queries}",
            "",
            "## Test",
            "",
            f"- No reranking MRR@{summary['metric_k']}: {test['no_rerank']['mrr_at_5']:.6f}",
            (
                f"- Rerank all top {summary['top_k']} MRR@{summary['metric_k']}: "
                f"{test[rerank_all_key]['mrr_at_5']:.6f} "
                f"(queries={test[rerank_all_key]['reranked_queries']}, "
                f"pairs={test[rerank_all_key]['reranked_pairs']})"
            ),
        ]
        for combo in _policy_comparison_combinations():
            name = _condition_name(combo)
            label = " + ".join(SELECTOR_LABELS[selector] for selector in combo)
            metrics = test[name]
            lines.append(
                f"- {label} MRR@{summary['metric_k']}: {metrics['mrr_at_5']:.6f} "
                f"(queries={metrics['reranked_queries']}, pairs={metrics['reranked_pairs']})"
            )
        return "\n".join(lines)


def default_reranker_spec(model_name: str) -> CrossEncoderRerankerSpec:
    normalized = _canonical_cross_encoder_name(model_name)
    if normalized in {"cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2"}:
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=512, batch_size=16)
    if normalized == "jinaai/jina-reranker-v2-base-multilingual":
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=1024, batch_size=8)
    if normalized == "cross-encoder/ettin-reranker-17m-v1":
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=512, batch_size=16)
    if normalized == "cross-encoder/ettin-reranker-32m-v1":
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=512, batch_size=8)
    if normalized == "cross-encoder/ettin-reranker-150m-v1":
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=512, batch_size=8)
    if normalized == "Qwen/Qwen3-Reranker-8B":
        return CrossEncoderRerankerSpec(model_name=normalized, max_length=1024, batch_size=8)
    return CrossEncoderRerankerSpec(model_name=normalized, max_length=512, batch_size=8)


def default_flagembedding_reranker_spec(
    model_name: str,
    quantize: bool = False,
) -> FlagEmbeddingRerankerSpec:
    normalized = model_name.strip()
    if normalized == "BAAI/bge-reranker-v2-m3":
        return FlagEmbeddingRerankerSpec(
            model_name=normalized,
            max_length=1024,
            batch_size=8,
            use_fp16=False,
            normalize=False,
            quantize_dynamic_int8=quantize,
        )
    return FlagEmbeddingRerankerSpec(
        model_name=normalized,
        max_length=512,
        batch_size=8,
        use_fp16=False,
        normalize=False,
        quantize_dynamic_int8=quantize,
    )


def default_jina_compute_score_reranker_spec(
    model_name: str,
    quantize: bool = False,
) -> JinaComputeScoreRerankerSpec:
    normalized = model_name.strip()
    if normalized == "jinaai/jina-reranker-v2-base-multilingual":
        return JinaComputeScoreRerankerSpec(
            model_name=normalized,
            max_length=1024,
            batch_size=8,
            trust_remote_code=True,
            use_flash_attn=False,
            quantize_dynamic_int8=quantize,
        )
    return JinaComputeScoreRerankerSpec(
        model_name=normalized,
        max_length=1024,
        batch_size=8,
        trust_remote_code=True,
        use_flash_attn=False,
        quantize_dynamic_int8=quantize,
    )


def canonical_multistage_reranker_specs() -> dict[str, AnyRerankerSpec]:
    return {
        "jina": replace(
            default_jina_compute_score_reranker_spec("jinaai/jina-reranker-v2-base-multilingual", quantize=True),
            candidate_fields="title_abstract",
        ),
        "minilm": default_reranker_spec("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "bge": replace(
            default_flagembedding_reranker_spec("BAAI/bge-reranker-v2-m3", quantize=True),
            candidate_fields="title_abstract",
        ),
    }


def default_llama_nemotron_reranker_spec(
    model_name: str,
    quantize: bool = False,
) -> LlamaNemotronRerankerSpec:
    normalized = model_name.strip()
    return LlamaNemotronRerankerSpec(
        model_name=normalized,
        max_length=1024,
        batch_size=8,
        trust_remote_code=True,
        quantize_dynamic_int8=quantize,
    )


def default_claude_listwise_reranker_spec(
    model_name: str = "claude-haiku-4-5",
) -> ClaudeListwiseRerankerSpec:
    normalized = model_name.strip()
    return ClaudeListwiseRerankerSpec(
        model_name=normalized,
        window_size=20,
        window_step=10,
        max_output_tokens=512,
        temperature=0.0,
        timeout_seconds=60,
        max_retries=3,
        max_concurrency=5,
        candidate_fields="title_abstract",
    )


def default_qwen3_reranker_spec(model_name: str = "Qwen/Qwen3-Reranker-0.6B") -> Qwen3RerankerSpec:
    return Qwen3RerankerSpec(
        model_name=model_name.strip(),
        instruction=QWEN3_RERANKER_TASK_INSTRUCTION,
        max_length=8192,
        batch_size=8,
        candidate_fields="title_abstract",
    )


def default_ollama_listwise_top5_reranker_spec(model_name: str = "deepseek-r1:8b") -> OllamaListwiseTop5RerankerSpec:
    normalized = model_name.strip()
    if normalized == "deepseek-r1:8b":
        return OllamaListwiseTop5RerankerSpec(
            model_name=normalized,
            tokenizer_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            ollama_url="http://localhost:11434/api/generate",
            think=True,
            num_predict=32768,
            temperature=0.6,
            top_p=0.95,
            seed=13,
            candidate_fields="title_abstract",
        )
    raise ValueError(f"No default Ollama listwise top-5 reranker settings for {normalized}")


def default_ollama_claim_decomposition_reranker_spec(
    model_name: str = "qwen3.5:4b",
    fallback_model_name: str | None = "qwen3:8b",
    abstract_mode: str = "full",
) -> OllamaClaimDecompositionRerankerSpec:
    if abstract_mode not in CLAIM_ABSTRACT_MODES:
        raise ValueError(f"Unsupported abstract_mode {abstract_mode}; expected one of {CLAIM_ABSTRACT_MODES}")
    return OllamaClaimDecompositionRerankerSpec(
        model_name=model_name.strip(),
        fallback_model_name=None if fallback_model_name is None else fallback_model_name.strip(),
        ollama_url="http://localhost:11434/api/generate",
        think=False,
        temperature=0.0,
        seed=13,
        timeout_seconds=300,
        max_retries=3,
        keep_alive=-1,
        semantic_top_k=30,
        field_weights=(("entity", 2.0), ("question", 2.0), ("direct_support", 2.0)),
        anchor_weight=2.0,
        first_stage_weight=0.35,
        semantic_weight=0.65,
        pairwise_enabled=False,
        pairwise_max_gap=0.10,
        pairwise_top_n=3,
        abstract_mode=abstract_mode,
        abstract_sentence_count=3,
        sentence_encoder="qwen3_embedding_0_6b",
        judgment_batch_size=3,
        anchor_conflicts_enabled=False,
        candidate_fields="title_abstract",
    )


def default_ollama_claim_judgment_reranker_spec(
    model_name: str = "qwen3.5:4b",
    fallback_model_name: str | None = "qwen3:8b",
) -> OllamaClaimJudgmentRerankerSpec:
    return OllamaClaimJudgmentRerankerSpec(
        model_name=model_name.strip(),
        fallback_model_name=None if fallback_model_name is None else fallback_model_name.strip(),
        ollama_url="http://localhost:11434/api/generate",
        think=False,
        temperature=0.0,
        seed=13,
        timeout_seconds=300,
        max_retries=3,
        keep_alive=-1,
        semantic_top_k=30,
        pairwise_enabled=False,
        pairwise_max_group_size=None,
        candidate_fields="title_abstract",
    )


def default_ollama_listwise_claim_reranker_spec(
    model_name: str = "qwen3:8b",
    tokenizer_name: str = "Qwen/Qwen3-8B",
    fallback_model_name: str | None = None,
) -> OllamaListwiseClaimRerankerSpec:
    return OllamaListwiseClaimRerankerSpec(
        model_name=model_name.strip(),
        fallback_model_name=None if fallback_model_name is None else fallback_model_name.strip(),
        tokenizer_name=tokenizer_name.strip(),
        ollama_url="http://localhost:11434/api/generate",
        think=False,
        temperature=0.0,
        seed=13,
        timeout_seconds=600,
        max_retries=3,
        keep_alive=-1,
        semantic_top_k=30,
        abstract_stage_top_n=10,
    )


def default_listwise_reranker_spec(model_name: str) -> TransformersListwiseRerankerSpec:
    normalized = model_name.strip()
    if normalized == "jinaai/jina-reranker-v3":
        return TransformersListwiseRerankerSpec(
            model_name=normalized,
            trust_remote_code=True,
            max_documents_per_query=64,
            implementation_version=2,
        )
    return TransformersListwiseRerankerSpec(
        model_name=normalized,
        trust_remote_code=True,
        max_documents_per_query=None,
        implementation_version=2,
    )


def default_llm_reranker_spec(model_name: str) -> LLMPairwiseRerankerSpec:
    normalized = model_name.strip()
    if normalized in {"qwen3:8b", "Qwen3:8b", "qwen3:8b-q4_K_M"}:
        return LLMPairwiseRerankerSpec(
            model_name="qwen3:8b",
            max_input_tokens=2048,
            max_new_tokens=4,
            batch_size=1,
            ollama_url="http://localhost:11434/api/generate",
            timeout_seconds=300,
            max_retries=3,
            temperature=0.0,
            disable_thinking=True,
        )
    if normalized in {"Qwen/Qwen2.5-14B-Instruct", "qwen2.5:14b-instruct"}:
        return LLMPairwiseRerankerSpec(
            model_name="qwen2.5:14b-instruct",
            max_input_tokens=2048,
            max_new_tokens=4,
            batch_size=1,
            ollama_url="http://localhost:11434/api/generate",
            timeout_seconds=120,
            max_retries=5,
            temperature=0.0,
            disable_thinking=True,
        )
    return LLMPairwiseRerankerSpec(
        model_name=normalized,
        max_input_tokens=2048,
        max_new_tokens=4,
        batch_size=1,
        ollama_url="http://localhost:11434/api/generate",
        timeout_seconds=120,
        max_retries=5,
        temperature=0.0,
        disable_thinking=True,
    )
