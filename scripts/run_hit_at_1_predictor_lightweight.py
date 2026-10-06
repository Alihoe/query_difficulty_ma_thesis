# scripts/run_hit_at_1_predictor_lightweight.py
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.features_hit_at_1_lightweight import (
    CONSENSUS_MODE_BM25_AND_SELF,
    LightweightHitAtOnePredictorFeatureBuilder,
    ensure_manifest_entry,
)
from query_difficulty.modeling import _predictor_completed_retrievers
from query_difficulty.modeling_hit_at_1_lightweight import LightweightHitAtOnePredictorRunner
from query_difficulty.retrieval import compose_retriever_name
from query_difficulty.runtime import build_runtime

LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME = "lightweight/hit_at_1"
RESTRICTED_SCOPES = ["title", "abstract", "venue", "authors", "title_abstract_venue", "all_fields"]


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest: str = typer.Option(
        "config/lightweight_hit_at_1_features.json",
        help="Path to the frozen lightweight feature manifest.",
    ),
    retriever: list[str] = typer.Option(
        None,
        "--retriever",
        help="Run only the named retriever. Repeat for multiple retrievers. Defaults to all configured retrievers.",
    ),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Hit@1 predictor (lightweight)")
    runtime.console.log(f"Loading project configuration from {config}")
    runtime.console.log(f"Running lightweight hit@1 predictor for {language}")
    runtime.console.log(f"Using frozen feature manifest at {manifest}")
    runtime.console.log(
        "Consensus features are restricted to BM25 and the respective semantic retriever "
        "(instead of the full retriever battery)"
    )
    runtime.console.log("Using maximum dev F0.5 for hit@1 threshold selection")

    lightweight_features = LightweightHitAtOnePredictorFeatureBuilder(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features.bundle,
        runtime.retrieval,
        runtime.clustering,
        runtime.text_services,
        manifest_path=manifest,
        consensus_mode=CONSENSUS_MODE_BM25_AND_SELF,
    )

    runner = LightweightHitAtOnePredictorRunner(
        runtime.config,
        runtime.store,
        runtime.console,
        lightweight_features,
        runtime.retrieval,
        predictor_name=LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME,
    )

    def run_batch(batch: list[str]) -> dict[str, dict[str, float]]:
        for name in batch:
            ensure_manifest_entry(
                runtime.config, runtime.store, runtime.console, runtime.features, runtime.retrieval,
                language, name, lightweight_features,
            )
        return runner.run(language, retrievers=batch)

    if retriever:
        summary = run_batch(retriever)
    else:
        freeze_only = sorted(
            name
            for name, entry in lightweight_features._manifest.get("retrievers", {}).items()
            if entry.get("pipeline_stage", "freeze") != "rfe_pruned"
        )
        if freeze_only:
            runtime.console.log(
                f"{len(freeze_only)} manifest entr{'y is' if len(freeze_only) == 1 else 'ies are'} still "
                f"freeze-only (not correlation-pruned + RFE'd): {freeze_only}. Running "
                "scripts/promote_lightweight_hit_at_1_manifest_entries.py first to bring every entry to the "
                "same RFE-pruned status before sweeping further retrievers/scopes. This is the expensive "
                "step (RFE, ~2 min per elimination step per retriever) and runs as part of this command."
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "promote_lightweight_hit_at_1_manifest_entries.py"),
                    "--language", language,
                    "--config", config,
                    "--manifest", manifest,
                ],
                check=True,
            )
            lightweight_features.reload_manifest()
            runtime.console.log("Manifest promotion complete, resuming the lightweight sweep")
        base_retrievers = list(runtime.config.enabled_lexical_retrievers) + list(runtime.config.enabled_semantic_retrievers)
        field_scope_retrievers = list(runtime.config.full_scope_retrievers)
        batches: list[list[str]] = []
        batches.append([compose_retriever_name(base, "title_abstract") for base in base_retrievers])
        for scope in RESTRICTED_SCOPES:
            batches.append([compose_retriever_name(base, scope) for base in field_scope_retrievers])
        completed = _predictor_completed_retrievers(runtime.store, LIGHTWEIGHT_HIT_AT_1_PREDICTOR_NAME, language)
        batches = [[name for name in batch if name not in completed] for batch in batches]
        batches = [batch for batch in batches if batch]
        if not batches:
            runtime.console.log(
                "Nothing missing: every retriever already has lightweight hit@1 predictions at the "
                "default scope and restricted scopes (title, abstract, venue, authors, title_abstract_venue, "
                "all_fields)."
            )
            return
        runtime.console.log(
            f"Bare run: {sum(len(batch) for batch in batches)} retriever/scope combination(s) missing "
            f"lightweight hit@1 predictions, processed default-scope gaps first (title_abstract for "
            f"{base_retrievers}), then restricted scopes last (title/abstract/venue/authors/"
            f"title_abstract_venue/all_fields for {field_scope_retrievers}): {batches}"
        )
        summary = {}
        for batch in batches:
            summary.update(run_batch(batch))
    summary_path = runtime.store.path("predictions", "lightweight", "hit_at_1", language, "summary.json")
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved lightweight hit@1 summary to {summary_path}")
    runtime.console.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
