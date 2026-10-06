from __future__ import annotations

import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.modeling import HitAtOnePredictorRunner, hit_at_1_model_is_current
from query_difficulty.retrieval import compose_retriever_name
from query_difficulty.runtime import build_runtime

SINGLE_FIELD_SCOPES = ["title", "abstract", "venue", "authors"]


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    retriever: list[str] = typer.Option(None, "--retriever", help="Run only the named retriever. Repeat for multiple retrievers."),
    quiet: bool = typer.Option(False, help="Reduce console output."),
) -> None:
    runtime = build_runtime(config, verbose=not quiet)
    runtime.console.section("Hit@1 predictor")
    runtime.console.log(f"Loading project configuration from {config}")
    runtime.console.log(f"Running hit@1 predictor for {language}")
    runtime.console.log("Using maximum dev F0.5 for hit@1 threshold selection")
    runner = HitAtOnePredictorRunner(
        runtime.config,
        runtime.store,
        runtime.console,
        runtime.features,
        runtime.retrieval,
    )
    if retriever:
        target_retrievers = [runtime.retrieval.resolve_retriever_name(name) for name in retriever]
        summary = runner.run(language, retrievers=target_retrievers)
    else:
        base_retrievers = list(runtime.config.enabled_lexical_retrievers) + list(runtime.config.enabled_semantic_retrievers)
        field_scope_retrievers = list(runtime.config.full_scope_retrievers)
        batches: list[list[str]] = []
        for scope in SINGLE_FIELD_SCOPES:
            batches.append([compose_retriever_name(base, scope) for base in field_scope_retrievers])
        batches.append([compose_retriever_name(base, "title_abstract") for base in base_retrievers])
        batches = [
            [name for name in batch if not hit_at_1_model_is_current(runtime.store, name)]
            for batch in batches
        ]
        batches = [batch for batch in batches if batch]
        if not batches:
            runtime.console.log(
                "Nothing missing: every retriever already has a current hit@1 model at the default scope, "
                f"and {field_scope_retrievers} already have current models at the single-field scopes "
                "(title, abstract, venue, authors)."
            )
            return
        runtime.console.log(
            f"Bare run: {sum(len(batch) for batch in batches)} retriever/scope combination(s) missing a current "
            f"hit@1 model, processed single-field scopes first (title/abstract/venue/authors for "
            f"{field_scope_retrievers}), then default-scope gaps last: {batches}"
        )
        summary = {}
        for batch in batches:
            summary.update(runner.run(language, retrievers=batch))
    summary_path = runtime.store.path("predictions", "hit_at_1", language, "summary.json")
    runtime.store.write_json(summary_path, summary)
    runtime.console.log(f"Saved hit@1 summary to {summary_path}")
    runtime.console.log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
