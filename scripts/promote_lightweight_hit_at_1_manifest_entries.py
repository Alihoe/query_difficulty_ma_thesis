from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_MANIFEST_PATH = ROOT / "config" / "lightweight_hit_at_1_features.json"
CORRELATION_ROUND_1_THRESHOLD = 0.80
CORRELATION_ROUND_2_THRESHOLD = 0.70
SCRATCH_DIR = ROOT / "artifacts" / "predictions" / "lightweight_manifest_promotion"


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _freeze_only_retrievers(manifest: dict) -> list[str]:
    names = []
    for name, entry in manifest.get("retrievers", {}).items():
        if entry.get("pipeline_stage", "freeze") == "rfe_pruned":
            continue
        names.append(name)
    return sorted(names)


def _run_step(args: list[str]) -> None:
    print(f"$ {' '.join(args)}")
    subprocess.run(args, check=True)


def main(
    language: str = typer.Option("en", help="Language code such as en, de, or fr."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    manifest: str = typer.Option(str(DEFAULT_MANIFEST_PATH), help="Frozen feature manifest to promote entries in, in place."),
    retriever: list[str] = typer.Option(
        None,
        "--retriever",
        help="Promote only the named retriever(s). Repeat for multiple. Defaults to every freeze-only manifest entry.",
    ),
    n_bootstrap: int = typer.Option(
        2000,
        help="Bootstrap resamples for the RFE standard-error estimate (matches rfe_prune_lightweight_hit_at_1_features.py's default).",
    ),
) -> None:
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest_data = _load_manifest(manifest_path)

    targets = list(retriever) if retriever else _freeze_only_retrievers(manifest_data)
    if not targets:
        print(f"Nothing to promote: every entry in {manifest_path} is already RFE-pruned or hardcoded.")
        return

    print(
        f"Promoting {len(targets)} manifest entr{'y' if len(targets) == 1 else 'ies'} from freeze-only to "
        f"RFE-pruned status (correlation rho>={CORRELATION_ROUND_1_THRESHOLD:.2f} then "
        f">={CORRELATION_ROUND_2_THRESHOLD:.2f}, then RFE with {n_bootstrap} bootstrap resamples): {targets}"
    )
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    for index, name in enumerate(targets, start=1):
        print()
        print(f"=== [{index}/{len(targets)}] Promoting {name} ===")
        report_tag = f"promote_{name}"
        round1_path = SCRATCH_DIR / f"{name}.round1.json"
        round2_path = SCRATCH_DIR / f"{name}.round2.json"
        rfe_path = SCRATCH_DIR / f"{name}.rfe.json"

        _run_step([
            sys.executable, str(ROOT / "scripts" / "correlate_lightweight_hit_at_1_features.py"),
            "--language", language, "--config", config,
            "--manifest", str(manifest_path), "--threshold", str(CORRELATION_ROUND_1_THRESHOLD),
            "--retriever", name, "--output", str(round1_path), "--report-tag", report_tag,
        ])
        _run_step([
            sys.executable, str(ROOT / "scripts" / "correlate_lightweight_hit_at_1_features.py"),
            "--language", language, "--config", config,
            "--manifest", str(round1_path), "--threshold", str(CORRELATION_ROUND_2_THRESHOLD),
            "--retriever", name, "--output", str(round2_path), "--report-tag", report_tag,
        ])
        _run_step([
            sys.executable, str(ROOT / "scripts" / "rfe_prune_lightweight_hit_at_1_features.py"),
            "--language", language, "--config", config,
            "--manifest", str(round2_path), "--n-bootstrap", str(n_bootstrap),
            "--retriever", name, "--output", str(rfe_path), "--report-tag", report_tag,
        ])

        rfe_manifest = _load_manifest(rfe_path)
        rfe_entry = dict(rfe_manifest["retrievers"][name])
        rfe_entry["pipeline_stage"] = "rfe_pruned"
        rfe_entry["promoted_from"] = "freeze"

        manifest_data = _load_manifest(manifest_path)
        before_count = manifest_data["retrievers"][name]["qualifying_feature_count"]
        manifest_data["retrievers"][name] = rfe_entry
        _write_manifest_atomic(manifest_path, manifest_data)
        after_count = rfe_entry["qualifying_feature_count"]
        print(f"{name}: promoted {before_count} -> {after_count} qualifying feature(s), merged into {manifest_path}")

    print()
    print(
        f"Done. Promoted {len(targets)} retriever(s): {targets}. Their lightweight model "
        "artifacts are now stale (feature manifest changed under them) and will be retrained automatically "
        "the next time run_hit_at_1_predictor_lightweight.py runs, "
        "since the cached model's qualifying_features no longer match the manifest."
    )


if __name__ == "__main__":
    typer.run(main)
