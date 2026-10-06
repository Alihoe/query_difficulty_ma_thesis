from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from scipy.stats import kendalltau

from _unsolved_consensus_example_figures import ROOT
from _gold_correlation_matrix_tables import _feature_lookup, _format_number, _load_feature_groups
from _unsolved_vs_rest_distribution_tables import cliffs_delta, load_unsolved_and_rest_ids
from generate_task_difficulty_analysis_unsolved_query_inspection import (
    _document_field,
    _load_collection,
    _majority_top_document,
)
from query_difficulty.analysis import DIFFICULTY_CORRELATION_SIGNIFICANCE_LEVEL
from query_difficulty.runtime import build_runtime

POOL_RETRIEVERS = ["qwen3_embedding_0_6b", "gte_large_en_v1_5", "bge_base_en_v1_5", "bm25"]
PAIR_RETRIEVERS = ["qwen3_embedding_0_6b", "bm25"]
SOURCE_RETRIEVERS = ["qwen3_embedding_0_6b"]
EXCLUDED_FEATURES = {"distance_weighted_training_query_rr_at_k"}
AGREEMENT_REPLACEMENTS = {
    "bm25_top_1_agreement": "top_1_consensus_frequency",
    "bm25_top_100_agreement": "top_100_document_consensus",
}
REFERENCE_EXCLUDED_RETRIEVERS = {"qwen3_embedding_0_6b_finetuned"}
LIGHTWEIGHT_MODEL_DIR = "lightweight"
CORRELATIONS_PATH = ROOT / "artifacts" / "reports" / "retriever_difficulty_correlations" / "en_train" / "correlations.parquet"
CORRELATION_FEATURE_SET = "gold_independent"
FEATURE_STORE = "descriptive"
TRAIN_SPLIT = "en_train"
TEST_SPLIT = "en_test"
LANGUAGE = "en"
EXPECTED_TEST_QUERIES = 6076
EXPECTED_UNSOLVED = 430
OBSERVATION_COLUMNS = ["plausible_competing_document", "context_missing", "hard_but_solvable", "easy", "duplicate_document", "other", "comment"]
FAMILY_CODES = {
    "hit_at_1": "H",
    "in_top_max_depth": "M",
    "in_top_k": "S",
    "calibrated_ordinal": "O",
    "tuned_qpp": "Q",
}
FAMILY_NAMES = {
    "H": "Hit@1",
    "M": "In-Max-Depth",
    "S": "In-top-k sweep",
    "O": "Calibrated ordinal",
    "Q": "Tuned QPP",
}
OUTPUT_DIR = "spotting_difficult_queries"
LATEX_OUTPUT_DIR = "spotting_difficult_queries_screening"


def output_paths(runtime) -> dict[str, Path]:
    table_dir = runtime.config.latex_table_dir / LATEX_OUTPUT_DIR
    report_dir = runtime.store.root_dir / "reports" / OUTPUT_DIR
    return {
        "table": table_dir / f"table_spotting_feature_selection_{LANGUAGE}.tex",
        "inspection_table": table_dir / f"table_spotting_inspection_{LANGUAGE}.tex",
        "inspection_figure": table_dir / f"figure_spotting_inspection_examples_{LANGUAGE}_train.tex",
        "detection": report_dir / "spotting_detection.json",
        "selection": report_dir / "spotting_feature_selection.csv",
        "candidates": report_dir / "spotting_train_candidates.csv",
        "readme": report_dir / "README.md",
        "sheet": report_dir / "inspection_sheet.csv",
        "key": report_dir / "inspection_key.csv",
    }


def predictor_family(predictor_name: str) -> str:
    if predictor_name.startswith("in_top_k"):
        return "in_top_k"
    if predictor_name not in FAMILY_CODES:
        raise RuntimeError(f"Unknown lightweight predictor '{predictor_name}'")
    return predictor_name


def importance_entries(console, path: Path, retriever_name: str, family: str, model: str, default_depth: int | None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Lightweight model importances missing: {path}")
    frame = pd.read_parquet(path, columns=["feature", "importance"])
    total = float(frame["importance"].sum())
    if not total > 0:
        console.log(f"Skipping {model}: all importances are zero in {path.relative_to(ROOT).as_posix()} (model uses no feature)")
        return []
    entries = []
    for column, importance in zip(frame["feature"].astype(str), frame["importance"].astype(float), strict=True):
        if "::" in column:
            depth_label, feature_name = column.split("::", 1)
            depth = int(depth_label.removeprefix("depth_"))
        else:
            if default_depth is None:
                raise RuntimeError(f"No depth for {column} in {path}")
            feature_name, depth = column, default_depth
        entries.append(
            {
                "retriever": retriever_name,
                "family": family,
                "model": model,
                "feature": feature_name,
                "manifest_depth": depth,
                "importance": importance / total,
            }
        )
    return entries


def lightweight_model_entries(console, model_root: Path) -> pd.DataFrame:
    records = []
    for retriever_name in SOURCE_RETRIEVERS:
        for predictor_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            retriever_dir = predictor_dir / retriever_name
            if not retriever_dir.is_dir():
                continue
            name = predictor_dir.name
            family = predictor_family(name)
            if family == "hit_at_1":
                depth = int(joblib.load(retriever_dir / "model.joblib")["feature_depth"])
                found = importance_entries(console, retriever_dir / "feature_importances.parquet", retriever_name, family, name, depth)
            elif family in {"in_top_max_depth", "in_top_k"}:
                found = importance_entries(console, retriever_dir / "feature_importances.parquet", retriever_name, family, name, None)
            elif family == "calibrated_ordinal":
                sizes = [int(size) for size in json.loads((retriever_dir / "manifest.json").read_text(encoding="utf-8"))["candidate_sizes"]]
                found = []
                for size in sizes:
                    found += importance_entries(
                        console, retriever_dir / f"depth_{size:04d}_importances.parquet", retriever_name, family, f"{name}/head_{size}", size
                    )
                found += importance_entries(
                    console, retriever_dir / "solvability_importances.parquet", retriever_name, family, f"{name}/solvability", max(sizes)
                )
            else:
                depth = int(json.loads((retriever_dir / "manifest.json").read_text(encoding="utf-8"))["feature_depth"])
                found = []
                for head in ("failure", "score"):
                    found += importance_entries(console, retriever_dir / f"{head}_importances.parquet", retriever_name, family, f"{name}/{head}", depth)
            models = sorted({entry["model"] for entry in found})
            console.log(f"{retriever_name} / {LIGHTWEIGHT_MODEL_DIR}/{name}: {len(models)} model(s), {len(found)} feature entries")
            records += found
    return pd.DataFrame(records)


def candidate_specs(console, model_root: Path, correlations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = lightweight_model_entries(console, model_root)
    models_per_family = raw.groupby("family")["model"].nunique()
    if set(models_per_family.index) != set(FAMILY_CODES):
        raise RuntimeError(f"Expected models for families {sorted(FAMILY_CODES)}, found {sorted(models_per_family.index)}")
    for family, count in models_per_family.items():
        console.log(f"Family {FAMILY_NAMES[FAMILY_CODES[family]]}: {count} model(s)")
    console.log(f"Read {len(raw)} feature entries from {raw['model'].nunique()} trained models in {model_root.relative_to(ROOT).as_posix()}")
    excluded = raw[raw["feature"].isin(EXCLUDED_FEATURES)]
    for _, row in excluded.iterrows():
        console.log(f"Excluding {row['feature']} (k={row['manifest_depth']}, {row['model']}, normalized importance {row['importance']:.3f})")
    raw = raw[~raw["feature"].isin(EXCLUDED_FEATURES)].copy()
    raw["importance_from"] = raw["feature"].where(raw["feature"].isin(AGREEMENT_REPLACEMENTS), "")
    for source, target in AGREEMENT_REPLACEMENTS.items():
        count = int((raw["feature"] == source).sum())
        console.log(f"Replacing {source} by the all-retriever {target} ({count} model entries; importance carried over)")
    raw["feature"] = raw["feature"].replace(AGREEMENT_REPLACEMENTS)

    depth_dependence = (
        correlations.drop_duplicates(subset=["retriever_name", "feature_name"]).set_index(["retriever_name", "feature_name"])[
            "depth_dependent"
        ]
    )
    missing = [(r, f) for r, f in zip(raw["retriever"], raw["feature"], strict=True) if (r, f) not in depth_dependence.index]
    if missing:
        raise RuntimeError(f"No stored training correlation for: {missing}")
    raw["depth_dependent"] = [bool(depth_dependence[(r, f)]) for r, f in zip(raw["retriever"], raw["feature"], strict=True)]
    raw["k_key"] = [int(d) if dep else -1 for d, dep in zip(raw["manifest_depth"], raw["depth_dependent"], strict=True)]
    raw["family_weighted_importance"] = raw["importance"] / raw["family"].map(models_per_family) / len(models_per_family)

    specs = []
    for (retriever_name, feature_name), group in raw.groupby(["retriever", "feature"], sort=False):
        copies = group.groupby("k_key")["family_weighted_importance"].sum().reset_index()
        copies = copies.sort_values(["family_weighted_importance", "k_key"], ascending=[False, True])
        representative = int(copies["k_key"].iloc[0])
        depth_dependent = bool(group["depth_dependent"].iloc[0])
        load_rows = group[group["k_key"] == representative]
        specs.append(
            {
                "retriever": retriever_name,
                "feature": feature_name,
                "depth_dependent": depth_dependent,
                "k": float(representative) if depth_dependent else math.nan,
                "load_depth": int(load_rows["manifest_depth"].min()),
                "train_importance": float(group["family_weighted_importance"].sum()),
                "representative_importance": float(copies["family_weighted_importance"].iloc[0]),
                "importance_from": ";".join(sorted({value for value in group["importance_from"] if value})),
                "n_families": int(group["family"].nunique()),
                "n_models": int(group["model"].nunique()),
                "families": [code for code in FAMILY_NAMES if code in {FAMILY_CODES[f] for f in group["family"]}],
                "predictors": sorted(set(group["model"])),
                "manifest_depths": sorted({int(d) for d in group["manifest_depth"]}),
            }
        )
    specs_frame = pd.DataFrame(specs).sort_values("train_importance", ascending=False).reset_index(drop=True)
    console.log(
        f"{len(specs_frame)} distinct features; importance = family-weighted mean of per-model normalized importances, "
        "summed over depth copies; representative k = most important copy"
    )
    return specs_frame, raw


def stored_tau(correlations: pd.DataFrame, retriever_name: str, feature_name: str, k: float) -> dict[str, float]:
    rows = correlations[(correlations["retriever_name"] == retriever_name) & (correlations["feature_name"] == feature_name)]
    rows = rows[rows["candidate_size"] == k] if not math.isnan(k) else rows[rows["candidate_size"].isna()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one stored correlation row for {retriever_name}/{feature_name}/k={k}, found {len(rows)}")
    row = rows.iloc[0]
    return {
        "tau_b": float(row["kendall_tau_b"]),
        "adjusted_p_value": float(row["adjusted_p_value"]),
        "significant": bool(row["significant"]),
        "tau_n": int(row["n_used"]),
        "tau_source": f"stored ({CORRELATIONS_PATH.relative_to(ROOT).as_posix()}, feature_set={CORRELATION_FEATURE_SET})",
    }


class FeatureStore:
    def __init__(self, artifact_root: Path, console) -> None:
        self.artifact_root = artifact_root
        self.console = console
        self.cache: dict[tuple[str, str, int], pd.DataFrame] = {}

    def values(self, split: str, retriever_name: str, feature_name: str, depth: int) -> pd.Series:
        key = (split, retriever_name, depth)
        if key not in self.cache:
            path = self.artifact_root / "features" / FEATURE_STORE / split / retriever_name / f"depth_{depth:04d}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Stored feature file missing (not recomputed): {path}")
            frame = pd.read_parquet(path)
            if frame["query_id"].duplicated().any():
                raise RuntimeError(f"Duplicate query ids in {path}")
            self.cache[key] = frame.set_index(frame["query_id"].astype(int))
            self.console.log(f"Loaded {path.relative_to(ROOT).as_posix()} ({len(frame)} rows)")
        frame = self.cache[key]
        if feature_name not in frame.columns:
            raise KeyError(f"{feature_name} not in stored features for {split}/{retriever_name}/depth {depth}")
        return pd.to_numeric(frame[feature_name], errors="coerce").astype(np.float64)


def load_rankings(artifact_root: Path, split: str, retriever_name: str, max_depth: int, *, max_rank: int | None, query_ids=None) -> pd.DataFrame:
    path = artifact_root / "rankings" / split / f"{retriever_name}_top_{max_depth}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Stored ranking missing (not recomputed): {path}")
    filters = []
    if max_rank is not None:
        filters.append(("rank", "<=", int(max_rank)))
    if query_ids is not None:
        filters.append(("query_id", "in", [int(q) for q in query_ids]))
    return pd.read_parquet(path, columns=["query_id", "rank", "document_id"], filters=filters or None)


def gold_in_top(rankings: pd.DataFrame, gold: pd.Series) -> pd.Series:
    merged = rankings.merge(gold.rename("gold_document_id"), left_on="query_id", right_index=True, how="inner")
    hit_ids = set(merged.loc[merged["document_id"] == merged["gold_document_id"], "query_id"].astype(int))
    return pd.Series({int(q): int(q) in hit_ids for q in gold.index}, dtype=bool)


def pool_flags(console, artifact_root: Path, split: str, gold: pd.Series, max_depth: int, top_depth: int) -> pd.DataFrame:
    hits = {}
    for position, retriever_name in enumerate(POOL_RETRIEVERS, start=1):
        rankings = load_rankings(artifact_root, split, retriever_name, max_depth, max_rank=top_depth)
        covered = set(rankings["query_id"].astype(int))
        uncovered = len(set(gold.index) - covered)
        hits[retriever_name] = gold_in_top(rankings, gold)
        console.log(
            f"[{position}/{len(POOL_RETRIEVERS)}] {split} / {retriever_name}: gold in top {top_depth} for "
            f"{int(hits[retriever_name].sum())} of {len(gold)} queries ({uncovered} queries without a ranking)"
        )
    frame = pd.DataFrame(hits)
    frame["flag_pool"] = ~frame[POOL_RETRIEVERS].any(axis=1)
    frame["flag_pair"] = ~frame[PAIR_RETRIEVERS].any(axis=1)
    return frame


def display_name(row: pd.Series, feature_to_display: dict[str, str]) -> str:
    name = feature_to_display.get(row["feature"], row["feature"].replace("_", " "))
    if row["depth_dependent"]:
        name = f"{name} ($k={int(row['k'])}$)"
    return name


def feature_selection_table(selection: pd.DataFrame, agreement: dict, n_unsolved: int, n_rest: int, delta_threshold: float) -> str:
    families = ", ".join(f"{code}: {name}" for code, name in FAMILY_NAMES.items())
    p_text = f"{agreement['p_value']:.2g}" if np.isfinite(agreement["p_value"]) else "--"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Features of the ultra-lightweight Qwen3-Emb-0.6B predictors on the training split vs.\ their separation of unsolved test queries. Fam.: number of predictor families using the feature ({families}). Imp.: training importance (per-model normalized importances summed over depth copies, averaged within and then across families). $\tau_b$: Kendall correlation with reciprocal rank on the training split ($^{{*}}$ BH-adjusted $p<{DIFFICULTY_CORRELATION_SIGNIFICANCE_LEVEL:g}$). $\delta$: Cliff's $\delta$, unsolved ($n={n_unsolved}$) vs.\ remaining ($n={n_rest}$) test queries, positive = higher for unsolved. In line: $\delta$ has the sign implied by $\tau_b$; $^{{\dagger}}$ also $|\delta|\geq{delta_threshold:g}$ (used to rank training queries). $^{{\ddagger}}$ Consensus over the 13 other base retrievers; importance carried over from the \textit{{BM25}}-only agreement feature it replaces.}}",
        r"\label{tab:spotting_feature_selection}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.85}",
        r"\begin{tabular}{p{0.5\linewidth}rrrrc}",
        r"\toprule",
        r"Feature & Fam. & Imp.\ (\%) & $\tau_b$ & $\delta$ & In line \\",
        r"\midrule",
    ]
    for _, row in selection.iterrows():
        tau = _format_number(row["tau_b"]) + (r"$^{*}$" if row["significant"] else "")
        delta = _format_number(row["cliffs_delta"]) if np.isfinite(row["cliffs_delta"]) else "--"
        in_line = ("yes" + (r"$^{\dagger}$" if row["kept"] else "")) if row["direction_consistent"] else "no"
        marker = r"$^{\ddagger}$" if row["importance_from"] else ""
        lines.append(f"{row['display_name']}{marker} & {row['n_families']} & {100 * row['train_importance']:.1f} & {tau} & {delta} & {in_line} \\\\")
    lines += [
        r"\midrule",
        rf"\multicolumn{{6}}{{l}}{{In line: {agreement['in_line']} of {agreement['n_features']}; Kendall $\tau_b$(Imp., $|\delta|$) $= {_format_number(agreement['tau_b'])}$ ($p={p_text}$, $n={agreement['n_features_with_delta']}$)}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def reference_retrievers(runtime) -> list[str]:
    names = [*runtime.config.enabled_lexical_retrievers, *runtime.config.enabled_semantic_retrievers]
    return [name for name in names if name not in REFERENCE_EXCLUDED_RETRIEVERS]


def detection_metrics(flags: pd.Series, unsolved_ids: set[int]) -> dict[str, float]:
    flagged = {int(q) for q in flags.index[flags]}
    true_positives = len(flagged & unsolved_ids)
    return {
        "flagged": len(flagged),
        "true_positives": true_positives,
        "precision": true_positives / len(flagged) if flagged else math.nan,
        "recall": true_positives / len(unsolved_ids) if unsolved_ids else math.nan,
    }


def write_text(path: Path, text: str, console) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(apply_thesis_captions(text), encoding="utf-8")
    console.log(f"Wrote {path.relative_to(ROOT).as_posix()}")


def main(
    n_per_group: int = typer.Option(25, help="Number of flagged and of control queries exported for inspection."),
    seed: int = typer.Option(42, help="Random seed for the control sample and the sheet order."),
    delta_threshold: float = typer.Option(0.33, help="Minimum |Cliff's delta| for a feature to be kept."),
    top_depth: int = typer.Option(5, help="Rank cut-off for the unsolved definition and the flag."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
    overwrite_sheet: bool = typer.Option(False, help="Overwrite an existing inspection sheet and key."),
    reuse_random_from: str = typer.Option("", help="Key file whose random group is reused instead of drawing a new one."),
) -> None:
    runtime = build_runtime(config, verbose=True)
    console = runtime.console
    artifact_root = runtime.store.root_dir
    max_depth = int(runtime.config.max_ranking_depth)
    model_root = artifact_root / "models" / LIGHTWEIGHT_MODEL_DIR
    paths = output_paths(runtime)

    if (paths["sheet"].exists() or paths["key"].exists()) and not overwrite_sheet:
        raise SystemExit(
            f"{paths['sheet'].relative_to(ROOT).as_posix()} or {paths['key'].relative_to(ROOT).as_posix()} already exists; "
            "rerun with --overwrite-sheet to replace it (annotations would be lost)."
        )

    console.section("Step 0: inputs")
    datasets = runtime.dataset.load().queries
    train_queries = datasets[TRAIN_SPLIT].copy()
    test_queries = datasets[TEST_SPLIT].copy()
    train_queries["query_id"] = train_queries["query_id"].astype(int)
    test_queries["query_id"] = test_queries["query_id"].astype(int)
    train_gold = train_queries.set_index("query_id")["document_id"].astype(int)
    test_gold = test_queries.set_index("query_id")["document_id"].astype(int)
    console.log(f"{TRAIN_SPLIT}: {len(train_gold)} queries; {TEST_SPLIT}: {len(test_gold)} queries")
    if len(test_gold) != EXPECTED_TEST_QUERIES:
        raise SystemExit(f"Expected {EXPECTED_TEST_QUERIES} test queries, found {len(test_gold)}")

    unsolved_ids, rest_ids = load_unsolved_and_rest_ids(runtime, TEST_SPLIT, LANGUAGE)
    if len(unsolved_ids) != EXPECTED_UNSOLVED:
        raise SystemExit(
            f"Unsolved definition yields {len(unsolved_ids)} of {len(test_gold)} test queries, expected {EXPECTED_UNSOLVED}. Stopping."
        )
    console.log(f"Unsolved check passed: {len(unsolved_ids)} of {len(test_gold)} test queries")

    correlations = pd.read_parquet(CORRELATIONS_PATH)
    correlations = correlations[correlations["feature_set"] == CORRELATION_FEATURE_SET]
    console.log(f"Loaded {len(correlations)} stored {CORRELATION_FEATURE_SET} training correlations from {CORRELATIONS_PATH.relative_to(ROOT).as_posix()}")

    console.section("Step 1: feature selection")
    specs, model_entries = candidate_specs(console, model_root, correlations)
    store = FeatureStore(artifact_root, console)
    feature_to_display, _, _, _ = _feature_lookup(_load_feature_groups())
    test_ids = sorted(test_gold.index)
    train_ids = sorted(train_gold.index)
    test_values: dict[str, pd.Series] = {}
    train_values: dict[str, pd.Series] = {}
    rows = []
    for _, spec in specs.iterrows():
        key = f"{spec['retriever']}::{spec['feature']}::{'' if math.isnan(spec['k']) else int(spec['k'])}"
        test_values[key] = store.values(TEST_SPLIT, spec["retriever"], spec["feature"], spec["load_depth"]).reindex(test_ids)
        train_values[key] = store.values(TRAIN_SPLIT, spec["retriever"], spec["feature"], spec["load_depth"]).reindex(train_ids)
        rows.append({**spec.to_dict(), "key": key, **stored_tau(correlations, spec["retriever"], spec["feature"], spec["k"])})

    selection = pd.DataFrame(rows)

    unsolved_sorted = sorted(unsolved_ids)
    rest_sorted = sorted(rest_ids)
    deltas, missing_u, missing_r, n_u, n_r = [], [], [], [], []
    for key in selection["key"]:
        values_u = test_values[key].reindex(unsolved_sorted).to_numpy(dtype=np.float64)
        values_r = test_values[key].reindex(rest_sorted).to_numpy(dtype=np.float64)
        deltas.append(cliffs_delta(values_u, values_r))
        missing_u.append(int((~np.isfinite(values_u)).sum()))
        missing_r.append(int((~np.isfinite(values_r)).sum()))
        n_u.append(int(np.isfinite(values_u).sum()))
        n_r.append(int(np.isfinite(values_r).sum()))
    selection["cliffs_delta"] = deltas
    selection["missing_unsolved"] = missing_u
    selection["missing_rest"] = missing_r
    selection["n_unsolved_used"] = n_u
    selection["n_rest_used"] = n_r
    selection["direction_consistent"] = [
        bool(np.isfinite(d) and np.sign(d) != 0 and np.sign(d) == -np.sign(t))
        for d, t in zip(selection["cliffs_delta"], selection["tau_b"], strict=True)
    ]
    selection["kept"] = selection["direction_consistent"] & (selection["cliffs_delta"].abs() >= delta_threshold)
    selection["display_name"] = selection.apply(lambda row: display_name(row, feature_to_display), axis=1)
    selection = selection.sort_values(["train_importance", "display_name"], ascending=[False, True]).reset_index(drop=True)
    finite = selection[np.isfinite(selection["cliffs_delta"])]
    agreement_result = kendalltau(finite["train_importance"], finite["cliffs_delta"].abs(), variant="b", alternative="two-sided")
    agreement = {
        "n_features": int(len(selection)),
        "n_features_with_delta": int(len(finite)),
        "in_line": int(selection["direction_consistent"].sum()),
        "in_line_and_large": int(selection["kept"].sum()),
        "tau_b": float(agreement_result.statistic),
        "p_value": float(agreement_result.pvalue),
    }

    print("\nTraining importance vs. separation of unsolved test queries (ordered by training importance):")
    for _, row in selection.iterrows():
        print(
            f"  {row['display_name']:<75} fam={','.join(row['families']):<10} models={row['n_models']:>2} "
            f"imp={100 * row['train_importance']:5.1f}% tau={row['tau_b']:+.3f}{'*' if row['significant'] else ' '} "
            f"delta={row['cliffs_delta']:+.3f} missing U/R={row['missing_unsolved']}/{row['missing_rest']} "
            f"in_line={'yes' if row['direction_consistent'] else 'no'} kept={'yes' if row['kept'] else 'no'}"
        )
    print(
        f"  In line: {agreement['in_line']} of {agreement['n_features']} features "
        f"({agreement['in_line_and_large']} with |delta| >= {delta_threshold}); "
        f"Kendall tau_b(importance, |delta|) = {agreement['tau_b']:+.3f} (p={agreement['p_value']:.3g}, n={agreement['n_features_with_delta']})"
    )
    write_text(paths["table"], feature_selection_table(selection, agreement, len(unsolved_ids), len(rest_ids), delta_threshold) + "\n", console)
    paths["selection"].parent.mkdir(parents=True, exist_ok=True)
    selection.assign(
        predictors=lambda f: f["predictors"].map(lambda v: ";".join(v)),
        families=lambda f: f["families"].map(lambda v: ";".join(v)),
        manifest_depths=lambda f: f["manifest_depths"].map(lambda v: ";".join(str(d) for d in v)),
    ).to_csv(paths["selection"], index=False)
    console.log(f"Wrote {paths['selection'].relative_to(ROOT).as_posix()}")
    kept = selection[selection["kept"]].reset_index(drop=True)
    console.log(f"Kept {len(kept)} of {len(selection)} candidate features")
    if kept.empty:
        raise SystemExit("No feature passed the selection; stopping before ranking.")

    console.section("Step 2: detection on the test split")
    test_flags = pool_flags(console, artifact_root, TEST_SPLIT, test_gold, max_depth, top_depth)
    pool_metrics = detection_metrics(test_flags["flag_pool"], unsolved_ids)
    pair_metrics = detection_metrics(test_flags["flag_pair"], unsolved_ids)
    unflagged_unsolved = len(unsolved_ids - {int(q) for q in test_flags.index[test_flags["flag_pool"]]})
    detection = {
        "split": TEST_SPLIT,
        "test_queries": len(test_gold),
        "unsolved_queries": len(unsolved_ids),
        "unsolved_definition": f"gold not in top {top_depth} of any submitted system (data/submissions, cross_system_unsolved_indicator)",
        "flag_definition": f"gold not in top {top_depth} of any listed retriever",
        "pool": {"retrievers": POOL_RETRIEVERS, **pool_metrics},
        "qwen_bm25_only": {"retrievers": PAIR_RETRIEVERS, **pair_metrics},
        "unsolved_not_flagged_by_pool": unflagged_unsolved,
    }
    write_text(paths["detection"], json.dumps(detection, indent=2) + "\n", console)
    summary = (
        f"Flagging test queries whose gold document is not in the top {top_depth} of any of the four pool retrievers "
        f"marks {pool_metrics['flagged']} of {len(test_gold)} queries, {pool_metrics['true_positives']} of which are among the "
        f"{len(unsolved_ids)} queries no submitted system solved (precision {pool_metrics['precision']:.3f}, recall {pool_metrics['recall']:.3f}). "
        f"Using only Qwen3-Emb-0.6B and \\textit{{BM25}} flags {pair_metrics['flagged']} queries with {pair_metrics['true_positives']} true positives "
        f"(precision {pair_metrics['precision']:.3f}, recall {pair_metrics['recall']:.3f}). "
        f"No threshold or parameter was tuned on the test split."
    )
    print("\nSummary to paste:\n" + summary + "\n")

    console.section("Step 3: ranking all training queries")
    candidate_ids = train_ids
    console.log(f"Ranking all {len(candidate_ids)} training queries by the kept features (no retrieval pre-filter)")
    if len(candidate_ids) < 2 * n_per_group:
        raise SystemExit(f"Only {len(candidate_ids)} candidates; need at least {2 * n_per_group}.")

    oriented = {}
    score_features = []
    for _, row in kept.iterrows():
        values = train_values[row["key"]].reindex(candidate_ids)
        duplicate_of = next(
            (other for other in score_features if np.array_equal(values.to_numpy(), train_values[other].reindex(candidate_ids).to_numpy(), equal_nan=True)),
            None,
        )
        if duplicate_of is not None:
            console.log(f"{row['key']} has identical values to {duplicate_of} on the candidates; counted once in the score")
            continue
        orientation = -1.0 if row["tau_b"] > 0 else 1.0
        oriented[row["key"]] = (orientation * values).rank(pct=True, method="average")
        score_features.append(row["key"])
        console.log(
            f"Score feature {row['key']}: higher = harder when value is {'lower' if orientation < 0 else 'higher'} "
            f"(training tau_b={row['tau_b']:+.3f}); missing for {int(values.isna().sum())} candidates"
        )
    percentiles = pd.DataFrame(oriented, index=candidate_ids)
    available = percentiles.notna().sum(axis=1)
    candidates = pd.DataFrame(
        {
            "score": percentiles.mean(axis=1, skipna=True),
            "features_available": available,
            "features_used_in_score": len(score_features),
        }
    )
    any_missing = int((available < len(score_features)).sum())
    all_missing = int((available == 0).sum())
    console.log(f"Candidates with at least one missing score feature: {any_missing}; with all missing (no score): {all_missing}")
    candidates = pd.concat([candidates, percentiles.add_prefix("pct::")], axis=1)
    candidates.index.name = "query_id"
    candidates.to_csv(paths["candidates"])
    console.log(f"Wrote {paths['candidates'].relative_to(ROOT).as_posix()}")

    scored = candidates[candidates["score"].notna()].reset_index().sort_values(["score", "query_id"], ascending=[False, True])
    if len(scored) < n_per_group:
        raise SystemExit(f"Only {len(scored)} candidates have a score; need {n_per_group}.")
    flagged_ids = scored["query_id"].head(n_per_group).astype(int).tolist()
    train_flags = pool_flags(console, artifact_root, TRAIN_SPLIT, train_gold, max_depth, top_depth)
    missed_ids = {int(q) for q in train_flags.index[train_flags["flag_pool"]]}
    console.log(
        f"Training queries missed by all pool retrievers (gold not in top {top_depth} of {', '.join(POOL_RETRIEVERS)}): "
        f"{len(missed_ids)} of {len(train_gold)}; {len(missed_ids & set(flagged_ids))} of the strongest {n_per_group} are among them"
    )
    remaining_ids = sorted(missed_ids - set(flagged_ids))
    reused_ids: list[int] = []
    if reuse_random_from:
        previous_key = pd.read_csv(ROOT / reuse_random_from)
        reused_ids = sorted(int(q) for q in previous_key.loc[previous_key["group"] == "control", "query_id"])
        invalid = [q for q in reused_ids if q not in set(remaining_ids)]
        if len(reused_ids) > n_per_group or invalid:
            raise SystemExit(
                f"Cannot reuse random group from {reuse_random_from}: {len(reused_ids)} queries (at most {n_per_group}); "
                f"not among the remaining missed queries: {invalid}"
            )
        console.log(f"Reusing the {len(reused_ids)} random queries from {reuse_random_from} (all among the remaining missed queries)")
    pool_ids = sorted(set(remaining_ids) - set(reused_ids))
    additional = n_per_group - len(reused_ids)
    drawn_ids = [int(q) for q in np.random.default_rng(seed).choice(pool_ids, size=additional, replace=False)] if additional else []
    if additional:
        console.log(f"Drew {additional} additional random queries from {len(pool_ids)} remaining missed training queries (seed {seed})")
    control_ids = sorted(reused_ids + drawn_ids)
    console.log(
        f"Flagged: top {n_per_group} by score (score range {scored['score'].head(n_per_group).min():.4f}-{scored['score'].head(n_per_group).max():.4f}); "
        f"control: {len(reused_ids)} reused, {len(drawn_ids)} drawn (seed {seed}) from training queries missed by all pool retrievers"
    )

    console.section("Step 4: blind inspection sheet")
    selected = pd.DataFrame(
        {"query_id": flagged_ids + control_ids, "group": ["flagged"] * n_per_group + ["control"] * n_per_group}
    )
    selected["score"] = candidates["score"].reindex(selected["query_id"]).to_numpy()
    selected = selected.iloc[np.random.default_rng(seed).permutation(len(selected))].reset_index(drop=True)
    selected["sheet_id"] = np.arange(1, len(selected) + 1)

    reference_names = reference_retrievers(runtime)
    console.log(
        f"Reference document = rank-1 document of most of {len(reference_names)} retrievers ({', '.join(reference_names)}); "
        f"excluded: {', '.join(sorted(REFERENCE_EXCLUDED_RETRIEVERS))}"
    )
    rankings = {}
    for retriever_name in reference_names:
        frame = load_rankings(artifact_root, TRAIN_SPLIT, retriever_name, max_depth, max_rank=None, query_ids=selected["query_id"])
        frame = frame.sort_values(["query_id", "rank"])
        rankings[retriever_name] = {int(q): group["document_id"].astype(int).tolist() for q, group in frame.groupby("query_id")}
    collection = _load_collection(ROOT / "data" / "SciWeb2026")
    console.log(f"Loaded {len(collection)} documents")
    query_text = train_queries.set_index("query_id")["text"]
    sheet_rows = []
    tied = 0
    for _, row in selected.iterrows():
        query_id = int(row["query_id"])
        majority = _majority_top_document(query_id, rankings)
        tied += int(majority["top_document_tied_candidates"] > 1)
        gold_id = int(train_gold.loc[query_id])
        top_id = majority["top_document_id"]
        sheet_rows.append(
            {
                "sheet_id": int(row["sheet_id"]),
                "query_id": query_id,
                "query_text": " ".join(str(query_text.loc[query_id]).split()),
                "gold_title": _document_field(collection, gold_id, "title"),
                "gold_abstract": _document_field(collection, gold_id, "abstract"),
                "reference_document_id": "" if top_id is None else int(top_id),
                "reference_title": _document_field(collection, top_id, "title"),
                "reference_abstract": _document_field(collection, top_id, "abstract"),
                "reference_retrievers": int(majority["top_document_votes"]),
                "retrievers_total": int(majority["ranking_systems"]),
                **{column: "" for column in OBSERVATION_COLUMNS},
            }
        )
    console.log(f"Reference document: {tied} of {len(selected)} queries needed the tie-break (sum of reciprocal ranks, then lower id)")
    paths["sheet"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sheet_rows).to_csv(paths["sheet"], index=False, encoding="utf-8-sig")
    console.log(f"Wrote {paths['sheet'].relative_to(ROOT).as_posix()}")
    selected[["sheet_id", "query_id", "group", "score"]].to_csv(paths["key"], index=False)
    console.log(f"Wrote {paths['key'].relative_to(ROOT).as_posix()} (do not open before annotating)")

    readme = [
        "# Spotting difficult queries: run record",
        "",
        "Generated by `latex/scripts/spotting_feature_selection_and_sheet.py`.",
        "",
        "## Inputs",
        f"- Queries and gold documents: runtime dataset (`{TRAIN_SPLIT}`, `{TEST_SPLIT}`)",
        "- Documents: `data/SciWeb2026/collection_data.json`",
        "- Unsolved test queries: `data/submissions/*.zip` via `cross_system_unsolved_indicator`",
        f"- Pool rankings: `artifacts/rankings/<split>/<retriever>_top_{max_depth}.parquet` for {', '.join(POOL_RETRIEVERS)}",
        f"- Feature values: `artifacts/features/{FEATURE_STORE}/<split>/<retriever>/depth_<k>.parquet`",
        f"- Ultra-lightweight predictor features: `{model_root.relative_to(ROOT).as_posix()}/<predictor>/<retriever>/*importances.parquet` "
        "(identical to the fitted models' input columns); k from the feature's depth prefix, the Hit@1 model's feature_depth, "
        "the ordinal head size (solvability head: largest head), or the tuned-QPP feature_depth",
        f"- Training correlations: `{CORRELATIONS_PATH.relative_to(ROOT).as_posix()}` (feature_set={CORRELATION_FEATURE_SET})",
        f"- Excluded: {', '.join(sorted(EXCLUDED_FEATURES))}",
        "",
        "## Candidate features",
        "| Feature | Retriever | Predictor model(s) | Model k | Train importance | k used | tau_b source | Missing U/R (test) | Kept |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in selection.iterrows():
        readme.append(
            f"| {row['feature']} | {row['retriever']} | {', '.join(row['predictors'])} | "
            f"{', '.join(str(d) for d in row['manifest_depths']) or '-'} | {100 * row['train_importance']:.1f}% | "
            f"{'-' if not row['depth_dependent'] else int(row['k'])} | {row['tau_source']} | "
            f"{row['missing_unsolved']}/{row['missing_rest']} | {'yes' if row['kept'] else 'no'} |"
        )
    readme += [
        "",
        "## Sample sizes",
        f"- Test: {len(test_gold)} queries, {len(unsolved_ids)} unsolved (U), {len(rest_ids)} rest (R)",
        f"- Ranked training queries: all {len(candidate_ids)} (no retrieval pre-filter); random group drawn from the {len(missed_ids)} missed by all pool retrievers",
        f"- Score features: {len(score_features)} ({', '.join(score_features)})",
        f"- Candidates with >=1 missing score feature: {any_missing}; with no score: {all_missing}",
        f"- Inspection: {n_per_group} flagged + {n_per_group} control",
        f"- Reference document: rank-1 document of most of {len(reference_names)} retrievers ({', '.join(reference_names)}); "
        f"tie-break needed for {tied} queries",
        "",
        "## Parameters",
        f"- seed={seed} (control sample and sheet order, numpy default_rng), N={n_per_group}, |delta| threshold={delta_threshold}, top depth={top_depth}",
        "- Percentile ranks: pandas rank(pct=True, method='average') within the candidates, oriented so that higher = harder",
        "",
    ]
    write_text(paths["readme"], "\n".join(readme), console)

    print("\nSummary")
    print(f"  Candidate features: {len(selection)}; kept: {len(kept)} ({', '.join(kept['key'])})")
    print(f"  Pool flag on test: {pool_metrics}")
    print(f"  Qwen+BM25 flag on test: {pair_metrics}")
    print(f"  Ranked training queries: {len(candidate_ids)}; strongest {n_per_group}, random {n_per_group}")
    for name, path in paths.items():
        if name in {"inspection_table", "inspection_figure"}:
            continue
        print(f"  {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    typer.run(main)
