from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import pandas as pd
import typer

from _unsolved_consensus_example_figures import ROOT
from spotting_feature_selection_and_sheet import output_paths
from query_difficulty.runtime import build_runtime

GROUPS = [("flagged", "Strongest"), ("control", "Missed"), ("random", "Random")]
OBSERVATIONS = [
    ("plausible_competing_document", "Plausible competing document"),
    ("context_missing", "Context missing"),
    ("hard_but_solvable", "Hard but solvable"),
    ("easy", "Easy"),
    ("duplicate_document", "Duplicate document"),
    ("other", "Other"),
]
POSITIVE_VALUES = {"1", "x", "yes", "y", "true"}
NEGATIVE_VALUES = {"", "0", "no", "n", "false"}


def parse_observations(sheet: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame(index=sheet.index)
    invalid = []
    for column, _ in OBSERVATIONS:
        values = sheet[column].fillna("").astype(str).str.strip().str.lower()
        for sheet_id, value in zip(sheet["sheet_id"], values, strict=True):
            if value not in POSITIVE_VALUES | NEGATIVE_VALUES:
                invalid.append(f"sheet_id {sheet_id}, {column}: '{value}'")
        parsed[column] = values.isin(POSITIVE_VALUES)
    if invalid:
        raise SystemExit(
            "Unrecognized annotation values (use one of "
            f"{sorted(POSITIVE_VALUES)} for yes, empty or {sorted(NEGATIVE_VALUES - {''})} for no):\n  " + "\n  ".join(invalid)
        )
    return parsed


def inspection_table(counts: pd.DataFrame, sizes: dict[str, int]) -> str:
    header_groups = " & ".join(rf"\multicolumn{{2}}{{c}}{{{label} ($N={sizes[key]}$)}}" for key, label in GROUPS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Blind manual inspection of training queries: the strongest difficulty-feature scores among all training queries (Strongest), a random sample of the remaining queries whose gold document none of \textit{BM25}, Qwen3-Emb-0.6B, gte-large-en-v1.5 and bge-base-en-v1.5 ranks in the top 5 (Missed), and a random sample of all remaining training queries (Random). A query can carry several observations.}",
        r"\label{tab:spotting_inspection}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.85}",
        rf"\begin{{tabular}}{{l{'rr' * len(GROUPS)}}}",
        r"\toprule",
        rf"Observation & {header_groups} \\",
        "".join(rf"\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(len(GROUPS))),
        " & " + " & ".join([r"$n$ & \%"] * len(GROUPS)) + r" \\",
        r"\midrule",
    ]
    for column, label in OBSERVATIONS:
        cells = []
        for key, _ in GROUPS:
            count = int(counts.loc[key, column])
            cells.append(f"{count} & {100 * count / sizes[key]:.1f}")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file.")) -> None:
    paths = output_paths(build_runtime(config, verbose=False))
    print(f"Reading {paths['sheet'].relative_to(ROOT).as_posix()} and {paths['key'].relative_to(ROOT).as_posix()}")
    sheet = pd.read_csv(paths["sheet"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(paths["key"])
    sheet["sheet_id"] = sheet["sheet_id"].astype(int)
    sheet["query_id"] = sheet["query_id"].astype(int)
    merged = sheet.merge(key, on="sheet_id", how="outer", suffixes=("", "_key"), indicator=True)
    if (merged["_merge"] != "both").any():
        raise SystemExit(f"Sheet and key do not match on sheet_id: {merged.loc[merged['_merge'] != 'both', 'sheet_id'].tolist()}")
    if (merged["query_id"] != merged["query_id_key"]).any():
        raise SystemExit("query_id differs between sheet and key for some sheet_id")
    print(f"Matched {len(merged)} annotated queries")

    observations = parse_observations(merged)
    observations["group"] = merged["group"]
    sizes = {key_name: int((merged["group"] == key_name).sum()) for key_name, _ in GROUPS}
    unknown = set(merged["group"]) - set(sizes)
    if unknown:
        raise SystemExit(f"Unknown groups in key: {sorted(unknown)}")
    counts = observations.groupby("group")[[column for column, _ in OBSERVATIONS]].sum().reindex([k for k, _ in GROUPS]).fillna(0)
    unannotated = int((~observations[[column for column, _ in OBSERVATIONS]].any(axis=1)).sum())
    print(f"Queries without any marked observation: {unannotated}")

    print("\nInspection results:")
    for column, label in OBSERVATIONS:
        parts = [f"{name} {int(counts.loc[k, column])}/{sizes[k]} ({100 * counts.loc[k, column] / sizes[k]:.1f}%)" for k, name in GROUPS]
        print(f"  {label:<32} " + "  ".join(parts))

    paths["inspection_table"].parent.mkdir(parents=True, exist_ok=True)
    paths["inspection_table"].write_text(apply_thesis_captions(inspection_table(counts, sizes) + "\n"), encoding="utf-8")
    print(f"\nWrote {paths['inspection_table'].relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    typer.run(main)
