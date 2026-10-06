from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_task_difficulty_analysis_unsolved_query_inspection import (
    _document_field,
    _load_collection,
    _load_gold_queries,
    _load_rankings,
    _majority_top_document,
)
from query_difficulty.analysis import (
    task_difficulty_analysis_submission_miss_matrix,
    task_difficulty_analysis_universally_unsolved_query_ids,
)

ABSTRACT_WORD_BUDGET = 55
FIGURE_EXAMPLE_COUNT = 6

LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "“": r"\textquotedblleft{}",
    "”": r"\textquotedblright{}",
    '"': r"\textquotedbl{}",
    "‘": "`",
    "’": "'",
    "‐": "-",
    "‑": "-",
    "–": "--",
    "—": "---",
    "…": "...",
    " ": " ",
}


def escape_latex(text: str) -> str:
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in " ".join(str(text).split()))


def truncate_words(text: str, word_budget: int) -> str:
    words = str(text).split()
    if len(words) <= word_budget:
        return " ".join(words)
    return " ".join(words[:word_budget]).rstrip(".,;:") + "..."


def build_unsolved_consensus_frame(language: str = "en") -> pd.DataFrame:
    data_root = ROOT / "data" / "SciWeb2026"
    submission_root = ROOT / "data" / "submissions"

    print(f"Loading gold queries for '{language}' from {data_root}")
    gold_queries = _load_gold_queries(data_root, language)
    print(f"Loaded {len(gold_queries)} gold queries")

    print(f"Loading collection from {data_root / 'collection_data.json'}")
    collection = _load_collection(data_root)
    print(f"Loaded {len(collection)} documents")

    print(f"Building submission miss matrix from {submission_root}")
    miss_matrix = task_difficulty_analysis_submission_miss_matrix(submission_root, gold_queries, language)
    if miss_matrix.empty:
        raise RuntimeError(f"No matching submission files found for language '{language}'.")
    unsolved_ids = task_difficulty_analysis_universally_unsolved_query_ids(miss_matrix)
    print(f"Found {len(unsolved_ids)} universally unsolved queries across {len(miss_matrix.index)} systems")

    print("Loading full system rankings")
    rankings = _load_rankings(submission_root, language)

    unsolved = (
        gold_queries.loc[gold_queries["query_id"].isin(unsolved_ids)]
        .sort_values("query_id", ascending=True)
        .reset_index(drop=True)
    )
    records = []
    for position, row in enumerate(unsolved.itertuples(index=False), start=1):
        majority = _majority_top_document(int(row.query_id), rankings)
        records.append(
            {
                "query_id": int(row.query_id),
                "query_text": row.text,
                "gold_document_id": int(row.document_id),
                "gold_title": _document_field(collection, row.document_id, "title"),
                "gold_abstract": _document_field(collection, row.document_id, "abstract"),
                **majority,
                "top_title": _document_field(collection, majority["top_document_id"], "title"),
                "top_abstract": _document_field(collection, majority["top_document_id"], "abstract"),
            }
        )
        if position % 50 == 0 or position == len(unsolved):
            print(f"Processed {position}/{len(unsolved)} unsolved queries")
    frame = pd.DataFrame(records)
    print("Rank-1 agreement distribution (votes -> number of unsolved queries):")
    for votes, count in frame["top_document_votes"].value_counts().sort_index().items():
        print(f"  {int(votes):>2} -> {int(count)}")
    return frame


def _render_example(row: pd.Series) -> str:
    return "\n".join(
        [
            r"\begin{minipage}{\textwidth}",
            rf"    {{\bf Query ID: {int(row['query_id'])}}} \\",
            r"    \nopagebreak",
            r"    \begin{tabular}{@{}lp{11.5cm}@{}}",
            r"        \hline",
            rf"        {{\bf Query:}} & ``{escape_latex(row['query_text'])}'' \\",
            r"        \hline",
            rf"        {{\bf Gold Doc ({int(row['gold_document_id'])}):}} & {{\bf Title:}} {escape_latex(row['gold_title'])} \\",
            rf"         & {{\bf Abstract:}} {escape_latex(truncate_words(row['gold_abstract'], ABSTRACT_WORD_BUDGET))} \\",
            r"        \hline",
            rf"        {{\bf Top Retr. ({int(row['top_document_id'])}):}} & {{\bf Title:}} {escape_latex(row['top_title'])} \\",
            rf"        ({int(row['top_document_votes'])}/{int(row['ranking_systems'])} systems) & {{\bf Abstract:}} {escape_latex(truncate_words(row['top_abstract'], ABSTRACT_WORD_BUDGET))} \\",
            r"        \hline",
            r"    \end{tabular}",
            r"\end{minipage}",
        ]
    )


def render_figure(examples: pd.DataFrame, label: str) -> str:
    blocks = [_render_example(row) for _, row in examples.iterrows()]
    body = "\n\n\\vspace{0.6cm}\n\n\n".join(blocks)
    return "\n".join(
        [
            r"\begin{center}",
            r"\small",
            r"\centering",
            "",
            body,
            "",
            rf"\label{{{label}}}",
            r"\end{center}",
            "",
        ]
    )


def print_examples(examples: pd.DataFrame) -> None:
    for position, row in enumerate(examples.itertuples(index=False), start=1):
        print(
            f"[{position}/{len(examples)}] query {row.query_id}: gold {row.gold_document_id}, "
            f"top {row.top_document_id} ranked first by {row.top_document_votes}/{row.ranking_systems} systems "
            f"(tied candidates: {row.top_document_tied_candidates})"
        )


def write_figure(latex: str, file_name: str) -> Path:
    output_path = ROOT / "latex" / "tables" / "difficulty_analysis" / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(f"Wrote figure to {output_path}")
    return output_path
