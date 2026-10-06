from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import pandas as pd
import typer

from _unsolved_consensus_example_figures import ROOT, escape_latex, truncate_words
from generate_task_difficulty_analysis_unsolved_query_inspection import _document_field, _load_collection, _majority_top_document
from spotting_feature_selection_and_sheet import POOL_RETRIEVERS, TRAIN_SPLIT, load_rankings, output_paths
from table_spotting_inspection import GROUPS, OBSERVATIONS, parse_observations
from query_difficulty.runtime import build_runtime

ABSTRACT_WORD_BUDGET = 55


def assessment_text(row: pd.Series) -> str:
    labels = [label for column, label in OBSERVATIONS if row[f"is_{column}"]]
    return "; ".join(labels) if labels else "--"


def render_case(position: int, row: pd.Series, group_label: str) -> str:
    reference_id = str(row["reference_document_id"])
    reference_label = int(reference_id) if reference_id else "--"
    return "\n".join(
        [
            r"\begin{minipage}{\textwidth}",
            rf"    {{\bf Case {position}, Query ID: {int(row['query_id'])}}} ({group_label}, score {float(row['score']):.3f}) \\",
            r"    \nopagebreak",
            r"    \begin{tabular}{@{}lp{11.5cm}@{}}",
            r"        \hline",
            rf"        {{\bf Query:}} & ``{escape_latex(row['query_text'])}'' \\",
            r"        \hline",
            rf"        {{\bf Gold Doc:}} & {{\bf Title:}} {escape_latex(row['gold_title'])} \\",
            rf"         & {{\bf Abstract:}} {escape_latex(truncate_words(row['gold_abstract'], ABSTRACT_WORD_BUDGET))} \\",
            r"        \hline",
            rf"        {{\bf Top Retr. ({reference_label}):}} & {{\bf Title:}} {escape_latex(row['reference_title'])} \\",
            rf"        ({int(row['reference_retrievers'])}/{int(row['retrievers_total'])} retrievers) & {{\bf Abstract:}} {escape_latex(truncate_words(row['reference_abstract'], ABSTRACT_WORD_BUDGET))} \\",
            r"        \hline",
            rf"        {{\bf Manual assessment:}} & {escape_latex(assessment_text(row))} \\",
            r"        \hline",
            r"    \end{tabular}",
            r"\end{minipage}",
        ]
    )


def main(
    per_group: int = typer.Option(5, help="Examples shown per group: highest scores for strongest, a seeded sample otherwise."),
    seed: int = typer.Option(42, help="Random seed for sampling examples within the missed and random groups."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=False)
    paths = output_paths(runtime)
    print(f"Reading {paths['sheet'].relative_to(ROOT).as_posix()} and {paths['key'].relative_to(ROOT).as_posix()}")
    sheet = pd.read_csv(paths["sheet"], dtype=str, keep_default_na=False, encoding="utf-8-sig")
    key = pd.read_csv(paths["key"], dtype=str, keep_default_na=False)
    merged = sheet.merge(key, on=["sheet_id", "query_id"], how="inner", validate="one_to_one")
    if len(merged) != len(sheet) or len(merged) != len(key):
        raise SystemExit(f"Sheet ({len(sheet)}) and key ({len(key)}) do not match one-to-one ({len(merged)} merged)")
    observations = parse_observations(merged)
    for column, _ in OBSERVATIONS:
        merged[f"is_{column}"] = observations[column]
    merged["score"] = merged["score"].astype(float)

    selected = {}
    for group_key, group_label in GROUPS:
        members = merged[merged["group"] == group_key].sort_values(["score", "query_id"], ascending=[False, True])
        if group_key == "flagged":
            members = members.head(per_group)
        else:
            members = members.sample(n=min(per_group, len(members)), random_state=seed).sort_values(["score", "query_id"], ascending=[False, True])
        selected[group_key] = members.copy()

    query_ids = [int(q) for members in selected.values() for q in members["query_id"]]
    rankings = {}
    for retriever_name in POOL_RETRIEVERS:
        frame = load_rankings(runtime.store.root_dir, TRAIN_SPLIT, retriever_name, int(runtime.config.max_ranking_depth), max_rank=None, query_ids=query_ids)
        frame = frame.sort_values(["query_id", "rank"])
        rankings[retriever_name] = {int(q): group["document_id"].astype(int).tolist() for q, group in frame.groupby("query_id")}
    collection = _load_collection(ROOT / "data" / "SciWeb2026")
    print(f"Top Retr. = rank-1 document of most of the {len(POOL_RETRIEVERS)} pool retrievers ({', '.join(POOL_RETRIEVERS)})")
    for members in selected.values():
        for index, row in members.iterrows():
            majority = _majority_top_document(int(row["query_id"]), rankings)
            top_id = majority["top_document_id"]
            members.at[index, "reference_document_id"] = "" if top_id is None else str(int(top_id))
            members.at[index, "reference_title"] = _document_field(collection, top_id, "title")
            members.at[index, "reference_abstract"] = _document_field(collection, top_id, "abstract")
            members.at[index, "reference_retrievers"] = str(int(majority["top_document_votes"]))
            members.at[index, "retrievers_total"] = str(int(majority["ranking_systems"]))

    blocks = []
    position = 0
    for group_key, group_label in GROUPS:
        members = selected[group_key]
        print(f"{group_label}: {len(members)} examples")
        for _, row in members.iterrows():
            position += 1
            blocks.append(render_case(position, row, group_label.lower()))
            print(f"  [{position}] query {row['query_id']}: score {row['score']:.3f}, assessment: {assessment_text(row)}")

    group_text = ", ".join(label.lower() for _, label in GROUPS)
    latex = "\n".join(
        [
            r"\begin{center}",
            r"\footnotesize",
            "",
            "\n\\\\\n".join(blocks),
            "",
            r"\vspace{0.2cm}",
            rf"\captionof{{figure}}{{Example inspected training queries, {per_group} per group ({group_text}), ordered by difficulty score within each group. Strongest: the highest scores among all training queries. Missed: randomly selected from the inspected queries missed by all four pool retrievers. Random: randomly selected from the inspected random training queries. Top Retr.: rank-1 document of the most of the four pool retrievers.}}",
            r"\label{fig:spotting_inspection_examples}",
            r"\end{center}",
            "",
        ]
    )
    paths["inspection_figure"].parent.mkdir(parents=True, exist_ok=True)
    paths["inspection_figure"].write_text(apply_thesis_captions(latex), encoding="utf-8")
    print(f"Wrote {paths['inspection_figure'].relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    typer.run(main)
