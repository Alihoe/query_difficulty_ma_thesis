from __future__ import annotations

import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
import sys

import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import (
    read_submission_archive,
    task_difficulty_analysis_submission_miss_matrix,
    task_difficulty_analysis_system_display_name,
    task_difficulty_analysis_universally_unsolved_query_ids,
)


def _load_gold_queries(data_dir: Path, language: str) -> pd.DataFrame:
    path = data_dir / f"final_{language}_test.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload).rename(columns={"index": "query_id", "pubkey": "document_id"})
    frame["query_id"] = frame["query_id"].astype(int)
    frame["document_id"] = frame["document_id"].astype(int)
    frame["text"] = frame["text"].fillna("")
    return frame[["query_id", "document_id", "text"]]


def _load_collection(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "collection_data.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload).rename(columns={"pubkey": "document_id"})
    frame["document_id"] = frame["document_id"].astype(int)
    frame["title"] = frame["title"].fillna("")
    frame["abstract"] = frame["abstract"].fillna("")
    return frame.set_index("document_id")[["title", "abstract"]]


def _load_rankings(submission_dir: Path, language: str) -> dict[str, dict[int, list[int]]]:
    rankings: dict[str, dict[int, list[int]]] = {}
    archive_paths = sorted(submission_dir.glob("*.zip"))
    for position, archive_path in enumerate(archive_paths, start=1):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            typer.echo(f"[{position}/{len(archive_paths)}] Skipping {archive_path.name}: no '{language}' predictions")
            continue
        frame = archives[language].rename(columns={"index": "query_id"}).copy()
        frame["query_id"] = frame["query_id"].astype(int)
        frame = frame.drop_duplicates(subset=["query_id"], keep="first")
        rankings[archive_path.stem] = {
            int(query_id): list(preds) for query_id, preds in zip(frame["query_id"], frame["preds"], strict=True)
        }
        typer.echo(f"[{position}/{len(archive_paths)}] Loaded {len(frame)} rankings from {archive_path.name}")
    return rankings


def _majority_top_document(query_id: int, rankings: dict[str, dict[int, list[int]]]) -> dict[str, object]:
    top_votes: Counter[int] = Counter()
    top_voters: dict[int, list[str]] = defaultdict(list)
    reciprocal_rank_sums: Counter[int] = Counter()
    ranking_systems = 0
    for system_id, system_rankings in rankings.items():
        preds = system_rankings.get(query_id, [])
        if not preds:
            continue
        ranking_systems += 1
        top_votes[preds[0]] += 1
        top_voters[preds[0]].append(task_difficulty_analysis_system_display_name(system_id))
        for rank, document_id in enumerate(preds, start=1):
            reciprocal_rank_sums[document_id] += 1.0 / rank
    if not top_votes:
        return {
            "ranking_systems": 0,
            "top_document_id": None,
            "top_document_votes": 0,
            "top_document_mean_reciprocal_rank": float("nan"),
            "top_document_tied_candidates": 0,
            "top_document_voters": "",
        }
    max_votes = max(top_votes.values())
    tied = [document_id for document_id, votes in top_votes.items() if votes == max_votes]
    top_document_id = max(tied, key=lambda document_id: (reciprocal_rank_sums[document_id], -document_id))
    return {
        "ranking_systems": ranking_systems,
        "top_document_id": int(top_document_id),
        "top_document_votes": int(max_votes),
        "top_document_mean_reciprocal_rank": reciprocal_rank_sums[top_document_id] / ranking_systems,
        "top_document_tied_candidates": len(tied),
        "top_document_voters": ", ".join(sorted(top_voters[top_document_id])),
    }


def _document_field(collection: pd.DataFrame, document_id: object, field: str) -> str:
    if document_id is None or pd.isna(document_id) or int(document_id) not in collection.index:
        return ""
    return str(collection.at[int(document_id), field])


def _render_card(row: pd.Series) -> str:
    tie_note = (
        f" Tie between {int(row['top_document_tied_candidates'])} papers, broken by mean reciprocal rank."
        if int(row["top_document_tied_candidates"]) > 1
        else ""
    )
    top_document_label = "none" if pd.isna(row["top_document_id"]) else str(int(row["top_document_id"]))
    return "\n".join(
        [
            "<section class='card'>",
            f"<h2>{escape(str(row['query_label']))}</h2>",
            "<div class='grid'>",
            "<div class='cell'>",
            "<h3>Query</h3>",
            f"<p>{escape(str(row['query_text']))}</p>",
            "</div>",
            "<div class='cell gold'>",
            f"<h3>Gold document ({int(row['gold_document_id'])})</h3>",
            f"<p class='title'>{escape(str(row['gold_title']))}</p>",
            f"<p>{escape(str(row['gold_abstract']))}</p>",
            "</div>",
            "<div class='cell top'>",
            f"<h3>Most frequent rank-1 document ({top_document_label})</h3>",
            (
                f"<p class='meta'>Ranked first by {int(row['top_document_votes'])} of {int(row['ranking_systems'])} systems; "
                f"mean reciprocal rank {row['top_document_mean_reciprocal_rank']:.3f}.{escape(tie_note)}</p>"
            ),
            f"<p class='title'>{escape(str(row['top_title']))}</p>",
            f"<p>{escape(str(row['top_abstract']))}</p>",
            f"<p class='meta'>Systems: {escape(str(row['top_document_voters']))}</p>",
            "</div>",
            "</div>",
            "</section>",
        ]
    )


def main(
    data_dir: str = typer.Option("data/SciWeb2026", help="Directory containing the final test JSON files and the collection."),
    submission_dir: str = typer.Option("data/submissions", help="Directory containing zipped system submissions."),
    output_dir: str = typer.Option("artifacts/reports", help="Directory where the report and CSV file are written."),
    language: str = typer.Option("en", "--language", help="Language code to include."),
) -> None:
    data_root = ROOT / data_dir
    submission_root = ROOT / submission_dir
    output_root = ROOT / output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading gold queries for '{language}' from {data_root}")
    gold_queries = _load_gold_queries(data_root, language)
    typer.echo(f"Loaded {len(gold_queries)} gold queries")

    typer.echo(f"Loading collection from {data_root / 'collection_data.json'}")
    collection = _load_collection(data_root)
    typer.echo(f"Loaded {len(collection)} documents")

    typer.echo(f"Building submission miss matrix from {submission_root}")
    miss_matrix = task_difficulty_analysis_submission_miss_matrix(submission_root, gold_queries, language)
    if miss_matrix.empty:
        raise typer.Exit(f"No matching submission files found for language '{language}'.")
    unsolved_ids = task_difficulty_analysis_universally_unsolved_query_ids(miss_matrix)
    typer.echo(f"Found {len(unsolved_ids)} universally unsolved queries across {len(miss_matrix.index)} systems")

    typer.echo("Loading full system rankings")
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
                "query_label": f"{language}:{int(row.query_id)}",
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
            typer.echo(f"Processed {position}/{len(unsolved)} unsolved queries")
    inspection = pd.DataFrame(records)
    inspection["top_document_id"] = inspection["top_document_id"].astype("Int64")

    csv_path = output_root / f"task_difficulty_analysis_unsolved_query_inspection_{language}.csv"
    inspection.to_csv(csv_path, index=False)
    typer.echo(f"Wrote task difficulty analysis unsolved query inspection CSV to {csv_path}")

    report_html = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Unsolved query inspection</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }",
            ".card { background: #fff; border: 1px solid #d0d0d0; border-radius: 6px; padding: 12px 16px; margin-bottom: 24px; }",
            ".grid { display: grid; grid-template-columns: 1fr 1.5fr 1.5fr; gap: 16px; }",
            ".cell { font-size: 13px; line-height: 1.45; }",
            ".cell h3 { font-size: 14px; margin: 0 0 8px 0; }",
            ".gold { border-left: 4px solid #1b7837; padding-left: 10px; }",
            ".top { border-left: 4px solid #b2182b; padding-left: 10px; }",
            ".title { font-weight: bold; }",
            ".meta { color: #555; font-size: 12px; }",
            "@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Task difficulty analysis: {escape(language.upper())} universally unsolved query inspection</h1>",
            (
                f"<p>{len(inspection)} queries for which none of the {len(miss_matrix.index)} submitted systems "
                "ranked the gold document in the top 5. For each query, the full query text, the gold document, and the "
                "document most frequently ranked first across systems are shown. Ties in rank-1 votes are broken by "
                "the higher mean reciprocal rank across all systems' full rankings.</p>"
            ),
            *[_render_card(row) for _, row in inspection.iterrows()],
            "</body>",
            "</html>",
        ]
    )
    report_path = output_root / f"task_difficulty_analysis_unsolved_query_inspection_{language}.html"
    report_path.write_text(report_html, encoding="utf-8")
    typer.echo(f"Wrote task difficulty analysis unsolved query inspection report to {report_path}")


if __name__ == "__main__":
    typer.run(main)
