from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = ROOT / "annotations" / "difficult_query_annotations.csv"
QUERIES_PATH = ROOT / "data" / "SciWeb2026" / "en_train.json"
COLLECTION_PATH = ROOT / "data" / "SciWeb2026" / "collection_data.json"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "spotting_difficult_queries"
SHEET_PATH = OUTPUT_DIR / "inspection_sheet.csv"
KEY_PATH = OUTPUT_DIR / "inspection_key.csv"
GROUP_CODES = {"strongest": "flagged", "missed": "control", "random": "random"}
OBSERVATIONS = ["plausible_competing_document", "context_missing", "hard_but_solvable", "easy", "duplicate_document", "other"]


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS_PATH)
    print(f"Read {len(annotations)} annotations from {ANNOTATIONS_PATH}")
    queries = {int(row["index"]): row["text"] for row in json.loads(QUERIES_PATH.read_text(encoding="utf-8"))}
    print(f"Read {len(queries)} queries from {QUERIES_PATH}")
    documents = {int(row["pubkey"]): row for row in json.loads(COLLECTION_PATH.read_text(encoding="utf-8"))}
    print(f"Read {len(documents)} documents from {COLLECTION_PATH}")
    missing = sorted(set(annotations["query_id"]) - set(queries))
    if missing:
        raise ValueError(f"{len(missing)} annotated queries are missing from {QUERIES_PATH}: {missing}")
    sheet = pd.DataFrame(
        {
            "sheet_id": annotations["sheet_id"],
            "query_id": annotations["query_id"],
            "query_text": annotations["query_id"].map(queries),
            "gold_title": annotations["gold_document_id"].map(lambda value: documents[int(value)]["title"]),
            "gold_abstract": annotations["gold_document_id"].map(lambda value: documents[int(value)]["abstract"]),
            "reference_document_id": annotations["reference_document_id"],
            "reference_title": annotations["reference_document_id"].map(lambda value: documents[int(value)]["title"]),
            "reference_abstract": annotations["reference_document_id"].map(lambda value: documents[int(value)]["abstract"]),
            "reference_retrievers": annotations["reference_retrievers"],
            "retrievers_total": annotations["retrievers_total"],
        }
    )
    for column in OBSERVATIONS:
        sheet[column] = annotations[column].map({1: "1", 0: ""})
    sheet["comment"] = annotations["comment"].fillna("")
    key = pd.DataFrame(
        {
            "sheet_id": annotations["sheet_id"],
            "query_id": annotations["query_id"],
            "group": annotations["group"].map(GROUP_CODES),
            "score": annotations["difficulty_score"],
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig")
    key.to_csv(KEY_PATH, index=False)
    print(f"Wrote {len(sheet)} annotated queries to {SHEET_PATH}")
    print(f"Wrote {len(key)} group assignments to {KEY_PATH}")
    print(key["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
