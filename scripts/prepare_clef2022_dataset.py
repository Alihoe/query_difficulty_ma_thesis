# scripts/prepare_clef2022_dataset.py
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "_raw_clef2022"
SOURCE_VCLAIMS_ZIP = SOURCE_DIR / "vclaims_json.zip"
SOURCE_QUERIES_PATH = SOURCE_DIR / "CT2022-Task2A-EN-Dev-Test_Queries.tsv"
SOURCE_QRELS_PATH = SOURCE_DIR / "CT2022-Task2A-EN-Dev-Test_QRELs.tsv"
OUTPUT_DIR = ROOT / "data" / "CLEF2022"
OUTPUT_CORPUS_PATH = OUTPUT_DIR / "collection_data.json"
OUTPUT_QUERIES_PATH = OUTPUT_DIR / "final_en_test.json"


def convert_corpus() -> dict[str, int]:
    # vclaim_id is a string; the pipeline casts document ids to int throughout, so every
    # document gets a synthetic sequential integer id here, and queries below are remapped
    # through the same table to preserve the original vclaim_id -> document relationship exactly.
    vclaim_id_to_int_id: dict[str, int] = {}
    converted = []
    with zipfile.ZipFile(SOURCE_VCLAIMS_ZIP) as archive:
        names = [name for name in archive.namelist() if name.startswith("vclaims/") and name.endswith(".json")]
        print(f"Read {len(names)} vclaim documents from {SOURCE_VCLAIMS_ZIP}")
        for integer_id, name in enumerate(names):
            with archive.open(name) as handle:
                record = json.load(handle)
            vclaim_id_to_int_id[record["vclaim_id"]] = integer_id
            converted.append(
                {
                    "pubkey": integer_id,
                    "title": record.get("title", "") or "",
                    "abstract": " ".join(part for part in [record.get("subtitle", ""), record.get("vclaim", "")] if part),
                    "venue": "",
                    "authors": record.get("author", "") or "",
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} documents to {OUTPUT_CORPUS_PATH}")
    return vclaim_id_to_int_id


def convert_test_claims(vclaim_id_to_int_id: dict[str, int]) -> None:
    queries = pd.read_csv(SOURCE_QUERIES_PATH, sep="\t")
    qrels = pd.read_csv(SOURCE_QRELS_PATH, sep="\t", header=None, names=["iclaim_id", "iteration", "vclaim_id", "relevance"])
    print(f"Read {len(queries)} queries from {SOURCE_QUERIES_PATH}")
    print(f"Read {len(qrels)} qrels from {SOURCE_QRELS_PATH}")
    duplicate_query_gold = int(qrels["iclaim_id"].duplicated().sum())
    if duplicate_query_gold:
        raise ValueError(f"Expected single-gold queries; found {duplicate_query_gold} queries with more than one qrel row")
    merged = queries.merge(qrels[["iclaim_id", "vclaim_id"]], on="iclaim_id", how="inner")
    if len(merged) != len(queries):
        raise ValueError(f"Expected every query to have exactly one gold vclaim; got {len(merged)} of {len(queries)}")
    missing_gold = [vclaim_id for vclaim_id in merged["vclaim_id"] if vclaim_id not in vclaim_id_to_int_id]
    if missing_gold:
        raise ValueError(f"{len(missing_gold)} gold vclaim_id values are missing from the corpus mapping")
    converted = [
        {"index": row_index, "pubkey": vclaim_id_to_int_id[row.vclaim_id], "text": row.iclaim}
        for row_index, row in enumerate(merged.itertuples(index=False))
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QUERIES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} queries to {OUTPUT_QUERIES_PATH}")


def main() -> None:
    print("Preparing CLEF CheckThat! 2022 Task 2A (Previously Fact-Checked Claims) dataset for out-of-domain evaluation")
    vclaim_id_to_int_id = convert_corpus()
    convert_test_claims(vclaim_id_to_int_id)
    print("Done")


if __name__ == "__main__":
    main()
