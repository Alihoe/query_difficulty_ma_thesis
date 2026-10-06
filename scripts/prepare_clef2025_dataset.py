# scripts/prepare_clef2025_dataset.py
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "_raw_clef2025"
SOURCE_CORPUS_PATH = SOURCE_DIR / "subtask4b_collection_data.pkl"
SOURCE_TEST_PATH = SOURCE_DIR / "subtask4b_query_tweets_test_gold.tsv"
OUTPUT_DIR = ROOT / "data" / "CLEF2025"
OUTPUT_CORPUS_PATH = OUTPUT_DIR / "collection_data.json"
OUTPUT_QUERIES_PATH = OUTPUT_DIR / "final_en_test.json"


def convert_corpus() -> dict[str, int]:
    with SOURCE_CORPUS_PATH.open("rb") as handle:
        corpus = pickle.load(handle)
    print(f"Read {len(corpus)} documents from {SOURCE_CORPUS_PATH}")
    # cord_uid is a string; the pipeline casts document ids to int throughout, so every
    # document gets a synthetic sequential integer id here, and queries below are remapped
    # through the same table to preserve the original cord_uid -> document relationship exactly.
    cord_uid_to_int_id: dict[str, int] = {}
    converted = []
    for integer_id, row in enumerate(corpus.itertuples(index=False)):
        cord_uid_to_int_id[row.cord_uid] = integer_id
        converted.append(
            {
                "pubkey": integer_id,
                "title": "" if pd.isna(row.title) else row.title,
                "abstract": "" if pd.isna(row.abstract) else row.abstract,
                "venue": "" if pd.isna(row.journal) else row.journal,
                "authors": "" if pd.isna(row.authors) else row.authors,
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} documents to {OUTPUT_CORPUS_PATH}")
    return cord_uid_to_int_id


def convert_test_claims(cord_uid_to_int_id: dict[str, int]) -> None:
    test = pd.read_csv(SOURCE_TEST_PATH, sep="\t")
    print(f"Read {len(test)} claims from {SOURCE_TEST_PATH}")
    duplicate_post_ids = int(test["post_id"].duplicated().sum())
    if duplicate_post_ids:
        raise ValueError(f"Expected unique post_id values in the CLEF 2025 test split; found {duplicate_post_ids} duplicates")
    missing_gold = [cord_uid for cord_uid in test["cord_uid"] if cord_uid not in cord_uid_to_int_id]
    if missing_gold:
        raise ValueError(f"{len(missing_gold)} gold cord_uid values are missing from the corpus mapping")
    converted = [
        {"index": int(row.post_id), "pubkey": cord_uid_to_int_id[row.cord_uid], "text": row.tweet_text}
        for row in test.itertuples(index=False)
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QUERIES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} queries to {OUTPUT_QUERIES_PATH}")


def main() -> None:
    print("Preparing CLEF CheckThat! 2025 Subtask 4B dataset for out-of-domain evaluation (test split only)")
    cord_uid_to_int_id = convert_corpus()
    convert_test_claims(cord_uid_to_int_id)
    print("Done")


if __name__ == "__main__":
    main()
