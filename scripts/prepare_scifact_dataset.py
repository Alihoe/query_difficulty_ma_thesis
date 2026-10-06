# scripts/prepare_scifact_dataset.py
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ARCHIVE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
RAW_DIR = ROOT / "data" / "_raw_scifact"
RAW_ARCHIVE_PATH = RAW_DIR / "data.tar.gz"
RAW_DATA_DIR = RAW_DIR / "data"
OUTPUT_DIR = ROOT / "data" / "SciFact"
OUTPUT_CORPUS_PATH = OUTPUT_DIR / "collection_data.json"
OUTPUT_QUERIES_PATH = OUTPUT_DIR / "final_en_test.json"


def ensure_raw_data() -> None:
    if (
        (RAW_DATA_DIR / "corpus.jsonl").exists()
        and (RAW_DATA_DIR / "claims_train.jsonl").exists()
        and (RAW_DATA_DIR / "claims_dev.jsonl").exists()
    ):
        print(f"Raw SciFact release already present at {RAW_DATA_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading canonical SciFact release from {RAW_ARCHIVE_URL}")
    urllib.request.urlretrieve(RAW_ARCHIVE_URL, RAW_ARCHIVE_PATH)
    archive_size_mb = RAW_ARCHIVE_PATH.stat().st_size / (1024 * 1024)
    print(f"Downloaded {archive_size_mb:.1f} MB to {RAW_ARCHIVE_PATH}")
    print(f"Extracting archive to {RAW_DIR}")
    with tarfile.open(RAW_ARCHIVE_PATH, "r:gz") as archive:
        archive.extractall(RAW_DIR)
    print(f"Extracted raw SciFact release to {RAW_DATA_DIR}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_corpus() -> int:
    corpus_rows = read_jsonl(RAW_DATA_DIR / "corpus.jsonl")
    print(f"Read {len(corpus_rows)} documents from corpus.jsonl")
    converted = [
        {
            "pubkey": row["doc_id"],
            "title": row["title"],
            "abstract": " ".join(row["abstract"]),
            "venue": "",
            "authors": "",
        }
        for row in corpus_rows
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} documents to {OUTPUT_CORPUS_PATH}")
    return len(converted)


def convert_query_claims() -> None:
    # train+dev combined: none of our predictor models are ever fit on SciFact data (they are
    # trained exclusively on CLEF), so SciFact's own train/dev boundary carries no meaning for us.
    # test is excluded because it is genuinely unlabeled (leaderboard-only, no evidence field at all).
    converted: list[dict] = []
    no_evidence_count = 0
    multi_gold_count = 0
    for split_name in ("claims_train.jsonl", "claims_dev.jsonl"):
        claim_rows = read_jsonl(RAW_DATA_DIR / split_name)
        print(f"Read {len(claim_rows)} claims from {split_name}")
        for row in claim_rows:
            evidence = row.get("evidence", {})
            if len(evidence) == 0:
                no_evidence_count += 1
                continue
            if len(evidence) > 1:
                multi_gold_count += 1
                continue
            gold_doc_id = int(next(iter(evidence.keys())))
            converted.append({"index": row["id"], "pubkey": gold_doc_id, "text": row["claim"]})
    print(f"Excluded {no_evidence_count} claims with no evidence document (NOT ENOUGH INFO)")
    print(f"Excluded {multi_gold_count} claims with more than one gold evidence document")
    print(f"Kept {len(converted)} single-gold claims (train+dev combined)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QUERIES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} queries to {OUTPUT_QUERIES_PATH}")


def main() -> None:
    print("Preparing SciFact dataset for out-of-domain evaluation")
    ensure_raw_data()
    convert_corpus()
    convert_query_claims()
    print("Done")


if __name__ == "__main__":
    main()
