# scripts/prepare_climatecheck_dataset.py
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_CORPUS_SIZE = 10_000
RESTRICTED_CORPUS_RANDOM_SEED = 13
ABSTRACT_MAX_CHARACTERS_P99 = 6_000
CORPUS_PARQUET_URL = (
    "https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
CLAIMS_TEST_PARQUET_URL = (
    "https://huggingface.co/datasets/rabuahmad/climatecheck/"
    "resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)
CLAIMS_TRAIN_PARQUET_URL = (
    "https://huggingface.co/datasets/rabuahmad/climatecheck/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
RAW_DIR = ROOT / "data" / "_raw_climatecheck"
RAW_CORPUS_PATH = RAW_DIR / "publications_corpus.parquet"
RAW_CLAIMS_TEST_PATH = RAW_DIR / "claims_test.parquet"
RAW_CLAIMS_TRAIN_PATH = RAW_DIR / "claims_train.parquet"
OUTPUT_DIR = ROOT / "data" / "ClimateCheck"
OUTPUT_CORPUS_PATH = OUTPUT_DIR / "collection_data.json"
OUTPUT_QUERIES_PATH = OUTPUT_DIR / "final_en_test.json"
OUTPUT_GOLD_SETS_PATH = OUTPUT_DIR / "gold_sets_en_test.json"
EVIDENTIARY_ANNOTATIONS = {"Supports", "Refutes"}


def ensure_raw_file(url: str, path: Path) -> None:
    if path.exists():
        print(f"Raw file already present at {path}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB to {path}")


def convert_corpus() -> int:
    with OUTPUT_GOLD_SETS_PATH.open("r", encoding="utf-8") as handle:
        gold_sets: dict[str, list[int]] = json.load(handle)
    gold_ids = {int(abstract_id) for ids in gold_sets.values() for abstract_id in ids}
    print(
        f"Restricting the ClimateCheck corpus to a CLEF-scale candidate pool "
        f"(full corpus embedding is infeasible on this machine: measured throughput extrapolates to "
        f"multiple weeks for the full 394,269-abstract corpus)"
    )
    print(f"{len(gold_ids)} unique gold abstracts across {len(gold_sets)} evidentiary claims are always included")

    corpus = pd.read_parquet(RAW_CORPUS_PATH, columns=["abstract_id", "title", "abstract"])
    print(f"Read {len(corpus)} abstracts from the full ClimateCheck publications corpus")
    is_gold = corpus["abstract_id"].astype(int).isin(gold_ids)
    gold_rows = corpus.loc[is_gold]
    negative_pool = corpus.loc[~is_gold]
    negative_sample_size = max(0, RESTRICTED_CORPUS_SIZE - len(gold_rows))
    rng = np.random.default_rng(RESTRICTED_CORPUS_RANDOM_SEED)
    sampled_negative_positions = rng.choice(len(negative_pool), size=negative_sample_size, replace=False)
    negative_rows = negative_pool.iloc[sampled_negative_positions]
    print(
        f"Sampled {len(negative_rows)} non-gold abstracts (seed={RESTRICTED_CORPUS_RANDOM_SEED}, matching "
        f"runtime.random_seed in the project configs) to reach a target pool of {RESTRICTED_CORPUS_SIZE} documents"
    )
    restricted_corpus = pd.concat([gold_rows, negative_rows], ignore_index=True)
    print(f"Restricted corpus size: {len(restricted_corpus)} documents (matches CLEF's own 10,000-document scale)")

    abstract_lengths = restricted_corpus["abstract"].fillna("").map(len)
    truncated_count = int((abstract_lengths > ABSTRACT_MAX_CHARACTERS_P99).sum())
    print(
        f"Truncating {truncated_count} abstracts longer than {ABSTRACT_MAX_CHARACTERS_P99} characters "
        f"(p99 of this corpus's own abstract-length distribution; the longest raw abstract is "
        f"{int(abstract_lengths.max())} characters, almost certainly a full-text-in-abstract-field "
        "scraping artifact rather than a genuine abstract)"
    )

    converted = [
        {
            "pubkey": int(row.abstract_id),
            "title": "" if pd.isna(row.title) else row.title,
            "abstract": ("" if pd.isna(row.abstract) else row.abstract)[:ABSTRACT_MAX_CHARACTERS_P99],
            "venue": "",
            "authors": "",
        }
        for row in restricted_corpus.itertuples(index=False)
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CORPUS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    print(f"Wrote {len(converted)} documents to {OUTPUT_CORPUS_PATH}")
    return len(converted)


def convert_query_claims() -> None:
    # train+test combined: none of our predictor models are ever fit on ClimateCheck data (they are
    # trained exclusively on CLEF), so ClimateCheck's own train/test boundary carries no meaning for us.
    # Verified disjoint: 0 claim_id overlap and 0 claim-text overlap between the two splits.
    test_pairs = pd.read_parquet(RAW_CLAIMS_TEST_PATH, columns=["claim_id", "claim", "abstract_id", "annotation"])
    train_pairs = pd.read_parquet(RAW_CLAIMS_TRAIN_PATH, columns=["claim_id", "claim", "abstract_id", "annotation"])
    pairs = pd.concat([test_pairs, train_pairs], ignore_index=True)
    print(f"Read {len(test_pairs)} test pairs + {len(train_pairs)} train pairs = {len(pairs)} claim-abstract pairs")
    print(f"Annotation distribution: {pairs['annotation'].value_counts().to_dict()}")
    evidentiary_pairs = pairs[pairs["annotation"].isin(EVIDENTIARY_ANNOTATIONS)]
    print(
        f"Kept {len(evidentiary_pairs)} pairs annotated Supports/Refutes "
        f"(dropped {len(pairs) - len(evidentiary_pairs)} Not Enough Information pairs)"
    )
    gold_sets = evidentiary_pairs.groupby("claim_id")["abstract_id"].apply(lambda ids: sorted(set(int(i) for i in ids)))
    claim_texts = evidentiary_pairs.drop_duplicates(subset="claim_id").set_index("claim_id")["claim"]
    print(f"Kept {len(gold_sets)} claims with at least one evidentiary gold abstract")
    gold_count_distribution = gold_sets.map(len).value_counts().sort_index().to_dict()
    print(f"Gold-set size distribution (num_gold_docs -> num_claims): {gold_count_distribution}")

    queries = [
        {"index": int(claim_id), "pubkey": int(gold_ids[0]), "text": claim_texts[claim_id]}
        for claim_id, gold_ids in gold_sets.items()
    ]
    gold_sets_payload = {int(claim_id): gold_ids for claim_id, gold_ids in gold_sets.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QUERIES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(queries, handle, ensure_ascii=False)
    print(f"Wrote {len(queries)} queries to {OUTPUT_QUERIES_PATH}")
    print("Each query's pubkey is only a representative gold document (lowest abstract id in its gold set),")
    print("required by the existing single-gold DatasetManager schema; it is not the evaluation signal.")
    with OUTPUT_GOLD_SETS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(gold_sets_payload, handle, ensure_ascii=False)
    print(f"Wrote full gold sets for {len(gold_sets_payload)} claims to {OUTPUT_GOLD_SETS_PATH}")
    print("This side file (index -> full list of gold abstract ids) is what the later evaluation step uses")
    print("to score Hit@1 as set membership, since ClimateCheck claims are frequently multi-gold.")


def main() -> None:
    print("Preparing ClimateCheck dataset for out-of-domain evaluation")
    ensure_raw_file(CORPUS_PARQUET_URL, RAW_CORPUS_PATH)
    ensure_raw_file(CLAIMS_TEST_PARQUET_URL, RAW_CLAIMS_TEST_PATH)
    ensure_raw_file(CLAIMS_TRAIN_PARQUET_URL, RAW_CLAIMS_TRAIN_PATH)
    convert_query_claims()
    convert_corpus()
    print("Done")


if __name__ == "__main__":
    main()
