# Difficult Query Annotations

Blind manual inspection of 150 English training queries of the CLEF CheckThat! 2026 *Source Retrieval for Scientific Web Claims* task. One annotator. The 150 queries were inspected in random order without knowing their group.

## Groups

| `group` | Description |
|---|---|
| `strongest` | The 50 training queries with the highest difficulty score |
| `missed` | 50 random training queries whose gold publication none of BM25, Qwen3-Embedding-0.6B, GTE-large-en-v1.5 and BGE-base-en-v1.5 ranks in the top 5 |
| `random` | 50 random remaining training queries |

## Columns

| Column | Description |
|---|---|
| `sheet_id` | Position in the blind inspection sheet |
| `query_id` | Query `index` in `en_train.json` |
| `split` | Data split of the query |
| `group` | Sampling group, see above |
| `difficulty_score` | Mean oriented percentile rank of the 15 difficulty features |
| `gold_document_id` | `pubkey` of the gold publication |
| `reference_document_id` | `pubkey` of the publication ranked first by most of the 14 retrievers |
| `reference_retrievers` | Number of retrievers that rank the reference publication first |
| `retrievers_total` | Number of retrievers considered |
| `plausible_competing_document` | 1 if the observation *plausible competing document* applies |
| `context_missing` | 1 if the observation *context missing* applies |
| `hard_but_solvable` | 1 if the observation *hard but solvable* applies |
| `easy` | 1 if the query seems *easy* |
| `duplicate_document` | 1 if the collection contains a second version of the gold publication that is ranked first instead |
| `other` | 1 for other observations |
| `comment` | Free-text note of the annotator |

A query can carry several observations.

## Texts

Post texts and publication titles and abstracts are not included. After placing the task data in `data/SciWeb2026/` (see [`../data/README.md`](../data/README.md)), run

```
python annotations/restore_inspection_sheet.py
```

This writes the full inspection sheet with texts to `artifacts/reports/spotting_difficult_queries/inspection_sheet.csv` and the group key to `inspection_key.csv`, which the table and figure scripts read.

## License

CC BY 4.0, see [`LICENSE`](LICENSE).
