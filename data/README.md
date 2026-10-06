# Data

No dataset is included in this repository. Place the files as listed below.

## CLEF CheckThat! 2026, Source Retrieval for Scientific Web Claims

Obtain the task data from the CheckThat! Lab 2026 organizers and place it in `data/SciWeb2026/`:

```
data/SciWeb2026/collection_data.json
data/SciWeb2026/en_train.json
data/SciWeb2026/en_dev.json
data/SciWeb2026/final_en_test.json
```

`final_en_test.json` must contain the gold labels (`pubkey`). The German and French files of the task are optional and not used in the thesis.

## Submitted runs

The cross-system difficulty analysis uses the official English test submissions of all participating teams. Obtain them from the task organizers and place the 37 zip files unchanged in `data/submissions/`.

## Out-of-distribution datasets

| Dataset | Source | Preparation |
|---|---|---|
| SciFact | Downloaded automatically | `python scripts/prepare_scifact_dataset.py` |
| ClimateCheck | Downloaded automatically from Hugging Face (`rabuahmad/climatecheck`, `rabuahmad/climatecheck_publications_corpus`) | `python scripts/prepare_climatecheck_dataset.py` |
| CLEF 2022 CheckThat! Task 2A (English) | CheckThat! Lab 2022. Place `vclaims_json.zip`, `CT2022-Task2A-EN-Dev-Test_Queries.tsv` and `CT2022-Task2A-EN-Dev-Test_QRELs.tsv` in `data/_raw_clef2022/` | `python scripts/prepare_clef2022_dataset.py` |
| CLEF 2025 CheckThat! Subtask 4b | CheckThat! Lab 2025. Place `subtask4b_collection_data.pkl` and `subtask4b_query_tweets_test_gold.tsv` in `data/_raw_clef2025/` | `python scripts/prepare_clef2025_dataset.py` |

The preparation scripts write the converted files to `data/SciFact/`, `data/ClimateCheck/`, `data/CLEF2022/` and `data/CLEF2025/`.

## Models

All retrieval and reranking models are downloaded from Hugging Face on first use. The model names are listed in `config/default.yaml` and `src/query_difficulty/multistep.py`.
