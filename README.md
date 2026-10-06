# Adaptive Candidate-Set Size Selection and Retrieval Difficulty Analysis in Scientific Source Retrieval

Code to reproduce the tables and figures of the master thesis on the CLEF CheckThat! Lab 2026 task *Source Retrieval for Scientific Web Claims* (English).

## Annotations

[`annotations/difficult_query_annotations.csv`](annotations/difficult_query_annotations.csv) contains all 150 manually inspected training queries of the difficulty analysis (50 strongest difficulty scores, 50 missed by the pool retrievers, 50 random). The thesis shows only selected examples. See [`annotations/README.md`](annotations/README.md) for the columns.

## Setup

Python 3.11.

```
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
```

Datasets are not included. See [`data/README.md`](data/README.md).

## Reproduction

All scripts run without arguments and cache their results in `artifacts/`. Run them from the repository root in this order.

1. Out-of-distribution datasets

   ```
   python scripts/prepare_clef2022_dataset.py
   python scripts/prepare_clef2025_dataset.py
   python scripts/prepare_scifact_dataset.py
   python scripts/prepare_climatecheck_dataset.py
   ```

2. Full predictors (computes rankings and features on first use)

   ```
   python scripts/run_hit_at_1_predictor.py
   python scripts/run_in_top_max_depth_predictor.py
   python scripts/run_in_top_max_depth_sweep_predictor.py
   python scripts/run_calibrated_ordinal_predictor.py
   python scripts/run_tuned_qpp_predictor.py
   ```

3. Lightweight predictors

   ```
   python scripts/rebuild_lightweight_variants.py
   python scripts/run_hit_at_1_predictor_lightweight.py
   ```

4. Test evaluation

   ```
   python scripts/run_frozen_predictor_test_evaluation.py
   ```

5. Query-specific reranking, confidence routing and out-of-distribution evaluation

   ```
   python scripts/run_all_multistep_retrieval.py
   python scripts/run_confidence_routed_lightweight_experiments.py
   python scripts/run_all_ood_hit1_comparison.py
   ```

6. Cost of the Hit-at-1 gate

   ```
   python scripts/verify_fast_hit_at_1_gate.py
   python scripts/run_hit_at_1_gate_cost_overnight.py
   python scripts/run_hit_at_1_gate_setup_cost.py
   ```

7. Difficulty analysis and ablations

   ```
   python scripts/generate_retriever_difficulty_correlation_analysis.py
   python scripts/run_qpp_predictor_log_rank_qwen.py
   python scripts/run_hit_at_1_seed_sensitivity_qwen.py
   ```

8. Tables and figures (written to `latex/tables/` and `latex/figures/`)

   ```
   python latex/scripts/spotting_feature_selection_and_sheet.py
   python annotations/restore_inspection_sheet.py
   ```

   Then run every script in `latex/scripts/` whose name starts with `table_` or `figure_`, then

   ```
   python latex/scripts/write_lightweight_tables.py
   python latex/scripts/write_test_performance_tables.py
   ```

   `write_test_performance_tables.py` must run last because it replaces some tables written by earlier scripts with the versions used in the thesis.

Reranking (step 5) and feature computation are expensive. All experiments of the thesis were run on CPU.

## License

Code: MIT, see [`LICENSE`](LICENSE). Annotations: CC BY 4.0, see [`annotations/LICENSE`](annotations/LICENSE).
