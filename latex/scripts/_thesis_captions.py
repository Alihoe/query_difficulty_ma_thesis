from __future__ import annotations

import re

BS = "\\"
BM25 = BS + "textit{BM25}"
QWEN = BS + "textit{Qwen3-Embedding-0.6B}"
MINUS = "$-$"
DEV = "on the English development set"
TEST = "on the English test set"
TRAIN = "on the English training set"
TASK = BS + "textit{Source Retrieval for Scientific Web Claims}"
DEPTHS = "$k " + BS + "in " + BS + "{5,10,20,30,50,60,100,120,200,1000" + BS + "}$"
RERANKERS = {
    "minilm": BS + "textit{ms-marco-MiniLM-L-6-v2}",
    "jina": BS + "textit{jina-reranker-v2-base-multilingual}",
    "bge": BS + "textit{bge-reranker-v2-m3}",
}
RETRIEVERS = {"bm25": BM25, "qwen": QWEN}
LABEL_SUFFIX = {"bm25": "bm25", "qwen": "qwen3-embedding-0-6b"}
LIGHTWEIGHT_SUFFIX = {"bm25": "bm25", "qwen": "qwen3_embedding_0_6b"}
RERANKING_SUFFIX = {"bm25": "bm25", "qwen": "qwen3_embedding_0_6b"}
CORRELATION_SUFFIX = {"bm25": "bm25", "qwen": "qwen"}


def _direction(outcome: str) -> str:
    return f"Direction indicates whether higher feature values are associated with {outcome} (+) or not ({MINUS})."


HIT_DIRECTION = _direction("the relevant publication being ranked first")
DEPTH_DIRECTION = _direction("the relevant publication being ranked within the top 1000")
QPP_DIRECTION = (
    "Direction indicates whether higher feature values are associated with a higher (+) or lower "
    f"({MINUS}) reciprocal rank of the relevant publication."
)
SIGNED_PER_K = (
    "signed by direction (+ if higher feature values are associated with the relevant publication being ranked "
    f"within the top $k$, {MINUS} otherwise)"
)
RERANK_MARKS = (
    "Bold marks the best MRR@5 per $k$, underlining the best MRR@5 overall, and a bold policy name the policy that "
    "reaches it. $" + BS + "wedge$ denotes that a query is only reranked if all combined predictors select it."
)
CORRELATION_MEASURES = (
    "measured as Kendall's $" + BS + "tau_b$ and Spearman's $" + BS + "rho$ with the reciprocal rank of the relevant "
    "publication."
)
SYMBOL_NOTE = "The symbols are explained below the table."
UNSOLVED_GROUPS = (
    f"Median feature values of {QWEN} for the 430 English test queries that no submitted system solves within the top 5 "
    "(U) and the 5,646 remaining queries (R), with Cliff's $" + BS + "delta$."
)


def _appendix(label: str, plural: bool = False) -> str:
    return f"The complete {'tables are' if plural else 'table is'} given in Appendix {BS}ref{{{label}}}."


def _columns_as(label: str) -> str:
    return f"Columns as in Table {BS}ref{{{label}}}."


def _correlation_matrix(predictor: str) -> str:
    return (
        "Spearman rank correlation ($" + BS + f"rho$) between the {predictor} feature importance rankings of {QWEN}, "
        f"{BS}textit{{GTE-large-en-v1.5}}, and {BS}textit{{E5-large-v2}}. Bold marks the highest correlation in each row."
    )


CAPTIONS: dict[str, str] = {
    "tab:dataset_statistics":
        "Number of English queries per data split and number of scientific publications in the document collection of "
        f"the {TASK} task.",
    "tab:submission_info_en_official":
        f"Retrieval performance of the documented systems {TEST}. R@$k$ is the proportion of queries whose gold "
        "publication is ranked within the top $k$. Solved@5 and Missed@5 count the queries with and without the gold "
        "publication in the top 5, and Unique Solved@5 counts the queries that no other system solves within the top 5. "
        "Bold marks the two systems with the most uniquely solved queries.",
    "tab:qwen_deep_recall_en_dev":
        f"Recall@$k$ of {QWEN} {DEV} for increasing retrieval depths $k$. Bold marks the maximum depth of 1000 used in "
        "this work.",
    "tab:candidate_depths":
        f"Number of candidates passed to neural and LLM rerankers by the systems participating in the {TASK} task, per "
        "language.",
    "tab:system_architectures":
        f"Retrieval and reranking configurations of the documented systems of the {TASK} task.",
    "tab:gold_free_lexical_pre_retrieval_features":
        "Gold-independent lexical pre-retrieval features used for query performance prediction.",
    "tab:gold_free_lexical_post_retrieval_features":
        "Gold-independent lexical top-ranked-document and top-$k$ document-set features used for query performance "
        f"prediction. The features are computed for {BM25}, TF-IDF, exact-keyword matching, and Jaccard similarity, and "
        f"depth-dependent features for {DEPTHS}.",
    "tab:gold_free_lexical_post_retrieval_score_features":
        "Gold-independent lexical retrieval-score features used for query performance prediction. The features are "
        f"computed for {BM25}, TF-IDF, exact-keyword matching, and Jaccard similarity, and depth-dependent features for "
        f"{DEPTHS}.",
    "tab:retriever_test_mrr_en":
        f"MRR@5 of all considered retrievers {TEST}, in descending order.",
    "tab:cluster_overview":
        "Number of clusters and minimum, mean, and maximum cluster size (number of documents) found by HDBSCAN and "
        "spherical $k$-means in the document collection for each semantic retriever.",
    "tab:gold_free_semantic_pre_retrieval_features":
        "Gold-independent semantic pre-retrieval features used for query performance prediction. The features are "
        "computed separately for each semantic retriever.",
    "tab:gold_free_semantic_features_score":
        "Gold-independent semantic score features used for query performance prediction. The features are computed "
        f"separately for each semantic retriever, and depth-dependent features for {DEPTHS}.",
    "tab:gold_free_semantic_features_geometry":
        "Gold-independent semantic ranking-geometry features used for query performance prediction. The features are "
        f"computed separately for each semantic retriever, and depth-dependent features for {DEPTHS}.",
    "tab:gold_free_cluster_features":
        "Gold-independent cluster features used for query performance prediction. The features are computed separately "
        f"for each semantic retriever and clustering method for {DEPTHS}.",
    "tab:gold_free_agreement_features":
        "Gold-independent retriever-agreement features used for query performance prediction. For each query and "
        "semantic retriever $r$, agreement is computed with the other included retrievers.",
    "tab:gold_free_perturbation_features":
        "Gold-independent perturbation-based robustness feature used for query performance prediction. The feature is "
        "computed separately for each semantic retriever and candidate-set size $k$.",
    "tab:gold_aware_lexical_features":
        "Gold-aware lexical features used for the retriever-specific difficulty analysis.",
    "tab:gold_aware_semantic_features":
        "Gold-aware semantic score and ranking-geometry features used for the retriever-specific difficulty analysis.",
    "tab:gold_aware_semantic_cluster_perturbation_features":
        "Gold-aware cluster and perturbation features used for the retriever-specific difficulty analysis.",
    "tab:hit_at_1_test_performance_en":
        f"Prediction performance of the full Hit-at-1 predictors {TEST}. Retrieval MRR@5 is the first-stage performance "
        f"of the respective retriever and {BS}#Feat. the number of features. Bold marks the best value per column.",
    "tab:hit_at_1_lightweight_test_performance_en":
        f"Prediction performance of the full and lightweight Hit-at-1 predictors for {BM25} and {QWEN} {TEST}. "
        f"{BS}#Feat. is the number of features. Bold marks the better value of the full and lightweight variant.",
    "tab:in_top_max_depth_test_performance_en":
        f"Prediction performance of the full Maximum Depth predictors {TEST} with a maximum depth of 1000. Retrieval "
        f"MRR@5 is the first-stage performance of the respective retriever and {BS}#Feat. the number of features. Bold "
        "marks the best value per column.",
    "tab:in_top_max_depth_lightweight_test_performance_en":
        f"Prediction performance of the full and lightweight Maximum Depth predictors for {BM25} and {QWEN} {TEST} with "
        f"a maximum depth of 1000. {BS}#Feat. is the number of features. Bold marks the better value of the full and "
        "lightweight variant.",
    "tab:in_top_max_depth_confusion_breakdown_en":
        f"Confusion breakdown of the full Maximum Depth predictors {TEST} with a maximum depth of 1000. Actual gives the "
        "number of queries whose relevant publication is not (Not-k) or is (In-k) ranked within the top 1000. Correct "
        "and Incorrect count the queries predicted to be outside the top 1000 that actually are outside or inside it, "
        "and Missed counts the queries outside the top 1000 that are predicted to be inside.",
    "tab:calibrated_ordinal_test_performance_en":
        f"Prediction performance of the full ordinal candidate-set-size predictors {TEST}. "
        "Recall is the proportion of queries whose selected candidate set contains the relevant publication, where "
        "queries whose relevant publication is not ranked within the top 1000 count as correct if they are predicted "
        "as unreachable. Exact accuracy, QWK, and MAE are computed over the queries whose relevant publication "
        f"is ranked within the top 1000, and {BS}#Feat. gives the range of feature counts across the binary classifiers. "
        "Bold marks the best value per column, where lower is better for MAE.",
    "tab:calibrated_ordinal_lightweight_test_performance_en":
        f"Prediction performance of the full and lightweight ordinal candidate-set-size predictors for {BM25} and {QWEN} "
        f"{TEST}. Columns as in Table {BS}ref{{tab:calibrated_ordinal_test_performance_en}}. Bold marks the better value "
        "of the full and lightweight variant, where lower is better for MAE.",
    "tab:calibrated_ordinal_candidate_size_breakdown_en":
        "Predicted and actual smallest sufficient candidate-set sizes of the full ordinal candidate-set-size predictors "
        f"{TEST}, with per-class F1 scores. Predicted and Actual give the number of queries per candidate-set size, and "
        "Unreachable covers the queries whose relevant publication is not ranked within the top 1000.",
    "tab:tuned_qpp_test_performance_en":
        f"Prediction performance of the full QPP-based candidate-set-size predictors {TEST}. "
        "Recall is the proportion of queries whose selected candidate set contains the relevant publication, where "
        "queries whose relevant publication is not ranked within the top 1000 count as correct if they are predicted "
        "as unreachable. Failure is the proportion of queries predicted as unreachable, and Mean and Median "
        f"describe the predicted candidate-set sizes of the remaining queries. {BS}#Feat. is the number of features. "
        "Bold marks the best value per column, where lower is better for Mean, Median, and Failure.",
    "tab:tuned_qpp_lightweight_test_performance_en":
        f"Prediction performance of the full and lightweight QPP-based candidate-set-size predictors for {BM25} and "
        f"{QWEN} {TEST}. Columns as in Table {BS}ref{{tab:tuned_qpp_test_performance_en}}. Bold marks the better value of "
        "the full and lightweight variant, where lower is better for Mean, Median, and Failure.",
    "tab:tuned_qpp_candidate_size_breakdown_en":
        "Predicted and actual smallest sufficient candidate-set sizes of the full QPP-based candidate-set-size predictors "
        f"{TEST}, with per-class F1 scores. Predicted and Actual give the number of queries per candidate-set size, and "
        "Unreachable covers the queries whose relevant publication is not ranked within the top 1000. Predicted sizes "
        "are assigned to the next larger evaluated candidate-set size.",
    "tab:in_top_k_sweep_test_performance_en":
        f"Prediction performance (MCC) of the full binary candidate-set-size predictors {TEST} for each candidate-set "
        "size $k$. Bold marks the best value per $k$.",
    "tab:in_top_k_sweep_lightweight_test_performance_en":
        f"Prediction performance (MCC) of the full and lightweight binary candidate-set-size predictors for {BM25} and "
        f"{QWEN} {TEST} for each candidate-set size $k$. {BS}#Feat. is the number of features. Bold marks the better "
        "value of the full and lightweight variant.",
    "tab:candidate_size_confusion_qwen3_embedding_0_6b_en":
        f"Prediction performance of the full Hit-at-1, Maximum Depth, binary, ordinal, and QPP-based predictors for {QWEN} "
        f"{TEST}, grouped by candidate-set size $k$. BalAcc is the balanced accuracy. Bold marks the best value per $k$, "
        "underlining the metric each predictor is tuned for, and a box the best value of each predictor per metric across "
        "candidate-set sizes.",
    "tab:hit_at_1_lightweight_feature_selection_steps_en":
        "Number of features after each step of the lightweight feature selection for the Hit-at-1 predictor and the "
        f"resulting $F_{{0.5}}$ {TEST}. Full is the number of features the selection starts from, which excludes the "
        f"multi-retriever consensus features. The steps are an importance filter, the removal of correlated features at $|"
        + BS + "rho| " + BS + "geq 0.80$ and $|" + BS + "rho| " + BS + "geq 0.70$ on the English training set, and "
        "recursive feature elimination (RFE) evaluated on the English development set. Bold marks the better $F_{0.5}$.",
    "tab:qpp_log_rank_candidate_size_calibration_en":
        "Candidate-set-size prediction of the QPP-based predictor with a log-rank regression target ($1/" + BS
        + "log_2(" + BS + f"mathrm{{rank}}+1)$) for {QWEN} {DEV}. $N$ is the number of queries per smallest sufficient "
        "candidate-set size, Exact Match the proportion of queries whose predicted size equals it, Median Pred. the "
        "median predicted size, and Coverage the proportion of queries whose predicted candidate set contains the "
        "relevant publication.",
    "tab:hit_at_1_importances_correlation": _correlation_matrix("Hit-at-1"),
    "tab:in_top_max_depth_importances_correlation": _correlation_matrix("Maximum Depth"),
    "tab:calibrated_ordinal_importances_correlation": _correlation_matrix("averaged ordinal candidate-set-size"),
    "tab:tuned_qpp_importances_score_correlation": _correlation_matrix("QPP regressor"),
    "tab:in_top_k_sweep_importances_correlation": _correlation_matrix("binary candidate-set-size"),
    "tab:multistage_reranking_oracle_comparison_qwen3_embedding_0_6b_en":
        f"MRR@5 of {QWEN} {TEST} when the candidate set selected by each reranking policy is reranked perfectly. "
        + RERANK_MARKS,
    "tab:multistage_reranking_best_adaptive_bootstrap_en":
        f"Post-hoc paired bootstrap analysis of the selected adaptive reranking configurations {TEST}. $" + BS
        + "Delta$ is the MRR@5 difference between adaptive and full reranking with its 95" + BS + "% confidence "
        "interval, and Saved is the proportion of reranker calls saved. $" + BS + "wedge$ denotes that a query is only "
        "reranked if all combined predictors select it.",
    "tab:hit_at_1_gate_online_cost_qwen3_embedding_0_6b_jina_en":
        f"Online runtime of full and adaptive Hit-at-1 reranking with {QWEN} and {RERANKERS['jina']} at $k=30$ on a "
        "stratified subset of 100 English test queries. Values are means $" + BS + "pm$ standard deviations over five "
        "runs, and Workload total is the runtime for all 100 queries.",
    "tab:hit_at_1_gate_cost_robustness_qwen3_embedding_0_6b_jina_en":
        "Runtime of full and adaptive Hit-at-1 reranking across five subsets of English test queries. Values are the "
        "mean, standard deviation, minimum, and maximum across the subsets.",
    "tab:hit_at_1_gate_setup_cost_qwen3_embedding_0_6b_jina_en":
        f"One-time setup cost of the Hit-at-1 predictor for {QWEN} before deployment.",
    "tab:ood_dataset_statistics":
        "Number of queries per data split, number of considered queries, and number of documents of the "
        "out-of-distribution datasets.",
    "tab:ood_hit1_comparison_en":
        "Zero-shot transfer of the lightweight Hit-at-1 predictor to the out-of-distribution datasets, with "
        f"{QWEN} for retrieval and {RERANKERS['jina']} for reranking. $" + BS + "Delta$ is the MRR@5 difference "
        "between Hit-at-1 and full reranking with its 95" + BS + "% paired bootstrap confidence interval. Bold marks the "
        "better of full and Hit-at-1 reranking.",
    "tab:confidence_routed_shapley_values_en":
        f"Shapley value of each retriever's contribution to the confidence-routed MRR@5 {TEST}, with its own MRR@5 and "
        "its worst-case contribution to a single retriever combination. The best-case contribution equals the own "
        "MRR@5 for every retriever and is therefore not shown. Bold marks the best value per column.",
    "tab:confidence_routed_best_configurations_en":
        f"Best confidence-routed MRR@5 and MRR@5 with perfect routing for each pool size {TEST}, found by exhaustive "
        "search over all combinations of the 14 candidate retrievers. Perfect routing selects the best retriever for "
        "each query, a dot marks the retrievers in the best combination of each size, and bold marks the best routed "
        "result.",
    "tab:hardness_by_solution_en":
        "Distribution of the English test queries by the proportion and number of the 37 submitted systems that "
        "retrieve the gold publication within the top 5.",
    "tab:distributions_qwen_unsolved_large_effect_en_test":
        f"{UNSOLVED_GROUPS} Only features with a large difference ($|" + BS + "delta| " + BS + "geq 0.474$) are shown. "
        + _appendix("appendix:unsolved-vs-rest"),
    "tab:distributions_qwen_unsolved_medium_effect_en_test":
        f"{UNSOLVED_GROUPS} All features with at least a medium difference ($|" + BS + "delta| " + BS + "geq 0.33$) "
        "are shown.",
    "tab:spotting_feature_selection":
        f"Transfer of the lightweight {QWEN} features to cross-system difficult queries. Families is the number of "
        "predictor families that keep the feature and Importance its average share of the training importance. Train "
        "effect gives the direction and Kendall's $" + BS + f"tau_b$ with the reciprocal rank {TRAIN}, and Test effect "
        "the direction and Cliff's $" + BS + "delta$ between unsolved and remaining English test queries. Selected marks "
        "features with the same direction and at least a medium difference ($|" + BS + "delta| " + BS + "geq 0.33$).",
    "tab:spotting_inspection":
        "Observations from the blind manual inspection of the ten flagged training queries with the strongest "
        "difficulty scores (Strongest) and ten randomly selected training queries (Random). A query can carry several "
        "observations.",
}

for key, name in RETRIEVERS.items():
    label = LABEL_SUFFIX[key]
    light = LIGHTWEIGHT_SUFFIX[key]
    corr = CORRELATION_SUFFIX[key]

    hit = f"tab:hit_at_1_importances_{label}"
    CAPTIONS[hit] = (
        f"Hit-at-1 feature importances for {name}, showing features above 0.5{BS}% of the total importance. "
        f"{HIT_DIRECTION} {_appendix('appendix:importances-hit-at-1')}"
    )
    CAPTIONS[hit + "_full"] = f"Complete Hit-at-1 feature importances for {name}. {_columns_as(hit)}"
    CAPTIONS[f"tab:hit_at_1_lightweight_importances_{light}"] = f"Lightweight Hit-at-1 feature importances for {name}. {HIT_DIRECTION}"

    for part, kind, extra in (
        ("static", "depth-independent", ""),
        ("dynamic", "depth-dependent", " Each candidate-set size $k$ of a depth-dependent feature is treated as a separate feature."),
    ):
        main = f"tab:in_top_max_depth_importances_{label}_{part}"
        CAPTIONS[main] = (
            f"Maximum Depth {kind} feature importances for {name}, showing features above 0.5{BS}% of the total "
            f"importance.{extra} {DEPTH_DIRECTION} {_appendix('appendix:importances-max-depth')}"
        )
        CAPTIONS[main + "_full"] = f"Complete Maximum Depth {kind} feature importances for {name}. {_columns_as(main)}"
    CAPTIONS[f"tab:in_top_max_depth_lightweight_importances_{light}"] = (
        f"Lightweight Maximum Depth feature importances for {name}. {DEPTH_DIRECTION}"
    )

    ordinal = f"tab:calibrated_ordinal_importances_{label}"
    CAPTIONS[ordinal] = (
        f"Ordinal candidate-set-size feature importances for {name}, showing features with an average importance above "
        f"0.015. Values are the importances in each binary classifier of the cascade, {SIGNED_PER_K}. Average is the "
        "mean unsigned importance, and bold marks the most important feature per $k$. "
        + _appendix("appendix:importances-ordinal")
    )
    CAPTIONS[ordinal + "_full"] = f"Complete ordinal candidate-set-size feature importances for {name}. {_columns_as(ordinal)}"
    CAPTIONS[f"tab:calibrated_ordinal_lightweight_importances_{light}"] = (
        f"Lightweight ordinal candidate-set-size feature importances for {name}, in percent of the total importance per "
        f"classifier and {SIGNED_PER_K}. Solv. is the classifier for the relevant publication being ranked within the "
        "top 1000, and Average is the mean unsigned importance."
    )

    qpp = f"tab:tuned_qpp_importances_score_{label}"
    CAPTIONS[qpp] = (
        f"QPP regressor feature importances for {name}, showing features above 0.5{BS}% of the total importance. "
        f"{QPP_DIRECTION} {_appendix('appendix:importances-qpp')}"
    )
    CAPTIONS[qpp + "_full"] = f"Complete QPP regressor feature importances for {name}. {_columns_as(qpp)}"
    CAPTIONS[f"tab:tuned_qpp_lightweight_importances_score_{light}"] = f"Lightweight QPP regressor feature importances for {name}. {QPP_DIRECTION}"

    binary = f"tab:in_top_k_sweep_importances_{label}"
    CAPTIONS[binary] = (
        f"Binary candidate-set-size feature importances for {name}, showing features with an average importance above "
        f"1{BS}% of the total importance per $k$. Values are in percent of the total importance per $k$, {SIGNED_PER_K}. "
        "Depth is the feature depth with the highest total importance, Average the mean unsigned importance, and bold "
        "marks the most important feature per $k$. " + _appendix("appendix:importances-binary")
    )
    CAPTIONS[binary + "_full"] = f"Complete binary candidate-set-size feature importances for {name}. {_columns_as(binary)}"
    CAPTIONS[f"tab:in_top_k_sweep_lightweight_importances_{light}"] = (
        f"Lightweight binary candidate-set-size feature importances for {name}, in percent of the total importance per "
        f"$k$ and {SIGNED_PER_K}. Depth is the feature depth with the highest total importance, Average the mean "
        "unsigned importance, and bold marks the most important feature per $k$."
    )

    CAPTIONS[f"tab:selected_gold_independent_correlations_{corr}"] = (
        f"Correlations of selected gold-independent features with retrieval difficulty for {name} {TRAIN}, measured as "
        "Kendall's $" + BS + "tau_b$ with the reciprocal rank of the relevant publication."
    )
    for scope, scope_text in (("gold_aware", "gold-aware"), ("gold_independent", "gold-independent")):
        CAPTIONS[f"tab:{scope}_correlations_{corr}_static_en_train"] = (
            f"Correlations of {scope_text} depth-independent features with retrieval difficulty for {name} {TRAIN}, "
            + CORRELATION_MEASURES
        )
        CAPTIONS[f"tab:{scope}_correlations_{corr}_dynamic_symbols_en_train"] = (
            f"Correlations of {scope_text} depth-dependent features with retrieval difficulty for {name} {TRAIN} for "
            f"each candidate-set size $k$. {SYMBOL_NOTE}"
        )

for key, reranker in RERANKERS.items():
    CAPTIONS[f"tab:multistage_reranking_comparison_{key}_en"] = (
        f"MRR@5 and number of reranker calls ({BS}#Calls) of {BM25} and {QWEN} reranked with {reranker} under different "
        f"reranking policies {TEST}. Besides the reference policies, only policies that reach the best MRR@5 for at "
        f"least one $k$ are shown. {RERANK_MARKS} " + _appendix("appendix:reranking-full", plural=True)
    )
    for retriever_key, name in RETRIEVERS.items():
        CAPTIONS[f"tab:multistage_reranking_full_comparison_{RERANKING_SUFFIX[retriever_key]}_{key}_en"] = (
            f"MRR@5 and number of reranker calls ({BS}#Calls) of {name} reranked with {reranker} under all reranking "
            f"policies {TEST}. {RERANK_MARKS}"
        )


def short_caption(caption: str) -> str | None:
    parts = re.split(r"(?<=\.)\s+(?=[A-Z\\$])", caption, maxsplit=1)
    return parts[0] if len(parts) > 1 else None


def caption_command(label: str) -> str | None:
    caption = CAPTIONS.get(label)
    if caption is None:
        return None
    short = short_caption(caption)
    return f"{BS}caption[{short}]{{{caption}}}" if short else f"{BS}caption{{{caption}}}"


def _caption_span(latex: str, label_position: int) -> tuple[int, int] | None:
    table_start = max(latex.rfind(BS + "begin{table", 0, label_position), latex.rfind(BS + "begin{longtable", 0, label_position))
    start = latex.rfind(BS + "caption", max(table_start, 0), label_position)
    if start < 0:
        return None
    k = start + len(BS + "caption")
    if latex[k] == "[":
        depth = 0
        while not (latex[k] == "]" and depth == 0):
            depth += {"{": 1, "}": -1}.get(latex[k], 0)
            k += 1
        k += 1
    if latex[k] != "{":
        return None
    depth = 1
    k += 1
    while depth:
        depth += {"{": 1, "}": -1}.get(latex[k], 0)
        k += 1
    return start, k


def apply_thesis_captions(latex: str) -> str:
    spans = []
    for match in re.finditer(r"\\label\{([^}]*)\}", latex):
        command = caption_command(match.group(1))
        span = _caption_span(latex, match.start()) if command else None
        if span:
            spans.append((span, command))
    for (start, end), command in sorted(spans, reverse=True):
        latex = latex[:start] + command + latex[end:]
    return latex
