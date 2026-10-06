from __future__ import annotations

import ast
import json
import math
import re
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hdbscan.validity import validity_index
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    silhouette_score,
)


TASK_DIFFICULTY_ANALYSIS_SYSTEM_DISPLAY_NAMES: dict[str, str] = {
    "claim2source": "Claim2Source",
    "team-outsider": "Outsider",
    "mattiaskvist": "Mattiaskvist",
    "taha_m_habib": "Taha M Habib",
    "seupd-retrix": "RETRIX",
    "farnaz": "Farnaz",
    "varytrieval": "Varytrieval",
    "mever_checkthat": "MeVer",
    "amr_abdelsamea": "Amr Abdelsamea",
    "thechuongchu": "Thechuongchu",
    "awakened": "Awakened",
    "mohotarema": "Mohotarema",
    "mg_ct2026": "MGCT2026",
    "seupd2526-fella": "FELLA",
    "ivanvynyavskyy": "Ivan Vynyavskyy",
    "aditya": "Aditya",
    "bhargab_jb": "Bhargab JB",
    "richart": "Richart",
    "trevi": "Trevi",
    "joseph": "Joseph",
    "huesos": "Huesos",
    "meth3group2": "Meth3Group2",
    "sagasu": "SAGASU",
    "mailuefterl": "Mailuefterl",
    "rpapusoi": "Rpapusoi",
    "clara": "Clara",
    "valentin-tian": "Valentin Tian",
    "checkmates": "Checkmates",
    "elineevje": "Elineevje",
    "seupd2526-dell": "SEUPD2526 Dell",
    "cam_le": "Cam Le",
    "harshpatel": "Harshpatel",
    "mary_toapanta": "Mary Toapanta",
    "fmi-bionlp-aac": "FMI-BioNLP-AAC",
    "seupd2526-dspy": "SEUPD2526 DSPy",
    "lamungu": "Lamungu",
    "teamxclef2026": "TeamXCLEF2026",
}

SUBMISSION_ARCHIVE_STEM_PATTERN = re.compile(r"^\d+_(.+?)_PhaseId\d+_")
LEXICAL_RETRIEVER_BASE_NAMES = {"bm25", "tfidf", "keyword_overlap", "jaccard"}
DIFFICULTY_CORRELATION_SIGNIFICANCE_LEVEL = 0.05
DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT = "gold_independent"
DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE = "gold_aware"
DIFFICULTY_CORRELATION_FEATURE_SET_ALL = "all_features"
DIFFICULTY_CORRELATION_EXCLUDED_COLUMNS = {
    "query_id",
    "document_id",
    "language",
    "split_name",
    "retriever_name",
    "gold_rank",
    "smallest_sufficient_size",
    "retrieval_failure_within_max_depth",
    "rr_at_max_depth",
    "candidate_size",
    "gold_in_top_k",
}
GOLD_INDEPENDENT_SHARED_QUERY_FEATURES = {
    "query_length_characters",
    "query_length_tokens",
    "x_artifact_ratio",
    "x_mention_ratio",
    "english_indicator",
    "german_indicator",
    "french_indicator",
    "mean_query_term_idf",
    "maximum_query_term_idf",
    "query_term_idf_standard_deviation",
    "mean_query_term_ictf",
    "query_scope",
    "query_scope_non_stopwords",
    "out_of_vocabulary_ratio",
    "stopword_ratio",
    "repeated_term_ratio",
    "simplified_clarity_score",
    "idf_maximum_to_minimum_ratio",
    "numeral_ratio",
    "acronym_ratio",
    "proper_noun_ratio",
    "mean_query_term_polysemy",
    "mean_dependency_link_span",
    "sentiment_polarity",
    "sentiment_intensity",
}
GOLD_INDEPENDENT_LEXICAL_TOP_DOCUMENT_FEATURES = {
    "top_1_title_length_characters",
    "top_1_title_length_tokens",
    "top_1_title_query_coverage",
    "top_1_abstract_length_characters",
    "top_1_abstract_length_tokens",
    "top_1_abstract_query_coverage",
    "top_1_authors_length_characters",
    "top_1_authors_length_tokens",
    "top_1_authors_query_coverage",
    "top_1_venue_length_characters",
    "top_1_venue_length_tokens",
    "top_1_venue_query_coverage",
    "top_1_title_abstract_entity_overlap",
    "top_1_query_term_coverage",
    "top_1_unmatched_idf_mass_ratio",
}
GOLD_INDEPENDENT_LEXICAL_STATIC_RANKING_FEATURES = {
    "top_1_score",
    "top_1_top_2_score_margin",
}
GOLD_INDEPENDENT_LEXICAL_DYNAMIC_FEATURES = {
    "minimum_top_k_pairwise_tfidf_similarity",
    "maximum_top_k_pairwise_tfidf_similarity",
    "mean_top_k_pairwise_tfidf_similarity",
    "minimum_top_k_pairwise_entity_similarity",
    "maximum_top_k_pairwise_entity_similarity",
    "mean_top_k_pairwise_entity_similarity",
    "clarity_score",
    "retrieval_coherency_graph_density",
    "largest_retrieval_coherency_component_share",
    "score_at_rank_k",
    "score_at_rank_k_plus_1",
    "rank_k_rank_k_plus_1_score_margin",
    "top_1_rank_k_score_drop",
    "mean_top_k_score",
    "median_top_k_score",
    "top_k_score_standard_deviation",
    "top_k_score_coefficient_of_variation",
    "top_k_score_entropy",
    "top_k_score_skewness",
    "top_k_score_slope",
    "boundary_score_slope",
    "close_competitor_fraction",
    "cumulative_top_k_score_mass",
    "boundary_surprise_score",
    "surprising_document_fraction",
    "normalized_query_commitment",
    "weighted_information_gain",
    "top_1_rank_k_score_ratio",
    "mean_score_decrease",
}
GOLD_INDEPENDENT_SEMANTIC_PRE_STATIC_FEATURES = {
    "query_collection_centroid_cosine",
    "nearest_centroid_cosine_spherical",
    "nearest_second_nearest_centroid_margin_spherical",
    "nearest_centroid_cosine_hdbscan",
    "nearest_second_nearest_centroid_margin_hdbscan",
    "cluster_weighted_corpus_familiarity",
}
GOLD_INDEPENDENT_SEMANTIC_PRE_DYNAMIC_FEATURES = {
    "distance_weighted_training_query_rr_at_k",
}
GOLD_INDEPENDENT_SEMANTIC_SCORE_STATIC_FEATURES = {
    "top_1_cosine_similarity",
    "top_1_top_2_cosine_margin",
}
GOLD_INDEPENDENT_SEMANTIC_SCORE_DYNAMIC_FEATURES = {
    "cosine_similarity_at_rank_k",
    "cosine_similarity_at_rank_k_plus_1",
    "rank_k_rank_k_plus_1_cosine_margin",
    "top_1_rank_k_cosine_drop",
    "mean_top_k_cosine_similarity",
    "median_top_k_cosine_similarity",
    "top_k_cosine_standard_deviation",
    "top_k_cosine_entropy",
    "top_k_cosine_skewness",
    "top_k_cosine_slope",
    "boundary_cosine_slope",
    "close_competitor_fraction",
    "above_mean_cosine_fraction",
    "boundary_semantic_surprise_score",
    "surprising_document_fraction",
    "mean_cosine_decrease",
    "boundary_cosine_z_score",
    "standardized_top_1_rank_k_cosine_drop",
    "dense_weighted_information_gain",
    "dense_normalized_query_commitment",
}
GOLD_INDEPENDENT_SEMANTIC_GEOMETRY_DYNAMIC_FEATURES = {
    "query_top_k_centroid_cosine",
    "top_k_pairwise_coherence",
    "mean_top_k_centroid_distance",
    "top_k_centroid_distance_standard_deviation",
    "maximum_top_k_centroid_distance",
    "minimum_top_k_pairwise_cosine_similarity",
    "maximum_top_k_pairwise_cosine_similarity",
    "top_k_pairwise_cosine_standard_deviation",
    "preceding_candidate_set_coherence_ratio",
    "query_adjusted_coherence_ratio",
    "average_neighbour_degree",
    "graph_density",
    "embedding_spatial_autocorrelation",
    "semantic_retrieval_coherency_graph_density",
    "largest_semantic_retrieval_coherency_component_share",
    "log_query_document_hyper_volume",
}
GOLD_INDEPENDENT_SEMANTIC_CLUSTER_DYNAMIC_FEATURES = {
    "spherical_query_nearest_cluster_share_in_top_k",
    "spherical_retrieval_score_mass_of_query_nearest_cluster_in_top_k",
    "spherical_distinct_clusters_in_top_k",
    "spherical_cluster_entropy_in_top_k",
    "spherical_score_weighted_cluster_entropy_in_top_k",
    "spherical_largest_cluster_share_in_top_k",
    "spherical_dominant_cluster_enrichment_in_top_k",
    "hdbscan_query_nearest_cluster_share_in_top_k",
    "hdbscan_retrieval_score_mass_of_query_nearest_cluster_in_top_k",
    "hdbscan_distinct_clusters_in_top_k",
    "hdbscan_cluster_entropy_in_top_k",
    "hdbscan_score_weighted_cluster_entropy_in_top_k",
    "hdbscan_largest_cluster_share_in_top_k",
    "hdbscan_dominant_cluster_enrichment_in_top_k",
    "hdbscan_noise_fraction_in_top_k",
}
GOLD_INDEPENDENT_RETRIEVER_AGREEMENT_FEATURES = {
    "mean_semantic_top_1_agreement",
    "maximum_semantic_top_1_agreement",
    "mean_semantic_top_100_agreement",
    "maximum_semantic_top_100_agreement",
    "bm25_top_1_agreement",
    "bm25_top_100_agreement",
    "top_1_consensus_frequency",
    "top_100_document_consensus",
}
GOLD_INDEPENDENT_PERTURBATION_FEATURES = {
    "perturbation_ranking_stability",
}
GOLD_AWARE_SHARED_CONTEXT_FEATURES = {
    "query_length_tokens",
    "gold_title_length_characters",
    "gold_title_length_tokens",
    "gold_abstract_length_characters",
    "gold_abstract_length_tokens",
    "gold_authors_length_characters",
    "gold_authors_length_tokens",
    "gold_venue_length_characters",
    "gold_venue_length_tokens",
    "gold_title_query_coverage",
    "gold_abstract_query_coverage",
    "gold_authors_query_coverage",
    "gold_venue_query_coverage",
    "gold_title_abstract_entity_overlap",
    "x_artifact_ratio",
    "x_mention_ratio",
}
GOLD_AWARE_SHARED_SEMANTIC_CONTEXT_FEATURES = {
    "top_1_cosine_similarity",
    "top_2_cosine_similarity",
    "top_3_cosine_similarity",
    "top_4_cosine_similarity",
    "top_5_cosine_similarity",
    "top_1_top_10_mean_gap",
    "top_1_agreement_vs_bge_m3",
    "top_1_agreement_vs_multilingual_e5_base",
}
GOLD_AWARE_LEXICAL_MATCHING_FEATURES = {
    "gold_query_term_coverage",
    "gold_unmatched_idf_mass_ratio",
}
GOLD_AWARE_LEXICAL_RANKING_STATIC_FEATURES = {
    "gold_document_retrieval_score",
    "top_1_gold_score_gap",
    "top_1_gold_score_ratio",
    "gold_document_surprise_score",
    "gold_document_surprise_threshold_indicator",
    "top_1_gold_query_term_coverage_gap",
    "top_1_gold_unmatched_idf_mass_gap",
}
GOLD_AWARE_LEXICAL_RANKING_DYNAMIC_FEATURES = {
    "rank_k_gold_score_gap",
    "gold_score_top_k_z_score",
}
GOLD_AWARE_SEMANTIC_SCORE_STATIC_FEATURES = {
    "gold_document_cosine_similarity",
    "top_1_gold_cosine_gap",
    "gold_document_semantic_surprise_score",
    "gold_document_semantic_surprise_threshold_indicator",
}
GOLD_AWARE_SEMANTIC_SCORE_DYNAMIC_FEATURES = {
    "mean_top_k_gold_cosine_gap",
    "rank_k_gold_cosine_gap",
    "gold_cosine_top_k_z_score",
    "gold_cosine_above_top_k_mean",
}
GOLD_AWARE_SEMANTIC_GEOMETRY_DYNAMIC_FEATURES = {
    "gold_top_k_centroid_cosine",
    "mean_gold_top_k_document_cosine",
    "minimum_gold_top_k_document_cosine",
    "maximum_gold_top_k_document_cosine",
    "gold_top_k_cosine_standard_deviation",
    "gold_document_graph_degree",
    "gold_document_retrieval_coherency_degree",
    "gold_connected_to_largest_coherency_component",
}
GOLD_AWARE_SPHERICAL_CLUSTER_STATIC_FEATURES = {
    "spherical_gold_cluster_centroid_cosine",
    "spherical_gold_cluster_centroid_rank",
}
GOLD_AWARE_SPHERICAL_CLUSTER_DYNAMIC_FEATURES = {
    "spherical_gold_cluster_present_in_top_k",
    "spherical_gold_cluster_share_in_top_k",
    "spherical_gold_cluster_score_mass_in_top_k",
    "spherical_gold_cluster_enrichment_in_top_k",
}
GOLD_AWARE_HDBSCAN_CLUSTER_STATIC_FEATURES = {
    "hdbscan_gold_cluster_centroid_cosine",
    "hdbscan_gold_cluster_centroid_rank",
    "hdbscan_gold_document_labelled_as_noise",
}
GOLD_AWARE_HDBSCAN_CLUSTER_DYNAMIC_FEATURES = {
    "hdbscan_gold_cluster_present_in_top_k",
    "hdbscan_gold_cluster_share_in_top_k",
    "hdbscan_gold_cluster_score_mass_in_top_k",
    "hdbscan_gold_cluster_enrichment_in_top_k",
}
GOLD_AWARE_PERTURBATION_STATIC_FEATURES = {
    "gold_document_cosine_perturbation",
}
GOLD_AWARE_PERTURBATION_DYNAMIC_FEATURES = {
    "gold_document_perturbation_retention",
}


@dataclass(frozen=True, slots=True)
class DifficultyCorrelationFeatureSpec:
    feature_name: str
    feature_group: str
    depth_dependent: bool
    feature_set: str
    retriever_family: str


def reciprocal_rank_at_k(predictions: list[int], gold_id: int, k: int) -> float:
    for rank, candidate in enumerate(predictions[:k], start=1):
        if candidate == gold_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(predictions: list[int], gold_id: int, k: int) -> float:
    return float(gold_id in predictions[:k])


def read_submission_archive(archive_path: Path) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.endswith(".tsv") or member.startswith("__MACOSX"):
                continue
            language = member.split("_")[-1].split(".")[0]
            frame = pd.read_csv(archive.open(member), sep="\t")
            frame["preds"] = frame["preds"].map(lambda value: [int(pred) for pred in ast.literal_eval(value)])
            output[language] = frame
    return output


def task_difficulty_analysis_system_display_name(system_name: str) -> str:
    slug = _task_difficulty_analysis_submission_slug(system_name)
    if slug in TASK_DIFFICULTY_ANALYSIS_SYSTEM_DISPLAY_NAMES:
        return TASK_DIFFICULTY_ANALYSIS_SYSTEM_DISPLAY_NAMES[slug]
    return _task_difficulty_analysis_clean_slug(slug)


def submission_performance_table(submission_dir: Path, gold_queries: pd.DataFrame, language: str) -> pd.DataFrame:
    gold = gold_queries[["query_id", "document_id"]].copy()
    records = []
    solved_counter: Counter[int] = Counter()
    per_system_hits: dict[str, set[int]] = {}
    for archive_path in sorted(submission_dir.glob("*.zip")):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            continue
        frame = archives[language].rename(columns={"index": "query_id"})
        merged = gold.merge(frame, on="query_id", how="inner")
        merged["rr5"] = [reciprocal_rank_at_k(preds, gold_id, 5) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)]
        merged["r1"] = [recall_at_k(preds, gold_id, 1) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)]
        merged["r5"] = [recall_at_k(preds, gold_id, 5) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)]
        solved = set(merged.loc[merged["r5"] == 1.0, "query_id"].tolist())
        per_system_hits[archive_path.stem] = solved
        solved_counter.update(solved)
        records.append(
            {
                "system": archive_path.stem,
                "mrr@5": merged["rr5"].mean(),
                "r@1": merged["r1"].mean(),
                "r@5": merged["r5"].mean(),
                "solved@5": int(merged["r5"].sum()),
                "missed@5": int((merged["r5"] == 0.0).sum()),
            }
        )
    frame = pd.DataFrame(records).sort_values(["mrr@5", "r@5", "r@1"], ascending=False).reset_index(drop=True)
    unique_counts = []
    for system in frame["system"]:
        solved = per_system_hits[system]
        unique_counts.append(sum(1 for query_id in solved if solved_counter[query_id] == 1))
    frame["unique_solved@5"] = unique_counts
    return frame


def task_difficulty_analysis_submission_miss_matrix(
    submission_dir: Path, gold_queries: pd.DataFrame, language: str, k: int = 5
) -> pd.DataFrame:
    gold = gold_queries[["query_id", "document_id"]].copy()
    gold["query_id"] = gold["query_id"].astype(int)
    gold["document_id"] = gold["document_id"].astype(int)

    rows: dict[str, dict[int, int]] = {}
    for archive_path in sorted(submission_dir.glob("*.zip")):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            continue
        frame = archives[language].rename(columns={"index": "query_id"}).copy()
        frame["query_id"] = frame["query_id"].astype(int)
        frame = frame.drop_duplicates(subset=["query_id"], keep="first")

        merged = gold.merge(frame[["query_id", "preds"]], on="query_id", how="left")
        merged["preds"] = merged["preds"].map(lambda value: value if isinstance(value, list) else [])
        merged["hit"] = [recall_at_k(preds, gold_id, k) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)]
        rows[archive_path.stem] = {
            int(query_id): int(hit == 0.0) for query_id, hit in zip(merged["query_id"], merged["hit"], strict=True)
        }

    if not rows:
        return pd.DataFrame()
    matrix = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)
    matrix.index.name = "system"
    return matrix.astype(np.int8)


def task_difficulty_analysis_uniquely_solved_query_matrix(miss_matrix: pd.DataFrame) -> pd.DataFrame:
    if miss_matrix.empty:
        return pd.DataFrame()
    solved_matrix = (1 - miss_matrix).astype(np.int8)
    unique_columns = solved_matrix.columns[solved_matrix.sum(axis=0) == 1]
    unique_matrix = solved_matrix.loc[:, unique_columns].copy()
    unique_matrix.index.name = miss_matrix.index.name
    return unique_matrix.astype(np.int8)


def task_difficulty_analysis_query_solving_counts(miss_matrix: pd.DataFrame) -> pd.Series:
    if miss_matrix.empty:
        return pd.Series(dtype=np.int32, name="solving_systems")
    solving_counts = (len(miss_matrix.index) - miss_matrix.sum(axis=0).astype(int)).astype(np.int32)
    solving_counts.index.name = "query_id"
    solving_counts.name = "solving_systems"
    return solving_counts.sort_index()


def task_difficulty_analysis_system_rarity_profile(miss_matrix: pd.DataFrame) -> pd.DataFrame:
    if miss_matrix.empty:
        return pd.DataFrame()
    solving_counts = task_difficulty_analysis_query_solving_counts(miss_matrix)
    solved_matrix = (1 - miss_matrix).astype(np.int8)
    rows: dict[str, dict[int, int]] = {}
    for system_id in solved_matrix.index.astype(str):
        solved_query_ids = solved_matrix.columns[solved_matrix.loc[system_id] == 1]
        counts = solving_counts.reindex(solved_query_ids).value_counts().sort_index()
        rows[system_id] = {int(bucket): int(value) for bucket, value in counts.items()}
    profile = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(np.int32)
    profile = profile.reindex(columns=list(range(1, len(miss_matrix.index) + 1)), fill_value=0)
    profile.index.name = miss_matrix.index.name
    profile.columns.name = "solving_systems"
    return profile


def task_difficulty_analysis_system_rarity_summary(
    miss_matrix: pd.DataFrame, rare_thresholds: tuple[int, ...] = (2, 3, 5)
) -> pd.DataFrame:
    if miss_matrix.empty:
        return pd.DataFrame(
            columns=[
                "system_id",
                "system",
                "rare_solve_score",
                "unique_solved_queries",
                "solved_queries",
                "missed_queries",
                "miss_rate",
            ]
        )
    profile = task_difficulty_analysis_system_rarity_profile(miss_matrix)
    profile_columns = np.asarray(profile.columns.to_list(), dtype=np.float64)
    weighted_profile = profile.to_numpy(dtype=np.float64) / profile_columns
    summary = pd.DataFrame(
        {
            "system_id": profile.index.astype(str),
            "system": [task_difficulty_analysis_system_display_name(system_id) for system_id in profile.index.astype(str)],
            "rare_solve_score": weighted_profile.sum(axis=1),
            "unique_solved_queries": profile.get(1, pd.Series(0, index=profile.index)).to_numpy(dtype=np.int32),
            "solved_queries": profile.sum(axis=1).to_numpy(dtype=np.int32),
            "missed_queries": miss_matrix.sum(axis=1).reindex(profile.index).to_numpy(dtype=np.int32),
            "miss_rate": miss_matrix.mean(axis=1).reindex(profile.index).to_numpy(dtype=np.float64),
        }
    )
    for threshold in rare_thresholds:
        matching_columns = [int(column) for column in profile.columns.tolist() if int(column) <= int(threshold)]
        column_name = f"solved_by_{int(threshold)}_or_fewer_systems"
        if not matching_columns:
            summary[column_name] = 0
            continue
        summary[column_name] = profile.loc[:, matching_columns].sum(axis=1).to_numpy(dtype=np.int32)
    return summary


def task_difficulty_analysis_universally_unsolved_query_ids(miss_matrix: pd.DataFrame) -> list[int]:
    if miss_matrix.empty:
        return []
    unsolved_columns = miss_matrix.columns[miss_matrix.sum(axis=0) == len(miss_matrix.index)]
    return [int(query_id) for query_id in unsolved_columns.tolist()]


def best_vs_combined_table(submission_dir: Path, gold_queries: pd.DataFrame, language: str) -> pd.DataFrame:
    table = submission_performance_table(submission_dir, gold_queries, language)
    best_single = table.iloc[0]
    combined = defaultdict(list)
    for archive_path in sorted(submission_dir.glob("*.zip")):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            continue
        frame = archives[language].rename(columns={"index": "query_id"})
        for query_id, preds in zip(frame["query_id"], frame["preds"], strict=True):
            combined[int(query_id)].append(list(preds))
    merged_records = []
    for row in gold_queries[["query_id", "document_id"]].itertuples(index=False):
        rankings = combined.get(int(row.query_id), [])
        best_rank = None
        for ranking in rankings:
            try:
                rank = ranking.index(int(row.document_id)) + 1
            except ValueError:
                continue
            best_rank = rank if best_rank is None else min(best_rank, rank)
        merged_records.append(
            {
                "rr5": 0.0 if best_rank is None or best_rank > 5 else 1.0 / best_rank,
                "r1": float(best_rank == 1),
                "r5": float(best_rank is not None and best_rank <= 5),
            }
        )
    combined_frame = pd.DataFrame(merged_records)
    return pd.DataFrame(
        [
            {
                "condition": "Best single system",
                "system": best_single["system"],
                "mrr@5": best_single["mrr@5"],
                "r@1": best_single["r@1"],
                "r@5": best_single["r@5"],
                "solved@5": int(best_single["solved@5"]),
                "missed@5": int(best_single["missed@5"]),
            },
            {
                "condition": "Best-of-all submissions",
                "system": "All systems combined",
                "mrr@5": combined_frame["rr5"].mean(),
                "r@1": combined_frame["r1"].mean(),
                "r@5": combined_frame["r5"].mean(),
                "solved@5": int(combined_frame["r5"].sum()),
                "missed@5": int((combined_frame["r5"] == 0.0).sum()),
            },
        ]
    )


def difficulty_distribution_table(submission_dir: Path, gold_queries: pd.DataFrame, language: str) -> pd.DataFrame:
    gold = gold_queries[["query_id", "document_id"]].copy()
    solved_counts = Counter()
    system_count = 0
    for archive_path in sorted(submission_dir.glob("*.zip")):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            continue
        system_count += 1
        frame = archives[language].rename(columns={"index": "query_id"})
        merged = gold.merge(frame, on="query_id", how="inner")
        merged["hit"] = [recall_at_k(preds, gold_id, 5) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)]
        solved_counts.update(merged.loc[merged["hit"] == 1.0, "query_id"].astype(int).tolist())
    bins = [
        ("Unsolved", lambda value: value == 0),
        ("Rarely solved", lambda value: 0 < value / system_count <= 0.25),
        ("Minority solved", lambda value: 0.25 < value / system_count <= 0.5),
        ("Majority solved", lambda value: 0.5 < value / system_count <= 0.75),
        ("Widely solved", lambda value: 0.75 < value / system_count < 1.0),
        ("Universally solved", lambda value: value == system_count),
    ]
    records = []
    total_queries = len(gold_queries)
    for label, predicate in bins:
        values = [count for _, count in solved_counts.items() if predicate(count)]
        matching_query_ids = [query_id for query_id in gold_queries["query_id"].tolist() if predicate(solved_counts.get(int(query_id), 0))]
        records.append(
            {
                "solvability_label": label,
                "systems_range_percent": _systems_range_text(label, system_count),
                "systems_range_count": _systems_count_text(label, system_count),
                "queries": len(matching_query_ids),
                "percentage": len(matching_query_ids) / total_queries,
            }
        )
    return pd.DataFrame(records)


def cross_system_unsolved_indicator(submission_dir: Path, gold_queries: pd.DataFrame, language: str) -> pd.Series:
    gold = gold_queries[["query_id", "document_id"]].copy()
    solved_counts: Counter[int] = Counter()
    for archive_path in sorted(submission_dir.glob("*.zip")):
        archives = read_submission_archive(archive_path)
        if language not in archives:
            continue
        frame = archives[language].rename(columns={"index": "query_id"})
        merged = gold.merge(frame, on="query_id", how="inner")
        merged["hit"] = [
            recall_at_k(preds, gold_id, 5) for preds, gold_id in zip(merged["preds"], merged["document_id"], strict=True)
        ]
        solved_counts.update(merged.loc[merged["hit"] == 1.0, "query_id"].astype(int).tolist())
    is_unsolved = pd.Series(
        {
            int(query_id): int(solved_counts.get(int(query_id), 0) == 0)
            for query_id in gold_queries["query_id"].tolist()
        },
        name="is_unsolved",
    )
    is_unsolved.index.name = "query_id"
    return is_unsolved


def cluster_overview_table(artifact_dir: Path) -> pd.DataFrame:
    rows = []
    cluster_root = artifact_dir / "clusters"
    if not cluster_root.exists():
        return pd.DataFrame(columns=["retriever", "algorithm", "cluster_count", "minimum_cluster_size", "mean_cluster_size", "maximum_cluster_size"])
    for retriever_path in sorted(cluster_root.iterdir()):
        if not retriever_path.is_dir():
            continue
        for algorithm_path in sorted(retriever_path.iterdir()):
            summary_path = algorithm_path / "summary.json"
            if not summary_path.exists():
                continue
            summary = pd.read_json(summary_path, typ="series").to_dict()
            rows.append(
                {
                    "retriever": retriever_path.name,
                    "algorithm": algorithm_path.name,
                    "cluster_count": summary.get("cluster_count", 0),
                    "minimum_cluster_size": summary.get("minimum_cluster_size", 0),
                    "mean_cluster_size": summary.get("mean_cluster_size", 0.0),
                    "maximum_cluster_size": summary.get("maximum_cluster_size", 0),
                }
            )
    return pd.DataFrame(rows)


def cluster_diagnostics_table(
    artifact_dir: Path,
    retriever_names: list[str] | None = None,
    algorithms: list[str] | None = None,
    random_seed: int = 13,
    silhouette_sample_size: int = 2048,
) -> pd.DataFrame:
    rows = []
    cluster_root = artifact_dir / "clusters"
    if not cluster_root.exists():
        return pd.DataFrame(
            columns=[
                "retriever",
                "algorithm",
                "protocol_version",
                "embedding_view",
                "document_count",
                "cluster_count",
                "clustered_document_count",
                "noise_fraction",
                "largest_cluster_share_all",
                "largest_cluster_share_clustered",
                "effective_cluster_count",
                "normalized_cluster_entropy",
                "clusters_leq_10",
                "doc_share_in_clusters_leq_10",
                "sampled_silhouette_cosine",
                "hdbscan_dbcv_unit_sphere_euclidean",
                "hdbscan_dbcv_warning_count",
                "selected_k",
                "selection_mean_score",
                "selection_second_mean_score",
                "selection_margin",
            ]
        )
    selected_retrievers = set(retriever_names) if retriever_names else None
    selected_algorithms = set(algorithms) if algorithms else None
    for retriever_path in sorted(cluster_root.iterdir()):
        if not retriever_path.is_dir():
            continue
        if selected_retrievers is not None and retriever_path.name not in selected_retrievers:
            continue
        embedding_path = artifact_dir / "embeddings" / retriever_path.name / "documents.npy"
        if not embedding_path.exists():
            continue
        embeddings = np.load(embedding_path)
        document_count = int(len(embeddings))
        for algorithm_path in sorted(retriever_path.iterdir()):
            if not algorithm_path.is_dir():
                continue
            algorithm = algorithm_path.name
            if selected_algorithms is not None and algorithm not in selected_algorithms:
                continue
            assignments_path = algorithm_path / "assignments.parquet"
            summary_path = algorithm_path / "summary.json"
            if not assignments_path.exists() or not summary_path.exists():
                continue
            labels = pd.read_parquet(assignments_path)["label"].to_numpy(dtype=np.int32)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            view_path = algorithm_path / "view.json"
            view_state_path = algorithm_path / "view_state.joblib"
            view_name = "raw_l2"
            view_state: dict[str, np.ndarray] = {}
            if view_path.exists():
                view_name = str(json.loads(view_path.read_text(encoding="utf-8")).get("name", "raw_l2"))
            if view_state_path.exists():
                view_state = joblib.load(view_state_path)
            evaluation_embeddings = _project_embeddings_for_cluster_view(embeddings, view_name, view_state)
            row = {
                "retriever": retriever_path.name,
                "algorithm": algorithm,
                "protocol_version": summary.get("protocol_version"),
                "embedding_view": view_name,
                "document_count": document_count,
                "cluster_count": int(summary.get("cluster_count", 0)),
                **_cluster_distribution_metrics(labels, document_count),
                "sampled_silhouette_cosine": _sampled_cluster_silhouette(
                    embeddings=evaluation_embeddings,
                    labels=labels,
                    algorithm=algorithm,
                    metric=str(view_state.get("analysis_metric", summary.get("analysis_metric", "cosine"))),
                    random_seed=random_seed,
                    sample_size=silhouette_sample_size,
                ),
                "hdbscan_dbcv_unit_sphere_euclidean": np.nan,
                "hdbscan_dbcv_warning_count": 0,
                "selected_k": np.nan,
                "selection_mean_score": np.nan,
                "selection_second_mean_score": np.nan,
                "selection_margin": np.nan,
            }
            if algorithm == "hdbscan":
                dbcv_score, warning_count = _hdbscan_density_validity(evaluation_embeddings, labels)
                row["hdbscan_dbcv_unit_sphere_euclidean"] = dbcv_score
                row["hdbscan_dbcv_warning_count"] = warning_count
            if algorithm == "spherical_kmeans":
                row.update(_spherical_selection_metrics(algorithm_path / "selection_trace.json"))
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = ["retriever", "algorithm"]
    return frame.sort_values(sort_columns).reset_index(drop=True)


def embedding_anisotropy_table(
    artifact_dir: Path,
    retriever_names: list[str] | None = None,
    random_seed: int = 13,
    document_sample_size: int = 800,
    pair_sample_size: int = 3000,
) -> pd.DataFrame:
    embedding_root = artifact_dir / "embeddings"
    if not embedding_root.exists():
        return pd.DataFrame(
            columns=[
                "retriever",
                "base_retriever",
                "document_scope",
                "document_count",
                "embedding_dimension",
                "norm_min",
                "norm_max",
                "centroid_norm",
                "pair_cosine_mean",
                "pair_cosine_p90",
                "pc1_ratio_sample",
                "pc5_ratio_sample",
                "centroid_norm_rank",
                "pair_cosine_mean_rank",
                "pc1_ratio_sample_rank",
                "anisotropy_score",
                "anisotropy_rank",
                "deanisotropization_priority",
            ]
        )
    available = sorted(path.name for path in embedding_root.iterdir() if path.is_dir())
    selected = retriever_names or available
    rows = []
    for index, retriever_name in enumerate(selected):
        embedding_path = embedding_root / retriever_name / "documents.npy"
        if not embedding_path.exists():
            continue
        rows.append(
            _embedding_anisotropy_row(
                retriever_name=retriever_name,
                embedding_path=embedding_path,
                random_seed=random_seed + index,
                document_sample_size=document_sample_size,
                pair_sample_size=pair_sample_size,
            )
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ["centroid_norm", "pair_cosine_mean", "pc1_ratio_sample"]:
        frame[f"{column}_rank"] = frame[column].rank(ascending=False, method="min")
    frame["anisotropy_score"] = frame[
        ["centroid_norm_rank", "pair_cosine_mean_rank", "pc1_ratio_sample_rank"]
    ].sum(axis=1)
    frame["anisotropy_rank"] = frame["anisotropy_score"].rank(method="dense").astype(int)
    total = len(frame)
    frame["deanisotropization_priority"] = [
        _deanisotropization_priority(int(rank), total)
        for rank in frame["anisotropy_rank"].tolist()
    ]
    return frame.sort_values(
        ["anisotropy_rank", "centroid_norm", "pair_cosine_mean", "retriever"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _cluster_distribution_metrics(labels: np.ndarray, document_count: int) -> dict[str, float | int]:
    non_noise_labels = labels[labels != -1]
    clustered_document_count = int(len(non_noise_labels))
    if clustered_document_count == 0:
        return {
            "clustered_document_count": 0,
            "noise_fraction": float(np.mean(labels == -1)) if document_count else 0.0,
            "largest_cluster_share_all": 0.0,
            "largest_cluster_share_clustered": 0.0,
            "effective_cluster_count": 0.0,
            "normalized_cluster_entropy": 0.0,
            "clusters_leq_10": 0,
            "doc_share_in_clusters_leq_10": 0.0,
        }
    _, counts = np.unique(non_noise_labels, return_counts=True)
    proportions = counts.astype(np.float64) / float(clustered_document_count)
    entropy = float(-np.sum(proportions * np.log(proportions)))
    normalized_entropy = 0.0 if len(counts) <= 1 else float(entropy / np.log(len(counts)))
    effective_cluster_count = float(np.exp(entropy))
    tiny_mask = counts <= 10
    return {
        "clustered_document_count": clustered_document_count,
        "noise_fraction": float(np.mean(labels == -1)),
        "largest_cluster_share_all": float(counts.max() / float(document_count)),
        "largest_cluster_share_clustered": float(counts.max() / float(clustered_document_count)),
        "effective_cluster_count": effective_cluster_count,
        "normalized_cluster_entropy": normalized_entropy,
        "clusters_leq_10": int(np.sum(tiny_mask)),
        "doc_share_in_clusters_leq_10": float(counts[tiny_mask].sum() / float(clustered_document_count)),
    }


def _sampled_cluster_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    metric: str,
    random_seed: int,
    sample_size: int,
) -> float:
    if algorithm == "hdbscan":
        mask = labels != -1
        evaluation_embeddings = embeddings[mask]
        evaluation_labels = labels[mask]
    else:
        evaluation_embeddings = embeddings
        evaluation_labels = labels
    if len(evaluation_labels) < 3 or len(np.unique(evaluation_labels)) < 2:
        return float("nan")
    try:
        capped_sample_size = min(len(evaluation_embeddings), sample_size)
        return float(
            silhouette_score(
                evaluation_embeddings,
                evaluation_labels,
                metric=metric,
                sample_size=capped_sample_size if capped_sample_size < len(evaluation_embeddings) else None,
                random_state=random_seed,
            )
        )
    except ValueError:
        return float("nan")


def _hdbscan_density_validity(embeddings: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    non_noise_unique = np.unique(labels[labels != -1])
    if len(non_noise_unique) < 2:
        return float("nan"), 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            score = validity_index(embeddings.astype(np.float64, copy=False), labels, metric="euclidean")
        except ValueError:
            return float("nan"), len(caught)
    return float(score), len(caught)


def _spherical_selection_metrics(trace_path: Path) -> dict[str, float]:
    if not trace_path.exists():
        return {}
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    validation_records = trace.get("validation_records", [])
    if validation_records:
        mean_scores = sorted(
            [
                (int(record["k"]), float(record["mean_silhouette_cosine"]))
                for record in validation_records
                if record.get("mean_silhouette_cosine") is not None
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        if not mean_scores:
            return {
                "selected_k": float(trace.get("selected_k", np.nan)),
                "selection_mean_score": float("nan"),
                "selection_second_mean_score": float("nan"),
                "selection_margin": float("nan"),
            }
        best_mean = mean_scores[0][1]
        second_mean = mean_scores[1][1] if len(mean_scores) > 1 else float("nan")
        margin = best_mean - second_mean if len(mean_scores) > 1 else float("nan")
        return {
            "selected_k": float(trace.get("selected_k", mean_scores[0][0])),
            "selection_mean_score": best_mean,
            "selection_second_mean_score": second_mean,
            "selection_margin": margin,
        }
    rerun_candidates = trace.get("rerun_candidates", [])
    if rerun_candidates:
        ranked = []
        for candidate in rerun_candidates:
            metrics = candidate.get("metrics", {})
            ranked.append(
                (
                    int(candidate.get("selected_k", candidate.get("parameters", {}).get("k", trace.get("selected_k", 0)))),
                    float(metrics.get("mean_sampled_silhouette_cosine"))
                    if metrics.get("mean_sampled_silhouette_cosine") is not None
                    else float("nan"),
                )
            )
        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)
        best_mean = ranked[0][1]
        second_mean = ranked[1][1] if len(ranked) > 1 else float("nan")
        margin = best_mean - second_mean if len(ranked) > 1 else float("nan")
        selected = trace.get("selected", {})
        selected_parameters = selected.get("parameters", {})
        selected_k = selected.get("selected_k", selected_parameters.get("k", trace.get("selected_k", np.nan)))
        return {
            "selected_k": float(selected_k),
            "selection_mean_score": best_mean,
            "selection_second_mean_score": second_mean,
            "selection_margin": margin,
        }
    reruns = trace.get("final_reruns", {})
    if not reruns:
        return {
            "selected_k": float(trace.get("selected_k", np.nan)),
            "selection_mean_score": float("nan"),
            "selection_second_mean_score": float("nan"),
            "selection_margin": float("nan"),
        }
    mean_scores = sorted(
        [(int(k), float(np.mean(scores))) for k, scores in reruns.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    best_mean = mean_scores[0][1]
    second_mean = mean_scores[1][1] if len(mean_scores) > 1 else float("nan")
    margin = best_mean - second_mean if len(mean_scores) > 1 else float("nan")
    return {
        "selected_k": float(trace.get("selected_k", mean_scores[0][0])),
        "selection_mean_score": best_mean,
        "selection_second_mean_score": second_mean,
        "selection_margin": margin,
    }


def _project_embeddings_for_cluster_view(
    embeddings: np.ndarray,
    view_name: str,
    view_state: dict[str, Any],
) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings.astype(np.float32, copy=False)
    if view_name == "umap":
        rows = embeddings.astype(np.float32, copy=False)
        if view_state.get("input_normalization") == "l2":
            norms = np.linalg.norm(rows.astype(np.float64, copy=False), axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            rows = (rows.astype(np.float64, copy=False) / norms).astype(np.float32, copy=False)
        reducer = view_state["reducer"]
        return reducer.transform(rows).astype(np.float32, copy=False)
    rows = embeddings.astype(np.float64, copy=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (rows / norms).astype(np.float32)


def candidate_size_grid_table(candidate_sizes: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"candidate_set_size": size, "log_candidate_set_size": 0 if size <= 0 else np.log(size)} for size in candidate_sizes]
    )


def _safe_metric_value(metric, *args, **kwargs) -> float:
    try:
        return float(metric(*args, **kwargs))
    except ValueError:
        return math.nan


def _format_metric_value(value: float, decimals: int = 4) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{decimals}f}"


def _metric_table(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value"])


def _binary_predictor_metric_summary(frame: pd.DataFrame, actual_column: str, predicted_column: str) -> dict[str, float]:
    actual = frame[actual_column].astype(bool)
    predicted = frame[predicted_column].astype(bool)
    scores = frame["confidence"].astype(float)
    true_positives = int((actual & predicted).sum())
    false_positives = int((~actual & predicted).sum())
    true_negatives = int((~actual & ~predicted).sum())
    false_negatives = int((actual & ~predicted).sum())
    positive_predictions = int(predicted.sum())
    positive_labels = int(actual.sum())
    precision = float(true_positives / positive_predictions) if positive_predictions else math.nan
    recall = float(true_positives / positive_labels) if positive_labels else math.nan
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall):
        f1 = float((2.0 * precision * recall) / (precision + recall))
    else:
        f1 = math.nan
    if np.isfinite(precision) and np.isfinite(recall) and ((0.25 * precision) + recall):
        f0_5 = float(((1.0 + 0.25) * precision * recall) / ((0.25 * precision) + recall))
    else:
        f0_5 = math.nan
    actual_int = actual.astype(int)
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f0_5": f0_5,
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "true_negatives": float(true_negatives),
        "false_negatives": float(false_negatives),
        "pr_auc": _safe_metric_value(average_precision_score, actual_int, scores),
        "roc_auc": _safe_metric_value(roc_auc_score, actual_int, scores),
        "balanced_accuracy": _safe_metric_value(balanced_accuracy_score, actual, predicted),
        "mcc": _safe_metric_value(matthews_corrcoef, actual, predicted),
    }


def hit_at_1_predictor_metric_summary(frame: pd.DataFrame) -> dict[str, float]:
    return _binary_predictor_metric_summary(frame, "hit_at_1", "predicted_hit_at_1")


def in_top_max_depth_predictor_metric_summary(frame: pd.DataFrame) -> dict[str, float]:
    return _binary_predictor_metric_summary(frame, "in_top_max_depth", "predicted_in_top_max_depth")


def hit_at_1_predictor_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = hit_at_1_predictor_metric_summary(frame)
    return _metric_table(
        [
            ("Precision", _format_metric_value(metrics["precision"])),
            ("Recall", _format_metric_value(metrics["recall"])),
            ("F1", _format_metric_value(metrics["f1"])),
            ("F0.5", _format_metric_value(metrics["f0_5"])),
            ("TP", str(int(metrics["true_positives"]))),
            ("FP", str(int(metrics["false_positives"]))),
            ("FN", str(int(metrics["false_negatives"]))),
            ("TN", str(int(metrics["true_negatives"]))),
        ]
    )


def _ordinal_candidate_recall(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    hits = []
    for row in frame.itertuples(index=False):
        if pd.isna(row.gold_rank):
            hits.append(float(bool(row.predicted_failure)))
            continue
        selected_size = float(row.selected_size) if not pd.isna(row.selected_size) else math.nan
        hits.append(float((not bool(row.predicted_failure)) and np.isfinite(selected_size) and selected_size >= float(row.gold_rank)))
    return float(np.mean(hits))


def _ordinal_class_index(value: float, class_index_by_size: dict[int, int], column_name: str) -> int:
    normalized = int(round(float(value)))
    if normalized not in class_index_by_size:
        raise ValueError(f"Unexpected {column_name} value {value!r}; expected one of {sorted(class_index_by_size)}")
    return class_index_by_size[normalized]


def ordinal_predictor_metric_summary(frame: pd.DataFrame, candidate_sizes: list[int]) -> dict[str, float]:
    ordered_sizes = list(dict.fromkeys(int(size) for size in candidate_sizes))
    class_index_by_size = {size: index for index, size in enumerate(ordered_sizes)}
    failure_class_index = len(ordered_sizes)

    non_failure = frame.loc[~frame["predicted_failure"] & frame["selected_size"].notna(), "selected_size"].astype(float)
    solvable = frame.loc[frame["smallest_sufficient_size"].notna()].copy()

    gold_classes: list[int] = []
    predicted_classes: list[int] = []
    exact_matches: list[bool] = []
    within_one_matches: list[bool] = []
    for row in solvable.itertuples(index=False):
        gold_class = _ordinal_class_index(row.smallest_sufficient_size, class_index_by_size, "smallest_sufficient_size")
        gold_classes.append(gold_class)
        if bool(row.predicted_failure) or pd.isna(row.selected_size):
            predicted_classes.append(failure_class_index)
            exact_matches.append(False)
            within_one_matches.append(False)
            continue
        predicted_class = _ordinal_class_index(row.selected_size, class_index_by_size, "selected_size")
        predicted_classes.append(predicted_class)
        distance = abs(predicted_class - gold_class)
        exact_matches.append(distance == 0)
        within_one_matches.append(distance <= 1)

    if gold_classes:
        gold_array = np.asarray(gold_classes, dtype=np.int32)
        predicted_array = np.asarray(predicted_classes, dtype=np.int32)
        ordinal_mae = float(np.abs(predicted_array - gold_array).mean())
        qwk = _safe_metric_value(
            cohen_kappa_score,
            gold_array,
            predicted_array,
            weights="quadratic",
            labels=list(range(failure_class_index + 1)),
        )
        exact_class_accuracy = float(np.mean(exact_matches))
        within_one_class_accuracy = float(np.mean(within_one_matches))
    else:
        ordinal_mae = math.nan
        qwk = math.nan
        exact_class_accuracy = math.nan
        within_one_class_accuracy = math.nan

    return {
        "candidate_recall": _ordinal_candidate_recall(frame),
        "mean_predicted_k": float(non_failure.mean()) if not non_failure.empty else math.nan,
        "median_predicted_k": float(non_failure.median()) if not non_failure.empty else math.nan,
        "exact_class_accuracy": exact_class_accuracy,
        "within_one_class_accuracy": within_one_class_accuracy,
        "ordinal_mae": ordinal_mae,
        "quadratic_weighted_cohen_kappa": qwk,
    }


def ordinal_predictor_metric_table(frame: pd.DataFrame, candidate_sizes: list[int]) -> pd.DataFrame:
    metrics = ordinal_predictor_metric_summary(frame, candidate_sizes)
    return _metric_table(
        [
            ("Candidate Recall", _format_metric_value(metrics["candidate_recall"])),
            ("Mean Predicted k", _format_metric_value(metrics["mean_predicted_k"], decimals=2)),
            ("Median Predicted k", _format_metric_value(metrics["median_predicted_k"], decimals=2)),
            ("Exact Class Accuracy", _format_metric_value(metrics["exact_class_accuracy"])),
            ("Within-One-Class Accuracy", _format_metric_value(metrics["within_one_class_accuracy"])),
            ("Ordinal MAE", _format_metric_value(metrics["ordinal_mae"])),
            ("Quadratic Weighted Cohen's Kappa", _format_metric_value(metrics["quadratic_weighted_cohen_kappa"])),
        ]
    )


def _candidate_size_classes(frame: pd.DataFrame, ordered_sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    size_to_index = {size: index for index, size in enumerate(ordered_sizes)}
    unreachable_index = len(ordered_sizes)

    actual_raw = frame["smallest_sufficient_size"].to_numpy(dtype=np.float64)
    actual = np.full(len(frame), unreachable_index, dtype=np.int64)
    finite_actual = np.isfinite(actual_raw)
    actual[finite_actual] = [size_to_index[int(round(value))] for value in actual_raw[finite_actual]]

    predicted_failure = frame["predicted_failure"].to_numpy(dtype=bool)
    selected_raw = frame["selected_size"].to_numpy(dtype=np.float64)
    reachable_predicted = (~predicted_failure) & np.isfinite(selected_raw)
    predicted = np.full(len(frame), unreachable_index, dtype=np.int64)
    predicted[reachable_predicted] = [
        size_to_index[int(round(value))] for value in selected_raw[reachable_predicted]
    ]
    return actual, predicted


def candidate_size_class_breakdown_table(
    frames_by_retriever: dict[str, pd.DataFrame],
    candidate_sizes: list[int],
    retriever_display_names: dict[str, str],
    retriever_order: list[str],
    caption: str,
    label: str,
) -> str:
    ordered_sizes = list(dict.fromkeys(int(size) for size in candidate_sizes))
    class_labels = [str(size) for size in ordered_sizes] + ["Unreachable"]
    class_indices = list(range(len(class_labels)))

    available_retrievers = [name for name in retriever_order if name in frames_by_retriever]
    if not available_retrievers:
        raise ValueError("No retriever frames provided for candidate size class breakdown table")

    per_retriever_counts: dict[str, dict[str, np.ndarray]] = {}
    for retriever_name in available_retrievers:
        actual, predicted = _candidate_size_classes(frames_by_retriever[retriever_name], ordered_sizes)
        per_retriever_counts[retriever_name] = {
            "actual": np.bincount(actual, minlength=len(class_labels)),
            "predicted": np.bincount(predicted, minlength=len(class_labels)),
            "f1": f1_score(actual, predicted, labels=class_indices, average=None, zero_division=0),
        }

    column_spec = "l" + "|ccc" * len(available_retrievers)
    header_groups = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{retriever_display_names[name]}}}" for name in available_retrievers
    )
    header_sub = " & ".join("Predicted & Actual & F1" for _ in available_retrievers)

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        f" & {header_groups} \\\\",
        f"Candidate Size & {header_sub} \\\\",
        "\\midrule",
    ]
    for class_index, class_label in enumerate(class_labels):
        cells = [class_label]
        for retriever_name in available_retrievers:
            counts = per_retriever_counts[retriever_name]
            cells.append(str(int(counts["predicted"][class_index])))
            cells.append(str(int(counts["actual"][class_index])))
            cells.append(_format_metric_value(float(counts["f1"][class_index])))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def retriever_difficulty_correlation_results(
    frames_by_depth: dict[int, pd.DataFrame],
    *,
    retriever_name: str,
    feature_set: str,
) -> pd.DataFrame:
    if feature_set not in {
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
    }:
        raise ValueError(f"Unsupported feature_set '{feature_set}'")
    if not frames_by_depth:
        return pd.DataFrame(
            columns=[
                "split_name",
                "retriever_name",
                "retriever_family",
                "feature_set",
                "feature_group",
                "feature_name",
                "depth_dependent",
                "candidate_size",
                "n_total",
                "n_used",
                "missing_count",
                "unique_value_count",
                "kendall_tau_b",
                "kendall_p_value",
                "adjusted_p_value",
                "significant",
                "spearman_rho",
                "spearman_p_value",
                "correlation_direction",
            ]
        )

    retriever_family = _difficulty_correlation_retriever_family(retriever_name)
    anchor_depth = min(int(depth) for depth in frames_by_depth)
    anchor_frame = frames_by_depth[anchor_depth]
    split_name = str(anchor_frame["split_name"].iloc[0]) if not anchor_frame.empty else ""
    feature_specs = _difficulty_correlation_feature_specs(
        feature_names=[str(column) for column in anchor_frame.columns],
        feature_set=feature_set,
        retriever_family=retriever_family,
    )
    rows: list[dict[str, Any]] = []
    for spec in feature_specs:
        depths = sorted(frames_by_depth) if spec.depth_dependent else [anchor_depth]
        for depth in depths:
            frame = frames_by_depth[int(depth)]
            row = _difficulty_correlation_row(
                frame=frame,
                split_name=split_name,
                retriever_name=retriever_name,
                spec=spec,
                candidate_size=int(depth) if spec.depth_dependent else math.nan,
            )
            if row is not None:
                rows.append(row)
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return _apply_difficulty_correlation_adjustments(results)


def retriever_difficulty_correlation_analysis_for_runtime(
    runtime: Any,
    split_key: str,
    retriever_names: list[str],
    *,
    feature_sets: tuple[str, ...] = (
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
        DIFFICULTY_CORRELATION_FEATURE_SET_ALL,
    ),
    descriptive_candidate_sizes: list[int] | tuple[int, ...] | None = None,
    gold_aware_candidate_sizes: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    normalized_feature_sets = tuple(dict.fromkeys(str(item) for item in feature_sets))
    invalid_feature_sets = [
        item
        for item in normalized_feature_sets
        if item
        not in {
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
            DIFFICULTY_CORRELATION_FEATURE_SET_ALL,
        }
    ]
    if invalid_feature_sets:
        raise ValueError(f"Unsupported feature sets: {invalid_feature_sets}")

    all_rows: list[pd.DataFrame] = []
    for retriever_name in retriever_names:
        gold_independent = pd.DataFrame()
        gold_aware = pd.DataFrame()
        if (
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT in normalized_feature_sets
            or DIFFICULTY_CORRELATION_FEATURE_SET_ALL in normalized_feature_sets
        ):
            descriptive_frames = runtime.features.build_features(
                split_key,
                retriever_name,
                candidate_sizes=list(descriptive_candidate_sizes) if descriptive_candidate_sizes is not None else None,
            )
            gold_independent = retriever_difficulty_correlation_results(
                descriptive_frames,
                retriever_name=retriever_name,
                feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
            )
            if DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT in normalized_feature_sets and not gold_independent.empty:
                all_rows.append(gold_independent)
        if (
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE in normalized_feature_sets
            or DIFFICULTY_CORRELATION_FEATURE_SET_ALL in normalized_feature_sets
        ):
            gold_aware_frames = runtime.features.build_gold_aware_features(
                split_key,
                retriever_name,
                candidate_sizes=list(gold_aware_candidate_sizes) if gold_aware_candidate_sizes is not None else None,
            )
            gold_aware = retriever_difficulty_correlation_results(
                gold_aware_frames,
                retriever_name=retriever_name,
                feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
            )
            if DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE in normalized_feature_sets and not gold_aware.empty:
                all_rows.append(gold_aware)
        if DIFFICULTY_CORRELATION_FEATURE_SET_ALL in normalized_feature_sets:
            merged = _difficulty_correlation_union_results(gold_independent, gold_aware)
            if not merged.empty:
                all_rows.append(merged)
    if not all_rows:
        return pd.DataFrame()
    frame = pd.concat(all_rows, ignore_index=True)
    return frame.sort_values(
        ["feature_set", "retriever_family", "retriever_name", "feature_group", "feature_name", "candidate_size"],
        ascending=[True, True, True, True, True, True],
        na_position="first",
    ).reset_index(drop=True)


def retriever_difficulty_correlation_top_table(
    results: pd.DataFrame,
    *,
    feature_set: str,
    retriever_family: str = "all",
    top_k_per_retriever: int = 8,
    significant_only: bool = True,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["retriever", "group", "feature", "k", "tau_b", "rho", "adj_p", "n"])

    filtered = results.loc[results["feature_set"] == str(feature_set)].copy()
    if retriever_family != "all":
        filtered = filtered.loc[filtered["retriever_family"] == str(retriever_family)].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["retriever", "group", "feature", "k", "tau_b", "rho", "adj_p", "n"])
    if significant_only:
        significant = filtered.loc[filtered["significant"]].copy()
        if not significant.empty:
            filtered = significant
    filtered["abs_kendall_tau_b"] = filtered["kendall_tau_b"].abs()
    filtered = filtered.sort_values(
        ["retriever_name", "abs_kendall_tau_b", "adjusted_p_value", "feature_group", "feature_name", "candidate_size"],
        ascending=[True, False, True, True, True, True],
        na_position="last",
    )
    if top_k_per_retriever > 0:
        filtered = filtered.groupby("retriever_name", sort=False).head(int(top_k_per_retriever)).reset_index(drop=True)
    display = pd.DataFrame(
        {
            "retriever": filtered["retriever_name"],
            "group": filtered["feature_group"],
            "feature": filtered["feature_name"],
            "k": filtered["candidate_size"].map(_difficulty_correlation_format_depth),
            "tau_b": filtered["kendall_tau_b"].map(_difficulty_correlation_format_signed),
            "rho": filtered["spearman_rho"].map(_difficulty_correlation_format_signed),
            "adj_p": filtered["adjusted_p_value"].map(_difficulty_correlation_format_p_value),
            "n": filtered["n_used"].map(lambda value: str(int(value))),
        }
    )
    return display.reset_index(drop=True)


def cross_system_difficulty_correlation_results(
    frames_by_depth: dict[int, pd.DataFrame],
    *,
    retriever_name: str,
    is_unsolved: pd.Series,
) -> pd.DataFrame:
    empty_columns = [
        "split_name",
        "retriever_name",
        "retriever_family",
        "feature_set",
        "feature_group",
        "feature_name",
        "depth_dependent",
        "candidate_size",
        "n_total",
        "n_used",
        "missing_count",
        "unique_value_count",
        "mean_unsolved",
        "mean_rest",
        "kendall_tau_b",
        "kendall_p_value",
        "adjusted_p_value",
        "significant",
        "spearman_rho",
        "spearman_p_value",
        "correlation_direction",
    ]
    if not frames_by_depth:
        return pd.DataFrame(columns=empty_columns)

    retriever_family = _difficulty_correlation_retriever_family(retriever_name)
    anchor_depth = min(int(depth) for depth in frames_by_depth)
    anchor_frame = frames_by_depth[anchor_depth]
    split_name = str(anchor_frame["split_name"].iloc[0]) if not anchor_frame.empty else ""
    feature_specs = _difficulty_correlation_feature_specs(
        feature_names=[str(column) for column in anchor_frame.columns],
        feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        retriever_family=retriever_family,
    )
    rows: list[dict[str, Any]] = []
    for spec in feature_specs:
        depths = sorted(frames_by_depth) if spec.depth_dependent else [anchor_depth]
        for depth in depths:
            frame = frames_by_depth[int(depth)]
            row = _cross_system_difficulty_correlation_row(
                frame=frame,
                split_name=split_name,
                retriever_name=retriever_name,
                spec=spec,
                candidate_size=int(depth) if spec.depth_dependent else math.nan,
                is_unsolved=is_unsolved,
            )
            if row is not None:
                rows.append(row)
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    adjusted = _apply_difficulty_correlation_adjustments(results)
    adjusted["correlation_direction"] = adjusted["kendall_tau_b"].map(_cross_system_difficulty_correlation_direction)
    return adjusted


def _cross_system_difficulty_correlation_row(
    *,
    frame: pd.DataFrame,
    split_name: str,
    retriever_name: str,
    spec: DifficultyCorrelationFeatureSpec,
    candidate_size: float,
    is_unsolved: pd.Series,
) -> dict[str, Any] | None:
    if spec.feature_name not in frame.columns or "query_id" not in frame.columns:
        return None
    feature_values = pd.to_numeric(frame[spec.feature_name], errors="coerce").to_numpy(dtype=np.float64)
    target = frame["query_id"].astype(int).map(is_unsolved).to_numpy(dtype=np.float64)
    if feature_values.shape != target.shape:
        return None
    valid_mask = np.isfinite(feature_values) & np.isfinite(target)
    n_total = int(len(frame))
    n_used = int(valid_mask.sum())
    missing_count = int(n_total - n_used)
    unique_value_count = int(np.unique(feature_values[valid_mask]).size) if n_used > 0 else 0
    kendall_tau_b = math.nan
    kendall_p_value = math.nan
    spearman_rho = math.nan
    spearman_p_value = math.nan
    mean_unsolved = math.nan
    mean_rest = math.nan
    if n_used >= 2 and unique_value_count >= 2 and np.unique(target[valid_mask]).size >= 2:
        used_features = feature_values[valid_mask]
        used_target = target[valid_mask]
        kendall_result = kendalltau(used_features, used_target, variant="b", alternative="two-sided")
        kendall_tau_b = float(kendall_result.statistic) if kendall_result.statistic is not None else math.nan
        kendall_p_value = float(kendall_result.pvalue) if kendall_result.pvalue is not None else math.nan
        spearman_result = spearmanr(used_features, used_target, alternative="two-sided")
        spearman_rho = float(spearman_result.statistic) if spearman_result.statistic is not None else math.nan
        spearman_p_value = float(spearman_result.pvalue) if spearman_result.pvalue is not None else math.nan
        mean_unsolved = float(used_features[used_target == 1.0].mean())
        mean_rest = float(used_features[used_target == 0.0].mean())
    return {
        "split_name": split_name,
        "retriever_name": retriever_name,
        "retriever_family": spec.retriever_family,
        "feature_set": spec.feature_set,
        "feature_group": spec.feature_group,
        "feature_name": spec.feature_name,
        "depth_dependent": bool(spec.depth_dependent),
        "candidate_size": float(candidate_size) if np.isfinite(candidate_size) else math.nan,
        "n_total": n_total,
        "n_used": n_used,
        "missing_count": missing_count,
        "unique_value_count": unique_value_count,
        "mean_unsolved": mean_unsolved,
        "mean_rest": mean_rest,
        "kendall_tau_b": kendall_tau_b,
        "kendall_p_value": kendall_p_value,
        "adjusted_p_value": math.nan,
        "significant": False,
        "spearman_rho": spearman_rho,
        "spearman_p_value": spearman_p_value,
        "correlation_direction": _cross_system_difficulty_correlation_direction(kendall_tau_b),
    }


def _cross_system_difficulty_correlation_direction(value: float) -> str:
    if not np.isfinite(value):
        return "undetermined"
    if value > 0.0:
        return "harder_when_higher"
    if value < 0.0:
        return "harder_when_lower"
    return "neutral"


def _cross_system_difficulty_format_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def cross_system_difficulty_correlation_top_table(
    results: pd.DataFrame,
    *,
    top_k_per_retriever: int = 12,
    significant_only: bool = True,
) -> pd.DataFrame:
    display_columns = [
        "retriever",
        "group",
        "feature",
        "k",
        "mean_unsolved",
        "mean_rest",
        "tau_b",
        "rho",
        "adj_p",
        "direction",
        "n",
    ]
    if results.empty:
        return pd.DataFrame(columns=display_columns)

    filtered = results.copy()
    if significant_only:
        significant = filtered.loc[filtered["significant"]].copy()
        if not significant.empty:
            filtered = significant
    filtered["abs_kendall_tau_b"] = filtered["kendall_tau_b"].abs()
    filtered = filtered.sort_values(
        ["retriever_name", "abs_kendall_tau_b", "adjusted_p_value", "feature_group", "feature_name", "candidate_size"],
        ascending=[True, False, True, True, True, True],
        na_position="last",
    )
    if top_k_per_retriever > 0:
        filtered = filtered.groupby("retriever_name", sort=False).head(int(top_k_per_retriever)).reset_index(drop=True)
    display = pd.DataFrame(
        {
            "retriever": filtered["retriever_name"],
            "group": filtered["feature_group"],
            "feature": filtered["feature_name"],
            "k": filtered["candidate_size"].map(_difficulty_correlation_format_depth),
            "mean_unsolved": filtered["mean_unsolved"].map(_cross_system_difficulty_format_value),
            "mean_rest": filtered["mean_rest"].map(_cross_system_difficulty_format_value),
            "tau_b": filtered["kendall_tau_b"].map(_difficulty_correlation_format_signed),
            "rho": filtered["spearman_rho"].map(_difficulty_correlation_format_signed),
            "adj_p": filtered["adjusted_p_value"].map(_difficulty_correlation_format_p_value),
            "direction": filtered["correlation_direction"],
            "n": filtered["n_used"].map(lambda value: str(int(value))),
        }
    )
    return display.reset_index(drop=True)


def frame_to_latex(frame: pd.DataFrame, caption: str, label: str, **kwargs: Any) -> str:
    kwargs.setdefault("position", "H")
    kwargs.setdefault("escape", True)
    return frame.to_latex(index=False, caption=caption, label=label, **kwargs)


def _difficulty_correlation_retriever_family(retriever_name: str) -> str:
    base_retriever, _ = _split_retriever_variant_name(str(retriever_name))
    return "lexical" if base_retriever in LEXICAL_RETRIEVER_BASE_NAMES else "semantic"


def _difficulty_correlation_feature_specs(
    feature_names: list[str],
    *,
    feature_set: str,
    retriever_family: str,
) -> list[DifficultyCorrelationFeatureSpec]:
    specs: list[DifficultyCorrelationFeatureSpec] = []
    seen: set[str] = set()
    for feature_name in feature_names:
        if feature_name in DIFFICULTY_CORRELATION_EXCLUDED_COLUMNS or feature_name in seen:
            continue
        seen.add(feature_name)
        spec = _difficulty_correlation_feature_spec(
            feature_name=feature_name,
            feature_set=feature_set,
            retriever_family=retriever_family,
        )
        if spec is not None:
            specs.append(spec)
    return specs


def _difficulty_correlation_feature_spec(
    *,
    feature_name: str,
    feature_set: str,
    retriever_family: str,
) -> DifficultyCorrelationFeatureSpec | None:
    if feature_name in DIFFICULTY_CORRELATION_EXCLUDED_COLUMNS:
        return None
    if feature_set == DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT:
        group, depth_dependent = _gold_independent_feature_group(feature_name, retriever_family)
    elif feature_set == DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE:
        group, depth_dependent = _gold_aware_feature_group(feature_name)
    else:
        raise ValueError(f"Unsupported feature_set '{feature_set}'")
    return DifficultyCorrelationFeatureSpec(
        feature_name=feature_name,
        feature_group=group,
        depth_dependent=depth_dependent,
        feature_set=feature_set,
        retriever_family=retriever_family,
    )


def _gold_independent_feature_group(feature_name: str, retriever_family: str) -> tuple[str, bool]:
    if feature_name in GOLD_INDEPENDENT_SHARED_QUERY_FEATURES:
        return "lexical_pre_retrieval", False
    if retriever_family == "lexical":
        if feature_name in GOLD_INDEPENDENT_LEXICAL_TOP_DOCUMENT_FEATURES:
            return "lexical_post_retrieval_top_document", False
        if feature_name in GOLD_INDEPENDENT_LEXICAL_STATIC_RANKING_FEATURES:
            return "lexical_post_retrieval_ranking", False
        if feature_name in GOLD_INDEPENDENT_LEXICAL_DYNAMIC_FEATURES:
            return "lexical_post_retrieval_ranking", True
        return "lexical_post_retrieval_other", _difficulty_correlation_depth_hint(feature_name)
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_PRE_STATIC_FEATURES:
        return "semantic_pre_retrieval", False
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_PRE_DYNAMIC_FEATURES:
        return "semantic_pre_retrieval", True
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_SCORE_STATIC_FEATURES:
        return "semantic_post_retrieval_score", False
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_SCORE_DYNAMIC_FEATURES:
        return "semantic_post_retrieval_score", True
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_GEOMETRY_DYNAMIC_FEATURES:
        return "semantic_post_retrieval_geometry", True
    if feature_name in GOLD_INDEPENDENT_SEMANTIC_CLUSTER_DYNAMIC_FEATURES:
        return "semantic_post_retrieval_cluster", True
    if feature_name in GOLD_INDEPENDENT_RETRIEVER_AGREEMENT_FEATURES:
        return "retriever_agreement", False
    if feature_name in GOLD_INDEPENDENT_PERTURBATION_FEATURES:
        return "perturbation", True
    return "semantic_post_retrieval_other", _difficulty_correlation_depth_hint(feature_name)


def _gold_aware_feature_group(feature_name: str) -> tuple[str, bool]:
    if feature_name in GOLD_AWARE_SHARED_CONTEXT_FEATURES:
        return "document_and_query_context", False
    if feature_name in GOLD_AWARE_SHARED_SEMANTIC_CONTEXT_FEATURES:
        return "semantic_context_signals", False
    if feature_name in GOLD_AWARE_LEXICAL_MATCHING_FEATURES:
        return "gold_aware_lexical_matching", False
    if feature_name in GOLD_AWARE_LEXICAL_RANKING_STATIC_FEATURES:
        return "gold_aware_lexical_ranking", False
    if feature_name in GOLD_AWARE_LEXICAL_RANKING_DYNAMIC_FEATURES:
        return "gold_aware_lexical_ranking", True
    if feature_name in GOLD_AWARE_SEMANTIC_SCORE_STATIC_FEATURES:
        return "gold_aware_semantic_score", False
    if feature_name in GOLD_AWARE_SEMANTIC_SCORE_DYNAMIC_FEATURES:
        return "gold_aware_semantic_score", True
    if feature_name in GOLD_AWARE_SEMANTIC_GEOMETRY_DYNAMIC_FEATURES:
        return "gold_aware_semantic_geometry", True
    if feature_name in GOLD_AWARE_SPHERICAL_CLUSTER_STATIC_FEATURES:
        return "gold_aware_cluster_spherical", False
    if feature_name in GOLD_AWARE_SPHERICAL_CLUSTER_DYNAMIC_FEATURES:
        return "gold_aware_cluster_spherical", True
    if feature_name in GOLD_AWARE_HDBSCAN_CLUSTER_STATIC_FEATURES:
        return "gold_aware_cluster_hdbscan", False
    if feature_name in GOLD_AWARE_HDBSCAN_CLUSTER_DYNAMIC_FEATURES:
        return "gold_aware_cluster_hdbscan", True
    if feature_name in GOLD_AWARE_PERTURBATION_STATIC_FEATURES:
        return "gold_aware_perturbation", False
    if feature_name in GOLD_AWARE_PERTURBATION_DYNAMIC_FEATURES:
        return "gold_aware_perturbation", True
    return "gold_aware_other", _difficulty_correlation_depth_hint(feature_name)


def _difficulty_correlation_depth_hint(feature_name: str) -> bool:
    dynamic_markers = (
        "top_k",
        "rank_k",
        "candidate_set",
        "boundary_",
        "at_rank_k",
        "mean_top_k",
        "median_top_k",
        "pairwise",
        "graph_",
        "coherence",
        "entropy",
        "perturbation_retention",
        "coherency",
        "present_in_top_k",
        "share_in_top_k",
        "score_mass_in_top_k",
        "enrichment_in_top_k",
        "cosine_drop",
        "cosine_z_score",
        "score_z_score",
        "cosine_slope",
        "score_slope",
        "query_rr_at_k",
    )
    return any(marker in feature_name for marker in dynamic_markers)


def _difficulty_correlation_row(
    *,
    frame: pd.DataFrame,
    split_name: str,
    retriever_name: str,
    spec: DifficultyCorrelationFeatureSpec,
    candidate_size: float,
) -> dict[str, Any] | None:
    if spec.feature_name not in frame.columns or "rr_at_max_depth" not in frame.columns:
        return None
    feature_values = pd.to_numeric(frame[spec.feature_name], errors="coerce").to_numpy(dtype=np.float64)
    reciprocal_rank = pd.to_numeric(frame["rr_at_max_depth"], errors="coerce").to_numpy(dtype=np.float64)
    if feature_values.shape != reciprocal_rank.shape:
        return None
    valid_mask = np.isfinite(feature_values) & np.isfinite(reciprocal_rank)
    n_total = int(len(frame))
    n_used = int(valid_mask.sum())
    missing_count = int(n_total - n_used)
    unique_value_count = int(np.unique(feature_values[valid_mask]).size) if n_used > 0 else 0
    kendall_tau_b = math.nan
    kendall_p_value = math.nan
    spearman_rho = math.nan
    spearman_p_value = math.nan
    if n_used >= 2 and unique_value_count >= 2 and np.unique(reciprocal_rank[valid_mask]).size >= 2:
        kendall_result = kendalltau(
            feature_values[valid_mask],
            reciprocal_rank[valid_mask],
            variant="b",
            alternative="two-sided",
        )
        kendall_tau_b = float(kendall_result.statistic) if kendall_result.statistic is not None else math.nan
        kendall_p_value = float(kendall_result.pvalue) if kendall_result.pvalue is not None else math.nan
        spearman_result = spearmanr(feature_values[valid_mask], reciprocal_rank[valid_mask], alternative="two-sided")
        spearman_rho = float(spearman_result.statistic) if spearman_result.statistic is not None else math.nan
        spearman_p_value = float(spearman_result.pvalue) if spearman_result.pvalue is not None else math.nan
    return {
        "split_name": split_name,
        "retriever_name": retriever_name,
        "retriever_family": spec.retriever_family,
        "feature_set": spec.feature_set,
        "feature_group": spec.feature_group,
        "feature_name": spec.feature_name,
        "depth_dependent": bool(spec.depth_dependent),
        "candidate_size": float(candidate_size) if np.isfinite(candidate_size) else math.nan,
        "n_total": n_total,
        "n_used": n_used,
        "missing_count": missing_count,
        "unique_value_count": unique_value_count,
        "kendall_tau_b": kendall_tau_b,
        "kendall_p_value": kendall_p_value,
        "adjusted_p_value": math.nan,
        "significant": False,
        "spearman_rho": spearman_rho,
        "spearman_p_value": spearman_p_value,
        "correlation_direction": _difficulty_correlation_direction(kendall_tau_b),
    }


def _apply_difficulty_correlation_adjustments(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    adjusted = frame.copy()
    adjusted["adjusted_p_value"] = _benjamini_hochberg_adjust(adjusted["kendall_p_value"].to_numpy(dtype=np.float64))
    adjusted["significant"] = adjusted["adjusted_p_value"] < DIFFICULTY_CORRELATION_SIGNIFICANCE_LEVEL
    adjusted["correlation_direction"] = adjusted["kendall_tau_b"].map(_difficulty_correlation_direction)
    return adjusted


def _difficulty_correlation_union_results(
    gold_independent: pd.DataFrame,
    gold_aware: pd.DataFrame,
) -> pd.DataFrame:
    if gold_independent.empty and gold_aware.empty:
        return pd.DataFrame()
    combined = pd.concat([gold_independent, gold_aware], ignore_index=True)
    if combined.empty:
        return combined
    priority = combined["feature_set"].map(
        {
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT: 0,
            DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE: 1,
        }
    ).fillna(10)
    deduped = (
        combined.assign(_source_priority=priority)
        .sort_values(
            ["retriever_name", "feature_name", "candidate_size", "_source_priority"],
            ascending=[True, True, True, True],
            na_position="first",
        )
        .drop_duplicates(subset=["retriever_name", "feature_name", "candidate_size"], keep="first")
        .drop(columns="_source_priority")
        .reset_index(drop=True)
    )
    deduped["feature_set"] = DIFFICULTY_CORRELATION_FEATURE_SET_ALL
    return _apply_difficulty_correlation_adjustments(deduped)


def _benjamini_hochberg_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(values.shape, math.nan, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return adjusted
    finite_values = values[finite_mask]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    count = len(ranked)
    scaled = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    clipped = np.clip(monotone, 0.0, 1.0)
    restored = np.empty_like(clipped)
    restored[order] = clipped
    adjusted[finite_mask] = restored
    return adjusted


def _difficulty_correlation_direction(value: float) -> str:
    if not np.isfinite(value):
        return "undetermined"
    if value < 0.0:
        return "greater_difficulty"
    if value > 0.0:
        return "better_retrieval"
    return "neutral"


def _difficulty_correlation_format_depth(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return str(int(value))


def _difficulty_correlation_format_signed(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def _difficulty_correlation_format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _systems_range_text(label: str, system_count: int) -> str:
    if label == "Unsolved":
        return "0 systems"
    if label == "Rarely solved":
        return ">0-25% of systems"
    if label == "Minority solved":
        return ">25-50% of systems"
    if label == "Majority solved":
        return ">50-75% of systems"
    if label == "Widely solved":
        return ">75-<100% of systems"
    return "100% of systems"


def _systems_count_text(label: str, system_count: int) -> str:
    if label == "Unsolved":
        return "0"
    if label == "Rarely solved":
        return f"1-{max(1, int(system_count * 0.25))}"
    if label == "Minority solved":
        return f"{max(1, int(system_count * 0.25)) + 1}-{max(1, int(system_count * 0.5))}"
    if label == "Majority solved":
        return f"{max(1, int(system_count * 0.5)) + 1}-{max(1, int(system_count * 0.75))}"
    if label == "Widely solved":
        return f"{max(1, int(system_count * 0.75)) + 1}-{system_count - 1}"
    return str(system_count)


def _task_difficulty_analysis_submission_slug(system_name: str) -> str:
    if system_name.startswith("predictions_"):
        return system_name.removeprefix("predictions_")
    match = SUBMISSION_ARCHIVE_STEM_PATTERN.match(system_name)
    if match:
        return match.group(1)
    return system_name


def _task_difficulty_analysis_clean_slug(slug: str) -> str:
    parts = [part for part in re.split(r"[_-]+", slug) if part]
    return " ".join(part.capitalize() for part in parts)


def _embedding_anisotropy_row(
    retriever_name: str,
    embedding_path: Path,
    random_seed: int,
    document_sample_size: int,
    pair_sample_size: int,
) -> dict[str, float | int | str]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix in {embedding_path}, found shape {embeddings.shape}")
    document_count, embedding_dimension = embeddings.shape
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(document_count, size=min(document_sample_size, document_count), replace=False)
    sample = np.asarray(embeddings[sample_indices], dtype=np.float32)
    centroid = np.asarray(embeddings.mean(axis=0), dtype=np.float32)
    pair_cosine_mean, pair_cosine_p90 = _pairwise_cosine_summary(sample, rng, pair_sample_size)
    pc1_ratio, pc5_ratio = _sample_principal_component_mass(sample)
    base_retriever, document_scope = _split_retriever_variant_name(retriever_name)
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float32), axis=1)
    return {
        "retriever": retriever_name,
        "base_retriever": base_retriever,
        "document_scope": document_scope,
        "document_count": int(document_count),
        "embedding_dimension": int(embedding_dimension),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "pair_cosine_mean": pair_cosine_mean,
        "pair_cosine_p90": pair_cosine_p90,
        "pc1_ratio_sample": pc1_ratio,
        "pc5_ratio_sample": pc5_ratio,
    }


def _pairwise_cosine_summary(sample: np.ndarray, rng: np.random.Generator, pair_sample_size: int) -> tuple[float, float]:
    if len(sample) == 0:
        return float("nan"), float("nan")
    count = max(1, pair_sample_size)
    left = rng.integers(0, len(sample), size=count)
    right = rng.integers(0, len(sample), size=count)
    pair_cosines = np.sum(sample[left] * sample[right], axis=1, dtype=np.float32)
    return float(np.mean(pair_cosines)), float(np.quantile(pair_cosines, 0.9))


def _sample_principal_component_mass(sample: np.ndarray) -> tuple[float, float]:
    if len(sample) < 2:
        return float("nan"), float("nan")
    centered = sample - sample.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0, 0.0
    return float(eigenvalues[0] / total), float(eigenvalues[:5].sum() / total)


def _split_retriever_variant_name(retriever_name: str) -> tuple[str, str]:
    if "__" not in retriever_name:
        return retriever_name, "title_abstract"
    base_retriever, document_scope = retriever_name.rsplit("__", 1)
    return base_retriever, document_scope


def _deanisotropization_priority(rank: int, total: int) -> str:
    if rank <= max(1, int(np.ceil(total * 0.25))):
        return "high"
    if rank <= max(2, int(np.ceil(total * 0.5))):
        return "medium"
    return "lower"
