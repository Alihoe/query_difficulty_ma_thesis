from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _gold_correlation_matrix_tables import (
    DYNAMIC_FEATURE_NAME_COLUMN_WIDTH,
    ROW_RULE,
    SYMBOL_LEGEND,
    SYMBOL_NO_CORRELATION,
    _correlation_symbol,
    _feature_lookup,
    _group_header_row,
    _latex_escape,
    _legend_symbol,
    _load_feature_groups,
    _ordered_feature_groups,
    _resolved_group_key,
    _format_number,
)
from query_difficulty.analysis import (
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
    cross_system_unsolved_indicator,
    retriever_difficulty_correlation_results,
)
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, compose_retriever_name
from query_difficulty.runtime import build_runtime

STATIC_COMPARISON_FEATURE_NAME_COLUMN_WIDTH = "0.26\\linewidth"

COMPARISON_ROW_KEY_COLUMNS = ["feature_name", "feature_group", "depth_dependent", "candidate_size"]


def _split_frames_by_query_ids(frames_by_depth: dict[int, pd.DataFrame], query_ids: set[int]) -> dict[int, pd.DataFrame]:
    return {
        depth: frame.loc[frame["query_id"].astype(int).isin(query_ids)].copy()
        for depth, frame in frames_by_depth.items()
    }


def _diverges(tau_u: float, significant_u: bool, tau_r: float, significant_r: bool) -> bool:
    symbol_u = _correlation_symbol(tau_u, bool(significant_u)) if np.isfinite(tau_u) else SYMBOL_NO_CORRELATION
    symbol_r = _correlation_symbol(tau_r, bool(significant_r)) if np.isfinite(tau_r) else SYMBOL_NO_CORRELATION
    has_u = symbol_u != SYMBOL_NO_CORRELATION
    has_r = symbol_r != SYMBOL_NO_CORRELATION
    if has_u and has_r:
        return (tau_u > 0) != (tau_r > 0)
    return has_u != has_r


def _merge_comparison_frames(results_u: pd.DataFrame, results_r: pd.DataFrame) -> pd.DataFrame:
    if results_u.empty or results_r.empty:
        return pd.DataFrame()
    value_columns = ["kendall_tau_b", "significant", "spearman_rho", "n_used"]
    left = results_u[COMPARISON_ROW_KEY_COLUMNS + value_columns].copy()
    right = results_r[COMPARISON_ROW_KEY_COLUMNS + value_columns].copy()
    sentinel = -1.0
    left["candidate_size"] = left["candidate_size"].fillna(sentinel)
    right["candidate_size"] = right["candidate_size"].fillna(sentinel)
    merged = left.merge(right, on=COMPARISON_ROW_KEY_COLUMNS, how="inner", suffixes=("_u", "_r"))
    merged["candidate_size"] = merged["candidate_size"].replace(sentinel, math.nan)
    merged["delta_tau_b"] = merged["kendall_tau_b_u"] - merged["kendall_tau_b_r"]
    merged["diverges"] = [
        _diverges(tau_u, significant_u, tau_r, significant_r)
        for tau_u, significant_u, tau_r, significant_r in zip(
            merged["kendall_tau_b_u"],
            merged["significant_u"],
            merged["kendall_tau_b_r"],
            merged["significant_r"],
            strict=True,
        )
    ]
    return merged


def _comparison_features_with_any_signal(merged: pd.DataFrame, feature_names: set[str]) -> set[str]:
    defined = merged[
        merged["feature_name"].isin(feature_names)
        & (merged["kendall_tau_b_u"].notna() | merged["kendall_tau_b_r"].notna())
    ]
    return set(defined["feature_name"].unique())


def _static_comparison_feature_order(
    merged: pd.DataFrame,
    feature_names: set[str],
    *,
    group_order: dict[str, int],
) -> pd.DataFrame:
    feature_info = (
        merged[merged["feature_name"].isin(feature_names)].drop_duplicates(subset=["feature_name"]).copy()
    )
    feature_info["resolved_group"] = feature_info["feature_group"].map(_resolved_group_key)
    feature_info["group_order"] = feature_info["resolved_group"].map(group_order).fillna(10_000).astype(int)
    feature_info["abs_delta"] = feature_info["delta_tau_b"].abs().fillna(-1.0)
    return feature_info.sort_values(
        ["group_order", "resolved_group", "abs_delta", "feature_name"],
        ascending=[True, True, False, True],
    )


def _comparison_symbol_text(tau_b: float, significant: bool) -> str:
    if not np.isfinite(tau_b):
        return SYMBOL_NO_CORRELATION
    return _correlation_symbol(tau_b, bool(significant))


def _build_static_comparison_table(
    merged: pd.DataFrame,
    *,
    static_feature_names: set[str],
    feature_to_display: dict[str, str],
    group_display: dict[str, str],
    group_order: dict[str, int],
    caption: str,
    label: str,
) -> str:
    feature_info = _static_comparison_feature_order(merged, static_feature_names, group_order=group_order)

    header_row = r"Feature & Dir (U) & $\tau_b$ (U) & Dir (R) & $\tau_b$ (R) & $\Delta\tau_b$ \\"
    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{longtable}{p{" + STATIC_COMPARISON_FEATURE_NAME_COLUMN_WIDTH + r"}ccccc}")
    lines.append(rf"\caption{{{_latex_escape(caption)}}} \label{{{label}}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{6}{l}{\emph{(continued from previous page)}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endhead")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{6}{r}{\emph{continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")

    current_group: str | None = None
    for _, feature_row in feature_info.iterrows():
        feature_name = feature_row["feature_name"]
        resolved_group = feature_row["resolved_group"]
        if resolved_group != current_group:
            lines.append(_group_header_row(resolved_group, group_display, 6))
            current_group = resolved_group

        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        record = merged[merged["feature_name"] == feature_name].iloc[0]
        tau_u = float(record["kendall_tau_b_u"])
        tau_r = float(record["kendall_tau_b_r"])
        symbol_u = _legend_symbol(_comparison_symbol_text(tau_u, record["significant_u"]))
        symbol_r = _legend_symbol(_comparison_symbol_text(tau_r, record["significant_r"]))
        tau_u_text = _format_number(tau_u) if np.isfinite(tau_u) else "--"
        tau_r_text = _format_number(tau_r) if np.isfinite(tau_r) else "--"
        delta_value = record["delta_tau_b"]
        delta_text = _format_number(delta_value) if np.isfinite(delta_value) else "--"
        if bool(record["diverges"]):
            delta_text = r"\textbf{" + delta_text + "}"
        lines.append(
            display_name
            + " & "
            + symbol_u
            + " & "
            + tau_u_text
            + " & "
            + symbol_r
            + " & "
            + tau_r_text
            + " & "
            + delta_text
            + r" \\"
        )
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def _build_dynamic_comparison_table(
    merged: pd.DataFrame,
    *,
    dynamic_feature_names: set[str],
    depth_columns: list[int],
    feature_to_display: dict[str, str],
    feature_to_order: dict[str, int],
    group_display: dict[str, str],
    group_order: dict[str, int],
    caption: str,
    label: str,
) -> str:
    feature_info = _ordered_feature_groups(
        merged, dynamic_feature_names, group_order=group_order, feature_to_order=feature_to_order
    )
    n_columns = len(depth_columns) + 1

    header_row = " & ".join(["Feature"] + [f"$k={depth}$" for depth in depth_columns]) + r" \\"
    column_spec = "p{" + DYNAMIC_FEATURE_NAME_COLUMN_WIDTH + "}" + "c" * len(depth_columns)

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{" + column_spec + "}")
    lines.append(rf"\caption{{{_latex_escape(caption)}}} \label{{{label}}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{" + str(n_columns) + r"}{l}{\emph{(continued from previous page)}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endhead")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{" + str(n_columns) + r"}{r}{\emph{continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")

    current_group: str | None = None
    for _, feature_row in feature_info.iterrows():
        feature_name = feature_row["feature_name"]
        resolved_group = feature_row["resolved_group"]
        if resolved_group != current_group:
            lines.append(_group_header_row(resolved_group, group_display, n_columns))
            current_group = resolved_group

        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        feature_rows = merged[merged["feature_name"] == feature_name]

        cells = []
        for depth in depth_columns:
            depth_row = feature_rows[feature_rows["candidate_size"] == depth]
            if depth_row.empty or (
                pd.isna(depth_row.iloc[0]["kendall_tau_b_u"]) and pd.isna(depth_row.iloc[0]["kendall_tau_b_r"])
            ):
                cells.append("")
                continue
            record = depth_row.iloc[0]
            symbol_u = _comparison_symbol_text(float(record["kendall_tau_b_u"]), record["significant_u"])
            symbol_r = _comparison_symbol_text(float(record["kendall_tau_b_r"]), record["significant_r"])
            cell = symbol_u + "/" + symbol_r
            if bool(record["diverges"]):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(" & ".join([display_name] + cells) + r" \\")
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def _comparison_legend(unsolved_count: int, rest_count: int) -> str:
    return (
        r"\footnotesize "
        rf"U = queries unsolved by any participating system (gold not retrieved in the top~5 by any submitted run; $n={unsolved_count}$); "
        rf"R = the remaining (`rest') queries ($n={rest_count}$). Arrow symbols follow the single-subset correlation legend:\\[0.4em]"
        + SYMBOL_LEGEND
        + r"\\[0.4em]\textbf{Bold} entries mark features where the U and R correlation directions disagree "
        r"while at least one side is significant, or where the correlation is significant in only one of the two subsets."
    )


def build_unsolved_comparison_tables(
    merged: pd.DataFrame,
    *,
    static_caption: str,
    static_label: str,
    dynamic_caption: str,
    dynamic_label: str,
    unsolved_count: int,
    rest_count: int,
) -> str:
    if merged.empty:
        raise ValueError("No comparison rows available for the requested retriever/feature_set/split")

    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)

    static_all = set(merged.loc[~merged["depth_dependent"], "feature_name"].unique())
    dynamic_all = set(merged.loc[merged["depth_dependent"], "feature_name"].unique())
    static_feature_names = _comparison_features_with_any_signal(merged, static_all)
    dynamic_feature_names = _comparison_features_with_any_signal(merged, dynamic_all)
    depth_columns = sorted(
        int(value) for value in merged.loc[merged["depth_dependent"], "candidate_size"].dropna().unique()
    )

    parts: list[str] = []
    if static_feature_names:
        parts.append(
            _build_static_comparison_table(
                merged,
                static_feature_names=static_feature_names,
                feature_to_display=feature_to_display,
                group_display=group_display,
                group_order=group_order,
                caption=static_caption,
                label=static_label,
            )
        )
    if dynamic_feature_names:
        parts.append(
            _build_dynamic_comparison_table(
                merged,
                dynamic_feature_names=dynamic_feature_names,
                depth_columns=depth_columns,
                feature_to_display=feature_to_display,
                feature_to_order=feature_to_order,
                group_display=group_display,
                group_order=group_order,
                caption=dynamic_caption,
                label=dynamic_label,
            )
        )
    parts.append(_comparison_legend(unsolved_count, rest_count))
    return "\n\n".join(parts) + "\n"


def write_unsolved_comparison_tables(
    *,
    config: str,
    language: str,
    split: str,
    feature_set: str,
    retriever_base_name: str,
    output_name: str,
    static_caption_template: str,
    static_label_template: str,
    dynamic_caption_template: str,
    dynamic_label_template: str,
    descriptive_candidate_size: list[int] | None = None,
    gold_aware_candidate_size: list[int] | None = None,
) -> Path:
    runtime = build_runtime(config, verbose=True)
    console = runtime.console

    console.section(f"Unsolved-comparison tables: {retriever_base_name} / {feature_set} / {split}")
    console.log("Loading gold queries and submission archives to determine unsolved queries")
    gold_queries = runtime.dataset.load().queries[split]
    submission_dir = runtime.config.root_dir / "data" / "submissions"
    is_unsolved = cross_system_unsolved_indicator(submission_dir, gold_queries, language)
    unsolved_ids = {int(query_id) for query_id in is_unsolved[is_unsolved == 1].index}
    rest_ids = {int(query_id) for query_id in is_unsolved[is_unsolved == 0].index}
    console.log(f"Found {len(unsolved_ids)} unsolved and {len(rest_ids)} rest queries")

    retriever_name = compose_retriever_name(retriever_base_name, DEFAULT_DOCUMENT_SCOPE)
    if feature_set == DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE:
        console.log(f"Building gold-aware features for {split} / {retriever_name} (this may take a while if uncached)")
        frames = runtime.features.build_gold_aware_features(
            split,
            retriever_name,
            candidate_sizes=list(gold_aware_candidate_size) if gold_aware_candidate_size is not None else None,
        )
    else:
        console.log(f"Building gold-independent features for {split} / {retriever_name}")
        frames = runtime.features.build_features(
            split,
            retriever_name,
            candidate_sizes=list(descriptive_candidate_size) if descriptive_candidate_size is not None else None,
        )
    console.log(f"Features ready for {len(frames)} candidate-set depth(s)")

    console.log("Splitting feature frames into unsolved (U) and rest (R) subsets")
    frames_u = _split_frames_by_query_ids(frames, unsolved_ids)
    frames_r = _split_frames_by_query_ids(frames, rest_ids)

    console.log("Computing difficulty correlations on the unsolved (U) subset")
    results_u = retriever_difficulty_correlation_results(frames_u, retriever_name=retriever_name, feature_set=feature_set)
    console.log("Computing difficulty correlations on the rest (R) subset")
    results_r = retriever_difficulty_correlation_results(frames_r, retriever_name=retriever_name, feature_set=feature_set)
    console.log(f"Merging {len(results_u)} (U) and {len(results_r)} (R) correlation rows")
    merged = _merge_comparison_frames(results_u, results_r)

    unsolved_count = len(unsolved_ids)
    rest_count = len(rest_ids)
    console.log("Building LaTeX comparison tables")
    latex = build_unsolved_comparison_tables(
        merged,
        static_caption=static_caption_template.format(split=split, unsolved_count=unsolved_count, rest_count=rest_count),
        static_label=static_label_template.format(split=split),
        dynamic_caption=dynamic_caption_template.format(split=split, unsolved_count=unsolved_count, rest_count=rest_count),
        dynamic_label=dynamic_label_template.format(split=split),
        unsolved_count=unsolved_count,
        rest_count=rest_count,
    )
    output_path = runtime.config.latex_table_dir / output_name.format(split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    console.log(f"Wrote {output_path}")
    return output_path
