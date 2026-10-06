from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from _gold_correlation_matrix_tables import (
    DYNAMIC_FEATURE_NAME_COLUMN_WIDTH,
    ROW_RULE,
    STATIC_FEATURE_NAME_COLUMN_WIDTH,
    _feature_lookup,
    _format_number,
    _group_header_row,
    _latex_escape,
    _load_feature_groups,
    _ordered_feature_groups,
    _resolved_group_key,
)
from _gold_correlation_unsolved_comparison_tables import _split_frames_by_query_ids
from query_difficulty.analysis import (
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
    _difficulty_correlation_feature_specs,
    _difficulty_correlation_retriever_family,
    cross_system_unsolved_indicator,
)
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, compose_retriever_name
from query_difficulty.runtime import build_runtime


def cliffs_delta(values_u: np.ndarray, values_r: np.ndarray) -> float:
    values_u = np.asarray(values_u, dtype=np.float64)
    values_r = np.asarray(values_r, dtype=np.float64)
    values_u = values_u[np.isfinite(values_u)]
    values_r = values_r[np.isfinite(values_r)]
    n_u = values_u.size
    n_r = values_r.size
    if n_u == 0 or n_r == 0:
        return math.nan
    ranks = rankdata(np.concatenate([values_u, values_r]), method="average")
    rank_sum_u = float(ranks[:n_u].sum())
    u_statistic = rank_sum_u - n_u * (n_u + 1) / 2.0
    return float(2.0 * u_statistic / (n_u * n_r) - 1.0)


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def distribution_summary(values_u: np.ndarray, values_r: np.ndarray) -> dict[str, float]:
    finite_u = np.asarray(values_u, dtype=np.float64)
    finite_r = np.asarray(values_r, dtype=np.float64)
    return {
        "median_u": finite_median(finite_u),
        "median_r": finite_median(finite_r),
        "n_u": int(np.isfinite(finite_u).sum()),
        "n_r": int(np.isfinite(finite_r).sum()),
        "cliffs_delta": cliffs_delta(finite_u, finite_r),
    }


def _number_or_dash(value: float) -> str:
    return _format_number(value) if np.isfinite(value) else "--"


def build_static_distribution_table(
    grouped_rows: list[tuple[str, list[dict[str, float | str]]]],
    *,
    caption: str,
    label: str,
) -> str:
    header_row = r"Feature & Median (U) & Median (R) & Cliff's $\delta$ \\"
    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{longtable}{p{" + STATIC_FEATURE_NAME_COLUMN_WIDTH + r"}ccc}")
    lines.append(rf"\caption{{{caption}}} \label{{{label}}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{4}{l}{\emph{(continued from previous page)}} \\")
    lines.append(r"\hline")
    lines.append(header_row)
    lines.append(r"\hline")
    lines.append(r"\endhead")
    lines.append(r"\hline")
    lines.append(r"\multicolumn{4}{r}{\emph{continued on next page}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\hline")
    lines.append(r"\endlastfoot")
    for group_display_text, rows in grouped_rows:
        lines.append(r"\multicolumn{4}{l}{\emph{" + group_display_text + r"}} \\")
        for row in rows:
            lines.append(
                str(row["display_name"])
                + " & "
                + _number_or_dash(float(row["median_u"]))
                + " & "
                + _number_or_dash(float(row["median_r"]))
                + " & "
                + _number_or_dash(float(row["cliffs_delta"]))
                + r" \\"
            )
            lines.append(ROW_RULE)
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def _build_dynamic_delta_table(
    summary: pd.DataFrame,
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
        summary, dynamic_feature_names, group_order=group_order, feature_to_order=feature_to_order
    )
    n_columns = len(depth_columns) + 1
    header_row = " & ".join(["Feature"] + [rf"$\delta_{{k={depth}}}$" for depth in depth_columns]) + r" \\"
    column_spec = "p{" + DYNAMIC_FEATURE_NAME_COLUMN_WIDTH + "}" + "c" * len(depth_columns)

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{" + column_spec + "}")
    lines.append(rf"\caption{{{caption}}} \label{{{label}}} \\")
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
        feature_rows = summary[summary["feature_name"] == feature_name]
        cells = []
        for depth in depth_columns:
            depth_row = feature_rows[feature_rows["candidate_size"] == depth]
            if depth_row.empty:
                cells.append("")
                continue
            value = float(depth_row.iloc[0]["cliffs_delta"])
            cells.append(_format_number(value) if np.isfinite(value) else "")
        lines.append(" & ".join([display_name] + cells) + r" \\")
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def distribution_legend(unsolved_count: int, rest_count: int) -> str:
    return (
        r"\footnotesize "
        rf"U = queries unsolved by any participating system (gold not retrieved in the top~5 by any submitted run; $n={unsolved_count}$); "
        rf"R = the remaining (`rest') queries ($n={rest_count}$). "
        r"Medians are computed over queries with a defined feature value. "
        r"Cliff's $\delta = P(X_U > X_R) - P(X_U < X_R)$ with ties counted as one half; "
        r"$\delta > 0$ indicates larger feature values in U, $\delta < 0$ larger values in R."
    )


def _static_grouped_rows(
    summary: pd.DataFrame,
    static_feature_names: set[str],
    *,
    feature_to_display: dict[str, str],
    group_display: dict[str, str],
    group_order: dict[str, int],
) -> list[tuple[str, list[dict[str, float | str]]]]:
    feature_info = summary[summary["feature_name"].isin(static_feature_names)].drop_duplicates(subset=["feature_name"]).copy()
    feature_info["resolved_group"] = feature_info["feature_group"].map(_resolved_group_key)
    feature_info["group_order"] = feature_info["resolved_group"].map(group_order).fillna(10_000).astype(int)
    feature_info["abs_delta"] = feature_info["cliffs_delta"].abs().fillna(-1.0)
    feature_info = feature_info.sort_values(
        ["group_order", "resolved_group", "abs_delta", "feature_name"],
        ascending=[True, True, False, True],
    )
    grouped: list[tuple[str, list[dict[str, float | str]]]] = []
    for resolved_group, group_frame in feature_info.groupby("resolved_group", sort=False):
        rows = [
            {
                "display_name": _latex_escape(feature_to_display.get(record["feature_name"], record["feature_name"])),
                "median_u": record["median_u"],
                "median_r": record["median_r"],
                "cliffs_delta": record["cliffs_delta"],
            }
            for _, record in group_frame.iterrows()
        ]
        grouped.append((_latex_escape(group_display.get(resolved_group, resolved_group)), rows))
    return grouped


def unsolved_vs_rest_distribution_summary(
    frames_by_depth: dict[int, pd.DataFrame],
    *,
    unsolved_ids: set[int],
    rest_ids: set[int],
    retriever_name: str,
    feature_set: str,
    console,
) -> pd.DataFrame:
    if not frames_by_depth:
        return pd.DataFrame()
    frames_u = _split_frames_by_query_ids(frames_by_depth, unsolved_ids)
    frames_r = _split_frames_by_query_ids(frames_by_depth, rest_ids)
    anchor_depth = min(int(depth) for depth in frames_by_depth)
    specs = _difficulty_correlation_feature_specs(
        feature_names=[str(column) for column in frames_by_depth[anchor_depth].columns],
        feature_set=feature_set,
        retriever_family=_difficulty_correlation_retriever_family(retriever_name),
    )
    console.log(f"Summarising {len(specs)} feature(s) across U and R")
    rows = []
    for position, spec in enumerate(specs, start=1):
        depths = sorted(frames_by_depth) if spec.depth_dependent else [anchor_depth]
        for depth in depths:
            frame_u = frames_u[int(depth)]
            frame_r = frames_r[int(depth)]
            if spec.feature_name not in frame_u.columns or spec.feature_name not in frame_r.columns:
                continue
            values_u = pd.to_numeric(frame_u[spec.feature_name], errors="coerce").to_numpy(dtype=np.float64)
            values_r = pd.to_numeric(frame_r[spec.feature_name], errors="coerce").to_numpy(dtype=np.float64)
            rows.append(
                {
                    "feature_name": spec.feature_name,
                    "feature_group": spec.feature_group,
                    "depth_dependent": spec.depth_dependent,
                    "candidate_size": int(depth) if spec.depth_dependent else math.nan,
                    **distribution_summary(values_u, values_r),
                }
            )
        console.log(
            f"[{position}/{len(specs)}] {spec.feature_name} "
            f"({'depth-dependent' if spec.depth_dependent else 'static'}, {len(depths)} depth(s))"
        )
    return pd.DataFrame(rows)


def build_unsolved_distribution_tables(
    summary: pd.DataFrame,
    *,
    static_caption: str,
    static_label: str,
    dynamic_caption: str,
    dynamic_label: str,
    unsolved_count: int,
    rest_count: int,
) -> str:
    if summary.empty:
        raise ValueError("No distribution rows available for the requested retriever/feature_set/split")
    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)

    defined = summary[summary["cliffs_delta"].notna() | summary["median_u"].notna() | summary["median_r"].notna()]
    static_feature_names = set(defined.loc[~defined["depth_dependent"], "feature_name"].unique())
    dynamic_feature_names = set(defined.loc[defined["depth_dependent"], "feature_name"].unique())
    depth_columns = sorted(
        int(value) for value in summary.loc[summary["depth_dependent"], "candidate_size"].dropna().unique()
    )

    parts: list[str] = []
    if static_feature_names:
        parts.append(
            build_static_distribution_table(
                _static_grouped_rows(
                    summary,
                    static_feature_names,
                    feature_to_display=feature_to_display,
                    group_display=group_display,
                    group_order=group_order,
                ),
                caption=_latex_escape(static_caption),
                label=static_label,
            )
        )
    if dynamic_feature_names:
        parts.append(
            _build_dynamic_delta_table(
                summary,
                dynamic_feature_names=dynamic_feature_names,
                depth_columns=depth_columns,
                feature_to_display=feature_to_display,
                feature_to_order=feature_to_order,
                group_display=group_display,
                group_order=group_order,
                caption=_latex_escape(dynamic_caption),
                label=dynamic_label,
            )
        )
    parts.append(distribution_legend(unsolved_count, rest_count))
    return "\n\n".join(parts) + "\n"


def load_unsolved_and_rest_ids(runtime, split: str, language: str) -> tuple[set[int], set[int]]:
    console = runtime.console
    console.log("Loading gold queries and submission archives to determine unsolved queries")
    gold_queries = runtime.dataset.load().queries[split]
    submission_dir = runtime.config.root_dir / "data" / "submissions"
    is_unsolved = cross_system_unsolved_indicator(submission_dir, gold_queries, language)
    unsolved_ids = {int(query_id) for query_id in is_unsolved[is_unsolved == 1].index}
    rest_ids = {int(query_id) for query_id in is_unsolved[is_unsolved == 0].index}
    console.log(f"Found {len(unsolved_ids)} unsolved and {len(rest_ids)} rest queries")
    return unsolved_ids, rest_ids


def write_unsolved_distribution_tables(
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

    console.section(f"Unsolved-vs-rest distribution tables: {retriever_base_name} / {feature_set} / {split}")
    unsolved_ids, rest_ids = load_unsolved_and_rest_ids(runtime, split, language)

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

    summary = unsolved_vs_rest_distribution_summary(
        frames,
        unsolved_ids=unsolved_ids,
        rest_ids=rest_ids,
        retriever_name=retriever_name,
        feature_set=feature_set,
        console=console,
    )
    console.log(f"Computed {len(summary)} distribution row(s)")

    unsolved_count = len(unsolved_ids)
    rest_count = len(rest_ids)
    console.log("Building LaTeX distribution tables")
    latex = build_unsolved_distribution_tables(
        summary,
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
