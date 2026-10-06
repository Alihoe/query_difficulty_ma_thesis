from __future__ import annotations

import pandas as pd

from _gold_correlation_matrix_tables import ROW_RULE, _group_header_row, _latex_escape, _ordered_feature_groups
from _in_top_max_depth_importance_tables import _signed_importance, split_depth_prefixed_feature
from query_difficulty.analysis import _gold_independent_feature_group

FEATURE_NAME_COLUMN_WIDTH = "0.30\\linewidth"
COMPACT_FEATURE_NAME_COLUMN_WIDTH = "0.28\\linewidth"


def _signed_share(share: float, direction: str) -> str:
    magnitude = f"{share:.1f}"
    if direction == "negative":
        return f"$-${magnitude}"
    if direction == "positive":
        return f"+{magnitude}"
    return magnitude


def direction_from_training(training_frame: pd.DataFrame, target: pd.Series, identifier_columns) -> dict[str, str]:
    directions: dict[str, str] = {}
    for col in training_frame.columns:
        if col in identifier_columns:
            continue
        series = training_frame[col]
        if series.nunique() <= 1:
            directions[col] = "flat"
            continue
        mean_1 = series[target].mean()
        mean_0 = series[~target].mean()
        directions[col] = "positive" if mean_1 > mean_0 else "negative"
    return directions


def aggregate_importance_by_base_feature(importances: pd.DataFrame, directions: dict[str, str]) -> pd.DataFrame:
    working = importances.copy()
    split = working["feature"].map(split_depth_prefixed_feature)
    working["base_feature"] = split.map(lambda pair: pair[1])
    working["direction"] = working["feature"].map(directions).fillna("unknown")
    summed = working.groupby("base_feature", as_index=False).agg(
        importance=("importance", "sum"), direction=("direction", "first")
    )
    return summed


def build_across_depth_importance_table(
    per_depth_summed: dict[int, pd.DataFrame],
    *,
    importance_threshold: float,
    retriever_family: str,
    feature_to_display: dict[str, str],
    feature_to_order: dict[str, int],
    group_display: dict[str, str],
    group_order: dict[str, int],
    caption: str,
    label: str,
    include_average: bool = False,
    group_rows: bool = True,
    main_depths: dict[str, str] | None = None,
    average_threshold: float | None = None,
) -> str:
    combined = []
    for depth, summed in per_depth_summed.items():
        frame = summed.copy()
        frame["depth"] = depth
        combined.append(frame)
    combined_frame = pd.concat(combined, ignore_index=True)

    depth_columns = sorted(per_depth_summed.keys())
    importance_by_depth = (
        combined_frame.pivot_table(index="base_feature", columns="depth", values="importance", aggfunc="first")
        .reindex(columns=depth_columns)
        .fillna(0.0)
    )
    if main_depths is not None:
        importance_by_depth = importance_by_depth / combined_frame.groupby("depth")["importance"].sum() * 100
    average_by_feature = importance_by_depth.mean(axis=1)

    if average_threshold is None:
        max_by_feature = combined_frame.groupby("base_feature")["importance"].max()
        qualifying_features = set(max_by_feature[max_by_feature >= importance_threshold].index)
    else:
        qualifying_features = set(average_by_feature[average_by_feature > average_threshold].index)
    if not qualifying_features:
        return ""

    combined_frame["feature_group"] = combined_frame["base_feature"].map(
        lambda name: _gold_independent_feature_group(name, retriever_family)[0]
    )
    lookup_frame = combined_frame.rename(columns={"base_feature": "feature_name"})
    feature_info = _ordered_feature_groups(
        lookup_frame, qualifying_features, group_order=group_order, feature_to_order=feature_to_order
    )

    if include_average:
        feature_info["average_importance"] = feature_info["feature_name"].map(average_by_feature)
        group_keys = ["group_order", "resolved_group"] if group_rows else []
        feature_info = feature_info.sort_values(
            group_keys + ["average_importance", "feature_name"],
            ascending=[True] * len(group_keys) + [False, True],
        )
    qualifying_frame = combined_frame[combined_frame["base_feature"].isin(qualifying_features)]
    column_best = qualifying_frame.groupby("depth")["importance"].max()
    column_total = combined_frame.groupby("depth")["importance"].sum()
    if main_depths is None:
        header_cells = ["Feature"] + [f"$k={depth}$" for depth in depth_columns] + (["Average"] if include_average else [])
        n_columns = len(header_cells)
        header_row = " & ".join(header_cells) + r" \\"
        column_spec = "p{" + FEATURE_NAME_COLUMN_WIDTH + "}" + "c" * (n_columns - 1)
    else:
        n_columns = 2 + len(depth_columns) + (1 if include_average else 0)
        header_row = (
            r" & & \multicolumn{" + str(len(depth_columns)) + r"}{c}{$k$}" + (" &" if include_average else "") + r" \\" + "\n"
            + " & ".join(["Feature", "Depth"] + [str(depth) for depth in depth_columns] + (["Average"] if include_average else [])) + r" \\"
        )
        column_spec = "p{" + COMPACT_FEATURE_NAME_COLUMN_WIDTH + "}c" + "r" * len(depth_columns) + ("|r" if include_average else "")

    lines: list[str] = []
    lines.append(r"\begingroup")
    lines.append(r"\footnotesize" if main_depths is None else r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}" if main_depths is None else r"\setlength{\tabcolsep}{2pt}")
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
        if group_rows and resolved_group != current_group:
            lines.append(_group_header_row(resolved_group, group_display, n_columns))
            current_group = resolved_group

        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        feature_rows = combined_frame[combined_frame["base_feature"] == feature_name]

        cells = []
        for depth in depth_columns:
            depth_row = feature_rows[feature_rows["depth"] == depth]
            if depth_row.empty:
                cells.append("")
                continue
            record = depth_row.iloc[0]
            if main_depths is None:
                text = _signed_importance(float(record["importance"]), str(record["direction"]))
            else:
                text = _signed_share(float(record["importance"]) / float(column_total[depth]) * 100, str(record["direction"]))
            cells.append(f"\\textbf{{{text}}}" if float(record["importance"]) == column_best[depth] else text)
        if include_average:
            cells.append(f"{float(average_by_feature[feature_name]):{'.4f' if main_depths is None else '.1f'}}")
        leading = [display_name] if main_depths is None else [display_name, main_depths.get(feature_name, "")]
        lines.append(" & ".join(leading + cells) + r" \\")
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)
