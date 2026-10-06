from __future__ import annotations

import pandas as pd

from _gold_correlation_matrix_tables import ROW_RULE, _latex_escape, _ordered_feature_groups
from query_difficulty.analysis import _gold_independent_feature_group

STATIC_FEATURE_NAME_COLUMN_WIDTH = "0.45\\linewidth"
RETRIEVER_DISPLAY = {"bm25": "\\textit{BM25}", "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B"}


def retriever_display(retriever: str) -> str:
    return RETRIEVER_DISPLAY.get(retriever, retriever)


def importance_longtable(rows: list[tuple[str, str, float, float]], *, caption: str, label: str) -> str:
    header_row = r"Feature & Direction & Importance & \% of total \\"
    lines: list[str] = []
    lines.append(r"\begin{longtable}{p{" + STATIC_FEATURE_NAME_COLUMN_WIDTH + r"}crr}")
    lines.append(rf"\caption{{{_latex_escape(caption)}}} \label{{{label}}} \\")
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
    for display_name, direction, importance, percent in rows:
        lines.append(
            display_name
            + " & "
            + _direction_symbol(direction)
            + " & "
            + f"{importance:.6f}"
            + " & "
            + f"{percent:.2f}\\%"
            + r" \\"
        )
        lines.append(ROW_RULE)
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def build_flat_importance_table(
    importances: pd.DataFrame,
    *,
    min_percent: float,
    caption: str,
    label: str,
) -> str:
    total_importance = float(importances["importance"].sum())
    ranked = importances[importances["importance"] > 0].copy()
    ranked["percent"] = ranked["importance"] / total_importance * 100.0
    ranked = ranked[ranked["percent"] > min_percent].sort_values(["importance", "display_name"], ascending=[False, True])
    rows = [
        (_latex_escape(name), str(direction), float(importance), float(percent))
        for name, direction, importance, percent in zip(
            ranked["display_name"], ranked["direction"], ranked["importance"], ranked["percent"]
        )
    ]
    return importance_longtable(rows, caption=caption, label=label) if rows else ""


def split_depth_prefixed_feature(feature: str) -> tuple[int | None, str]:
    if "::" not in feature:
        return None, feature
    prefix, base_name = feature.split("::", 1)
    if not prefix.startswith("depth_"):
        return None, feature
    return int(prefix[len("depth_") :]), base_name


def classify_importances(
    importances: pd.DataFrame, directions: dict[str, str], retriever_family: str
) -> pd.DataFrame:
    classified = importances.copy()
    split = classified["feature"].map(split_depth_prefixed_feature)
    classified["depth"] = split.map(lambda pair: pair[0])
    classified["base_feature"] = split.map(lambda pair: pair[1])
    classified["direction"] = classified["feature"].map(directions).fillna("unknown")
    groups = classified["base_feature"].map(lambda name: _gold_independent_feature_group(name, retriever_family))
    classified["feature_group"] = groups.map(lambda pair: pair[0])
    classified["depth_dependent"] = groups.map(lambda pair: bool(pair[1]))
    return classified


def _direction_symbol(direction: str) -> str:
    if direction == "positive":
        return "+"
    if direction == "negative":
        return "$-$"
    return ""


def _signed_importance(value: float, direction: str) -> str:
    magnitude = f"{value:.4f}"
    if direction == "negative":
        return f"$-${magnitude}"
    if direction == "positive":
        return f"+{magnitude}"
    return magnitude


def _order_by_importance(feature_info: pd.DataFrame, importance_by_feature: pd.Series) -> pd.DataFrame:
    ordered = feature_info.copy()
    ordered["total_importance"] = ordered["feature_name"].map(importance_by_feature)
    return ordered.sort_values(["total_importance", "feature_name"], ascending=[False, True])


def build_static_importance_table(
    classified: pd.DataFrame,
    *,
    total_importance: float,
    feature_to_display: dict[str, str],
    feature_to_order: dict[str, int],
    group_display: dict[str, str],
    group_order: dict[str, int],
    caption: str,
    label: str,
    min_percent: float = 0.0,
) -> str:
    static = classified[~classified["depth_dependent"]]
    summed = static.groupby("base_feature", as_index=False).agg(
        importance=("importance", "sum"), direction=("direction", "first")
    )
    summed = summed[(summed["importance"] > 0) & (summed["importance"] / total_importance * 100.0 > min_percent)]
    if summed.empty:
        return ""

    lookup_frame = static.rename(columns={"base_feature": "feature_name"})
    feature_info = _ordered_feature_groups(
        lookup_frame,
        set(summed["base_feature"]),
        group_order=group_order,
        feature_to_order=feature_to_order,
    )
    feature_info = _order_by_importance(feature_info, summed.set_index("base_feature")["importance"])

    rows = []
    for _, feature_row in feature_info.iterrows():
        feature_name = feature_row["feature_name"]
        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        summed_row = summed.loc[summed["base_feature"] == feature_name].iloc[0]
        importance = float(summed_row["importance"])
        rows.append((display_name, str(summed_row["direction"]), importance, importance / total_importance * 100.0))
    return importance_longtable(rows, caption=caption, label=label)


def build_flat_dynamic_importance_table(
    classified: pd.DataFrame,
    *,
    total_importance: float,
    importance_threshold: float,
    feature_to_display: dict[str, str],
    caption: str,
    label: str,
) -> str:
    dynamic = classified[
        classified["depth_dependent"] & (classified["importance"] > 0) & (classified["importance"] >= importance_threshold)
    ]
    if dynamic.empty:
        return ""
    dynamic = dynamic.sort_values(["importance", "base_feature", "depth"], ascending=[False, True, True])

    rows = []
    for _, record in dynamic.iterrows():
        feature_name = record["base_feature"]
        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        display_name += f" ($k={int(record['depth'])}$)"
        importance = float(record["importance"])
        rows.append((display_name, str(record["direction"]), importance, importance / total_importance * 100.0))
    return importance_longtable(rows, caption=caption, label=label)
