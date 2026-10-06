from __future__ import annotations
from _thesis_captions import apply_thesis_captions

from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from query_difficulty.analysis import retriever_difficulty_correlation_analysis_for_runtime
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, compose_retriever_name
from query_difficulty.runtime import build_runtime

STATIC_FEATURE_NAME_COLUMN_WIDTH = "0.40\\linewidth"
DYNAMIC_FEATURE_NAME_COLUMN_WIDTH = "0.24\\linewidth"

CORRELATION_STRENGTH_THRESHOLD = 0.3

SYMBOL_STRONG_POSITIVE = r"$\Uparrow$"
SYMBOL_WEAK_POSITIVE = r"$\uparrow$"
SYMBOL_STRONG_NEGATIVE = r"$\Downarrow$"
SYMBOL_WEAK_NEGATIVE = r"$\downarrow$"
SYMBOL_NO_CORRELATION = r"--"

ROW_RULE = r"\arrayrulecolor{gray!50}\hline\arrayrulecolor{black}"

GROUP_KEY_ALIASES = {
    "lexical_post_retrieval_ranking": "lexical_post_retrieval",
    "lexical_post_retrieval_top_document": "lexical_post_retrieval",
    "semantic_post_retrieval_geometry": "semantic_post_retrieval_ranking_geometry",
    "perturbation": "perturbation_based_robustness",
}


def _load_feature_groups() -> dict:
    with open(ROOT / "config" / "feature_groups.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["feature_groups"]


def _feature_lookup(groups_config: dict) -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, int]]:
    feature_to_display: dict[str, str] = {}
    feature_to_order: dict[str, int] = {}
    group_display: dict[str, str] = {}
    group_order: dict[str, int] = {}
    for group_key, group_data in groups_config.items():
        group_display[group_key] = group_data["display_name"]
        group_order[group_key] = int(group_data["order"])
        for position, (feature_name, display_name) in enumerate(group_data["features"].items()):
            feature_to_display[feature_name] = display_name
            feature_to_order[feature_name] = position
    return feature_to_display, feature_to_order, group_display, group_order


def _resolved_group_key(feature_group: str) -> str:
    return GROUP_KEY_ALIASES.get(feature_group, feature_group)


def _latex_escape(text: str) -> str:
    for source, target in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(source, target)
    return text


def _format_number(value: float) -> str:
    formatted = f"{float(value):.3f}"
    if formatted.startswith("0."):
        formatted = formatted[1:]
    elif formatted.startswith("-0."):
        formatted = "-" + formatted[2:]
    return formatted


def _correlation_symbol(tau_b: float, significant: bool) -> str:
    if not significant:
        return SYMBOL_NO_CORRELATION
    strong = abs(tau_b) >= CORRELATION_STRENGTH_THRESHOLD
    if tau_b > 0:
        return SYMBOL_STRONG_POSITIVE if strong else SYMBOL_WEAK_POSITIVE
    return SYMBOL_STRONG_NEGATIVE if strong else SYMBOL_WEAK_NEGATIVE


def _legend_symbol(symbol: str) -> str:
    return r"{\Large " + symbol + "}"


SYMBOL_LEGEND = (
    r"\footnotesize"
    r"\begin{tabular}{@{}c p{0.34\linewidth} c p{0.34\linewidth}@{}}"
    + _legend_symbol(SYMBOL_STRONG_POSITIVE)
    + r" & strong positive: feature higher for easier queries, $|\tau_b| \geq "
    + _format_number(CORRELATION_STRENGTH_THRESHOLD)
    + r"$ (significant, adjusted $p<0.05$) & "
    + _legend_symbol(SYMBOL_STRONG_NEGATIVE)
    + r" & strong negative: feature higher for harder queries, $|\tau_b| \geq "
    + _format_number(CORRELATION_STRENGTH_THRESHOLD)
    + r"$ (significant) \\[0.4em]"
    + _legend_symbol(SYMBOL_WEAK_POSITIVE)
    + r" & weak positive: feature higher for easier queries, $|\tau_b| < "
    + _format_number(CORRELATION_STRENGTH_THRESHOLD)
    + r"$ (significant) & "
    + _legend_symbol(SYMBOL_WEAK_NEGATIVE)
    + r" & weak negative: feature higher for harder queries, $|\tau_b| < "
    + _format_number(CORRELATION_STRENGTH_THRESHOLD)
    + r"$ (significant) \\[0.4em]"
    + r"\multicolumn{2}{l}{"
    + _legend_symbol(SYMBOL_NO_CORRELATION)
    + r"\ \ no significant correlation (adjusted $p \geq 0.05$)} & "
    + r"\multicolumn{2}{l}{Blank: feature undefined or constant at that $k$} \\"
    + r"\end{tabular}"
)


def _features_with_any_signal(subset: pd.DataFrame, feature_names: set[str]) -> set[str]:
    defined = subset[subset["feature_name"].isin(feature_names) & subset["kendall_tau_b"].notna()]
    return set(defined["feature_name"].unique())


def _ordered_feature_groups(
    subset: pd.DataFrame,
    feature_names: set[str],
    *,
    group_order: dict[str, int],
    feature_to_order: dict[str, int],
) -> pd.DataFrame:
    feature_info = (
        subset[subset["feature_name"].isin(feature_names)][["feature_name", "feature_group"]]
        .drop_duplicates()
        .copy()
    )
    feature_info["resolved_group"] = feature_info["feature_group"].map(_resolved_group_key)
    feature_info["group_order"] = feature_info["resolved_group"].map(group_order).fillna(10_000).astype(int)
    feature_info["feature_order"] = feature_info["feature_name"].map(feature_to_order).fillna(10_000).astype(int)
    return feature_info.sort_values(["group_order", "resolved_group", "feature_order", "feature_name"])


def _static_feature_order_by_tau_b(
    subset: pd.DataFrame,
    feature_names: set[str],
    *,
    group_order: dict[str, int],
) -> pd.DataFrame:
    feature_info = (
        subset[subset["feature_name"].isin(feature_names)][["feature_name", "feature_group", "kendall_tau_b"]]
        .drop_duplicates(subset=["feature_name"])
        .copy()
    )
    feature_info["resolved_group"] = feature_info["feature_group"].map(_resolved_group_key)
    feature_info["group_order"] = feature_info["resolved_group"].map(group_order).fillna(10_000).astype(int)
    return feature_info.sort_values(
        ["group_order", "resolved_group", "kendall_tau_b", "feature_name"],
        ascending=[True, True, False, True],
    )


def _group_header_row(resolved_group: str, group_display: dict[str, str], n_columns: int) -> str:
    display_group = group_display.get(resolved_group, resolved_group)
    return r"\multicolumn{" + str(n_columns) + r"}{l}{\emph{" + _latex_escape(display_group) + r"}} \\"


def _build_static_table(
    subset: pd.DataFrame,
    *,
    static_feature_names: set[str],
    feature_to_display: dict[str, str],
    feature_to_order: dict[str, int],
    group_display: dict[str, str],
    group_order: dict[str, int],
    caption: str,
    label: str,
) -> str:
    feature_info = _static_feature_order_by_tau_b(subset, static_feature_names, group_order=group_order)

    header_row = r"Feature & Direction & $\tau_b$ & $\rho$ \\"
    lines: list[str] = []
    lines.append(r"\begin{longtable}{p{" + STATIC_FEATURE_NAME_COLUMN_WIDTH + r"}ccc}")
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

    current_group: str | None = None
    for _, feature_row in feature_info.iterrows():
        feature_name = feature_row["feature_name"]
        resolved_group = feature_row["resolved_group"]
        if resolved_group != current_group:
            lines.append(_group_header_row(resolved_group, group_display, 4))
            current_group = resolved_group

        display_name = _latex_escape(feature_to_display.get(feature_name, feature_name))
        record = subset[subset["feature_name"] == feature_name].iloc[0]
        tau_b = float(record["kendall_tau_b"])
        symbol = _legend_symbol(_correlation_symbol(tau_b, bool(record["significant"])))
        lines.append(
            display_name
            + " & "
            + symbol
            + " & "
            + _format_number(tau_b)
            + " & "
            + _format_number(record["spearman_rho"])
            + r" \\"
        )
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _build_dynamic_symbol_table(
    subset: pd.DataFrame,
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
        subset, dynamic_feature_names, group_order=group_order, feature_to_order=feature_to_order
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
        feature_rows = subset[subset["feature_name"] == feature_name]

        cells = []
        for depth in depth_columns:
            depth_row = feature_rows[feature_rows["candidate_size"] == depth]
            if depth_row.empty or pd.isna(depth_row.iloc[0]["kendall_tau_b"]):
                cells.append("")
                continue
            record = depth_row.iloc[0]
            cells.append(_correlation_symbol(float(record["kendall_tau_b"]), bool(record["significant"])))
        lines.append(" & ".join([display_name] + cells) + r" \\")
        lines.append(ROW_RULE)

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def build_depth_matrix_tables(
    results: pd.DataFrame,
    *,
    retriever_name: str,
    feature_set: str,
    static_caption: str,
    static_label: str,
    dynamic_symbol_caption: str,
    dynamic_symbol_label: str,
) -> str:
    subset = results[
        (results["retriever_name"] == retriever_name) & (results["feature_set"] == feature_set)
    ].copy()
    if subset.empty:
        raise ValueError(f"No correlation results for retriever={retriever_name!r}, feature_set={feature_set!r}")

    groups_config = _load_feature_groups()
    feature_to_display, feature_to_order, group_display, group_order = _feature_lookup(groups_config)

    static_all = set(subset.loc[~subset["depth_dependent"], "feature_name"].unique())
    dynamic_all = set(subset.loc[subset["depth_dependent"], "feature_name"].unique())
    static_feature_names = _features_with_any_signal(subset, static_all)
    dynamic_feature_names = _features_with_any_signal(subset, dynamic_all)
    depth_columns = sorted(int(value) for value in subset.loc[subset["depth_dependent"], "candidate_size"].dropna().unique())

    parts: list[str] = []
    if static_feature_names:
        parts.append(
            _build_static_table(
                subset,
                static_feature_names=static_feature_names,
                feature_to_display=feature_to_display,
                feature_to_order=feature_to_order,
                group_display=group_display,
                group_order=group_order,
                caption=static_caption,
                label=static_label,
            )
        )
    if dynamic_feature_names:
        parts.append(
            _build_dynamic_symbol_table(
                subset,
                dynamic_feature_names=dynamic_feature_names,
                depth_columns=depth_columns,
                feature_to_display=feature_to_display,
                feature_to_order=feature_to_order,
                group_display=group_display,
                group_order=group_order,
                caption=dynamic_symbol_caption,
                label=dynamic_symbol_label,
            )
        )
        parts.append(SYMBOL_LEGEND)
    else:
        parts.append(SYMBOL_LEGEND)
    return "\n\n".join(parts) + "\n"


def write_depth_matrix_tables(
    *,
    config: str,
    split: str,
    feature_set: str,
    retriever_base_name: str,
    output_name: str,
    static_caption_template: str,
    static_label_template: str,
    dynamic_symbol_caption_template: str,
    dynamic_symbol_label_template: str,
    descriptive_candidate_size: list[int] | None = None,
    gold_aware_candidate_size: list[int] | None = None,
) -> Path:
    runtime = build_runtime(config, verbose=False)
    retriever_name = compose_retriever_name(retriever_base_name, DEFAULT_DOCUMENT_SCOPE)
    results = retriever_difficulty_correlation_analysis_for_runtime(
        runtime,
        split,
        [retriever_name],
        feature_sets=(feature_set,),
        descriptive_candidate_sizes=descriptive_candidate_size,
        gold_aware_candidate_sizes=gold_aware_candidate_size,
    )
    latex = build_depth_matrix_tables(
        results,
        retriever_name=retriever_name,
        feature_set=feature_set,
        static_caption=static_caption_template.format(split=split),
        static_label=static_label_template.format(split=split),
        dynamic_symbol_caption=dynamic_symbol_caption_template.format(split=split),
        dynamic_symbol_label=dynamic_symbol_label_template.format(split=split),
    )
    output_path = runtime.config.latex_table_dir / output_name.format(split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
    return output_path
