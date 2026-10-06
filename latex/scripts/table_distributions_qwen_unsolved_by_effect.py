from __future__ import annotations
from _thesis_captions import apply_thesis_captions

import numpy as np
import pandas as pd
import typer

from _gold_correlation_matrix_tables import (
    ROW_RULE,
    STATIC_FEATURE_NAME_COLUMN_WIDTH,
    _feature_lookup,
    _format_number,
    _latex_escape,
    _load_feature_groups,
)
from _unsolved_vs_rest_distribution_tables import (
    load_unsolved_and_rest_ids,
    unsolved_vs_rest_distribution_summary,
)
from query_difficulty.analysis import (
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
    DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
)
from query_difficulty.retrieval import DEFAULT_DOCUMENT_SCOPE, compose_retriever_name
from query_difficulty.runtime import build_runtime

EFFECT_SIZE_THRESHOLDS = {"medium": 0.33, "large": 0.474}
RETRIEVER_BASE_NAME = "qwen3_embedding_0_6b"
OUTPUT_NAME = "difficulty_analysis/table_distributions_qwen_unsolved_{effect}_effect_{split}.tex"
MAIN_EFFECT = "large"
APPENDIX_EFFECT = "medium"
APPENDIX_LABEL = "appendix:unsolved-vs-rest"


def _finite_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def collapse_summary(summary: pd.DataFrame, feature_to_display: dict[str, str]) -> pd.DataFrame:
    rows = []
    for feature_name, frame in summary.groupby("feature_name", sort=False):
        depth_dependent = bool(frame["depth_dependent"].iloc[0])
        rows.append(
            {
                "feature_name": feature_name,
                "display_name": _latex_escape(feature_to_display.get(feature_name, feature_name)),
                "depth_dependent": depth_dependent,
                "n_depths": int(frame["cliffs_delta"].notna().sum()) if depth_dependent else 1,
                "median_u": _finite_mean(frame["median_u"]),
                "median_r": _finite_mean(frame["median_r"]),
                "cliffs_delta": _finite_mean(frame["cliffs_delta"]),
            }
        )
    return pd.DataFrame(rows)


def _cell(value: float, depth_dependent: bool) -> str:
    if not np.isfinite(value):
        return "--"
    return _format_number(value) + (r"$^{\dagger}$" if depth_dependent else "")


def build_table(rows: pd.DataFrame, *, caption: str, label: str) -> str:
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
    for _, row in rows.iterrows():
        depth_dependent = bool(row["depth_dependent"])
        lines.append(
            str(row["display_name"])
            + " & "
            + _cell(float(row["median_u"]), depth_dependent)
            + " & "
            + _cell(float(row["median_r"]), depth_dependent)
            + " & "
            + _cell(float(row["cliffs_delta"]), depth_dependent)
            + r" \\"
        )
        lines.append(ROW_RULE)
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def legend(depth_columns: list[int]) -> str:
    depths = ", ".join(str(depth) for depth in depth_columns)
    return (
        "{\\scriptsize\n"
        rf"$^{{\dagger}}$ Depth-dependent feature. Values are averaged over the candidate-set sizes $k \in \{{{depths}\}}$ for which the feature is defined."
        "\\par}"
    )


def main(
    split: str = typer.Option("en_test", help="Query split."),
    language: str = typer.Option("en", help="Language code."),
    config: str = typer.Option("config/default.yaml", help="Path to the YAML configuration file."),
) -> None:
    runtime = build_runtime(config, verbose=True)
    console = runtime.console
    console.section(f"Unsolved-vs-rest distribution tables (main: {MAIN_EFFECT}, appendix: {APPENDIX_EFFECT}): {RETRIEVER_BASE_NAME} / {split}")
    unsolved_ids, rest_ids = load_unsolved_and_rest_ids(runtime, split, language)
    retriever_name = compose_retriever_name(RETRIEVER_BASE_NAME, DEFAULT_DOCUMENT_SCOPE)

    console.log(f"Building gold-independent features for {split} / {retriever_name}")
    independent_frames = runtime.features.build_features(split, retriever_name, candidate_sizes=None)
    console.log(f"Gold-independent features ready for {len(independent_frames)} candidate-set depth(s)")
    independent_summary = unsolved_vs_rest_distribution_summary(
        independent_frames,
        unsolved_ids=unsolved_ids,
        rest_ids=rest_ids,
        retriever_name=retriever_name,
        feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_INDEPENDENT,
        console=console,
    )

    console.log(f"Building gold-aware features for {split} / {retriever_name} (this may take a while if uncached)")
    aware_frames = runtime.features.build_gold_aware_features(split, retriever_name, candidate_sizes=None)
    console.log(f"Gold-aware features ready for {len(aware_frames)} candidate-set depth(s)")
    aware_summary = unsolved_vs_rest_distribution_summary(
        aware_frames,
        unsolved_ids=unsolved_ids,
        rest_ids=rest_ids,
        retriever_name=retriever_name,
        feature_set=DIFFICULTY_CORRELATION_FEATURE_SET_GOLD_AWARE,
        console=console,
    )

    summary = pd.concat([independent_summary, aware_summary], ignore_index=True)
    depth_columns = sorted(
        int(value) for value in summary.loc[summary["depth_dependent"], "candidate_size"].dropna().unique()
    )
    feature_to_display, _, _, _ = _feature_lookup(_load_feature_groups())
    collapsed = collapse_summary(summary, feature_to_display)
    console.log(f"Collapsed to {len(collapsed)} feature(s) before de-duplication")
    collapsed = collapsed.drop_duplicates(subset=["display_name"], keep="first")
    collapsed = collapsed[collapsed["cliffs_delta"].notna()]
    console.log(
        f"{len(collapsed)} unique feature(s) with a defined Cliff's delta "
        f"({int(collapsed['depth_dependent'].sum())} depth-dependent)"
    )
    for _, row in collapsed[collapsed["depth_dependent"]].iterrows():
        if row["n_depths"] < len(depth_columns):
            console.log(f"{row['feature_name']}: averaged over {row['n_depths']}/{len(depth_columns)} depth(s)")

    captions = {
        MAIN_EFFECT: (
            f"Unsolved vs.\\ rest (Qwen3-Emb-0.6B), showing features with a large difference "
            f"($|\\delta| \\geq {EFFECT_SIZE_THRESHOLDS[MAIN_EFFECT]}$). The complete table is given in "
            f"Appendix \\ref{{{APPENDIX_LABEL}}}."
        ),
        APPENDIX_EFFECT: (
            f"Unsolved vs.\\ rest (Qwen3-Emb-0.6B), showing all features with at least a medium difference "
            f"($|\\delta| \\geq {EFFECT_SIZE_THRESHOLDS[APPENDIX_EFFECT]}$)."
        ),
    }
    for effect in (MAIN_EFFECT, APPENDIX_EFFECT):
        threshold = EFFECT_SIZE_THRESHOLDS[effect]
        selected = collapsed[collapsed["cliffs_delta"].abs() >= threshold].copy()
        selected["abs_delta"] = selected["cliffs_delta"].abs()
        selected = selected.sort_values(["abs_delta", "display_name"], ascending=[False, True])
        console.log(
            f"{effect}: kept {len(selected)} feature(s) with |delta| >= {threshold} "
            f"({int(selected['depth_dependent'].sum())} depth-dependent)"
        )
        label = f"tab:distributions_qwen_unsolved_{effect}_effect_{split}"
        latex = build_table(selected, caption=captions[effect], label=label) + "\n\n" + legend(depth_columns) + "\n"
        output_path = runtime.config.latex_table_dir / OUTPUT_NAME.format(effect=effect, split=split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(apply_thesis_captions(latex), encoding="utf-8")
        console.log(f"Wrote {output_path}")
        print(output_path)


if __name__ == "__main__":
    typer.run(main)
