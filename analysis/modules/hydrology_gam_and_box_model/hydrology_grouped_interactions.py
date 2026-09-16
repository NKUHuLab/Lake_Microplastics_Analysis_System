# -*- coding: utf-8 -*-
"""Hydrology Fig. 2A global categorical-interaction GLM dumbbell.

The visual grammar follows the original dumbbell style. The statistical
estimand is changed to a full-sample categorical-interaction GLM:

    ln(MP) ~ all predictors + moderator_group + feature:moderator_group

For each feature, low/medium/high slopes are estimated as linear contrasts
from one full-sample model with HC3 robust covariance. P values are two-sided
Wald p values from the same contrast covariance.

v8 visual rule:
- highlighted group-specific slopes must satisfy p < 0.05 and have at least
  MIN_NONZERO_FOR_HIGHLIGHT non-zero observations in that moderator group;
- non-significant or sparsely supported slopes are shown in grey;
- feature order is determined only by highlighted slopes, using the largest
  absolute highlighted coefficient. Features without highlighted slopes get a
  sorting score of zero.
"""

from __future__ import annotations

import math
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(os.environ.get("LAKE_MP_REPO_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.environ.get("LAKE_MP_DATA_ROOT", REPO_ROOT / "data"))
TRAIN = Path(os.environ.get("LAKE_MP_TRAIN_DATA", DATA_ROOT / "model_products" / "train_data.csv"))
OUT_ROOT = Path(os.environ.get("LAKE_MP_OUTPUT_DIR", REPO_ROOT / "outputs")) / "hydrology_gam_and_box_model"
BASE = OUT_ROOT
OUT_FIG = BASE / "re_fig" / "Hydrological_moderation"
OUT_DATA = OUT_ROOT / "data"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    "Primary_Waste_Discharge",
    "Secondary_Waste_Discharge",
    "Advanced_Waste_Discharge",
    "RSE_paved",
    "RSE_gravel",
    "RSE_other",
    "Mismanaged",
    "fish_gdp_sqkm",
    "Cultivated_land",
    "Artificial_surface",
]
MODERATORS = ["Res_time", "Shore_dev"]
LOG_TRANSFORM = {"Mismanaged", "fish_gdp_sqkm", "Res_time", "Lake_area", "Vol_total"}
GROUPS = ["Low", "Medium", "High"]
MIN_NONZERO_FOR_HIGHLIGHT = 10

COLORS_ORIGINAL = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MOD_LABELS = {
    "Res_time": "Residence time",
    "Shore_dev": "Shoreline development",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TRAIN)
    needed = ["ln"] + PREDICTORS + MODERATORS
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=needed).copy()


def transform_standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        if col in LOG_TRANSFORM:
            x = np.log1p(x.clip(lower=0))
        sd = float(x.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[col] = (x - float(x.mean())) / sd
    return out


def ols_hc3(X: np.ndarray, y: np.ndarray):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    adj = resid / np.maximum(1.0 - leverage, 1e-8)
    meat = X.T @ ((adj[:, None] ** 2) * X)
    cov = xtx_inv @ meat @ xtx_inv
    return beta, cov


def contrast_stats(contrast: np.ndarray, beta: np.ndarray, cov: np.ndarray) -> dict[str, float]:
    est = float(contrast @ beta)
    se = math.sqrt(max(float(contrast @ cov @ contrast), 0.0))
    z = est / max(se, 1e-12)
    p = 2.0 * norm_sf(abs(z))
    return {
        "Estimate": est,
        "Std_Error": se,
        "Z_value": z,
        "P_value_two_sided": p,
        "CI_Lower_95": est - 1.96 * se,
        "CI_Upper_95": est + 1.96 * se,
    }


def moderator_groups(df: pd.DataFrame, moderator: str) -> tuple[pd.Series, pd.DataFrame]:
    labels = pd.qcut(df[moderator].rank(method="first"), q=3, labels=GROUPS)
    ranges = (
        df.groupby(labels, observed=True)[moderator]
        .agg(["min", "median", "max", "count"])
        .reset_index()
        .rename(columns={moderator: "Context", "min": "Min", "median": "Representative_value", "max": "Max", "count": "N_group"})
    )
    ranges.insert(0, "Moderator", moderator)
    return labels.astype(str), ranges


def categorical_interaction_table(df: pd.DataFrame, moderator: str):
    labels, ranges = moderator_groups(df, moderator)
    med = (labels == "Medium").astype(float).to_numpy()
    high = (labels == "High").astype(float).to_numpy()
    reps = dict(zip(ranges["Context"], ranges["Representative_value"]))
    ns = dict(zip(ranges["Context"], ranges["N_group"]))
    nonzero_by_feature = {}
    for feature in PREDICTORS:
        nonzero_by_feature[feature] = {
            group: int(((df[feature] > 0) & (labels == group)).sum())
            for group in GROUPS
        }

    Xbase = transform_standardize(df, PREDICTORS)
    y = df["ln"].to_numpy(float)
    effect_rows = []
    contrast_rows = []

    for feature in PREDICTORS:
        Xdf = Xbase.copy()
        Xdf[f"{moderator}_Medium"] = med
        Xdf[f"{moderator}_High"] = high
        Xdf[f"{feature}:{moderator}_Medium"] = Xdf[feature].to_numpy(float) * med
        Xdf[f"{feature}:{moderator}_High"] = Xdf[feature].to_numpy(float) * high
        names = ["Intercept"] + list(Xdf.columns)
        X = np.column_stack([np.ones(len(Xdf)), Xdf.to_numpy(float)])
        beta, cov = ols_hc3(X, y)

        fi = names.index(feature)
        mi = names.index(f"{feature}:{moderator}_Medium")
        hi = names.index(f"{feature}:{moderator}_High")

        group_contrasts = {}
        c = np.zeros(len(names)); c[fi] = 1.0
        group_contrasts["Low"] = c
        c = np.zeros(len(names)); c[fi] = 1.0; c[mi] = 1.0
        group_contrasts["Medium"] = c
        c = np.zeros(len(names)); c[fi] = 1.0; c[hi] = 1.0
        group_contrasts["High"] = c

        for group in GROUPS:
            stats = contrast_stats(group_contrasts[group], beta, cov)
            est = stats["Estimate"]
            lo = stats["CI_Lower_95"]
            hi95 = stats["CI_Upper_95"]
            effect_rows.append({
                "Model_Type": "Gaussian_GLM_lnMP_HC3_global_categorical_interaction",
                "Moderator": moderator,
                "Context": group,
                "N_model": len(df),
                "N_group": int(ns[group]),
                "N_nonzero_group": int(nonzero_by_feature[feature][group]),
                "Feature": feature,
                "Representative_value": float(reps[group]),
                "Coefficient": est,
                "Std_Error": stats["Std_Error"],
                "Z_value": stats["Z_value"],
                "P_value_two_sided_delta": stats["P_value_two_sided"],
                "CI_Lower_95": lo,
                "CI_Upper_95": hi95,
                "Effect_percent": (math.exp(est) - 1.0) * 100.0,
                "CI_Lower_percent": (math.exp(lo) - 1.0) * 100.0,
                "CI_Upper_percent": (math.exp(hi95) - 1.0) * 100.0,
                "Test_note": "Two-sided Wald test for group-specific slope from full-sample categorical-interaction GLM.",
            })

        pair_contrasts = {
            "Medium_minus_Low": np.eye(len(names))[mi],
            "High_minus_Low": np.eye(len(names))[hi],
            "High_minus_Medium": np.eye(len(names))[hi] - np.eye(len(names))[mi],
        }
        for contrast_name, c in pair_contrasts.items():
            stats = contrast_stats(c, beta, cov)
            contrast_rows.append({
                "Model_Type": "Gaussian_GLM_lnMP_HC3_global_categorical_interaction",
                "Moderator": moderator,
                "Feature": feature,
                "Contrast": contrast_name,
                **stats,
                "Test_note": "Pairwise contrast between group-specific slopes; reported for diagnostics, not plotted.",
            })

    long = pd.DataFrame(effect_rows)
    contrasts = pd.DataFrame(contrast_rows)
    effects = long.pivot(index="Feature", columns="Context", values="Coefficient")[GROUPS]
    ci_low = long.pivot(index="Feature", columns="Context", values="CI_Lower_95")[GROUPS]
    ci_high = long.pivot(index="Feature", columns="Context", values="CI_Upper_95")[GROUPS]
    return long, contrasts, effects, ci_low, ci_high, ranges


def save_pub(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(
            OUT_FIG / f"{stem}.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Creator": "Codex matplotlib; editable text PDF"},
        )
    plt.close(fig)


def plot_dumbbell_focus_ci(
    effects_df: pd.DataFrame,
    group_by_var: str,
    ci_low_df: pd.DataFrame,
    ci_high_df: pd.DataFrame,
    p_df: pd.DataFrame,
    highlight_df: pd.DataFrame,
    xlim: tuple[float, float] = (-2.2, 2.2),
    feature_order: list[str] | None = None,
    compact_ci_pairs: set[tuple[str, str]] | None = None,
    compact_ci_min_width: float = 2.5,
    compact_ci_half_width: float = 0.42,
    figsize: tuple[float, float] = (13.09, 11.6),
) -> None:
    effects_df = effects_df.copy()
    if feature_order is None:
        effects_df["abs_max"] = effects_df.abs().max(axis=1)
        df_sorted = effects_df.sort_values("abs_max", ascending=True).drop(columns="abs_max")
    else:
        ordered = [f for f in feature_order if f in effects_df.index]
        ordered += [f for f in effects_df.index if f not in ordered]
        df_sorted = effects_df.loc[ordered]

    ci_low_sorted = ci_low_df.loc[df_sorted.index]
    ci_high_sorted = ci_high_df.loc[df_sorted.index]
    p_sorted = p_df.loc[df_sorted.index]
    highlight_sorted = highlight_df.loc[df_sorted.index]
    y_pos = np.arange(len(df_sorted))
    group_names = list(df_sorted.columns)
    xmin, xmax = xlim

    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(
        y=y_pos,
        xmin=df_sorted.min(axis=1),
        xmax=df_sorted.max(axis=1),
        color="black",
        alpha=0.7,
        linewidth=2.5,
        zorder=1,
    )

    any_truncated = False
    any_compacted = False
    cap_half_height = 0.045
    for i, group in enumerate(group_names):
        color = COLORS_ORIGINAL[i % len(COLORS_ORIGINAL)]
        coefs = df_sorted[group].to_numpy(float)
        lows = ci_low_sorted[group].to_numpy(float)
        highs = ci_high_sorted[group].to_numpy(float)
        ps = p_sorted[group].to_numpy(float)
        highlights = highlight_sorted[group].to_numpy(bool)
        for feature, y, coef, lo, hi, pval, highlighted in zip(df_sorted.index, y_pos, coefs, lows, highs, ps, highlights):
            draw_color = color if highlighted else "#B8B8B8"
            point_edge = "black" if highlighted else "#7A7A7A"
            point_alpha = 0.95 if highlighted else 0.72
            line_alpha = 0.92 if highlighted else 0.55
            compact_this_ci = (
                compact_ci_pairs is not None
                and (feature, group) in compact_ci_pairs
                and (hi - lo) >= compact_ci_min_width
            )
            if compact_this_ci:
                any_compacted = True
                lo_compact = max(coef - compact_ci_half_width, xmin)
                hi_compact = min(coef + compact_ci_half_width, xmax)
                ax.plot([lo_compact, hi_compact], [y, y], color=draw_color, lw=1.35, alpha=min(line_alpha, 0.72), linestyle=(0, (2.0, 1.7)), zorder=2)
                ax.scatter(lo_compact, y, marker="<", s=42, color=draw_color, edgecolor="none", alpha=point_alpha, zorder=2.6)
                ax.scatter(hi_compact, y, marker=">", s=42, color=draw_color, edgecolor="none", alpha=point_alpha, zorder=2.6)
                continue
            lo_clip = max(lo, xmin)
            hi_clip = min(hi, xmax)
            ax.plot([lo_clip, hi_clip], [y, y], color=draw_color, lw=1.45, alpha=line_alpha, zorder=2)
            if lo >= xmin:
                ax.plot([lo, lo], [y - cap_half_height, y + cap_half_height], color=draw_color, lw=1.15, alpha=line_alpha, zorder=2)
            else:
                any_truncated = True
                ax.scatter(xmin, y, marker="<", s=42, color=draw_color, edgecolor="none", alpha=point_alpha, zorder=2.6)
            if hi <= xmax:
                ax.plot([hi, hi], [y - cap_half_height, y + cap_half_height], color=draw_color, lw=1.15, alpha=line_alpha, zorder=2)
            else:
                any_truncated = True
                ax.scatter(xmax, y, marker=">", s=42, color=draw_color, edgecolor="none", alpha=point_alpha, zorder=2.6)

        point_colors = [color if h else "#B8B8B8" for h in highlights]
        point_edges = ["black" if h else "#7A7A7A" for h in highlights]
        ax.scatter(coefs, y_pos, color=point_colors, s=180, label="_nolegend_", zorder=3, edgecolors=point_edges, linewidth=0.5, alpha=0.95)

    ax.set_xlim(xmin, xmax)
    ax.axvspan(xmin, 0, color="royalblue", alpha=0.05, zorder=0, label="_nolegend_")
    ax.axvspan(0, xmax, color="tomato", alpha=0.05, zorder=0, label="_nolegend_")
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.3)
    ax.yaxis.grid(False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Effect coefficient (standardized ln scale; 95% CI)", fontsize=14)
    ax.set_ylabel("")
    ax.set_title(f"Moderation Effect by {MOD_LABELS.get(group_by_var, group_by_var)}", fontsize=19, pad=20)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9.5, markerfacecolor=COLORS_ORIGINAL[i], markeredgecolor="black", markeredgewidth=0.5, label=group)
        for i, group in enumerate(group_names)
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", linestyle="none", markersize=9.5, markerfacecolor="#B8B8B8", markeredgecolor="#7A7A7A", markeredgewidth=0.5, label="Not significant")
    )
    legend = ax.legend(
        handles=legend_handles,
        title=f"Stratified by {MOD_LABELS.get(group_by_var, group_by_var)}",
        fontsize=9.5,
        loc="upper left",
        bbox_to_anchor=(1.015, 0.94),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.45,
    )
    plt.setp(legend.get_title(), fontsize=10.5)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted.index, fontdict={"family": "Arial", "size": 12})
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_pub(fig, f"Fig_Hydrology_Dumbbell_{group_by_var}_GLM_originalStyle_categoricalInteraction_stableSigOnlyOrder_v8")


def main() -> None:
    df = load_data()
    all_effects = []
    all_contrasts = []
    all_ranges = []
    for moderator in MODERATORS:
        long, contrasts, effects, ci_low, ci_high, ranges = categorical_interaction_table(df, moderator)
        pvals = long.pivot(index="Feature", columns="Context", values="P_value_two_sided_delta")[GROUPS]
        nonzero = long.pivot(index="Feature", columns="Context", values="N_nonzero_group")[GROUPS]
        highlight = (pvals < 0.05) & (nonzero >= MIN_NONZERO_FOR_HIGHLIGHT)
        all_effects.append(long)
        all_contrasts.append(contrasts)
        all_ranges.append(ranges)
        long.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_effects_{moderator}_v8.csv", index=False, encoding="utf-8-sig")
        contrasts.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_contrasts_{moderator}_v8.csv", index=False, encoding="utf-8-sig")
        effects.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_dumbbell_coefficients_{moderator}_v8.csv", encoding="utf-8-sig")
        pvals.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_dumbbell_pvalues_{moderator}_v8.csv", encoding="utf-8-sig")
        nonzero.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_dumbbell_nonzero_counts_{moderator}_v8.csv", encoding="utf-8-sig")
        highlight.astype(int).to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_dumbbell_highlight_mask_{moderator}_v8.csv", encoding="utf-8-sig")
        significant_effects = effects.where(highlight, 0.0)
        order = (
            significant_effects.abs()
            .max(axis=1)
            .rename("max_abs_significant_effect")
            .sort_values(ascending=True)
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        order["plot_rank_ascending"] = np.arange(1, len(order) + 1)
        order.to_csv(OUT_DATA / f"Hydrology_GLM_global_categorical_interaction_dumbbell_stableSigOnly_order_{moderator}_v8.csv", index=False, encoding="utf-8-sig")
        plot_dumbbell_focus_ci(
            effects,
            moderator,
            ci_low,
            ci_high,
            pvals,
            highlight,
            feature_order=order["Feature"].tolist(),
            compact_ci_pairs={("Primary_Waste_Discharge", "Low")},
        )

    all_effects_df = pd.concat(all_effects, ignore_index=True)
    all_contrasts_df = pd.concat(all_contrasts, ignore_index=True)
    all_ranges_df = pd.concat(all_ranges, ignore_index=True)
    all_effects_df.to_csv(OUT_DATA / "Hydrology_GLM_global_categorical_interaction_effects_all_v8.csv", index=False, encoding="utf-8-sig")
    all_contrasts_df.to_csv(OUT_DATA / "Hydrology_GLM_global_categorical_interaction_contrasts_all_v8.csv", index=False, encoding="utf-8-sig")
    all_ranges_df.to_csv(OUT_DATA / "Hydrology_GLM_global_categorical_interaction_group_ranges_v8.csv", index=False, encoding="utf-8-sig")

    print("Saved v8 categorical-interaction GLM outputs")
    print(OUT_DATA / "Hydrology_GLM_global_categorical_interaction_effects_all_v8.csv")
    print(OUT_DATA / "Hydrology_GLM_global_categorical_interaction_contrasts_all_v8.csv")
    print(OUT_FIG / "Fig_Hydrology_Dumbbell_Res_time_GLM_originalStyle_categoricalInteraction_stableSigOnlyOrder_v8.png")
    print(OUT_FIG / "Fig_Hydrology_Dumbbell_Shore_dev_GLM_originalStyle_categoricalInteraction_stableSigOnlyOrder_v8.png")
    focus = all_effects_df[all_effects_df["Feature"].eq("fish_gdp_sqkm")]
    print(focus[["Moderator", "Context", "N_group", "Representative_value", "Coefficient", "CI_Lower_95", "CI_Upper_95", "P_value_two_sided_delta", "Effect_percent"]].to_string(index=False))
    print(all_contrasts_df[(all_contrasts_df["Feature"].eq("fish_gdp_sqkm"))][["Moderator", "Contrast", "Estimate", "CI_Lower_95", "CI_Upper_95", "P_value_two_sided"]].to_string(index=False))


if __name__ == "__main__":
    main()
