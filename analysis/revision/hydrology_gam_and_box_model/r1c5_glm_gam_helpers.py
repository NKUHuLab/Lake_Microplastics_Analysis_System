# -*- coding: utf-8 -*-
"""Reusable GLM/GAM helpers for the R1C5 Fig. 2 revision.

The plotting scripts keep their original visual grammar. This module only
replaces ridge/linear-regression internals with inferential models that provide
coefficients, robust confidence intervals, p values, and nonlinear GAM curves.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


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

MODERATORS = ["Lake_area", "Shore_dev", "Vol_total", "Res_time"]
LOG_TRANSFORM = {"Mismanaged", "fish_gdp_sqkm", "Res_time", "Lake_area", "Vol_total"}


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out


def transform_predictors(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        if col in LOG_TRANSFORM:
            x = np.log1p(x.clip(lower=0))
        X[col] = x
    return X


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    Xs = pd.DataFrame(index=X.index)
    for col in X.columns:
        x = X[col].astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        Xs[col] = (x - mu) / sd
    return Xs


def _fit_lm_hc3(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, dict]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    resid = y - fitted
    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    adj = resid / np.maximum(1.0 - leverage, 1e-8)
    meat = X.T @ ((adj[:, None] ** 2) * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = beta / np.maximum(se, 1e-12)
    p = 2 * stats.norm.sf(np.abs(z))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    rows = []
    for name, b, s, zv, pv, lo, hi in zip(names, beta, se, z, p, ci_low, ci_high):
        rows.append({
            "Feature": name,
            "Coefficient": b,
            "Std_Error": s,
            "Z_value": zv,
            "P_value": pv,
            "CI_Lower_95": lo,
            "CI_Upper_95": hi,
            "Effect_percent": (math.exp(b) - 1.0) * 100.0,
            "CI_Lower_percent": (math.exp(lo) - 1.0) * 100.0,
            "CI_Upper_percent": (math.exp(hi) - 1.0) * 100.0,
        })

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n = len(y)
    k = X.shape[1]
    sigma2 = max(ss_res / n, 1e-12)
    loglik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)
    diag = {
        "N": n,
        "df_resid": max(n - k, 1),
        "AIC": 2 * k - 2 * loglik,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "RMSE": math.sqrt(ss_res / n),
        "residual_skew": float(stats.skew(resid, bias=False)),
        "residual_kurtosis": float(stats.kurtosis(resid, fisher=False, bias=False)),
    }
    return pd.DataFrame(rows), diag


def fit_subgroup_glm_hc3(
    df_group: pd.DataFrame,
    predictors: list[str] | None = None,
    group_name: str | None = None,
) -> tuple[pd.Series | None, pd.DataFrame | None, dict | None]:
    """Gaussian GLM on ln(MP) with HC3 robust SE, preserving original effect scale."""
    predictors = PREDICTORS if predictors is None else predictors
    if len(df_group) < 20 or "ln" not in df_group.columns:
        return None, None, None
    data = ensure_columns(df_group, predictors).dropna(subset=["ln"]).copy()
    if len(data) < 20:
        return None, None, None
    Xs = standardize(transform_predictors(data, predictors))
    X = np.column_stack([np.ones(len(Xs)), Xs.to_numpy(float)])
    names = ["Intercept"] + predictors
    table, diag = _fit_lm_hc3(X, data["ln"].to_numpy(float), names)
    table.insert(0, "Model_Type", "Gaussian_GLM_lnMP_HC3")
    table.insert(1, "Context", group_name or "Subgroup")
    table.insert(2, "N", len(data))
    coef = table[table["Feature"] != "Intercept"].set_index("Feature")["Coefficient"]
    return coef, table, diag


def fit_global_moderation_glm_hc3(
    df: pd.DataFrame,
    moderator: str,
    predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    predictors = PREDICTORS if predictors is None else predictors
    cols = predictors + [moderator]
    data = ensure_columns(df, cols).dropna(subset=["ln"]).copy()
    Xs = standardize(transform_predictors(data, cols))
    interaction = f"fish_gdp_sqkm:{moderator}"
    Xs[interaction] = Xs["fish_gdp_sqkm"] * Xs[moderator]
    X = np.column_stack([np.ones(len(Xs)), Xs.to_numpy(float)])
    names = ["Intercept"] + list(Xs.columns)
    table, diag = _fit_lm_hc3(X, data["ln"].to_numpy(float), names)
    table.insert(0, "Model_Type", "Gaussian_GLM_lnMP_HC3")
    table.insert(1, "Context", moderator)
    table.insert(2, "N", len(data))
    return table, diag


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    Xs = standardize(transform_predictors(ensure_columns(df, cols), cols))
    rows = []
    X = Xs.to_numpy(float)
    for i, col in enumerate(Xs.columns):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        beta = np.linalg.lstsq(others, y, rcond=None)[0]
        pred = others @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({"Feature": col, "VIF": 1.0 / max(1.0 - r2, 1e-12)})
    return pd.DataFrame(rows)


def make_tertiles(df: pd.DataFrame, var: str, labels: list[str] | None = None) -> pd.Series:
    labels = ["Low", "Medium", "High"] if labels is None else labels
    return pd.qcut(df[var].rank(method="first"), q=3, labels=labels)


def fit_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def gam_fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    n_splines: int = 6,
    grid_size: int = 200,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 12 or np.unique(x).size < 5:
        raise ValueError("Not enough unique observations for a GAM curve.")
    from pygam import LinearGAM, s

    splines = max(5, min(n_splines, np.unique(x).size - 1))
    gam = LinearGAM(s(0, n_splines=splines)).fit(x.reshape(-1, 1), y)
    xx = np.linspace(x.min(), x.max(), grid_size)
    yy = gam.predict(xx.reshape(-1, 1))
    ci = gam.confidence_intervals(xx.reshape(-1, 1), width=0.95)
    return {
        "x": xx,
        "fit": yy,
        "ci_low": ci[:, 0],
        "ci_high": ci[:, 1],
        "pseudo_r2": float(gam.statistics_["pseudo_r2"]["explained_deviance"]),
        "linear_r2": fit_linear_r2(x, y),
        "n": len(x),
    }
