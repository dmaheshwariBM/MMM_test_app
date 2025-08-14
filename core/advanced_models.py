# core/advanced_models.py
from __future__ import annotations
import os, json, math, itertools
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# We reuse your core.modeling.ols_model for consistency
from core import modeling

# ---------------------------
# Utilities
# ---------------------------
def _add_const(X: pd.DataFrame, add_const: bool = True) -> pd.DataFrame:
    if add_const and "const" not in X.columns:
        X = X.copy()
        X.insert(0, "const", 1.0)
    return X

def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out

def _product_name(a: str, b: str) -> str:
    return f"{a}__x__{b}"

def _make_interactions(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    Xix = pd.DataFrame(index=df.index)
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            Xix[_product_name(a,b)] = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
                                      pd.to_numeric(df[b], errors="coerce").fillna(0.0)
    return Xix

def _metrics(y: np.ndarray, yhat: np.ndarray, p: int) -> Dict[str, float]:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2)) if n > 0 else 0.0
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    rmse = math.sqrt(sse / n) if n > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    return {"n": int(n), "p": int(p), "r2": float(r2), "adj_r2": float(adj_r2), "rmse": float(rmse)}

# ---------------------------
# Breakout (grouped) models
# ---------------------------
def run_breakout_models(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    group_col: str,
    min_group_n: int = 30,
    add_const: bool = True,
    force_nonnegative: bool = True,
) -> Dict[str, Any]:
    """
    Fits a separate OLS model per group value of group_col.
    Returns a results dict keyed by group value, each with metrics and coef.
    """
    res: Dict[str, Any] = {}
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in dataframe")

    for g, dfg in df.groupby(group_col):
        if len(dfg) < min_group_n:
            continue
        y = pd.to_numeric(dfg[target], errors="coerce").fillna(0.0).values
        X = _safe_numeric(dfg, features)
        if add_const:
            X.insert(0, "const", 1.0)

        # Reuse your OLS routine (supports force_nonnegative)
        out = modeling.ols_model(
            X, y,
            add_const=False,  # already added
            compute_vif=False,
            force_nonnegative=force_nonnegative
        )
        # pack
        res[str(g)] = {
            "group": str(g),
            "metrics": out["metrics"],
            "coef": out["coef"].to_dict() if hasattr(out["coef"], "to_dict") else out["coef"],
            "yhat": out["yhat"].tolist(),
            "features": list(X.columns),
        }
    return {"type": "breakout", "group_col": group_col, "results": res}

# ---------------------------
# Pathway / Interaction model
# ---------------------------
def suggest_interactions_by_corr(
    df: pd.DataFrame, target: str, features: List[str], top_k: int = 5
) -> List[Tuple[str, str]]:
    """
    Heuristic: choose pairs whose product term has highest |corr| with target.
    """
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    pairs = list(itertools.combinations(features, 2))
    scores = []
    for a, b in pairs:
        prod = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
               pd.to_numeric(df[b], errors="coerce").fillna(0.0)
        if prod.std(ddof=0) == 0:
            continue
        c = prod.corr(y)
        if pd.notna(c):
            scores.append((abs(float(c)), (a, b)))
    scores.sort(reverse=True, key=lambda z: z[0])
    return [pair for _, pair in scores[:top_k]]

def run_interaction_model(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    interaction_pairs: List[Tuple[str, str]],
    add_const: bool = True,
    force_nonnegative: bool = False,
) -> Dict[str, Any]:
    """
    Adds specified interaction columns to X and fits OLS.
    """
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).values
    X_base = _safe_numeric(df, features)
    X_ix = _make_interactions(df, interaction_pairs)
    X = pd.concat([X_base, X_ix], axis=1)
    if add_const:
        X.insert(0, "const", 1.0)

    out = modeling.ols_model(
        X, y,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative
    )
    return {
        "type": "interaction",
        "metrics": out["metrics"],
        "coef": out["coef"].to_dict() if hasattr(out["coef"], "to_dict") else out["coef"],
        "yhat": out["yhat"].tolist(),
        "features": list(X.columns),
        "interaction_pairs": interaction_pairs,
    }

# ---------------------------
# Residual (two-stage) model
# ---------------------------
def _lag_series(s: pd.Series, k: int) -> pd.Series:
    if k <= 0: return s.copy()
    return s.shift(k).fillna(0.0)

def run_residual_model(
    df: pd.DataFrame,
    target: str,
    base_features: List[str],
    residual_features: List[str],
    base_add_const: bool = True,
    resid_add_const: bool = True,
    ar_lags: int = 0,
    force_nonnegative_base: bool = False,
    force_nonnegative_resid: bool = False,
) -> Dict[str, Any]:
    """
    Stage 1: fit base OLS on target ~ base_features  (returns yhat_base)
    Stage 2: fit residual OLS on (y - yhat_base) ~ residual_features (+ AR lags of residuals)
    Final prediction = yhat_base + yhat_resid
    """
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)

    # -- Stage 1: base
    X1 = _safe_numeric(df, base_features)
    if base_add_const:
        X1.insert(0, "const", 1.0)
    out1 = modeling.ols_model(
        X1, y.values,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative_base
    )
    yhat1 = pd.Series(out1["yhat"], index=df.index)

    # residuals
    resid = y - yhat1

    # -- Stage 2: residuals
    X2 = _safe_numeric(df, residual_features)
    # add AR lags of residuals if requested
    if ar_lags > 0:
        for k in range(1, ar_lags + 1):
            X2[f"resid_lag{k}"] = _lag_series(resid, k)
    if resid_add_const:
        X2.insert(0, "const", 1.0)
    out2 = modeling.ols_model(
        X2, resid.values,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative_resid
    )
    yhat2 = pd.Series(out2["yhat"], index=df.index)

    # final
    yhat_final = yhat1 + yhat2
    m = _metrics(y.values, yhat_final.values, p=len(X1.columns) + len(X2.columns))
    return {
        "type": "residual",
        "metrics": m,
        "stage1": {"metrics": out1["metrics"], "coef": out1["coef"].to_dict() if hasattr(out1["coef"],"to_dict") else out1["coef"]},
        "stage2": {"metrics": out2["metrics"], "coef": out2["coef"].to_dict() if hasattr(out2["coef"],"to_dict") else out2["coef"]},
        "yhat": yhat_final.tolist(),
        "features_stage1": list(X1.columns),
        "features_stage2": list(X2.columns),
    }
