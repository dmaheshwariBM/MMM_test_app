# core/transforms.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------
# Basic helpers
# ---------------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _nan_to_zero(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return a

# ---------------------------
# Curves & suggestions
# ---------------------------
def transform_none(x: pd.Series, k: float = 0.0) -> pd.Series:
    return to_num(x).fillna(0.0)

def transform_log(x: pd.Series, k: float = 0.01) -> pd.Series:
    """
    f(x) = log(1 + k * max(x,0))
    """
    x = to_num(x).fillna(0.0)
    x = np.maximum(x, 0.0)
    k = float(max(k, 1e-12))
    return pd.Series(np.log1p(k * x), index=x.index)

def transform_negexp(x: pd.Series, k: float = 0.01) -> pd.Series:
    """
    f(x) = 1 - exp(-k * max(x,0))
    """
    x = to_num(x).fillna(0.0)
    x = np.maximum(x, 0.0)
    k = float(max(k, 1e-12))
    return pd.Series(1.0 - np.exp(-k * x), index=x.index)

def suggest_k(series: pd.Series, tfm: str) -> float:
    """
    Suggest curvature k from data.
    Heuristics (on non-negative, numeric series):
      - Log:     k ≈ 1 / mean(x_pos)  → derivative at mean ≈ 0.5
      - NegExp:  k ≈ ln(2) / mean(x_pos) → half-saturation at mean
    Falls back to 0.01 if mean<=0 or empty.
    """
    s = to_num(series)
    s = s[s > 0]
    if s.empty:
        return 0.01
    m = float(s.mean())
    if m <= 0:
        return 0.01
    tfm = (tfm or "").strip().lower()
    if tfm in ("log", "logarithm", "logarithmic"):
        return float(1.0 / m)
    # negexp default
    return float(np.log(2.0) / m)

# ---------------------------
# Finite adstock (your formula)
# Effective_t = x_t + a*x_{t-1} + ... + a^K * x_{t-K}
# ---------------------------
def adstock_finite(x: pd.Series, alpha: float, K: int) -> pd.Series:
    alpha = float(alpha)
    K = int(max(0, K))
    x = to_num(x).fillna(0.0).values.astype(float)

    if K == 0 or alpha <= 0.0:
        return pd.Series(x, index=None).rename(None).set_axis(range(len(x)))

    # Build kernel [1, alpha, alpha^2, ..., alpha^K]
    kernel = np.power(alpha, np.arange(0, K + 1, dtype=float))
    # Convolve and truncate to original length
    conv = np.convolve(x, kernel, mode="full")[: len(x)]
    return pd.Series(conv, index=None).rename(None).set_axis(range(len(x)))

# ---------------------------
# Apply order: transform ↔ adstock+lag
# ---------------------------
def apply_with_order(
    x: pd.Series,
    transform: str,
    k: float,
    lag_months: int,
    adstock_alpha: float,
    order: str,
) -> pd.Series:
    """
    order options:
      - 'Transform→Adstock+Lag'
      - 'Adstock+Lag→Transform'
    Lag is applied by shifting the original x by K and summing with decay (finite adstock).
    """
    # build the adstocked series first (so we can reuse it)
    K = int(max(0, lag_months))
    a = float(np.clip(adstock_alpha, 0.0, 1.0))
    x_num = to_num(x).fillna(0.0)

    # adstock per your formula
    x_ad = adstock_finite(x_num, a, K)

    t = (transform or "none").strip().lower()
    if order.startswith("Transform"):
        # Transform on raw values, then adstock on transformed series
        if t in ("log", "logarithm", "logarithmic"):
            base = transform_log(x_num, k)
        elif t in ("negexp", "negative_exponential", "neg_exp"):
            base = transform_negexp(x_num, k)
        else:
            base = transform_none(x_num, 0.0)
        out = adstock_finite(base, a, K)
    else:
        # Adstock first, then transform
        if t in ("log", "logarithm", "logarithmic"):
            out = transform_log(x_ad, k)
        elif t in ("negexp", "negative_exponential", "neg_exp"):
            out = transform_negexp(x_ad, k)
        else:
            out = x_ad

    # always numeric, nan→0
    out = to_num(out).fillna(0.0)
    return out

# ---------------------------
# Vectorized apply for multiple metrics using a config list
# ---------------------------
def apply_bulk(
    df: pd.DataFrame,
    config_rows: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    config_rows: list of rows with keys:
      metric, transform, k, lag_months, adstock_alpha, order, use (bool)
    Returns:
      df_out with new columns "<metric>__tfm"
      meta dict with the same config
    """
    out = df.copy()
    cleaned_cfg = []
    for row in config_rows:
        try:
            metric = str(row["metric"])
        except Exception:
            continue
        use = bool(row.get("use", True))
        if not use:
            continue

        transform = str(row.get("transform", "None"))
        k = float(row.get("k", 0.01))
        lag = int(row.get("lag_months", 0))
        alpha = float(row.get("adstock_alpha", 0.0))
        order = str(row.get("order", "Transform→Adstock+Lag"))

        if metric not in out.columns:
            continue

        y = apply_with_order(out[metric], transform, k, lag, alpha, order)
        out[f"{metric}__tfm"] = y

        cleaned_cfg.append({
            "metric": metric,
            "transform": transform,
            "k": k,
            "lag_months": lag,
            "adstock_alpha": alpha,
            "order": order,
            "suggested_k": float(row.get("suggested_k", np.nan)),
            "use": True
        })

    meta = {"config": cleaned_cfg}
    return out, meta
