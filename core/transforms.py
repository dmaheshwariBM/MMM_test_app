# core/transforms.py
# v2.3.1  ASCII-only. Transform, lag/adstock, scaling + suggestions and pipeline apply.

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# ---------------- Basics ----------------

def ensure_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric and fill NaNs with 0.0 for stable math."""
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def adstock_finite(values: pd.Series, lag: int = 0, decay: float = 0.0) -> pd.Series:
    """
    Finite adstock with lag (L) and decay r in [0,1].
    Effective_t = x_t + r*x_{t-1} + r^2*x_{t-2} + ... + r^L*x_{t-L}
    """
    x = ensure_numeric_series(values).to_numpy(dtype=float)
    L = max(0, int(lag))
    r = float(max(0.0, min(1.0, decay)))
    if L == 0 or r == 0.0:
        return pd.Series(x, index=values.index)
    k = np.array([r**i for i in range(L + 1)], dtype=float)   # [1, r, r^2, ..., r^L]
    y_full = np.convolve(x, k, mode="full")
    y = y_full[: len(x)]
    return pd.Series(y, index=values.index)

# ---------------- Transformations ----------------

def tfm_none(x: pd.Series, **_) -> pd.Series:
    return ensure_numeric_series(x)

def tfm_log(x: pd.Series, k: float = 1.0) -> pd.Series:
    """
    y = log(1 + k*x), k>0
    Helpful for right-skewed spend/volume metrics.
    """
    k = float(max(k, 1e-12))
    return np.log1p(k * ensure_numeric_series(x))

def tfm_negexp(x: pd.Series, k: float = 0.01, beta: float = 1.0) -> pd.Series:
    """
    y = beta * (1 - exp(-k*x)), k>=0, beta>=0
    Saturating response with diminishing returns.
    """
    k = float(max(0.0, k))
    beta = float(max(0.0, beta))
    z = ensure_numeric_series(x)
    return beta * (1.0 - np.exp(-k * z))

def tfm_negexp_cannibalized(
    x: pd.Series,
    pool: Optional[pd.Series] = None,
    k: float = 0.01,
    beta: float = 1.0,
    gamma: float = 0.0
) -> pd.Series:
    """
    Negative exponential with simple cannibalization (0..1).
    y_base = beta * (1 - exp(-k*x))
    y = y_base * (1 - gamma * norm(pool)), norm(p) in [0,1]
    """
    y_base = tfm_negexp(x, k=k, beta=beta)
    g = float(max(0.0, min(1.0, gamma)))
    if pool is None or len(pool) != len(x):
        return y_base
    p = ensure_numeric_series(pool)
    pmin, pmax = float(np.min(p)), float(np.max(p))
    if pmax <= pmin:
        return y_base
    p_norm = (p - pmin) / (pmax - pmin)
    y = y_base * (1.0 - g * p_norm)
    y = np.maximum(y, 0.0)
    return pd.Series(y, index=x.index)

# ---------------- Scaling ----------------

def scale_none(s: pd.Series) -> pd.Series:
    return ensure_numeric_series(s)

def scale_minmax01(s: pd.Series) -> pd.Series:
    x = ensure_numeric_series(s)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax <= vmin:
        return pd.Series(np.zeros_like(x), index=s.index)
    return (x - vmin) / (vmax - vmin)

def scale_zscore(s: pd.Series) -> pd.Series:
    x = ensure_numeric_series(s)
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd <= 0:
        return pd.Series(np.zeros_like(x), index=s.index)
    return (x - mu) / sd

# ---------------- Suggestions ----------------

def _finite_median(x: pd.Series) -> float:
    arr = ensure_numeric_series(x).replace([np.inf, -np.inf], np.nan).dropna().values
    if arr.size == 0:
        return 1.0
    return float(np.median(arr))

def suggest_transform_type(x: pd.Series) -> str:
    """
    Heuristic:
    - If skewness > ~1.0, suggest 'log'
    - Else if max >> median (>= 10x), suggest 'negexp'
    - Else 'none'
    """
    z = ensure_numeric_series(x)
    med = _finite_median(z)
    try:
        skew = float(pd.Series(z).skew())
    except Exception:
        skew = 0.0
    ratio = (float(np.max(z)) / med) if med > 0 else 0.0
    if skew > 1.0:
        return "log"
    if ratio >= 10.0:
        return "negexp"
    return "none"

def suggest_k_for_negexp(x: pd.Series) -> float:
    """k ~ ln(2)/median(x) so median reaches ~50% of saturation."""
    med = _finite_median(x)
    if med <= 0:
        return 0.01
    return float(np.log(2.0) / med)

def suggest_k_for_log(x: pd.Series) -> float:
    """k ~ 1/max(x) so log argument stays O(1)."""
    z = ensure_numeric_series(x)
    mx = float(np.max(z)) if len(z) else 0.0
    if mx <= 0:
        return 1.0
    return float(1.0 / mx)

def suggest_adstock_decay(x: pd.Series) -> float:
    """Suggest decay ~= lag-1 autocorr, clipped to [0,0.9]."""
    z = ensure_numeric_series(x).values.astype(float)
    if len(z) < 3:
        return 0.5
    z0 = z[:-1]
    z1 = z[1:]
    num = float(np.sum((z0 - z0.mean()) * (z1 - z1.mean())))
    den = float(np.sqrt(np.sum((z0 - z0.mean())**2) * np.sum((z1 - z1.mean())**2)))
    r = num / den if den > 0 else 0.0
    return float(max(0.0, min(0.9, r)))

# ---------------- Pipeline ----------------

def apply_pipeline(
    df: pd.DataFrame,
    metric: str,
    params: Dict[str, Any],
    cannibal_pool_cols: Optional[List[str]] = None
) -> pd.Series:
    """
    Apply transform + (lag, adstock) + scaling to df[metric] based on params.
    params:
      - transform: 'none'|'log'|'negexp'|'negexp_cann'
      - k (float), beta (float), gamma (float for cannibalization)
      - order: 'transform_then_adstock' or 'adstock_then_transform'
      - lag (int), adstock (float in [0,1])
      - scaling: 'none'|'minmax'|'zscore'
    """
    x = df[metric] if metric in df.columns else pd.Series([0.0]*len(df))
    x = ensure_numeric_series(x)

    transform = str(params.get("transform", "none")).lower()
    order = str(params.get("order", "transform_then_adstock")).lower()
    lag = int(params.get("lag", 0) or 0)
    ad = float(params.get("adstock", 0.0) or 0.0)
    scaling = str(params.get("scaling", "none")).lower()

    k = float(params.get("k", 1.0) or 1.0)
    beta = float(params.get("beta", 1.0) or 1.0)
    gamma = float(params.get("gamma", 0.0) or 0.0)

    pool = None
    if transform == "negexp_cann" and cannibal_pool_cols:
        cols = [c for c in cannibal_pool_cols if c in df.columns and c != metric]
        if cols:
            pool = ensure_numeric_series(df[cols].sum(axis=1))

    def _transform(series: pd.Series) -> pd.Series:
        if transform == "none":
            return tfm_none(series)
        if transform == "log":
            return tfm_log(series, k=k)
        if transform == "negexp":
            return tfm_negexp(series, k=k, beta=beta)
        if transform == "negexp_cann":
            return tfm_negexp_cannibalized(series, pool=pool, k=k, beta=beta, gamma=gamma)
        return tfm_none(series)

    def _adstock(series: pd.Series) -> pd.Series:
        return adstock_finite(series, lag=lag, decay=ad)

    if order == "adstock_then_transform":
        y = _transform(_adstock(x))
    else:
        y = _adstock(_transform(x))

    if scaling == "minmax":
        y = scale_minmax01(y)
    elif scaling == "zscore":
        y = scale_zscore(y)
    else:
        y = scale_none(y)

    return y

def apply_many(
    df: pd.DataFrame,
    config_map: Dict[str, Dict[str, Any]],
    cannibal_pools: Optional[Dict[str, List[str]]] = None,
    suffix: str = "__tfm"
) -> pd.DataFrame:
    """
    Apply pipelines for multiple metrics. Returns a copy with new columns <metric><suffix>.
    """
    out = df.copy()
    for m, params in config_map.items():
        pool_cols = (cannibal_pools or {}).get(m, [])
        out[m + suffix] = apply_pipeline(df, m, params, pool_cols)
    return out
