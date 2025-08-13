import pandas as pd
import numpy as np

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# ---------- Lag (simple shift; still available if needed elsewhere) ----------
def lag(series: pd.Series, k: int):
    return _to_num(series).shift(k).fillna(0)

# ---------- Infinite (legacy) adstock: s'_t = x_t + alpha*s'_{t-1} ----------
# Kept for compatibility; not used by the new finite-lag logic unless you want it.
def adstock(series: pd.Series, alpha: float):
    result = []
    carry = 0.0
    for val in _to_num(series).fillna(0).astype(float):
        carry = val + alpha * carry
        result.append(carry)
    return pd.Series(result, index=series.index)

# ---------- FINITE adstocked distributed lag (your requested behavior) ----------
# Effective_t = sum_{i=0..K} alpha^i * x_{t-i}
def adstock_finite(series: pd.Series, alpha: float, K: int):
    s = _to_num(series).fillna(0).astype(float)
    K = int(max(0, K))
    if K == 0:
        return s
    out = s.copy()  # start with i=0 weight = 1
    w = 1.0
    for i in range(1, K + 1):
        w *= float(alpha)  # weight = alpha^i
        if w == 0.0:
            break
        out = out.add(s.shift(i).fillna(0) * w, fill_value=0.0)
    return out

# ---------- Other transforms ----------
def saturation(series: pd.Series, k: float, theta: float):
    s = _to_num(series).fillna(0).clip(lower=0).astype(float)
    return (s ** k) / (s ** k + theta ** k + 1e-12)

def log_transform(series: pd.Series):
    return np.log1p(_to_num(series).fillna(0).clip(lower=0))

def scale(series: pd.Series):
    s = _to_num(series).fillna(0).astype(float)
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std

def minmax_scale(series: pd.Series):
    s = _to_num(series).fillna(0).astype(float)
    mn, mx = s.min(), s.max()
    denom = (mx - mn) if mx != mn else 1.0
    return (s - mn) / denom

# ---------- Negative exponential transforms ----------
def negexp(series: pd.Series, beta: float = 0.01):
    """y = 1 - exp(-beta * x)  (diminishing returns)"""
    s = _to_num(series).fillna(0).clip(lower=0).astype(float)
    return 1.0 - np.exp(-beta * s)

def negexp_cannibal(series: pd.Series, beta: float, pool: pd.Series, gamma: float):
    """
    y = (1 - exp(-beta * x)) * exp(-gamma * pool)
    pool: normalized 'other activity' sum aligned to series index
    """
    base = negexp(series, beta=beta)
    p = pd.to_numeric(pool, errors="coerce").fillna(0.0)
    p = p.reindex(base.index, fill_value=0.0)
    return base * np.exp(-gamma * p)

# ---------- Base-transform helper (no lag/adstock inside) ----------
def base_transform(series: pd.Series, kind: str, beta: float = 0.01, pool: pd.Series | None = None, gamma: float = 0.0):
    """
    kind ∈ {"None","Log","NegExp","NegExp+Cannibalization"}
    """
    if kind == "None":
        return series
    if kind == "Log":
        return log_transform(series)
    if kind == "NegExp":
        return negexp(series, beta=beta)
    if kind == "NegExp+Cannibalization":
        if pool is None:
            pool = pd.Series(0.0, index=series.index)
        return negexp_cannibal(series, beta=beta, pool=pool, gamma=gamma)
    raise ValueError(f"Unknown base transform kind: {kind}")
