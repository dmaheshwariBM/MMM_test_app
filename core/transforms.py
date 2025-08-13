import pandas as pd
import numpy as np

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def lag(series: pd.Series, k: int):
    return _to_num(series).shift(k).fillna(0)

def adstock(series: pd.Series, alpha: float):
    result = []
    carry = 0.0
    for val in _to_num(series).fillna(0).astype(float):
        carry = val + alpha * carry
        result.append(carry)
    return pd.Series(result, index=series.index)

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

# --- New: Negative exponential transforms ---
def negexp(series: pd.Series, beta: float = 0.01):
    """
    Diminishing returns curve: f(x) = 1 - exp(-beta * x)
    """
    s = _to_num(series).fillna(0).clip(lower=0).astype(float)
    return 1.0 - np.exp(-beta * s)

def negexp_cannibal(series: pd.Series, beta: float, pool: pd.Series, gamma: float):
    """
    Negative exponential with cannibalization.
    Base effect: (1 - exp(-beta * x))
    Cannibalization factor: exp(-gamma * pool_norm)
      - pool is a normalized 'other activity' sum passed in by caller
    Final: base * exp(-gamma * pool)
    """
    base = negexp(series, beta=beta)
    # Align pool index
    p = pd.to_numeric(pool, errors="coerce").fillna(0.0)
    p = p.reindex(base.index, fill_value=0.0)
    return base * np.exp(-gamma * p)
