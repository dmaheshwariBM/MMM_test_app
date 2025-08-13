import pandas as pd
import numpy as np

def lag(series: pd.Series, k: int):
    return series.shift(k).fillna(0)

def adstock(series: pd.Series, alpha: float):
    result = []
    carry = 0.0
    for val in series.fillna(0).astype(float):
        carry = val + alpha*carry
        result.append(carry)
    return pd.Series(result, index=series.index)

def saturation(series: pd.Series, k: float, theta: float):
    s = series.clip(lower=0).astype(float)
    return (s**k) / (s**k + theta**k + 1e-12)

def log_transform(series: pd.Series):
    return np.log1p(series.clip(lower=0))

def scale(series: pd.Series):
    s = series.astype(float)
    return (s - s.mean())/(s.std()+1e-9)
