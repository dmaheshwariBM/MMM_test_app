# core/modeling.py
# v1.5.0  ASCII-only. Dependency-light OLS with optional nonnegative clipping.

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

def _to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0
    return X

def _ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # returns beta including intercept as first coeff if X already contains a 1s column
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta

def _metrics(y: np.ndarray, yhat: np.ndarray, p: int) -> Dict[str, float]:
    n = len(y)
    sse = float(np.sum((y - yhat) ** 2))
    ybar = float(np.mean(y)) if n > 0 else 0.0
    tss = float(np.sum((y - ybar) ** 2))
    r2 = float(1.0 - sse / tss) if tss > 0 else float("nan")
    adj = float(1.0 - (1.0 - r2) * (n - 1) / (n - p)) if (n > p and not np.isnan(r2)) else float("nan")
    rmse = float(np.sqrt(sse / n)) if n > 0 else float("nan")
    return {"r2": r2, "adj_r2": adj, "rmse": rmse, "n": n, "p": p}

def ols_model(
    X_df: pd.DataFrame,
    y: pd.Series,
    force_nonnegative: bool = True
) -> Tuple[Dict[str, float], Dict[str, float], List[float]]:
    """
    Returns (coef_dict, metrics, yhat_list)
    coef_dict has 'const' for intercept and feature->coef for each column in X_df.
    If force_nonnegative=True, negative betas (excluding intercept) are set to 0
    and intercept is refit to minimize MSE given fixed nonnegative betas.
    """
    # prepare matrices
    X = np.asarray(X_df.values, dtype=float)
    yv = np.asarray(pd.to_numeric(y, errors="coerce").fillna(0.0), dtype=float)

    # add intercept as first column of ones
    ones = np.ones((X.shape[0], 1), dtype=float)
    Xw = np.hstack([ones, X])

    # unconstrained OLS
    beta = _ols_closed_form(Xw, yv)  # [b0, b1, b2, ...]
    b0 = float(beta[0])
    b = beta[1:].astype(float)

    if force_nonnegative:
        # clip negatives to zero, then refit intercept only
        b = np.maximum(b, 0.0)
        yhat_wo_b0 = X.dot(b)
        # best intercept under squared error is mean of residual
        b0 = float(np.mean(yv - yhat_wo_b0))
        yhat = b0 + yhat_wo_b0
    else:
        yhat = Xw.dot(beta)

    # assemble outputs
    coef: Dict[str, float] = {"const": float(b0)}
    for j, col in enumerate(X_df.columns):
        coef[col] = float(b[j])

    mets = _metrics(yv, np.asarray(yhat), p=1 + X.shape[1])
    return coef, mets, list(map(float, yhat))

def build_design(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return _to_numeric_df(df, features)
