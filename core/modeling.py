# core/modeling.py
# v1.6.0  ASCII-only, dependency-light OLS with optional nonnegative betas
# Exposes:
#   - build_design(df, features) -> X_df
#   - ols_model(X_df, y, force_nonnegative=True) -> (coef_dict, metrics, yhat_list)
#   - compute_decomposition(df, coef, features, target=None, yhat=None) -> decomp dict

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

def build_design(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0
    return X

def _ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    - coef_dict includes "const" and one entry per X_df column.
    - If force_nonnegative is True, negative betas (excluding intercept) are clipped to 0,
      and the intercept is refit to minimize MSE given fixed nonnegative betas.
    """
    X = np.asarray(X_df.values, dtype=float)
    yv = np.asarray(pd.to_numeric(y, errors="coerce").fillna(0.0), dtype=float)

    ones = np.ones((X.shape[0], 1), dtype=float)
    Xw = np.hstack([ones, X])

    beta = _ols_closed_form(Xw, yv)  # [b0, b1, b2, ...]
    b0 = float(beta[0])
    b = beta[1:].astype(float)

    if force_nonnegative:
        b = np.maximum(b, 0.0)
        yhat_wo_b0 = X.dot(b)
        b0 = float(np.mean(yv - yhat_wo_b0))
        yhat = b0 + yhat_wo_b0
    else:
        yhat = Xw.dot(beta)

    coef: Dict[str, float] = {"const": float(b0)}
    for j, col in enumerate(X_df.columns):
        coef[col] = float(b[j])

    mets = _metrics(yv, np.asarray(yhat), p=1 + X.shape[1])
    return coef, mets, list(map(float, yhat))

def _intercept_key(coef: Dict[str, float]) -> Optional[str]:
    for k in ("const", "Intercept", "intercept", "CONST", "const_", "_const", "beta0", "b0"):
        if k in coef:
            return k
    return None

def compute_decomposition(
    df: pd.DataFrame,
    coef: Dict[str, float],
    features: List[str],
    target: Optional[str] = None,
    yhat: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Returns:
      { "base_pct": float, "carryover_pct": 0.0,
        "incremental_pct": float, "impactable_pct": {feature: pct, ...} }
    - Base is the intercept share.
    - Contributions from channels are positive-only when forming percentages.
    - Denominator preference: sum(y) if target provided, else sum(yhat) if provided,
      else sum of positive contributions, else 1.0.
    """
    n = len(df)
    # channel contributions
    contrib_sum: Dict[str, float] = {}
    ik = _intercept_key(coef)
    if ik is not None:
        contrib_sum["const"] = float(coef.get(ik, 0.0)) * n

    for f in features:
        if f == "const":
            continue
        c = float(coef.get(f, 0.0))
        x = pd.to_numeric(df[f], errors="coerce").fillna(0.0) if f in df.columns else pd.Series([0.0] * n)
        s = float(np.sum(np.maximum(c * x.values, 0.0)))
        contrib_sum[f] = s

    total_from_y = None
    if target and target in df.columns:
        total_from_y = float(pd.to_numeric(df[target], errors="coerce").fillna(0.0).sum())
    total_from_yhat = float(np.nansum(yhat)) if yhat is not None else 0.0
    total_from_contrib = float(sum(contrib_sum.values())) if contrib_sum else 0.0
    candidates = [t for t in (total_from_y, total_from_yhat, total_from_contrib) if t and t > 0]
    denom = candidates[0] if candidates else 1.0

    base_sum = float(contrib_sum.get("const", 0.0))
    base_pct = 100.0 * base_sum / denom

    impact_map: Dict[str, float] = {}
    for f, s in contrib_sum.items():
        if f == "const":
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        if s > 0:
            impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * s / denom

    incr_pct = float(sum(impact_map.values()))
    carry_pct = 0.0

    # light normalization to 100 pct
    total = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total - 100.0) > 0.05:
        target_incr = max(0.0, 100.0 - base_pct - carry_pct)
        scale = target_incr / incr_pct if incr_pct > 0 else 1.0
        for k in list(impact_map.keys()):
            impact_map[k] = impact_map[k] * scale
        incr_pct = float(sum(impact_map.values()))

    return {
        "base_pct": float(round(base_pct, 6)),
        "carryover_pct": float(round(carry_pct, 6)),
        "incremental_pct": float(round(incr_pct, 6)),
        "impactable_pct": {k: float(round(v, 6)) for k, v in impact_map.items()},
    }
