# core/advanced_models.py
# v1.4.0  ASCII-only. Deterministic, dependency-light advanced adjustments.
# Implements:
#   - _ensure_decomp_from_record_or_recompute(record, df)
#   - breakout_split(df, base_record, channel_to_split, sub_metrics)
#   - residual_reattribute(df, base_record, extra_channels, fraction=1.0)
#   - pathway_redistribute(df, base_record, channel_A, channel_B)
#   - apply_decomp_update(base_record, df, new_decomp)

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    from core import modeling
except Exception:
    modeling = None  # the page guards for this

ADV_MODELS_VERSION = "1.4.0"

def _norm_decomp(d: Dict[str, Any]) -> Dict[str, Any]:
    base = float(d.get("base_pct", 0.0))
    carry = float(d.get("carryover_pct", 0.0))
    impact = {str(k): float(v) for k, v in dict(d.get("impactable_pct", {})).items()}
    incr = float(sum(impact.values()))
    total = base + carry + incr
    if incr > 0 and abs(total - 100.0) > 0.05:
        target_incr = max(0.0, 100.0 - base - carry)
        scale = target_incr / incr if incr > 0 else 1.0
        for k in list(impact.keys()):
            impact[k] = impact[k] * scale
        incr = float(sum(impact.values()))
    return {
        "base_pct": float(round(base, 6)),
        "carryover_pct": float(round(carry, 6)),
        "incremental_pct": float(round(incr, 6)),
        "impactable_pct": {k: float(round(v, 6)) for k, v in impact.items()},
    }

def _intercept_key(coef: Dict[str, float]) -> Optional[str]:
    for k in ("const", "Intercept", "intercept", "CONST", "const_", "_const", "beta0", "b0"):
        if k in coef:
            return k
    return None

def _ensure_decomp_from_record_or_recompute(record: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Return a normalized decomp; compute from coef if not present."""
    d = record.get("decomp")
    if isinstance(d, dict) and "impactable_pct" in d:
        return _norm_decomp(d)

    # recompute from coef/features if modeling is available
    if modeling is None:
        return {"base_pct": float("nan"), "carryover_pct": 0.0, "incremental_pct": float("nan"), "impactable_pct": {}}

    features = list(record.get("features", []) or [])
    coef = dict(record.get("coef", {}) or {})
    target = record.get("target")
    yhat = record.get("yhat")
    try:
        return modeling.compute_decomposition(df, coef, features, target=target, yhat=yhat)
    except Exception:
        return {"base_pct": float("nan"), "carryover_pct": 0.0, "incremental_pct": float("nan"), "impactable_pct": {}}

def breakout_split(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    channel_to_split: str,
    sub_metrics: List[str]
) -> Dict[str, Any]:
    """
    Split the impact of channel_to_split across sub_metrics (no intercept involved).
    Weights are proportional to the positive sum of each sub metric over the train window.
    If all sums are zero, split equally.
    """
    # get current decomp
    cur = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact = dict(cur.get("impactable_pct", {}))
    parent = channel_to_split
    if parent not in impact:
        # nothing to split; return unchanged
        return cur

    parent_pct = float(impact.get(parent, 0.0))
    if parent_pct <= 0.0 or not sub_metrics:
        return cur

    # compute weights for subs
    sums = []
    for s in sub_metrics:
        if s in df.columns:
            v = pd.to_numeric(df[s], errors="coerce").fillna(0.0)
            sums.append(float(np.sum(np.maximum(v.values, 0.0))))
        else:
            sums.append(0.0)

    total = float(sum(sums))
    if total <= 0.0:
        # equal split if no signal
        w = [1.0 / len(sub_metrics)] * len(sub_metrics)
    else:
        w = [x / total for x in sums]

    # build new impact map
    new_imp = {k: float(v) for k, v in impact.items() if k != parent}
    for s, wi in zip(sub_metrics, w):
        new_imp[s] = new_imp.get(s, 0.0) + parent_pct * wi

    return _norm_decomp({
        "base_pct": cur.get("base_pct", 0.0),
        "carryover_pct": cur.get("carryover_pct", 0.0),
        "impactable_pct": new_imp
    })

def residual_reattribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    extra_channels: List[str],
    fraction: float = 1.0
) -> Dict[str, Any]:
    """
    Reallocate a fraction of the fitted Base (intercept) into extra channels that were not in the base model.
    Approach (deterministic and dependency-light):
      - Let base_pct be the current base impact percentage.
      - Build a "base series" per row: intercept contribution per row = coef['const'] (or 0).
      - Regress base_series on extras (OLS), clip negative betas at zero.
      - Weight each extra by its predicted share; allocate base_pct * fraction by those weights.
      - Reduce base_pct accordingly; add the allocated share to the extras (create if missing).
    """
    cur = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact = dict(cur.get("impactable_pct", {}))
    base_pct = float(cur.get("base_pct", 0.0))
    if base_pct <= 0.0 or not extra_channels:
        return cur

    coef = dict(base_record.get("coef", {}) or {})
    ik = _intercept_key(coef)
    const_val = float(coef.get(ik, 0.0)) if ik is not None else 0.0

    n = len(df)
    base_series = np.full(n, max(0.0, const_val), dtype=float)  # nonnegative per-row intercept contribution

    # design for extras
    X = []
    valid_cols = []
    for c in extra_channels:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values.astype(float)
            X.append(col)
            valid_cols.append(c)
    if not X:
        return cur
    X = np.vstack(X).T  # shape (n, k)

    # OLS then clip negatives to zero, refit intercept for base_series (not needed here for weights)
    beta, *_ = np.linalg.lstsq(X, base_series, rcond=None)
    beta = np.maximum(beta, 0.0)
    preds = X.dot(beta)  # k extras combined
    if float(np.sum(preds)) <= 0.0:
        # fallback: equal weights
        w = [1.0 / len(valid_cols)] * len(valid_cols)
    else:
        # weight each extra by its standalone positive contribution norm
        indiv = []
        for j in range(len(valid_cols)):
            pj = np.maximum(X[:, j] * beta[j], 0.0)
            indiv.append(float(np.sum(pj)))
        s = float(sum(indiv))
        if s <= 0.0:
            w = [1.0 / len(valid_cols)] * len(valid_cols)
        else:
            w = [x / s for x in indiv]

    frac = float(max(0.0, min(1.0, fraction)))
    allocate = base_pct * frac

    new_base = base_pct - allocate
    new_imp = {k: float(v) for k, v in impact.items()}
    for c, wi in zip(valid_cols, w):
        new_imp[c] = new_imp.get(c, 0.0) + allocate * wi

    return _norm_decomp({
        "base_pct": new_base,
        "carryover_pct": cur.get("carryover_pct", 0.0),
        "impactable_pct": new_imp
    })

def pathway_redistribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    channel_A: str,
    channel_B: str
) -> Dict[str, Any]:
    """
    Move a fraction s of A's impact to B, where s ~ R^2 from OLS(A ~ B, intercept).
    This keeps the total incremental share the same across A and B.
    """
    cur = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact = dict(cur.get("impactable_pct", {}))
    if channel_A not in impact or channel_B not in impact:
        return cur

    if channel_A not in df.columns or channel_B not in df.columns:
        return cur

    a = pd.to_numeric(df[channel_A], errors="coerce").fillna(0.0).values.astype(float)
    b = pd.to_numeric(df[channel_B], errors="coerce").fillna(0.0).values.astype(float)
    ones = np.ones((len(a), 1), dtype=float)
    X = np.hstack([ones, b.reshape(-1,1)])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    yhat = X.dot(beta)
    # R^2
    ss_res = float(np.sum((a - yhat) ** 2))
    ss_tot = float(np.sum((a - float(np.mean(a))) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

    s = float(r2)  # share to move
    a_old = float(impact.get(channel_A, 0.0))
    b_old = float(impact.get(channel_B, 0.0))
    move = a_old * s

    impact[channel_A] = max(0.0, a_old - move)
    impact[channel_B] = b_old + move

    return _norm_decomp({
        "base_pct": cur.get("base_pct", 0.0),
        "carryover_pct": cur.get("carryover_pct", 0.0),
        "impactable_pct": impact
    })

def apply_decomp_update(base_record: Dict[str, Any], df: pd.DataFrame, new_decomp: Dict[str, Any]) -> Dict[str, Any]:
    """Currently returns the provided new_decomp normalized. Future-safe for merging modes."""
    return _norm_decomp(new_decomp)
