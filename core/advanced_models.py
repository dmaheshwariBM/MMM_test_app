# core/advanced_models.py
"""
Advanced MMM post-processing utilities
Version: 2.2.0

Workflows:
- breakout_split: re-split one base channel’s impact into selected sub-metrics (no intercept; sum preserved)
- residual_reattribute: allocate a chosen FRACTION of Base % to new channels; Base % decreases by that amount
- pathway_redistribute: move a share of Channel A’s impact to Channel B; totals preserved
- apply_decomp_update: apply any of the above updates to a base record’s decomposition
- _ensure_decomp_from_record_or_recompute: robust decomp if missing in saved JSON (stable denominator + normalization)
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

__all__ = [
    "ADV_MODELS_VERSION",
    "breakout_split",
    "residual_reattribute",
    "pathway_redistribute",
    "apply_decomp_update",
    "_ensure_decomp_from_record_or_recompute",
]

ADV_MODELS_VERSION = "2.2.0"

# ---------------------------
# Basic helpers
# ---------------------------
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:64]

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _feature_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return feature series from df. If name=='const', return a 1s vector."""
    if name == "const":
        return pd.Series(np.ones(len(df)), index=df.index, name="const")
    if name in df.columns:
        return _to_num(df[name])
    if name.endswith("__tfm") and name[:-5] in df.columns:
        return _to_num(df[name[:-5]])
    return pd.Series(np.zeros(len(df)), index=df.index, name=name)

def _contrib_series(df: pd.DataFrame, coef: Dict[str, float], features: List[str]) -> Dict[str, pd.Series]:
    """Contribution time series for each feature: coef_j * x_j(t)."""
    out: Dict[str, pd.Series] = {}
    for f in features:
        c = float(coef.get(f, 0.0))
        out[f] = c * _feature_series(df, f)
    return out

# ---------------------------
# Minimal decomposition (robust fallback)
# ---------------------------
def _ensure_decomp_from_record_or_recompute(record: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ensure 'decomp' exists in result record. If missing, recompute a minimal one:
      - base_pct from 'const'
      - impactable_pct from positive contributions of non-const features
      - carryover_pct set to 0 (transform metadata not used here)

    Denominator priority (to avoid blow-ups):
      1) sum(actual target column) if available
      2) sum(yhat) if present and > 0
      3) sum(all feature contributions, incl. const)
      4) 1.0 (never tiny epsilons)
    """
    d = record.get("decomp")
    if isinstance(d, dict) and "impactable_pct" in d:
        # Also normalize/round incoming decomp for consistency
        return _normalize_and_round_decomp(d)

    coef = record.get("coef", {}) or {}
    features = record.get("features", []) or []

    contrib = _contrib_series(df, coef, features)

    # Denominator candidates
    total_from_y = None
    tgt = record.get("target")
    if tgt and tgt in df.columns:
        total_from_y = float(pd.to_numeric(df[tgt], errors="coerce").fillna(0.0).sum())

    yhat = np.asarray(record.get("yhat", []), float)
    total_from_yhat = float(np.nansum(yhat)) if yhat.size > 0 else 0.0

    total_from_contrib = float(sum(float(s.sum()) for s in contrib.values())) if contrib else 0.0

    candidates = [t for t in (total_from_y, total_from_yhat, total_from_contrib) if t and t > 0]
    total_pred = candidates[0] if candidates else 1.0

    # Base and impactable %
    base_sum = float(contrib.get("const", pd.Series(0.0, index=df.index)).sum())
    base_pct = 100.0 * base_sum / total_pred

    impact_map: Dict[str, float] = {}
    for f, s in contrib.items():
        if f == "const":
            continue
        val = float((s.clip(lower=0.0)).sum())  # clip negatives for impact %
        if val <= 0:
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * val / total_pred

    new_decomp = {
        "base_pct": base_pct,
        "carryover_pct": 0.0,
        "incremental_pct": float(sum(impact_map.values())),
        "impactable_pct": impact_map
    }
    return _normalize_and_round_decomp(new_decomp)

def _normalize_and_round_decomp(d: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Base% + Carryover% + Incremental% = 100 (± rounding), and round nicely."""
    base_pct = float(d.get("base_pct", 0.0))
    carry_pct = float(d.get("carryover_pct", 0.0))
    impact_map: Dict[str, float] = dict(d.get("impactable_pct", {}))
    incr_pct = float(sum(impact_map.values()))

    total = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total - 100.0) > 0.05:
        # scale incremental to fit 100 - base - carry
        target_incr = max(0.0, 100.0 - base_pct - carry_pct)
        scale = target_incr / incr_pct if incr_pct > 0 else 1.0
        for k in list(impact_map.keys()):
            impact_map[k] *= scale
        incr_pct = float(sum(impact_map.values()))

    # Round
    base_pct = float(round(base_pct, 6))
    carry_pct = float(round(carry_pct, 6))
    impact_map = {k: float(round(v, 6)) for k, v in impact_map.items()}
    incr_pct = float(round(sum(impact_map.values()), 6))
    return {
        "base_pct": base_pct,
        "carryover_pct": carry_pct,
        "incremental_pct": incr_pct,
        "impactable_pct": impact_map
    }

# ---------------------------
# Breakout split (re-split one base channel into sub-metrics; sum preserved)
# ---------------------------
def breakout_split(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    channel_to_split: str,
    sub_metrics: List[str],
) -> Dict[str, Any]:
    """
    Parent channel impact is redistributed across chosen sub-metrics.
    Fit: y = contrib_parent, X = sub_metric series (no intercept), w>=0 (via clipped OLS).
    Output 'allocated' holds absolute % points for each sub-metric; sums to parent %.
    """
    coef = base_record.get("coef", {}) or {}
    features = [f for f in base_record.get("features", []) or [] if f != "const"]
    # find real feature name for display channel
    parent_feat: Optional[str] = None
    for f in features:
        disp = f[:-5] if f.endswith("__tfm") else f
        if disp == channel_to_split:
            parent_feat = f; break
    if parent_feat is None:
        raise ValueError(f"Channel '{channel_to_split}' not found in base features.")

    decomp = _ensure_decomp_from_record_or_recompute(base_record, df)
    parent_pct = float(decomp.get("impactable_pct", {}).get(channel_to_split, 0.0))

    # y = contribution series of parent
    y = _contrib_series(df, coef, [parent_feat])[parent_feat].values.astype(float)
    y = np.maximum(y, 0.0)

    # X = sub metric series (prefer __tfm if present)
    X_cols: List[np.ndarray] = []
    sub_names: List[str] = []
    for m in sub_metrics:
        name = f"{m}__tfm" if f"{m}__tfm" in df.columns else m
        s = _feature_series(df, name).values.astype(float)
        s = np.maximum(s, 0.0)
        if s.sum() <= 0:
            continue
        X_cols.append(s); sub_names.append(m)
    if not X_cols:
        raise ValueError("No valid sub-metric series found (zero or missing).")

    X = np.column_stack(X_cols)  # n x k
    # nonnegative OLS via clip
    try:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        w = np.maximum(w, 0.0)
    except Exception:
        w = np.maximum(np.array([c.mean() for c in X_cols], dtype=float), 0.0)

    # shares via average contribution normalization
    num = np.array([w[j] * X_cols[j].mean() for j in range(len(X_cols))], dtype=float)
    denom = float(np.sum(num))
    shares = (num / denom) if denom > 1e-12 else np.ones(len(X_cols), dtype=float) / len(X_cols)

    allocated = {sub_names[j]: float(parent_pct * shares[j]) for j in range(len(sub_names))}
    return {
        "type": "breakout_split",
        "split_channel": channel_to_split,
        "original_channel_pct": parent_pct,
        "allocated": allocated,
        "note": "Sub-metric % add up to the original channel impact."
    }

# ---------------------------
# Residual re-attribution (allocate Base % to new channels; FRACTIONAL)
# ---------------------------
def residual_reattribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    extra_channels: List[str],
    fraction: float = 1.0,
) -> Dict[str, Any]:
    """
    Allocate a FRACTION (0–1] of Base % to new channels by fitting: 1 ≈ Σ_j w_j * X_j (no intercept).
    Columns are normalized to [0,1]. w clipped to >=0; shares = w/sum(w).
    Total allocation = fraction * Base%.
    """
    fraction = max(0.0, min(1.0, float(fraction)))
    decomp = _ensure_decomp_from_record_or_recompute(base_record, df)
    base_pct_before = float(decomp.get("base_pct", 0.0))
    allocate_total = base_pct_before * fraction

    mats: List[np.ndarray] = []
    names: List[str] = []
    for c in extra_channels:
        name = f"{c}__tfm" if f"{c}__tfm" in df.columns else c
        s = _feature_series(df, name).values.astype(float)
        s = np.maximum(s, 0.0)
        mx = float(s.max()) if s.size else 0.0
        if mx <= 0:
            continue
        mats.append(s / mx); names.append(c)
    if not mats:
        raise ValueError("No valid extra channel series found to reattribute base.")

    X = np.column_stack(mats)
    y = np.ones((X.shape[0],), dtype=float)
    try:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        w = np.maximum(w, 0.0)
    except Exception:
        w = np.ones((X.shape[1],), dtype=float)

    ssum = float(w.sum())
    shares = (w / ssum) if ssum > 1e-12 else np.ones_like(w) / len(w)

    allocated = {names[j]: float(allocate_total * shares[j]) for j in range(len(names))}
    base_pct_after = max(0.0, base_pct_before - allocate_total)

    return {
        "type": "residual_reattribute",
        "base_pct_before": base_pct_before,
        "fraction": fraction,
        "allocated": allocated,
        "base_pct_after": base_pct_after,
        "note": "Allocated pp come out of Base %; Incremental % increases by the same total."
    }

# ---------------------------
# Pathway redistribution (move impact from A to B; totals preserved)
# ---------------------------
def pathway_redistribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    channel_A: str,
    channel_B: str,
) -> Dict[str, Any]:
    """
    Share s inferred from nonnegative single-factor fit:
      contrib_A ≈ w * X_B (no intercept), s = min(1, sum(yhat)/sum(contrib_A))
    A_new = (1 - s)*A_old;  B_new = B_old + s*A_old
    """
    coef = base_record.get("coef", {}) or {}
    features = [f for f in base_record.get("features", []) or [] if f != "const"]
    decomp = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact_map = dict(decomp.get("impactable_pct", {}))

    def _find_feat(disp: str) -> Optional[str]:
        for f in features:
            name = f[:-5] if f.endswith("__tfm") else f
            if name == disp:
                return f
        return None

    fA = _find_feat(channel_A)
    fB = _find_feat(channel_B)
    if fA is None or fB is None:
        raise ValueError("Selected channels not found in base features.")

    y = _contrib_series(df, coef, [fA])[fA].values.astype(float)
    y = np.maximum(y, 0.0)
    Xb = _feature_series(df, fB).values.astype(float)
    Xb = np.maximum(Xb, 0.0)

    num = float(np.dot(Xb, y)); den = float(np.dot(Xb, Xb))
    w = 0.0 if den <= 1e-12 else max(0.0, num / den)
    yhat = w * Xb
    s_raw = 0.0 if y.sum() <= 1e-12 else float(yhat.sum() / y.sum())
    s = float(max(0.0, min(1.0, s_raw)))

    A_old = float(impact_map.get(channel_A, 0.0))
    B_old = float(impact_map.get(channel_B, 0.0))
    move = float(A_old * s)
    A_new = float(A_old - move)
    B_new = float(B_old + move)

    return {
        "type": "pathway_redistribute",
        "channel_A": channel_A,
        "channel_B": channel_B,
        "share_from_A_to_B": s,
        "moved_pct_points": move,
        "A_old": A_old, "A_new": A_new,
        "B_old": B_old, "B_new": B_new,
        "note": "A loses s×A_old; same amount added to B; totals preserved."
    }

# ---------------------------
# Apply a decomp update to a base record (returns NEW decomp dict)
# ---------------------------
def apply_decomp_update(
    base_record: Dict[str, Any],
    df: pd.DataFrame,
    update: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combine an update with the base decomp:
      - breakout_split: replace parent channel with allocated children
      - residual_reattribute: move a FRACTION of Base % to new channels
      - pathway_redistribute: move % from A to B
    """
    d0 = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact = dict(d0.get("impactable_pct", {}))
    base_pct = float(d0.get("base_pct", 0.0))
    carry_pct = float(d0.get("carryover_pct", 0.0))

    t = update.get("type")

    if t == "breakout_split":
        parent = update["split_channel"]
        alloc: Dict[str, float] = update.get("allocated", {})
        if parent in impact:
            del impact[parent]
        for k, v in alloc.items():
            impact[k] = impact.get(k, 0.0) + float(v)

    elif t == "residual_reattribute":
        alloc: Dict[str, float] = update.get("allocated", {})
        total = float(sum(alloc.values()))
        base_pct = max(0.0, base_pct - total)
        for k, v in alloc.items():
            impact[k] = impact.get(k, 0.0) + float(v)

    elif t == "pathway_redistribute":
        A = update["channel_A"]; B = update["channel_B"]
        move = float(update.get("moved_pct_points", 0.0))
        if A in impact:
            impact[A] = max(0.0, float(impact[A]) - move)
        impact[B] = impact.get(B, 0.0) + move

    # recompute incremental %
    incr_pct = float(sum(impact.values()))
    # normalize (keep base+carry fixed)
    total_pct = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total_pct - 100.0) > 0.05:
        scale = max(0.0, 100.0 - base_pct - carry_pct) / incr_pct
        for k in list(impact.keys()):
            impact[k] *= scale
        incr_pct = float(sum(impact.values()))

    # round
    base_pct = float(round(base_pct, 6))
    carry_pct = float(round(carry_pct, 6))
    impact = {k: float(round(v, 6)) for k, v in impact.items()}
    incr_pct = float(round(sum(impact.values()), 6))

    return {
        "base_pct": base_pct,
        "carryover_pct": carry_pct,
        "incremental_pct": incr_pct,
        "impactable_pct": impact
    }
