# core/advanced_models.py
"""
Advanced MMM post-processing utilities
Version: 2.1.0

Workflows implemented (no SciPy dependency):
- breakout_split: re-split one base channel’s impact into selected sub-metrics (no intercept; sum preserved)
- residual_reattribute: allocate a portion of Base % to new channels; Base % decreases by same total
- pathway_redistribute: move a share of Channel A’s impact to Channel B; totals preserved
- apply_decomp_update: apply any of the above updates to a base record’s decomposition
- _ensure_decomp_from_record_or_recompute: robust decomp if missing in saved JSON

All functions are pure-Pandas/NumPy and compatible with Streamlit Cloud.
"""
from __future__ import annotations
import os, re, math
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

ADV_MODELS_VERSION = "2.1.0"

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
    # Sometimes transforms saved as 'metric__tfm', but raw exists:
    if name.endswith("__tfm") and name[:-5] in df.columns:
        return _to_num(df[name[:-5]])
    # last resort: zeros
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
    """
    if isinstance(record.get("decomp"), dict) and "impactable_pct" in record["decomp"]:
        return record["decomp"]

    coef = record.get("coef", {}) or {}
    features = record.get("features", []) or []
    yhat = np.asarray(record.get("yhat", []), float)
    if yhat.size == 0:
        yhat = np.zeros(len(df), dtype=float)

    contrib = _contrib_series(df, coef, features)
    total_pred = float(np.maximum(yhat.sum(), 1e-12))
    if total_pred <= 0:
        total_pred = float(sum(s.sum() for s in contrib.values())) or 1.0

    base_sum = float(contrib.get("const", pd.Series(0.0, index=df.index)).sum())
    base_pct = 100.0 * base_sum / total_pred

    impact_map: Dict[str, float] = {}
    for f, s in contrib.items():
        if f == "const":
            continue
        val = float((s.clip(lower=0.0)).sum())
        if val <= 0:
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * val / total_pred

    carry_pct = 0.0
    incr_pct = max(0.0, 100.0 - base_pct - carry_pct)

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
    channel_to_split: str,      # display name, e.g. "Paid Search"
    sub_metrics: List[str],     # NOT in base model
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
# Residual re-attribution (allocate Base % to new channels)
# ---------------------------
def residual_reattribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    extra_channels: List[str],   # NOT in base model
) -> Dict[str, Any]:
    """
    Allocate a fraction of Base % to new channels by fitting ones ≈ sum_j w_j * X_j (no intercept).
    Columns are normalized to [0,1]. w clipped to >=0; shares = w/sum(w).
    """
    decomp = _ensure_decomp_from_record_or_recompute(base_record, df)
    base_pct_before = float(decomp.get("base_pct", 0.0))

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

    allocated = {names[j]: float(base_pct_before * shares[j]) for j in range(len(names))}
    total_alloc = float(sum(allocated.values()))
    base_pct_after = max(0.0, base_pct_before - total_alloc)

    return {
        "type": "residual_reattribute",
        "base_pct_before": base_pct_before,
        "allocated": allocated,
        "base_pct_after": base_pct_after,
        "note": "Allocated pp come out of Base % and increase Incremental %."
    }

# ---------------------------
# Pathway redistribution (move impact from A to B; totals preserved)
# ---------------------------
def pathway_redistribute(
    df: pd.DataFrame,
    base_record: Dict[str, Any],
    channel_A: str,  # loses some share
    channel_B: str,  # gains it
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
            if name == disp: return f
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
      - residual_reattribute: move % from Base to new channels
      - pathway_redistribute: move % from A to B
    """
    d = _ensure_decomp_from_record_or_recompute(base_record, df)
    impact = dict(d.get("impactable_pct", {}))
    base_pct = float(d.get("base_pct", 0.0))
    carry_pct = float(d.get("carryover_pct", 0.0))

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
    # normalize (keep base+carry fixed) to avoid rounding drift
    total_pct = base_pct + carry_pct + incr_pct
    if abs(total_pct - 100.0) > 0.01 and incr_pct > 0:
        scale = max(0.0, 100.0 - base_pct - carry_pct) / incr_pct
        for k in list(impact.keys()):
            impact[k] = impact[k] * scale
        incr_pct = float(sum(impact.values()))

    return {
        "base_pct": base_pct,
        "carryover_pct": carry_pct,
        "incremental_pct": incr_pct,
        "impactable_pct": impact
    }
