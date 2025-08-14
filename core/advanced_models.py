# core/advanced_models.py
from __future__ import annotations
import os, json, math, itertools, re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

# Reuse your main OLS routine
from core import modeling


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _add_const(X: pd.DataFrame, add_const: bool = True) -> pd.DataFrame:
    if add_const and "const" not in X.columns:
        X = X.copy()
        X.insert(0, "const", 1.0)
    return X

def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out

def _product_name(a: str, b: str) -> str:
    return f"{a}__x__{b}"

def _make_interactions(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    Xix = pd.DataFrame(index=df.index)
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            Xix[_product_name(a,b)] = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
                                      pd.to_numeric(df[b], errors="coerce").fillna(0.0)
    return Xix

def _metrics(y: np.ndarray, yhat: np.ndarray, p: int) -> Dict[str, float]:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    n = len(y)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2)) if n > 0 else 0.0
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    rmse = math.sqrt(sse / n) if n > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1)
    return {"n": int(n), "p": int(p), "r2": float(r2), "adj_r2": float(adj_r2), "rmse": float(rmse)}


# ------------------------------------------------------------------
# Decomposition: Base / Carryover / Incremental / Impactable by channel
# ------------------------------------------------------------------
def _strip_tfm(name: str) -> str:
    return name[:-5] if name.endswith("__tfm") else name

def _load_tfm_meta_map(dataset_csv_name: str) -> Dict[str, Tuple[float, int]]:
    """
    Returns mapping: '<metric>__tfm' -> (alpha, K)
    Looks for data/transforms_<stem>.json produced by the Transformations page.
    """
    stem = os.path.splitext(dataset_csv_name)[0]
    meta_path = os.path.join("data", f"transforms_{stem}.json")
    m: Dict[str, Tuple[float, int]] = {}
    if not os.path.exists(meta_path):
        return m
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        for row in meta.get("config", []):
            metric = str(row.get("metric", ""))
            alpha = float(row.get("adstock_alpha", 0.0))
            K = int(row.get("lag_months", 0))
            if metric:
                m[f"{metric}__tfm"] = (alpha, K)
    except Exception:
        pass
    return m

def _carry_share(alpha: float, K: int) -> float:
    """
    Fraction of total adstocked weight due to lags (t-1...t-K), i.e.,
    (a + a^2 + ... + a^K) / (1 + a + ... + a^K).
    """
    alpha = float(alpha); K = int(max(0, K))
    if K <= 0 or alpha <= 0.0:
        return 0.0
    if abs(alpha - 1.0) < 1e-10:
        denom = K + 1.0
    else:
        denom = (1.0 - alpha**(K + 1)) / (1.0 - alpha)
    denom = max(denom, 1e-12)
    return float(1.0 - (1.0 / denom))

def _decompose_linear(
    X: pd.DataFrame,
    coef: pd.Series | Dict[str, float],
    yhat: np.ndarray | pd.Series,
    dataset_csv_name: str,
    clip_negative_contrib: bool = True,
) -> Dict[str, Any]:
    """
    Linear contributions decomp with interaction allocation and carryover split.
    - Interactions 'A__x__B' are split 50/50 to A and B.
    - Carryover share per channel uses α,K from transforms metadata if available.
    """
    if isinstance(coef, dict):
        coef = pd.Series(coef)
    coef = coef.reindex(X.columns).fillna(0.0)
    yhat = np.asarray(yhat, float)

    # Precompute transform/carryover map
    tfm_map = _load_tfm_meta_map(dataset_csv_name)

    # Column-wise summed contribution
    contrib_sum: Dict[str, float] = {}
    for c in X.columns:
        s = float((pd.to_numeric(X[c], errors="coerce").fillna(0.0) * float(coef.get(c, 0.0))).sum())
        if clip_negative_contrib:
            s = max(0.0, s)
        contrib_sum[c] = s

    # Denominator
    total_pred = float(np.maximum(np.sum(yhat), 1e-12))
    if total_pred <= 0:
        total_pred = float(sum(contrib_sum.values())) or 1.0

    # Base (intercept if present)
    base_sum = contrib_sum.get("const", 0.0)

    # Channel aggregation (impactable & carry components)
    channel_incr: Dict[str, float] = {}
    channel_carry: Dict[str, float] = {}

    def _add_channel(name: str, value: float, carry_share: float):
        name_disp = _strip_tfm(name)
        ch_incr = value * max(0.0, 1.0 - carry_share)
        ch_carry = value * max(0.0, carry_share)
        channel_incr[name_disp] = channel_incr.get(name_disp, 0.0) + ch_incr
        channel_carry[name_disp] = channel_carry.get(name_disp, 0.0) + ch_carry

    for c, s in contrib_sum.items():
        if c == "const":
            continue
        if "__x__" in c:
            # interaction allocation
            a, b = c.split("__x__")
            half = 0.5 * s
            # carry shares based on each side if available
            ca = _carry_share(*tfm_map.get(a, tfm_map.get(f"{a}__tfm", (0.0, 0))))
            cb = _carry_share(*tfm_map.get(b, tfm_map.get(f"{b}__tfm", (0.0, 0))))
            _add_channel(a, half, ca)
            _add_channel(b, half, cb)
        else:
            # single feature
            cs = _carry_share(*tfm_map.get(c, (0.0, 0)))
            _add_channel(c, s, cs)

    carry_sum = float(sum(channel_carry.values()))
    incr_sum = float(sum(channel_incr.values()))

    # Percentages
    base_pct = 100.0 * base_sum / total_pred
    carry_pct = 100.0 * carry_sum / total_pred
    incr_pct = max(0.0, 100.0 - base_pct - carry_pct)

    impactable_pct = {k: (100.0 * v / total_pred) for k, v in sorted(channel_incr.items(), key=lambda z: -z[1])}

    return {
        "base_pct": base_pct,
        "carryover_pct": carry_pct,
        "incremental_pct": incr_pct,
        "impactable_pct": impactable_pct,
        "total_pred_sum": total_pred,
        "base_sum": base_sum,
        "carry_sum": carry_sum,
        "incremental_sum": incr_sum,
    }


# ------------------------------------------------------------------
# Breakout (grouped) models
# ------------------------------------------------------------------
def run_breakout_models(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    group_col: str,
    dataset_csv_name: str,
    min_group_n: int = 30,
    add_const: bool = True,
    force_nonnegative: bool = True,
) -> Dict[str, Any]:
    """
    Fits a separate OLS model per group value.
    Adds decomposition for each group model.
    """
    res: Dict[str, Any] = {}
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in dataframe")

    for g, dfg in df.groupby(group_col):
        if len(dfg) < min_group_n:
            continue
        y = pd.to_numeric(dfg[target], errors="coerce").fillna(0.0).values
        X = _safe_numeric(dfg, features)
        X = _add_const(X, add_const)

        out = modeling.ols_model(
            X, y,
            add_const=False,
            compute_vif=False,
            force_nonnegative=force_nonnegative
        )

        # Decomposition on this group's X/yhat
        coef_ser = out["coef"] if isinstance(out["coef"], pd.Series) else pd.Series(out["coef"])
        decomp = _decompose_linear(X, coef_ser, np.array(out["yhat"]), dataset_csv_name)

        res[str(g)] = {
            "group": str(g),
            "metrics": out["metrics"],
            "coef": coef_ser.to_dict(),
            "yhat": list(out["yhat"]),
            "features": list(X.columns),
            "decomp": decomp,
        }
    return {"type": "breakout", "group_col": group_col, "results": res}


# ------------------------------------------------------------------
# Interaction / Pathway models
# ------------------------------------------------------------------
def suggest_interactions_by_corr(
    df: pd.DataFrame, target: str, features: List[str], top_k: int = 5
) -> List[Tuple[str, str]]:
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
    pairs = list(itertools.combinations(features, 2))
    scores = []
    for a, b in pairs:
        prod = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
               pd.to_numeric(df[b], errors="coerce").fillna(0.0)
        if prod.std(ddof=0) == 0:
            continue
        c = prod.corr(y)
        if pd.notna(c):
            scores.append((abs(float(c)), (a, b)))
    scores.sort(reverse=True, key=lambda z: z[0])
    return [pair for _, pair in scores[:top_k]]

def run_interaction_model(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    interaction_pairs: List[Tuple[str, str]],
    dataset_csv_name: str,
    add_const: bool = True,
    force_nonnegative: bool = False,
) -> Dict[str, Any]:
    """
    Adds specified interaction columns to X and fits OLS.
    Includes decomposition where interaction contributions are split 50/50 to parents.
    """
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).values
    X_base = _safe_numeric(df, features)
    X_ix = _make_interactions(df, interaction_pairs)
    X = pd.concat([X_base, X_ix], axis=1)
    X = _add_const(X, add_const)

    out = modeling.ols_model(
        X, y,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative
    )

    coef_ser = out["coef"] if isinstance(out["coef"], pd.Series) else pd.Series(out["coef"])
    decomp = _decompose_linear(X, coef_ser, np.array(out["yhat"]), dataset_csv_name)

    return {
        "type": "interaction",
        "metrics": out["metrics"],
        "coef": coef_ser.to_dict(),
        "yhat": list(out["yhat"]),
        "features": list(X.columns),
        "interaction_pairs": interaction_pairs,
        "decomp": decomp,
    }


# ------------------------------------------------------------------
# Residual (two-stage) models
# ------------------------------------------------------------------
def _lag_series(s: pd.Series, k: int) -> pd.Series:
    if k <= 0: return s.copy()
    return s.shift(k).fillna(0.0)

def run_residual_model(
    df: pd.DataFrame,
    target: str,
    base_features: List[str],
    residual_features: List[str],
    dataset_csv_name: str,
    base_add_const: bool = True,
    resid_add_const: bool = True,
    ar_lags: int = 0,
    force_nonnegative_base: bool = False,
    force_nonnegative_resid: bool = False,
) -> Dict[str, Any]:
    """
    Stage 1: target ~ base_features    -> yhat_base
    Stage 2: (y - yhat_base) ~ residual_features (+ AR lags of residuals) -> yhat_resid
    Final yhat = yhat_base + yhat_resid
    Decomposition:
      - Channel impactable/carryover from Stage 1 features
      - Residual stage contributions grouped into 'ResidualAdj' (incremental bucket)
    """
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0)

    # Stage 1
    X1 = _safe_numeric(df, base_features)
    X1 = _add_const(X1, base_add_const)
    out1 = modeling.ols_model(
        X1, y.values,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative_base
    )
    yhat1 = pd.Series(out1["yhat"], index=df.index)
    coef1 = out1["coef"] if isinstance(out1["coef"], pd.Series) else pd.Series(out1["coef"]).reindex(X1.columns).fillna(0.0)

    # Residuals
    resid = y - yhat1

    # Stage 2
    X2 = _safe_numeric(df, residual_features)
    # AR lags of residuals
    if ar_lags > 0:
        for k in range(1, ar_lags + 1):
            X2[f"resid_lag{k}"] = _lag_series(resid, k)
    X2 = _add_const(X2, resid_add_const)
    out2 = modeling.ols_model(
        X2, resid.values,
        add_const=False,
        compute_vif=False,
        force_nonnegative=force_nonnegative_resid
    )
    yhat2 = pd.Series(out2["yhat"], index=df.index)
    coef2 = out2["coef"] if isinstance(out2["coef"], pd.Series) else pd.Series(out2["coef"]).reindex(X2.columns).fillna(0.0)

    # Final prediction
    yhat_final = (yhat1 + yhat2).values

    # Decomposition on Stage 1 (channel-facing) + add a ResidualAdj bucket from Stage 2
    decomp1 = _decompose_linear(X1, coef1, yhat1.values, dataset_csv_name)
    # Residual stage contributions (sum of all Stage 2 features including const)
    contrib2_sum = float((X2.mul(coef2, axis=1)).sum(axis=0).clip(lower=0.0).sum())
    total_pred = max(1e-12, float(np.sum(yhat_final)))
    # Allocate Stage 2 as an incremental bucket called 'ResidualAdj'
    impactable_pct = dict(decomp1["impactable_pct"])
    residual_adj_pct = 100.0 * contrib2_sum / total_pred
    if residual_adj_pct > 0:
        impactable_pct["ResidualAdj"] = impactable_pct.get("ResidualAdj", 0.0) + residual_adj_pct

    base_pct = decomp1["base_pct"]  # base intercept comes from stage 1 const
    carry_pct = decomp1["carryover_pct"]
    incr_pct = max(0.0, 100.0 - base_pct - carry_pct)  # includes ResidualAdj now

    return {
        "type": "residual",
        "metrics": _metrics(y.values, yhat_final, p=len(X1.columns) + len(X2.columns)),
        "stage1": {"metrics": out1["metrics"], "coef": coef1.to_dict()},
        "stage2": {"metrics": out2["metrics"], "coef": coef2.to_dict()},
        "yhat": list(yhat_final),
        "features_stage1": list(X1.columns),
        "features_stage2": list(X2.columns),
        "decomp": {
            "base_pct": base_pct,
            "carryover_pct": carry_pct,
            "incremental_pct": incr_pct,
            "impactable_pct": impactable_pct
        }
    }
