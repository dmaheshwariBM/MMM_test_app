# core/modeling.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional

# Optional p-values (SciPy). Safe if unavailable.
try:
    from scipy import stats as _scistats  # type: ignore
    _HAVE_SCIPY_STATS = True
except Exception:
    _HAVE_SCIPY_STATS = False

# Optional scikit-learn (for Ridge/Lasso). Code still works without it.
try:
    import sklearn  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# ----------------------------
# Data prep
# ----------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def prepare_xy(df: pd.DataFrame, target: str, features: List[str], fillna: float = 0.0
               ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Coerce X & y to numeric; fill NaNs with fillna; return coercion report."""
    if target not in df.columns:
        raise ValueError(f"Target `{target}` not found.")
    for c in features:
        if c not in df.columns:
            raise ValueError(f"Feature `{c}` not found.")

    y = _to_num(df[target])
    xdf = df[features].copy()

    report = {"coerced_to_nan": {}}
    for c in features:
        before = int(xdf[c].isna().sum())
        xdf[c] = _to_num(xdf[c])
        after = int(xdf[c].isna().sum())
        report["coerced_to_nan"][c] = max(0, after - before)

    before_y = int(y.isna().sum())
    y = _to_num(y)
    after_y = int(y.isna().sum())
    report["target_coerced_to_nan"] = max(0, after_y - before_y)

    xdf = xdf.fillna(fillna)
    y = y.fillna(fillna)
    return xdf, y, report


# ----------------------------
# Core metrics helpers
# ----------------------------
def _add_const(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _metrics_from_yhat(y: np.ndarray, yhat: np.ndarray, p: int) -> Dict[str, float]:
    n = len(y)
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    ybar = float(np.mean(y)) if n > 0 else 0.0
    sst = float(np.sum((y - ybar) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    adj_r2 = np.nan if (n <= p or np.isnan(r2)) else 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    rmse = float(np.sqrt(sse / n)) if n > 0 else np.nan
    mae = float(np.mean(np.abs(resid))) if n > 0 else np.nan
    mape = float(np.mean(np.abs(resid) / np.maximum(np.abs(y), 1e-12))) if n > 0 else np.nan
    aic = float(n * np.log(sse / n + 1e-12) + 2 * p) if n > 0 else np.nan
    bic = float(n * np.log(sse / n + 1e-12) + p * np.log(n + 1e-12)) if n > 0 else np.nan
    return {
        "n": n, "p": p, "df_resid": max(n - p, 1),
        "sse": sse, "sst": sst, "r2": r2, "adj_r2": adj_r2,
        "rmse": rmse, "mae": mae, "mape": mape, "aic": aic, "bic": bic,
    }


# ----------------------------
# OLS (unconstrained)
# ----------------------------
def _ols_closed_form(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat

    n = X.shape[0]
    p = X.shape[1]
    sse = float(np.sum(resid ** 2))
    ybar = float(np.mean(y)) if n > 0 else 0.0
    sst = float(np.sum((y - ybar) ** 2))
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    adj_r2 = np.nan
    if n > p and not np.isnan(r2):
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p)

    df_resid = max(n - p, 1)
    sigma2 = sse / df_resid
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))

    with np.errstate(divide="ignore", invalid="ignore"):
        tvals = beta / se
    if _HAVE_SCIPY_STATS:
        pvals = 2.0 * (1.0 - _scistats.t.cdf(np.abs(tvals), df=df_resid))
    else:
        pvals = np.full(beta.shape, np.nan)

    metrics = _metrics_from_yhat(y, yhat, p)
    return beta, yhat, {"metrics": metrics, "stderr": se, "tvalues": tvals, "pvalues": pvals, "residuals": y - yhat}


def compute_vif(X_df: pd.DataFrame) -> pd.Series:
    cols = list(X_df.columns)
    k = len(cols)
    if k < 2:
        return pd.Series([np.nan] * k, index=cols, name="VIF")
    vifs = []
    for j in range(k):
        y = X_df.iloc[:, j].values.astype(float)
        X_others = X_df.drop(columns=[cols[j]]).values.astype(float)
        Xo = _add_const(X_others)
        _, yhat, _ = _ols_closed_form(Xo, y)
        resid = y - yhat
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2j = np.nan if sst == 0 else 1.0 - sse / sst
        vifs.append(np.inf if (np.isnan(r2j) or r2j >= 1.0) else 1.0 / (1.0 - r2j))
    return pd.Series(vifs, index=cols, name="VIF")


# ----------------------------
# OLS (with optional non-negative constraint)
# ----------------------------
def ols_model(X_df: pd.DataFrame, y: pd.Series,
              add_constant: bool = True,
              compute_vif_flag: bool = True,
              force_nonnegative: bool = False) -> Dict[str, Any]:
    """
    If force_nonnegative=True:
      - Prefer SciPy NNLS if available for b>=0.
      - Else fit OLS then clamp negatives to 0.
      - Recompute yhat & ALL metrics from constrained beta.
      - std_err / t / p_value are set to NaN (not valid under constraint).
    """
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    X_df = X_df.copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

    X_mat = X_df.values.astype(float)
    if add_constant:
        X = _add_const(X_mat)
        names = ["const"] + list(X_df.columns)
    else:
        X = X_mat
        names = list(X_df.columns)

    if not force_nonnegative:
        beta, yhat, info = _ols_closed_form(X, y.values.astype(float))
        coef = pd.Series(beta, index=names, name="coef")
        stderr = pd.Series(info["stderr"], index=names, name="std_err")
        tvalues = pd.Series(info["tvalues"], index=names, name="t")
        pvalues = pd.Series(info["pvalues"], index=names, name="p_value")
        resid_s = pd.Series(info["residuals"], index=y.index, name="residual")
        yhat_s = pd.Series(yhat, index=y.index, name="yhat")
        metrics = info["metrics"]
    else:
        # Try SciPy NNLS
        beta = None
        try:
            from scipy.optimize import nnls  # type: ignore
            beta, _ = nnls(X, y.values.astype(float))
        except Exception:
            # Fallback: OLS then clamp
            beta, _, _ = np.linalg.lstsq(X, y.values.astype(float), rcond=None)

        beta = np.maximum(0.0, beta)  # ensure non-negative
        yhat = X @ beta
        metrics = _metrics_from_yhat(y.values.astype(float), yhat, X.shape[1])

        coef = pd.Series(beta, index=names, name="coef")
        # Under constraint, classic SE/t/p don't apply
        stderr = pd.Series([np.nan] * len(names), index=names, name="std_err")
        tvalues = pd.Series([np.nan] * len(names), index=names, name="t")
        pvalues = pd.Series([np.nan] * len(names), index=names, name="p_value")
        yhat_s = pd.Series(yhat, index=y.index, name="yhat")
        resid_s = pd.Series(y.values - yhat, index=y.index, name="residual")

    vif = compute_vif(X_df) if compute_vif_flag else None

    return {
        "coef": coef, "stderr": stderr, "tvalues": tvalues, "pvalues": pvalues,
        "yhat": yhat_s, "residuals": resid_s,
        "metrics": metrics, "vif": vif,
    }


# ----------------------------
# Optional Ridge/Lasso (sklearn) with optional non-negativity
# ----------------------------
def _wrap_sklearn_linear(model, X_df: pd.DataFrame, y: pd.Series,
                         add_constant: bool, compute_vif_flag: bool,
                         force_nonnegative: bool) -> Dict[str, Any]:
    X = X_df.copy()
    names = list(X.columns)
    if add_constant:
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
        names = ["const"] + names

    model.fit(X.values, y.values)
    yhat = model.predict(X.values)
    beta = None
    try:
        # Build combined coef vector including intercept if not add_constant
        if add_constant:
            # First column is our const; sklearn handled intercept internally if fit_intercept=False
            if hasattr(model, "coef_"):
                # We don't know which element corresponds to const in coef_ → treat intercept as 0
                beta = np.r_[0.0, np.ravel(model.coef_)]
            else:
                beta = np.zeros(X.shape[1])
        else:
            # add explicit const
            intercept = float(getattr(model, "intercept_", 0.0))
            coef_only = np.ravel(getattr(model, "coef_", np.zeros(X.shape[1])))
            beta = np.r_[intercept, coef_only]
            # and reflect const in names
            names = ["const"] + names
            # rebuild predictions accordingly
            yhat = (np.c_[np.ones((X.shape[0], 1)), X.values] @ beta)
    except Exception:
        beta = np.zeros(X.shape[1])

    # Optional positivity
    if force_nonnegative:
        beta = np.maximum(0.0, beta)
        # Recompute predictions/metrics from clamped beta
        X_design = np.c_[np.ones((X.shape[0], 1)), X.values] if not add_constant else X.values
        if add_constant:
            # our X already contains const as first column
            yhat = X_design @ beta
            p = X.shape[1]
        else:
            yhat = X_design @ beta
            p = X_design.shape[1]
        metrics = _metrics_from_yhat(y.values.astype(float), yhat, p)
        stderr = pd.Series([np.nan]*len(names), index=names, name="std_err")
        tvals  = pd.Series([np.nan]*len(names), index=names, name="t")
        pvals  = pd.Series([np.nan]*len(names), index=names, name="p_value")
    else:
        # Use original predictions
        p = (X.shape[1] if add_constant else X.shape[1] + 1)
        metrics = _metrics_from_yhat(y.values.astype(float), yhat, p)
        stderr = pd.Series([np.nan]*len(names), index=names, name="std_err")  # sklearn doesn't give SEs
        tvals  = pd.Series([np.nan]*len(names), index=names, name="t")
        pvals  = pd.Series([np.nan]*len(names), index=names, name="p_value")

    yhat_s = pd.Series(yhat, index=y.index, name="yhat")
    resid_s = pd.Series(y.values - yhat, index=y.index, name="residual")
    coef_s = pd.Series(beta, index=names, name="coef")
    vif = compute_vif(X_df) if compute_vif_flag else None

    return {
        "coef": coef_s, "stderr": stderr, "tvalues": tvals, "pvalues": pvals,
        "yhat": yhat_s, "residuals": resid_s,
        "metrics": metrics, "vif": vif,
    }


def ridge_model(X_df: pd.DataFrame, y: pd.Series, alpha: float = 1.0,
                add_constant: bool = True, compute_vif_flag: bool = True,
                force_nonnegative: bool = False) -> Dict[str, Any]:
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not installed")
    from sklearn.linear_model import Ridge
    try:
        mdl = Ridge(alpha=alpha, fit_intercept=not add_constant)
    except Exception:
        mdl = Ridge(alpha=alpha)
    return _wrap_sklearn_linear(mdl, X_df, y, add_constant, compute_vif_flag, force_nonnegative)


def lasso_model(X_df: pd.DataFrame, y: pd.Series, alpha: float = 1.0,
                add_constant: bool = True, compute_vif_flag: bool = True,
                force_nonnegative: bool = False) -> Dict[str, Any]:
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not installed")
    from sklearn.linear_model import Lasso
    try:
        mdl = Lasso(alpha=alpha, fit_intercept=not add_constant, max_iter=10000, positive=False)
    except Exception:
        mdl = Lasso(alpha=alpha, max_iter=10000)
    return _wrap_sklearn_linear(mdl, X_df, y, add_constant, compute_vif_flag, force_nonnegative)


# ----------------------------
# Decomposition & impactables
# ----------------------------
def _carryover_fraction(alpha: float, K: int) -> float:
    """Finite-adstock carryover share: sum_{i=1..K} a^i / sum_{i=0..K} a^i."""
    a = float(alpha); K = int(max(0, K))
    if K <= 0 or a <= 0.0:
        return 0.0
    num = a * (1.0 - a**K) / (1.0 - a) if a != 1.0 else float(K)
    den = (1.0 - a**(K+1)) / (1.0 - a) if a != 1.0 else float(K+1)
    frac = num / den if den != 0 else 0.0
    return float(np.clip(frac, 0.0, 1.0))


def _meta_lookup_alphaK(feature_name: str, transforms_meta: Optional[Dict[str, Any]]) -> Tuple[float, int]:
    """
    Expect feature like '<metric>__tfm'. Look up that metric in transforms meta to get (alpha, K).
    If not found, return (0,0).
    """
    if not transforms_meta:
        return 0.0, 0
    metric = feature_name.replace("__tfm", "")
    try:
        cfg = transforms_meta.get("config", [])
        for row in cfg:
            if str(row.get("metric")) == metric:
                a = float(row.get("adstock_alpha", 0.0))
                K = int(row.get("lag_months", 0))
                return a, K
    except Exception:
        pass
    return 0.0, 0


def impact_decomposition(y: pd.Series,
                         yhat: pd.Series,
                         coef: pd.Series,
                         X_df: pd.DataFrame,
                         add_constant: bool,
                         transforms_meta: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """
    Returns:
      - base_pct (of total = base + incremental_pos)
      - carryover_pct (of total; estimated using α,K per feature)
      - incremental_pct (of total)
      - impactable_pct (pd.Series) — per feature, normalized to 100 across channels
      - per_feature_contrib (pd.Series) — positive contributions over the dataset
    Notes:
      * We use positive contributions only for the incremental pool.
      * Denominator for % is (base_total + incremental_total_pos) so Base% + Incremental% = 100.
      * Intercept (base) is included only if add_constant and coef includes 'const'.
    """
    n = len(yhat)
    # Base contribution over the window
    base_total = 0.0
    if add_constant and "const" in coef.index:
        base_total = max(0.0, float(coef["const"]) * n)

    # Feature contributions (sum of beta_j * X_j across rows), clipped >=0
    coef_no_const = coef.drop(index="const") if "const" in coef.index else coef
    contrib = {}
    for col in coef_no_const.index:
        v = float(coef_no_const[col]) * pd.to_numeric(X_df[col], errors="coerce").fillna(0.0).astype(float)
        contrib[col] = float(v.clip(lower=0).sum())
    contrib_s = pd.Series(contrib, dtype=float)

    inc_total = float(contrib_s.sum())
    denom = max(base_total + inc_total, 1e-9)  # ensures Base% + Incremental% = 100

    # Impactable % by channel (normalized across channels to 100%)
    impactable = contrib_s.copy()
    if inc_total > 0:
        impactable = (impactable / inc_total) * 100.0
    else:
        impactable[:] = 0.0
    impactable.name = "Impactable %"

    # Carryover % (of total), using α,K (finite adstock) per feature
    carryover_total = 0.0
    if transforms_meta is not None and len(contrib_s) > 0:
        for col, cval in contrib_s.items():
            a, K = _meta_lookup_alphaK(col, transforms_meta)
            frac = _carryover_fraction(a, K)
            carryover_total += cval * frac
    carryover_pct = float((carryover_total / denom) * 100.0) if denom > 0 else 0.0

    res = {
        "base_pct": float((base_total / denom) * 100.0) if denom > 0 else 0.0,
        "carryover_pct": carryover_pct,
        "incremental_pct": float((inc_total / denom) * 100.0) if denom > 0 else 0.0,
        "impactable_pct": impactable,
        "per_feature_contrib": contrib_s,
        "denominator_total": denom,
    }
    return res
