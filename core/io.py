import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

# -------- Basic loaders --------
def load_csv_or_excel(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    x = pd.ExcelFile(path)
    return x.parse(x.sheet_names[0])

def detect_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date","week","month","day","period")):
            return c
    # fallback: first parsable
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            pass
    return None

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    n = len(df)
    # nulls
    for c in df.columns:
        cnt = int(df[c].isna().sum())
        if cnt > 0:
            issues.append((c, f"Nulls: {cnt} ({cnt/max(n,1)*100:.1f}%)"))
    # duplicates
    dup = int(df.duplicated().sum())
    if dup > 0:
        issues.append(("__rows__", f"Duplicate rows: {dup}"))
    # numeric-as-text
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                pd.to_numeric(df[c])
                issues.append((c, "Numeric values stored as text"))
            except Exception:
                pass
    # date gaps
    date_col = detect_date_col(df)
    if date_col is not None:
        try:
            s = pd.to_datetime(df[date_col])
            gaps = s.sort_values().diff().dropna()
            if not gaps.empty and gaps.max() > pd.Timedelta(days=10):
                issues.append((date_col, "Irregular or large date gaps"))
        except Exception:
            pass
    return pd.DataFrame(issues, columns=["Column","Issue"])

# -------- Type inference + coercion (with report) --------
def infer_dtype(series: pd.Series) -> str:
    s = series
    # datetime guess
    try:
        pd.to_datetime(s.dropna().astype(str).head(50), errors="raise")
        return "datetime"
    except Exception:
        pass
    if pd.api.types.is_bool_dtype(s): return "boolean"
    if pd.api.types.is_integer_dtype(s): return "integer"
    if pd.api.types.is_float_dtype(s): return "float"
    if pd.api.types.is_categorical_dtype(s): return "category"
    return "string"

def _coerce_one(series: pd.Series, ui_type: str, fmt_choice: str = "Auto", custom_fmt: str = "") -> Tuple[pd.Series, Dict[str, Any]]:
    """Coerce a single column and return (coerced_series, report)."""
    ui_type = ui_type.lower()
    s0 = series
    rep: Dict[str, Any] = {"requested_type": ui_type, "coerced_nulls": 0, "notes": ""}

    if ui_type == "integer":
        before_nulls = int(s0.isna().sum())
        s = pd.to_numeric(s0, errors="coerce")
        after_nulls = int(s.isna().sum())
        rep["coerced_nulls"] = max(0, after_nulls - before_nulls)
        s = s.round().astype("Int64")
        return s, rep

    if ui_type == "float":
        before_nulls = int(s0.isna().sum())
        s = pd.to_numeric(s0, errors="coerce").astype(float)
        after_nulls = int(s.isna().sum())
        rep["coerced_nulls"] = max(0, after_nulls - before_nulls)
        return s, rep

    if ui_type == "string":
        return s0.astype("string"), rep

    if ui_type == "boolean":
        s = s0
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            m = s.astype(str).str.strip().str.lower()
            s = m.replace(
                {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "1": True, "0": False}
            )
        before_nulls = int(pd.isna(s).sum())
        out = pd.Series(s, index=s0.index).astype("boolean")
        after_nulls = int(pd.isna(out).sum())
        rep["coerced_nulls"] = max(0, after_nulls - before_nulls)
        return out, rep

    if ui_type == "category":
        return s0.astype("string").astype("category"), rep

    if ui_type == "datetime":
        fmt_map = {"Auto": None, "DD/MM/YYYY": "%d/%m/%Y", "MM/DD/YYYY": "%m/%d/%Y", "YYYY-MM-DD": "%Y-%m-%d"}
        fmt = fmt_map.get(fmt_choice, None)
        if fmt_choice == "Custom":
            fmt = custom_fmt.strip() or None
        dayfirst = (fmt_choice == "DD/MM/YYYY")
        s = pd.to_datetime(s0, format=fmt, errors="coerce", dayfirst=dayfirst)
        rep["coerced_nulls"] = int(s.isna().sum()) - int(pd.isna(s0).sum())
        return s, rep

    # fallback
    return s0, rep

def coerce_with_report(
    df_raw: pd.DataFrame,
    schema_map: Dict[str, str],
    date_fmt_map: Dict[str, Tuple[str, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_typed, report_df) where report_df has per-column coercion stats.
    """
    df = df_raw.copy()
    rows = []
    for c, kind in schema_map.items():
        fmt_choice, custom_fmt = date_fmt_map.get(c, ("Auto",""))
        coerced, rep = _coerce_one(df[c], kind, fmt_choice, custom_fmt)
        rep["column"] = c
        # non-finites check for numeric
        if kind in ("integer","float"):
            ninf = int(np.isinf(pd.to_numeric(coerced, errors="coerce")).sum())
            rep["non_finite"] = ninf
        else:
            rep["non_finite"] = 0
        df[c] = coerced
        rows.append(rep)
    report = pd.DataFrame(rows, columns=["column","requested_type","coerced_nulls","non_finite","notes"])
    return df, report

# -------- Modeling readiness --------
def validate_for_modeling(df: pd.DataFrame, target: str, features: List[str]) -> Tuple[bool, List[str], List[str], pd.DataFrame, pd.Series]:
    """
    Checks types, NaNs, constants, non-finite values.
    Returns (ok, errors, warnings, X_clean, y_clean).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if target not in df.columns:
        return False, [f"Target '{target}' not found."], [], pd.DataFrame(), pd.Series(dtype=float)

    for f in features:
        if f not in df.columns:
            errors.append(f"Feature '{f}' not found.")

    if errors:
        return False, errors, [], pd.DataFrame(), pd.Series(dtype=float)

    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().any():
        errors.append(f"Target '{target}' contains NaNs after numeric coercion.")

    X = df[features].copy()

    # ensure numeric features
    for c in features:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().any():
                errors.append(f"Feature '{c}' contains non-numeric values; coercion produced NaNs.")

    # drop rows with NaN or non-finite
    mask_finite = np.isfinite(X.select_dtypes(include=[np.number])).all(axis=1)
    mask_notna = (~X.isna()).all(axis=1) & (~y.isna())
    mask = mask_finite & mask_notna
    dropped = int(len(X) - mask.sum())
    if dropped > 0:
        warnings.append(f"Dropped {dropped} row(s) due to NaN/inf in features or target.")

    X_clean = X.loc[mask].astype(float)
    y_clean = y.loc[mask].astype(float)

    # constant features
    consts = [c for c in X_clean.columns if X_clean[c].nunique(dropna=True) <= 1]
    if consts:
        warnings.append("Dropped constant feature(s): " + ", ".join(consts))
        X_clean = X_clean.drop(columns=consts)

    ok = len(errors) == 0 and X_clean.shape[1] > 0 and X_clean.shape[0] >= 10
    if X_clean.shape[0] < 10:
        errors.append("Insufficient usable rows after cleaning (< 10).")

    return ok, errors, warnings, X_clean, y_clean

# -------- Helper merge (optional elsewhere) --------
def merge_spend_sales(spend_df: pd.DataFrame, sales_df: pd.DataFrame, date_key="Week") -> pd.DataFrame:
    spend_df = spend_df.copy(); sales_df = sales_df.copy()
    spend_df[date_key] = pd.to_datetime(spend_df[date_key], errors="coerce")
    sales_df[date_key] = pd.to_datetime(sales_df[date_key], errors="coerce")
    spend_wide = spend_df.pivot(index=date_key, columns="Channel", values="Spend").fillna(0.0).sort_index().reset_index()
    master = pd.merge(sales_df.sort_values(date_key), spend_wide, on=date_key, how="inner")
    cols = [date_key, "Sales"] + [c for c in master.columns if c not in (date_key, "Sales")]
    return master[cols]
