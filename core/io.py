import pandas as pd
import numpy as np
import os

def load_csv_or_excel(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def detect_date_col(df: pd.DataFrame):
    for c in df.columns:
        if "date" in c.lower() or "week" in c.lower() or "month" in c.lower():
            return c
    # fallback try parse first column
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    # nulls
    for c in df.columns:
        nulls = int(df[c].isna().sum())
        if nulls>0:
            rows.append((c, f"Nulls: {nulls} ({nulls/n*100:.1f}%)"))
    # duplicates
    dups = int(df.duplicated().sum())
    if dups>0:
        rows.append(("__rows__", f"Duplicate rows: {dups}"))
    # numeric as object
    for c in df.columns:
        if df[c].dtype=="object":
            try:
                pd.to_numeric(df[c])
                rows.append((c, "Numeric values stored as text"))
            except Exception:
                pass
    # date coverage
    date_col = detect_date_col(df)
    if date_col is not None:
        try:
            s = pd.to_datetime(df[date_col])
            gaps = (s.sort_values().diff().dropna().value_counts())
            if not gaps.empty and gaps.index.max()>pd.Timedelta(days=10):
                rows.append((date_col, "Irregular date gaps"))
        except Exception:
            pass
    return pd.DataFrame(rows, columns=["Column","Issue"])

def merge_spend_sales(spend_df: pd.DataFrame, sales_df: pd.DataFrame, date_key="Week") -> pd.DataFrame:
    spend_pivot = spend_df.pivot(index=date_key, columns="Channel", values="Spend").fillna(0.0).reset_index()
    merged = pd.merge(sales_df, spend_pivot, on=date_key, how="inner")
    return merged
