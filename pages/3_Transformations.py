# pages/3_Transformations.py
# v3.3.0  Table-style config (no forms), robust picker, preview, validate, save.

import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

from core import transforms as T

PAGE_ID = "TRANSFORMATIONS_PAGE_v3_3_0"
st.title("Transformations")
st.caption(f"Page ID: {PAGE_ID}")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cross-page banners
if st.session_state.get("last_saved_path"):
    st.success(f"Saved: {st.session_state['last_saved_path']}")
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# -------- Guard: required core functions ----------
_required = [
    "suggest_transform_type", "suggest_k_for_log", "suggest_k_for_negexp",
    "suggest_adstock_decay", "apply_pipeline", "apply_many", "adstock_finite"
]
_missing = [fn for fn in _required if not hasattr(T, fn)]
if _missing:
    st.error("core/transforms.py is out of date. Missing: " + ", ".join(_missing))
    st.stop()

# ---------------- Dataset picker (CSV + XLSX with sheet) ----------------
def _list_datasets() -> List[str]:
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".xlsx"))])

def _p(fn: str) -> str:
    return os.path.join(DATA_DIR, fn)

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def _read_excel_cached(path: str, sheet: str, mtime: float) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)

st.subheader("Select dataset")
cL, cR = st.columns([3,1])
with cR:
    if st.button("Refresh files"):
        st.rerun()

datasets = _list_datasets()
if not datasets:
    st.info("No datasets in data/. Upload CSV/XLSX on Data Upload page.")
    st.stop()

default_idx = 0
if st.session_state.get("tfm_selected_file") in datasets:
    default_idx = datasets.index(st.session_state["tfm_selected_file"])

with cL:
    dataset_name = st.selectbox("Dataset (from data/)", options=datasets, index=default_idx, key="tfm_selected_file")

excel_sheet = None
if dataset_name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(_p(dataset_name))
        sheets = xls.sheet_names
    except Exception as e:
        sheets = []
        st.error(f"Could not read Excel sheets: {e}")
    if sheets:
        def_sheet_idx = 0
        if st.session_state.get("tfm_selected_sheet") in sheets:
            def_sheet_idx = sheets.index(st.session_state["tfm_selected_sheet"])
        excel_sheet = st.selectbox("Excel sheet", options=sheets, index=def_sheet_idx, key="tfm_selected_sheet")

# Load with cache keyed by file mtime
try:
    p = _p(dataset_name)
    mtime = os.path.getmtime(p)
    if dataset_name.lower().endswith(".csv"):
        df = _read_csv_cached(p, mtime)
    else:
        if not excel_sheet:
            excel_sheet = pd.ExcelFile(p).sheet_names[0]
        df = _read_excel_cached(p, excel_sheet, mtime)
except Exception as e:
    st.error(f"Failed to load {dataset_name}: {e}")
    st.stop()

st.caption("Loaded: data/{}".format(dataset_name) + (f" • sheet: {excel_sheet}" if excel_sheet else ""))

# ---------------- Roles & time window ----------------
st.subheader("Roles & window")

# Detect date-like columns
date_like = []
for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        date_like.append(c)
    else:
        try:
            pd.to_datetime(df[c].head(20), errors="raise")
            date_like.append(c)
        except Exception:
            pass

c1, c2, c3, c4 = st.columns(4)
with c1:
    time_col = st.selectbox("Time column (optional)", options=["(none)"] + date_like, index=0)
    if time_col != "(none)":
        df["_tfm_time"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["_tfm_time"] = pd.NaT
with c2:
    id_col = st.selectbox("ID column (optional)", options=["(none)"] + list(df.columns), index=0)
with c3:
    seg_col = st.selectbox("Segment column (optional)", options=["(none)"] + list(df.columns), index=0)
with c4:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    tgt = st.selectbox("Target column (for reference)", options=["(none)"] + numeric_cols, index=0)

# Clamp time window to available data
if time_col != "(none)":
    valid_times = df["_tfm_time"].dropna()
    if len(valid_times) == 0:
        st.warning("Selected time column has no parseable dates; window controls disabled.")
    else:
        dt_min = valid_times.min().date()
        dt_max = valid_times.max().date()
        start_dt, end_dt = st.date_input("Time window", value=(dt_min, dt_max), min_value=dt_min, max_value=dt_max)
        if isinstance(start_dt, (list, tuple)):  # streamlit older versions
            start_dt, end_dt = start_dt[0], end_dt[1]
        mask = (df["_tfm_time"].dt.date >= start_dt) & (df["_tfm_time"].dt.date <= end_dt)
        df = df.loc[mask].reset_index(drop=True)
        st.caption(f"Filtered rows: {len(df)} ({start_dt} to {end_dt})")

# ---------------- Metrics to transform ----------------
excluded = set()
for c in (id_col, seg_col, tgt, "_tfm_time"):
    if c and c != "(none)":
        excluded.add(c)
metrics_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]
if not metrics_all:
    st.info("No numeric metrics available for transformation in this window.")
    st.stop()

st.subheader("Configure transformations (table)")
sel_metrics = st.multiselect("Metrics to include", options=metrics_all, default=metrics_all[:min(8, len(metrics_all))])

cfg_key = f"TFM_CFG__{dataset_name}__{excel_sheet or ''}"
if cfg_key not in st.session_state:
    st.session_state[cfg_key] = {}

def _defaults_for(s: pd.Series) -> Dict[str, Any]:
    tfm = T.suggest_transform_type(s)
    if tfm == "log":
        k = T.suggest_k_for_log(s); beta = 1.0
    elif tfm == "negexp":
        k = T.suggest_k_for_negexp(s); beta = 1.0
    else:
        k = 1.0; beta = 1.0
    return {
        "transform": tfm,
        "k": round(float(k), 6),
        "beta": float(beta),
        "gamma": 0.0,
        "order": "transform_then_adstock",
        "lag": 0,
        "adstock": round(float(T.suggest_adstock_decay(s)), 3),
        "scaling": "none",
        "cannibal_pool": "",  # comma-separated list
    }

# Ensure defaults for selected
for m in sel_metrics:
    if m not in st.session_state[cfg_key]:
        st.session_state[cfg_key][m] = _defaults_for(df[m])

# Build editable table dataframe
rows = []
for m in sel_metrics:
    c = st.session_state[cfg_key][m]
    rows.append({
        "metric": m,
        "transform": c.get("transform", "none"),
        "k": float(c.get("k", 1.0)),
        "beta": float(c.get("beta", 1.0)),
        "gamma": float(c.get("gamma", 0.0)),
        "order": c.get("order", "transform_then_adstock"),
        "lag": int(c.get("lag", 0)),
        "adstock": float(c.get("adstock", 0.0)),
        "scaling": c.get("scaling", "none"),
        "cannibal_pool": c.get("cannibal_pool", "") if isinstance(c.get("cannibal_pool",""), str) else ",".join(c.get("cannibal_pool", [])),
    })
cfg_df = pd.DataFrame(rows)

# Editor with column configs
cfg_df_edited = st.data_editor(
    cfg_df,
    key="tfm_cfg_editor",
    num_rows="fixed",
    use_container_width=True,
    disabled=["metric"],
    column_config={
        "metric": st.column_config.TextColumn("Metric", help="Source column (fixed)"),
        "transform": st.column_config.SelectboxColumn(
            "Transform",
            options=["none", "log", "negexp", "negexp_cann"],
            help="Choose transform function"
        ),
        "k": st.column_config.NumberColumn("k", help="Curvature: log/negexp scale", step=1e-3, format="%.6f", min_value=0.0),
        "beta": st.column_config.NumberColumn("beta", help="Negexp amplitude", step=0.1, format="%.3f", min_value=0.0),
        "gamma": st.column_config.NumberColumn("gamma", help="Cannibalization 0..1 (negexp_cann only)", step=0.05, format="%.2f", min_value=0.0, max_value=1.0),
        "order": st.column_config.SelectboxColumn("Order", options=["transform_then_adstock","adstock_then_transform"]),
        "lag": st.column_config.NumberColumn("Lag", help="Periods to include in adstock window", min_value=0, max_value=52, step=1),
        "adstock": st.column_config.NumberColumn("Adstock", help="Decay 0..1", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
        "scaling": st.column_config.SelectboxColumn("Scaling", options=["none","minmax","zscore"]),
        "cannibal_pool": st.column_config.TextColumn("Cannibal pool", help="Comma-separated other metrics"),
    },
)

# Sync edited table back to session
for _, r in cfg_df_edited.iterrows():
    m = r["metric"]
    if m in st.session_state[cfg_key]:
        st.session_state[cfg_key][m].update({
            "transform": str(r["transform"]),
            "k": float(r["k"]),
            "beta": float(r["beta"]),
            "gamma": float(r["gamma"]),
            "order": str(r["order"]),
            "lag": int(r["lag"]),
            "adstock": float(r["adstock"]),
            "scaling": str(r["scaling"]),
            "cannibal_pool": [x.strip() for x in str(r["cannibal_pool"] or "").split(",") if x.strip()],
        })

# Suggested (all)
if st.button("Apply suggested (all)"):
    for m in sel_metrics:
        keep_pool = st.session_state[cfg_key][m].get("cannibal_pool", [])
        st.session_state[cfg_key][m] = {**_defaults_for(df[m]), "cannibal_pool": keep_pool}
    st.rerun()

st.divider()

# Build config_map + cannibal pools from session
def _build_config_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in sel_metrics:
        out[m] = dict(st.session_state[cfg_key][m])
    return out

config_map = _build_config_map()
cannib_pools = {m: st.session_state[cfg_key][m].get("cannibal_pool", []) for m in sel_metrics}

# Validate and previews
cV, cP, cS = st.columns([1,1,1])
with cV:
    if st.button("Validate"):
        issues = []
        for m, p in config_map.items():
            if m not in df.columns:
                issues.append(f"{m}: not in dataframe."); continue
            if not pd.api.types.is_numeric_dtype(df[m]):
                issues.append(f"{m}: not numeric.")
            if p.get("transform") in ("negexp", "negexp_cann") and float(p.get("k",0.0)) < 0:
                issues.append(f"{m}: k must be >= 0.")
            if p.get("transform") in ("negexp", "negexp_cann") and float(p.get("beta",0.0)) < 0:
                issues.append(f"{m}: beta must be >= 0.")
            ad = float(p.get("adstock", 0.0))
            if not (0.0 <= ad <= 1.0):
                issues.append(f"{m}: adstock must be in [0,1].")
            lag = int(p.get("lag", 0))
            if lag < 0:
                issues.append(f"{m}: lag must be >= 0.")
        if issues:
            st.error("Validation issues:"); [st.write("- " + e) for e in issues]
        else:
            st.success("Validation passed.")

with cP:
    metric_to_preview = st.selectbox("Preview metric", options=sel_metrics if sel_metrics else ["(none)"])
    if st.button("Preview selected metric"):
        if not sel_metrics:
            st.warning("Select metrics above first.")
        else:
            params = config_map[metric_to_preview]
            y = T.apply_pipeline(df, metric_to_preview, params, cannib_pools.get(metric_to_preview, []))
            prev = pd.DataFrame({metric_to_preview: df[metric_to_preview], metric_to_preview+"__tfm_preview": y})
            st.line_chart(prev[[metric_to_preview, metric_to_preview+"__tfm_preview"]])
            st.dataframe(prev.head(50), use_container_width=True)

with cS:
    if st.button("Preview transformed table"):
        try:
            out = T.apply_many(df, config_map, cannibal_pools=cannib_pools, suffix="__tfm")
            cols = []
            for m in sel_metrics:
                cols.extend([m, m + "__tfm"])
            st.dataframe(out[cols].head(100), use_container_width=True)
        except Exception as e:
            st.error(f"Preview failed: {e}")

st.subheader("Apply & Save")
base = os.path.splitext(os.path.basename(dataset_name))[0]
default_out = f"{base}__tfm.csv"
out_name = st.text_input("Output filename (data/)", value=default_out)

if st.button("Apply and save"):
    try:
        out = T.apply_many(df, config_map, cannibal_pools=cannib_pools, suffix="__tfm")
        dest = os.path.join(DATA_DIR, out_name)
        out.to_csv(dest, index=False)
        st.session_state["last_saved_path"] = dest
        st.session_state["last_save_error"] = ""
        st.success(f"Saved: {dest}")
    except Exception as e:
        st.session_state["last_saved_path"] = ""
        st.session_state["last_save_error"] = f"Save failed: {e}"
        st.error(st.session_state["last_save_error"])
