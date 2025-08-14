# pages/3_Transformations.py
# v3.2.1  Robust dataset picker, full per-metric transform UI, preview, validate, save.

import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core import transforms as T

PAGE_ID = "TRANSFORMATIONS_PAGE_v3_2_1"
st.title("Transformations")
st.caption("Page ID: {}".format(PAGE_ID))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cross-page banners
if st.session_state.get("last_saved_path"):
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# --------- Sanity check on core API (prevents silent AttributeError) ---------
_required = [
    "suggest_transform_type", "suggest_k_for_log", "suggest_k_for_negexp",
    "suggest_adstock_decay", "apply_pipeline", "apply_many", "adstock_finite"
]
_missing = [fn for fn in _required if not hasattr(T, fn)]
if _missing:
    st.error("core/transforms.py is out of date. Missing: {}".format(", ".join(_missing)))
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

colL, colR = st.columns([3,1])
with colR:
    if st.button("Refresh files"):
        st.rerun()

datasets = _list_datasets()
if not datasets:
    st.info("No datasets in data/. Upload CSV/XLSX on the Data Upload page.")
    st.stop()

default_idx = 0
if "tfm_selected_file" in st.session_state and st.session_state["tfm_selected_file"] in datasets:
    default_idx = datasets.index(st.session_state["tfm_selected_file"])

with colL:
    dataset_name = st.selectbox("Dataset (from data/)", options=datasets, index=default_idx, key="tfm_selected_file")

excel_sheet = None
if dataset_name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(_p(dataset_name))
        sheets = xls.sheet_names
    except Exception as e:
        sheets = []
        st.error("Could not read Excel sheets: {}".format(e))
    if sheets:
        def_sheet_idx = 0
        if "tfm_selected_sheet" in st.session_state and st.session_state["tfm_selected_sheet"] in sheets:
            def_sheet_idx = sheets.index(st.session_state["tfm_selected_sheet"])
        excel_sheet = st.selectbox("Excel sheet", options=sheets, index=def_sheet_idx, key="tfm_selected_sheet")

# Load with cache keyed by file mtime to avoid stale data
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
    st.error("Failed to load {}: {}".format(dataset_name, e))
    st.stop()

st.caption("Loaded: data/{}".format(dataset_name) + (f" • sheet: {excel_sheet}" if excel_sheet else ""))

# ---------------- Roles & time window ----------------

st.subheader("Roles & window")

# detect parseable datelike columns
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

if time_col != "(none)":
    valid_times = df["_tfm_time"].dropna()
    if len(valid_times) == 0:
        st.warning("Selected time column has no parseable dates; window controls disabled.")
    else:
        dt_min = valid_times.min().date()
        dt_max = valid_times.max().date()
        start_dt, end_dt = st.date_input(
            "Time window",
            value=(dt_min, dt_max),
            min_value=dt_min,
            max_value=dt_max
        )
        if isinstance(start_dt, (list, tuple)):
            start_dt, end_dt = start_dt[0], start_dt[1]
        mask = (df["_tfm_time"].dt.date >= start_dt) & (df["_tfm_time"].dt.date <= end_dt)
        df = df.loc[mask].reset_index(drop=True)
        st.caption("Filtered rows: {} ({} to {})".format(len(df), start_dt, end_dt))

# ---------------- Metrics to transform ----------------

excluded = set()
for c in (id_col, seg_col, tgt, "_tfm_time"):
    if c and c != "(none)":
        excluded.add(c)
metrics_all = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]

if not metrics_all:
    st.info("No numeric metrics available for transformation in this window.")
    st.stop()

cfg_key = f"TFM_CFG__{dataset_name}__{excel_sheet or ''}"
if cfg_key not in st.session_state:
    st.session_state[cfg_key] = {}

st.subheader("Configure transformations")
sel_metrics = st.multiselect("Metrics to transform", options=metrics_all, default=metrics_all[:min(6, len(metrics_all))])

# Reference: maths
with st.expander("Reference: transformation maths", expanded=False):
    st.markdown(
        """
        - **Log**:  y = log(1 + kx).  k scales input.
        - **Negative exponential**:  y = beta * (1 - exp(-k x)).
        - **Negative exponential with cannibalization**: same as above, times (1 - gamma * pool_norm).
        - **Adstock** (finite):  y_t = x_t + r x_{t-1} + ... + r^L x_{t-L}.
        - **Order**: Transform → Adstock, or Adstock → Transform.
        - **Scaling**: None, Min–Max [0,1], or Z-score.
        """
    )

def _metric_defaults(s: pd.Series) -> Dict[str, Any]:
    tfm = T.suggest_transform_type(s)
    if tfm == "log":
        k = T.suggest_k_for_log(s)
        beta = 1.0
    elif tfm == "negexp":
        k = T.suggest_k_for_negexp(s)
        beta = 1.0
    else:
        k = 1.0
        beta = 1.0
    return {
        "transform": tfm,
        "k": round(float(k), 6),
        "beta": float(beta),
        "gamma": 0.0,
        "order": "transform_then_adstock",
        "lag": 0,
        "adstock": round(float(T.suggest_adstock_decay(s)), 3),
        "scaling": "none",
        "cannibal_pool": [],
    }

# ensure defaults exist
for m in sel_metrics:
    if m not in st.session_state[cfg_key]:
        st.session_state[cfg_key][m] = _metric_defaults(df[m])

# Use a form to avoid flicker while typing
with st.form("tfm_form", clear_on_submit=False):
    for m in sel_metrics:
        conf = st.session_state[cfg_key][m]
        with st.expander(f"{m}", expanded=False):
            r1c1, r1c2, r1c3, r1c4 = st.columns([1.5, 1, 1, 1])
            conf["transform"] = r1c1.selectbox(
                "Transform",
                options=["none", "log", "negexp", "negexp_cann"],
                index=["none", "log", "negexp", "negexp_cann"].index(conf.get("transform","none")),
                key=f"{dataset_name}__{m}__transform"
            )
            if conf["transform"] == "log":
                conf["k"] = r1c2.number_input("k (log)", min_value=1e-12, value=float(conf.get("k",1.0)), step=1e-3, format="%.6f", key=f"{dataset_name}__{m}__klog")
                r1c3.caption("beta n/a")
                r1c4.caption("gamma n/a")
            elif conf["transform"] == "negexp":
                conf["k"] = r1c2.number_input("k (negexp)", min_value=0.0, value=float(conf.get("k",0.01)), step=1e-3, format="%.6f", key=f"{dataset_name}__{m}__kneg")
                conf["beta"] = r1c3.number_input("beta", min_value=0.0, value=float(conf.get("beta",1.0)), step=0.1, format="%.3f", key=f"{dataset_name}__{m}__beta")
                r1c4.caption("gamma n/a")
            elif conf["transform"] == "negexp_cann":
                conf["k"] = r1c2.number_input("k (negexp)", min_value=0.0, value=float(conf.get("k",0.01)), step=1e-3, format="%.6f", key=f"{dataset_name}__{m}__knegc")
                conf["beta"] = r1c3.number_input("beta", min_value=0.0, value=float(conf.get("beta",1.0)), step=0.1, format="%.3f", key=f"{dataset_name}__{m}__betac")
                conf["gamma"] = r1c4.number_input("gamma (0..1)", min_value=0.0, max_value=1.0, value=float(conf.get("gamma",0.0)), step=0.05, format="%.2f", key=f"{dataset_name}__{m}__gammac")
                pool_opts = [c for c in metrics_all if c != m]
                conf["cannibal_pool"] = st.multiselect(
                    "Cannibalization pool (optional)", options=pool_opts, default=conf.get("cannibal_pool", []),
                    key=f"{dataset_name}__{m}__pool"
                )
            else:
                r1c2.caption("no params"); r1c3.caption("no params"); r1c4.caption("no params")

            r2c1, r2c2, r2c3, r2c4 = st.columns([1.3, 1, 1, 1])
            conf["order"] = r2c1.selectbox(
                "Order",
                options=["transform_then_adstock", "adstock_then_transform"],
                index=["transform_then_adstock", "adstock_then_transform"].index(conf.get("order","transform_then_adstock")),
                key=f"{dataset_name}__{m}__order"
            )
            conf["lag"] = r2c2.number_input("Lag (periods)", min_value=0, max_value=52, step=1, value=int(conf.get("lag",0)), key=f"{dataset_name}__{m}__lag")
            conf["adstock"] = r2c3.number_input("Adstock (0..1)", min_value=0.0, max_value=1.0, step=0.05, value=float(conf.get("adstock",0.0)), key=f"{dataset_name}__{m}__ad")
            conf["scaling"] = r2c4.selectbox(
                "Scaling",
                options=["none", "minmax", "zscore"],
                index=["none", "minmax", "zscore"].index(conf.get("scaling","none")),
                key=f"{dataset_name}__{m}__scaling"
            )

            r3c1, r3c2 = st.columns([1, 2])
            if r3c1.button("Apply suggested", key=f"{dataset_name}__{m}__suggest"):
                st.session_state[cfg_key][m] = {
                    **_metric_defaults(df[m]),
                    "cannibal_pool": conf.get("cannibal_pool", [])
                }
            if r3c2.button("Preview", key=f"{dataset_name}__{m}__preview"):
                params = st.session_state[cfg_key][m]
                y = T.apply_pipeline(df, m, params, params.get("cannibal_pool", []))
                prev = pd.DataFrame({m: df[m], m+"__tfm_preview": y})
                st.line_chart(prev[[m, m+"__tfm_preview"]])
                st.dataframe(prev.head(50), use_container_width=True)

    submitted = st.form_submit_button("Update configuration (keeps values)")
    if submitted:
        st.success("Configuration updated.")

# ---------------- Validate & Preview All ----------------

st.subheader("Validate and Preview")

def _build_config_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in sel_metrics:
        out[m] = dict(st.session_state[cfg_key][m])
    return out

config_map = _build_config_map()

cV, cP = st.columns(2)
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
            st.error("Validation found issues:"); [st.write("- " + msg) for msg in issues]
        else:
            st.success("Validation passed.")

with cP:
    if st.button("Preview transformed table"):
        try:
            cannib_pools = {m: config_map[m].get("cannibal_pool", []) for m in sel_metrics}
            out = T.apply_many(df, config_map, cannibal_pools=cannib_pools, suffix="__tfm")
            cols = []
            for m in sel_metrics:
                cols.append(m); cols.append(m + "__tfm")
            st.dataframe(out[cols].head(100), use_container_width=True)
        except Exception as e:
            st.error("Preview failed: {}".format(e))

# ---------------- Apply & Save ----------------

st.subheader("Apply & Save")
base = os.path.splitext(os.path.basename(dataset_name))[0]
default_out = f"{base}__tfm.csv"
out_name = st.text_input("Output filename (data/)", value=default_out)

if st.button("Apply and save"):
    try:
        cannib_pools = {m: config_map[m].get("cannibal_pool", []) for m in sel_metrics}
        out = T.apply_many(df, config_map, cannibal_pools=cannib_pools, suffix="__tfm")
        dest = os.path.join(DATA_DIR, out_name)
        out.to_csv(dest, index=False)
        st.session_state["last_saved_path"] = dest
        st.session_state["last_save_error"] = ""
        st.success("Saved: {}".format(dest))
    except Exception as e:
        st.session_state["last_saved_path"] = ""
        st.session_state["last_save_error"] = "Save failed: {}".format(e)
        st.error(st.session_state["last_save_error"])
