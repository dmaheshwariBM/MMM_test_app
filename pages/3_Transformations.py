# pages/3_Transformations.py
# v3.4.0  Stable (form-based) Transformations editor:
# - CSV/XLSX picker (with sheet + Refresh)
# - Roles & time window clamped to available dates
# - Per-metric config inside a form (NO reruns until you click "Submit changes")
# - Conditional params per transform type + layman tooltips
# - Validate, Preview metric, Preview table, Apply & Save to data/<base>__tfm.csv

import os
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from core import transforms as T

PAGE_ID = "TRANSFORMATIONS_PAGE_v3_4_0"
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
    time_col = st.selectbox(
        "Time column (optional)",
        options=["(none)"] + date_like,
        index=0,
        help="If present, we use it to filter your data to a valid time window only."
    )
    if time_col != "(none)":
        df["_tfm_time"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["_tfm_time"] = pd.NaT
with c2:
    id_col = st.selectbox("ID column (optional)", options=["(none)"] + list(df.columns), index=0,
                          help="A unique identifier per row (e.g., HCP ID). Not transformed.")
with c3:
    seg_col = st.selectbox("Segment column (optional)", options=["(none)"] + list(df.columns), index=0,
                           help="Optional group label column; excluded from transformations.")
with c4:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    tgt = st.selectbox("Target column (for reference)", options=["(none)"] + numeric_cols, index=0,
                       help="The dependent variable you will model later; excluded from transformations.")

# Clamp time window to available data
if time_col != "(none)":
    valid_times = df["_tfm_time"].dropna()
    if len(valid_times) == 0:
        st.warning("Selected time column has no parseable dates; window controls disabled.")
    else:
        dt_min = valid_times.min().date()
        dt_max = valid_times.max().date()
        start_dt, end_dt = st.date_input("Time window", value=(dt_min, dt_max), min_value=dt_min, max_value=dt_max,
                                         help="Limits the rows we transform to your chosen period.")
        if isinstance(start_dt, (list, tuple)):  # guard older Streamlit behavior
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

st.subheader("Configure transformations (edits apply only on submit)")
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
        "cannibal_pool": [],  # list of columns
    }

# Ensure defaults for selected
for m in sel_metrics:
    if m not in st.session_state[cfg_key]:
        st.session_state[cfg_key][m] = _defaults_for(df[m])

# ---------------- Form: edit many metrics without reruns ----------------
with st.form("tfm_edit_form", clear_on_submit=False):
    st.caption("Tip: Nothing updates until you press **Submit changes** below.")
    edited: Dict[str, Dict[str, Any]] = {}

    for m in sel_metrics:
        conf = st.session_state[cfg_key][m].copy()
        with st.expander(f"{m}", expanded=False):
            # Transform choice
            conf["transform"] = st.selectbox(
                f"{m} — Transform",
                options=["none", "log", "negexp", "negexp_cann"],
                index=["none", "log", "negexp", "negexp_cann"].index(conf.get("transform","none")),
                key=f"{dataset_name}__{m}__transform",
                help="Pick how to reshape this channel's raw values."
            )

            # Conditional parameter row 1 (k, beta, gamma)
            c1, c2, c3 = st.columns(3)
            if conf["transform"] == "log":
                conf["k"] = c1.number_input(
                    "k (log)",
                    min_value=1e-12,
                    value=float(conf.get("k", 1.0)),
                    step=1e-3, format="%.6f",
                    key=f"{dataset_name}__{m}__klog",
                    help="Curvature for log(1 + k·x). Larger k compresses more (values saturate sooner)."
                )
                c2.caption("beta not used")
                c3.caption("gamma not used")
            elif conf["transform"] == "negexp":
                conf["k"] = c1.number_input(
                    "k (negexp)",
                    min_value=0.0,
                    value=float(conf.get("k", 0.01)),
                    step=1e-3, format="%.6f",
                    key=f"{dataset_name}__{m}__kneg",
                    help="Speed to saturation: higher k reaches the ceiling with less spend."
                )
                conf["beta"] = c2.number_input(
                    "beta",
                    min_value=0.0,
                    value=float(conf.get("beta", 1.0)),
                    step=0.1, format="%.3f",
                    key=f"{dataset_name}__{m}__beta",
                    help="Ceiling (max effect) for the channel's response."
                )
                c3.caption("gamma not used")
            elif conf["transform"] == "negexp_cann":
                conf["k"] = c1.number_input(
                    "k (negexp)",
                    min_value=0.0,
                    value=float(conf.get("k", 0.01)),
                    step=1e-3, format="%.6f",
                    key=f"{dataset_name}__{m}__knegc",
                    help="Speed to saturation: higher k reaches the ceiling with less spend."
                )
                conf["beta"] = c2.number_input(
                    "beta",
                    min_value=0.0,
                    value=float(conf.get("beta", 1.0)),
                    step=0.1, format="%.3f",
                    key=f"{dataset_name}__{m}__betac",
                    help="Ceiling (max effect) for the channel's response."
                )
                conf["gamma"] = c3.number_input(
                    "gamma (0–1)",
                    min_value=0.0, max_value=1.0,
                    value=float(conf.get("gamma", 0.0)),
                    step=0.05, format="%.2f",
                    key=f"{dataset_name}__{m}__gammac",
                    help="Cannibalization strength from selected competing channels."
                )
                # Pool chooser
                pool_opts = [c for c in metrics_all if c != m]
                conf["cannibal_pool"] = st.multiselect(
                    "Cannibalization pool (optional)",
                    options=pool_opts,
                    default=conf.get("cannibal_pool", []),
                    key=f"{dataset_name}__{m}__pool",
                    help="Other channels that can steal impact from this one."
                )
            else:
                c1.caption("no params"); c2.caption("no params"); c3.caption("no params")

            # Row 2: order / lag / adstock / scaling
            d1, d2, d3, d4 = st.columns(4)
            conf["order"] = d1.selectbox(
                "Order",
                options=["transform_then_adstock", "adstock_then_transform"],
                index=["transform_then_adstock", "adstock_then_transform"].index(conf.get("order","transform_then_adstock")),
                key=f"{dataset_name}__{m}__order",
                help="Choose whether to saturate then carry over to future periods, or vice versa."
            )
            conf["lag"] = d2.number_input(
                "Lag (periods)", min_value=0, max_value=52, step=1,
                value=int(conf.get("lag", 0)),
                key=f"{dataset_name}__{m}__lag",
                help="How many past periods contribute. With lag=2 and decay r, effect = x_t + r·x_{t-1} + r²·x_{t-2}."
            )
            conf["adstock"] = d3.number_input(
                "Adstock (0–1)", min_value=0.0, max_value=1.0, step=0.05,
                value=float(conf.get("adstock", 0.0)),
                key=f"{dataset_name}__{m}__ad",
                help="Memory/decay: 0=only current period, 1=full carryover. Typical range 0.3–0.8."
            )
            conf["scaling"] = d4.selectbox(
                "Scaling",
                options=["none", "minmax", "zscore"],
                index=["none", "minmax", "zscore"].index(conf.get("scaling","none")),
                key=f"{dataset_name}__{m}__scaling",
                help="Rescale the transformed series (often 'none' is fine for OLS)."
            )

        edited[m] = conf  # collect edits for this metric

    # Apply suggested (for selected metrics only) without leaving the form
    c_form_left, c_form_right = st.columns([1,1])
    with c_form_left:
        suggest_clicked = st.form_submit_button(
            "Apply suggested (selected)",
            help="Fill sensible defaults based on each metric's distribution.",
            type="secondary",
            use_container_width=True
        )
    with c_form_right:
        submit_clicked = st.form_submit_button(
            "Submit changes",
            help="Apply all the edits above to your configuration.",
            type="primary",
            use_container_width=True
        )

# Apply suggestions or persist edits after the form returns
if "edited" not in st.session_state:
    st.session_state["edited"] = {}

if 'suggest_clicked' in locals() and suggest_clicked:
    for m in sel_metrics:
        keep_pool = edited[m].get("cannibal_pool", st.session_state[cfg_key][m].get("cannibal_pool", []))
        st.session_state[cfg_key][m] = {**_defaults_for(df[m]), "cannibal_pool": keep_pool}
    st.success("Suggestions applied. Click Submit changes if you made further manual tweaks.")

if 'submit_clicked' in locals() and submit_clicked:
    for m in sel_metrics:
        st.session_state[cfg_key][m] = edited[m]
    st.success("Configuration saved.")

st.divider()

# Build config_map + cannibal pools from session
def _build_config_map() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in sel_metrics:
        out[m] = dict(st.session_state[cfg_key][m])
    return out

if not sel_metrics:
    st.stop()

config_map = _build_config_map()
cannib_pools = {m: config_map[m].get("cannibal_pool", []) for m in sel_metrics}

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
    metric_to_preview = st.selectbox("Preview metric", options=sel_metrics)
    if st.button("Preview selected metric"):
        try:
            params = config_map[metric_to_preview]
            y = T.apply_pipeline(df, metric_to_preview, params, cannib_pools.get(metric_to_preview, []))
            prev = pd.DataFrame({metric_to_preview: df[metric_to_preview], metric_to_preview+"__tfm": y})
            st.line_chart(prev[[metric_to_preview, metric_to_preview+"__tfm"]])
            st.dataframe(prev.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Preview failed: {e}")

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
out_name = st.text_input("Output filename (data/)", value=default_out,
                         help="Your transformed table will be written to the data/ folder.")

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
