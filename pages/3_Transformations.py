# pages/2_Transformations.py
import os, json
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from core import transforms

st.title("🧪 Transformations")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------
# Utilities
# ---------------------------------
def _list_csvs() -> List[str]:
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _infer_datetime(s: pd.Series) -> Optional[pd.Series]:
    try:
        dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if dt.notna().sum() >= max(5, int(0.5 * len(s))):
            return dt
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt.notna().sum() >= max(5, int(0.5 * len(s))):
            return dt
    except Exception:
        pass
    return None

def _suggest_k_for_series(s: pd.Series, tfm: str) -> float:
    return transforms.suggest_k(s, tfm)

def _new_name_for_transformed(stem: str) -> str:
    return f"{stem}__tfm.csv"

def _cfg_key(dataset: str) -> str:
    return f"tfm_cfg::{dataset}"

def _default_order() -> str:
    return "Transform→Adstock+Lag"

# ---------------------------------
# Dataset pick
# ---------------------------------
files = _list_csvs()
if not files:
    st.info("No CSVs in `data/` yet. Upload files in **Data Upload**.")
    st.stop()

ds_index = 0
dataset = st.selectbox("Dataset", files, index=ds_index, key="tfm_dataset_select")
df_raw = _read_csv(os.path.join(DATA_DIR, dataset))

st.caption(f"Rows: {len(df_raw):,} • Columns: {len(df_raw.columns)}")

# ---------------------------------
# Identify columns
# ---------------------------------
all_cols = list(df_raw.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_raw[c])]
non_num_cols = [c for c in all_cols if c not in num_cols]

with st.expander("📎 Key columns", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        time_col = st.selectbox("Time column (optional)", options=["— (none)"] + all_cols, index=0, key="tfm_time_col")
    with c2:
        id_col = st.selectbox("ID column (optional)", options=["— (none)"] + all_cols, index=0, key="tfm_id_col")
    with c3:
        target_col = st.selectbox("Target (Y)", options=num_cols if num_cols else ["—"], key="tfm_target_col")
    with c4:
        segment_col = st.selectbox("Segment column (optional)", options=["— (none)"] + all_cols, index=0, key="tfm_segment_col")

    # time filter (restricted to available range)
    dt_series = None
    if time_col and time_col != "— (none)":
        dt_series = _infer_datetime(df_raw[time_col])
        if dt_series is None:
            st.warning(f"Could not parse {time_col} as dates; no time filtering will be applied.")
        else:
            df_raw = df_raw.copy()
            df_raw["_tfm_dt"] = dt_series
            valid = df_raw["_tfm_dt"].dropna()
            if not valid.empty:
                min_d, max_d = valid.min(), valid.max()
                start, end = st.slider(
                    "Time window",
                    min_value=min_d.to_pydatetime(),
                    max_value=max_d.to_pydatetime(),
                    value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
                    format="YYYY-MM-DD",
                    key="tfm_time_window",
                )
                df_raw = df_raw[(df_raw["_tfm_dt"] >= start) & (df_raw["_tfm_dt"] <= end)].copy()
            else:
                st.warning("No valid dates found for time filtering.")

# ---------------------------------
# Candidate metrics (exclude key columns)
# ---------------------------------
exclude = set([c for c in (time_col, id_col, target_col, segment_col) if c and c != "— (none)"])
metric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and c not in exclude and not c.startswith("_tfm_")]

st.write(f"**Transformable metrics (numeric):** {len(metric_cols)}")

# ---------------------------------
# Build or load config table in session
# ---------------------------------
cfg_key = _cfg_key(dataset)
if cfg_key not in st.session_state:
    rows = []
    for m in metric_cols:
        s = pd.to_numeric(df_raw[m], errors="coerce").dropna()
        skew = float(s.skew()) if len(s) > 2 else 0.0
        tfm_default = "Log" if skew > 1.0 else "NegExp"
        k_sug = _suggest_k_for_series(s, tfm_default)
        rows.append({
            "use": True,
            "metric": m,
            "transform": tfm_default,
            "k": round(k_sug, 6),
            "suggested_k": round(k_sug, 6),
            "lag_months": 0,
            "adstock_alpha": 0.0,
            "order": _default_order(),
            # new scaling fields
            "scaling": "None",
            "scale_min": 0.0,
            "scale_max": 1.0,
        })
    st.session_state[cfg_key] = pd.DataFrame(rows)
else:
    cfg_df = st.session_state[cfg_key].copy()
    known = set(cfg_df["metric"])
    new_rows = []
    for m in metric_cols:
        if m not in known:
            s = pd.to_numeric(df_raw[m], errors="coerce").dropna()
            tfm_default = "NegExp"
            k_sug = _suggest_k_for_series(s, tfm_default)
            new_rows.append({
                "use": True,
                "metric": m,
                "transform": tfm_default,
                "k": round(k_sug, 6),
                "suggested_k": round(k_sug, 6),
                "lag_months": 0,
                "adstock_alpha": 0.0,
                "order": _default_order(),
                "scaling": "None",
                "scale_min": 0.0,
                "scale_max": 1.0,
            })
    if new_rows:
        st.session_state[cfg_key] = pd.concat([cfg_df, pd.DataFrame(new_rows)], ignore_index=True)

# keep only metrics present
cfg_df = st.session_state[cfg_key]
cfg_df = cfg_df[cfg_df["metric"].isin(metric_cols)].reset_index(drop=True)
st.session_state[cfg_key] = cfg_df

# ---------------------------------
# Editor
# ---------------------------------
st.markdown("#### Configure transformations & scaling")
st.caption(
    "Curvature **k** applies to Log/NegExp (Log: log(1+k·x), NegExp: 1−exp(−k·x)). "
    "Scaling is applied at the end. For **MinMax**, set `scale_min` & `scale_max`."
)

edited_cfg = st.data_editor(
    cfg_df,
    key=f"cfg_editor::{dataset}",
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "use": st.column_config.CheckboxColumn("Use"),
        "metric": st.column_config.TextColumn("Metric", disabled=True),
        "transform": st.column_config.SelectboxColumn("Transform", options=["None","Log","NegExp"],
                                                      help="Select a transform type."),
        "k": st.column_config.NumberColumn("k (curvature)", help="Shape parameter. Suggested based on mean."),
        "suggested_k": st.column_config.NumberColumn("k (suggested)", disabled=True),
        "lag_months": st.column_config.NumberColumn("Lag (months)", min_value=0, step=1, help="K in finite adstock."),
        "adstock_alpha": st.column_config.NumberColumn("Adstock α (0–1)", min_value=0.0, max_value=1.0, step=0.05,
                                                       help="Decay; effective_t = x_t + αx_{t-1} + ... + α^K x_{t-K}"),
        "order": st.column_config.SelectboxColumn("Order", options=["Transform→Adstock+Lag","Adstock+Lag→Transform"],
                                                  help="Choose whether to transform before or after adstock/lag."),
        # NEW: scaling
        "scaling": st.column_config.SelectboxColumn(
            "Scaling",
            options=[
                "None",
                "MinMax",
                "Standardize (z-score)",
                "Robust (median/IQR)",
                "Mean norm (÷ mean)",
                "Max norm (÷ max)",
                "Unit length (L2)",
            ],
            help="Applied after transform/adstock+lag."
        ),
        "scale_min": st.column_config.NumberColumn("scale_min (for MinMax)", help="Lower bound for MinMax scaling."),
        "scale_max": st.column_config.NumberColumn("scale_max (for MinMax)", help="Upper bound for MinMax scaling."),
    }
)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("🔄 Apply suggested k", use_container_width=True):
        tmp = edited_cfg.copy()
        for i, row in tmp.iterrows():
            tfm = str(row.get("transform", "None"))
            if tfm in ("Log","NegExp"):
                s = pd.to_numeric(df_raw[row["metric"]], errors="coerce")
                ksug = _suggest_k_for_series(s, tfm)
                tmp.at[i, "k"] = round(ksug, 6)
                tmp.at[i, "suggested_k"] = round(ksug, 6)
        st.session_state[cfg_key] = tmp
        st.experimental_rerun()
with c2:
    if st.button("Reset all", help="Reset to fresh suggestions.", use_container_width=True):
        del st.session_state[cfg_key]
        st.experimental_rerun()
with c3:
    st.caption("Tip: For **MinMax**, default [0,1] is common; for modeling, Standardize/Robust can help.")

# Persist edits
st.session_state[cfg_key] = edited_cfg

# ---------------------------------
# Preview
# ---------------------------------
st.markdown("#### Preview")
prev_cols = [r["metric"] for _, r in edited_cfg.iterrows() if r.get("use", True)]
if not prev_cols:
    st.info("Select at least one metric (Use = ✓) to preview.")
else:
    preview_metric = st.selectbox("Pick a metric to preview", options=prev_cols)
    row = edited_cfg[edited_cfg["metric"] == preview_metric].iloc[0]
    tfm = str(row["transform"])
    k = float(row["k"])
    lag = int(row["lag_months"])
    alpha = float(row["adstock_alpha"])
    order = str(row["order"])
    scaling = str(row.get("scaling", "None"))
    scale_min = float(row.get("scale_min", 0.0))
    scale_max = float(row.get("scale_max", 1.0))

    raw = pd.to_numeric(df_raw[preview_metric], errors="coerce").fillna(0.0)
    eff = transforms.apply_with_order(
        raw, tfm, k, lag, alpha, order,
        scaling=scaling, scale_min=scale_min, scale_max=scale_max
    )

    stats = pd.DataFrame({
        "Series": ["Original", "Transformed+Scaled"],
        "Mean": [raw.mean(), eff.mean()],
        "Std": [raw.std(), eff.std()],
        "Min": [raw.min(), eff.min()],
        "Max": [raw.max(), eff.max()],
        "Non-Null": [raw.notna().sum(), eff.notna().sum()]
    })
    st.dataframe(stats, use_container_width=True)

    if "_tfm_dt" in df_raw.columns:
        plot_df = pd.DataFrame({"t": df_raw["_tfm_dt"], "Original": raw, "Transformed+Scaled": eff}).set_index("t")
    else:
        plot_df = pd.DataFrame({"idx": np.arange(len(raw)), "Original": raw, "Transformed+Scaled": eff}).set_index("idx")
    st.line_chart(plot_df)

# ---------------------------------
# Apply & Save
# ---------------------------------
st.divider()
st.markdown("#### Save transformed dataset")

cfg_rows = []
for _, r in edited_cfg.iterrows():
    if bool(r.get("use", True)):
        cfg_rows.append({
            "metric": str(r["metric"]),
            "transform": str(r["transform"]),
            "k": float(r["k"]),
            "lag_months": int(r["lag_months"]),
            "adstock_alpha": float(r["adstock_alpha"]),
            "order": str(r["order"]),
            "scaling": str(r.get("scaling", "None")),
            "scale_min": float(r.get("scale_min", 0.0)),
            "scale_max": float(r.get("scale_max", 1.0)),
            "suggested_k": float(r.get("suggested_k", np.nan)),
            "use": True
        })

if st.button("💾 Save transformed CSV & metadata", type="primary"):
    try:
        df_applied, meta = transforms.apply_bulk(df_raw, cfg_rows)

        stem = os.path.splitext(os.path.basename(dataset))[0]
        out_csv = _new_name_for_transformed(stem)
        out_path = os.path.join(DATA_DIR, out_csv)
        df_applied.to_csv(out_path, index=False)

        meta_out = {
            "source_dataset": dataset,
            "time_col": (time_col if time_col != "— (none)" else None),
            "id_col": (id_col if id_col != "— (none)" else None),
            "target_col": target_col,
            "segment_col": (segment_col if segment_col != "— (none)" else None),
            "config": meta["config"],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        meta_path = os.path.join(DATA_DIR, f"transforms_{os.path.splitext(out_csv)[0]}.json")
        with open(meta_path, "w") as f:
            json.dump(meta_out, f, indent=2)

        st.session_state["mmm_current_dataset"] = out_csv
        st.session_state["mmm_target"] = target_col

        st.success(f"Saved **{out_csv}** and **{os.path.basename(meta_path)}** in `data/`.")
        st.caption("You can proceed to **Modeling** to build models on the transformed data.")
    except Exception as e:
        st.error(f"Save failed: {e}")
        st.exception(e)
