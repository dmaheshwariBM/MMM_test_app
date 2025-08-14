# pages/1_Data_Upload.py
# v1.7.0  Upload, delete, inspect, and correlation (Altair heatmap; no matplotlib needed)

import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

# Try Altair for heatmap (Streamlit ships with it); fall back to table-only if unavailable
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

st.title("Data Upload")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure guard state exists
if "delete_all_guard" not in st.session_state:
    st.session_state["delete_all_guard"] = False

# ---------------- Helpers ----------------
def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(max(0, n))
    for u in units:
        if s < 1024 or u == units[-1]:
            return "{:,.0f} {}".format(s, u)
        s /= 1024.0

def _list_files() -> List[str]:
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".xlsx"))])

def _load_df(path: str, excel_sheet: str = None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        if excel_sheet is None:
            xls = pd.ExcelFile(path)
            sheet = xls.sheet_names[0]
            return pd.read_excel(path, sheet_name=sheet)
        return pd.read_excel(path, sheet_name=excel_sheet)

# Cross-page save banners (from Modeling/Advanced/Results)
if st.session_state.get("last_saved_path"):
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# ---------------- Upload ----------------
st.subheader("Upload files")
uploaded = st.file_uploader(
    "Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True
)
if uploaded:
    saved = 0
    for uf in uploaded:
        try:
            dest = os.path.join(DATA_DIR, uf.name)
            # Overwrite if exists to keep things simple/explicit
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved += 1
        except Exception as e:
            st.error("Failed to save {}: {}".format(uf.name, e))
    if saved:
        st.success("Uploaded {} file(s) to data/".format(saved))
        st.rerun()

# ---------------- Inventory + Delete ----------------
st.subheader("Files in data/")
files = _list_files()
if not files:
    st.info("No files found. Upload CSV or XLSX above.")
else:
    rows: List[Dict[str, str]] = []
    for fn in files:
        p = os.path.join(DATA_DIR, fn)
        try:
            stt = os.stat(p)
            rows.append({
                "File": fn,
                "Type": "CSV" if fn.lower().endswith(".csv") else "Excel",
                "Size": _human_size(stt.st_size),
                "Modified": datetime.fromtimestamp(stt.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception:
            rows.append({"File": fn, "Type": "?", "Size": "?", "Modified": "?"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(360, 40 + 38*max(1, len(rows))))

    del_sel = st.multiselect("Select files to delete", options=files)
    c1, c2, c3 = st.columns([1,1,3])
    with c1:
        if st.button("Delete selected"):
            if not del_sel:
                st.warning("Select one or more files to delete.")
            else:
                deleted, errs = 0, 0
                for fn in del_sel:
                    try:
                        os.remove(os.path.join(DATA_DIR, fn))
                        deleted += 1
                    except Exception as e:
                        errs += 1
                        st.error("Could not delete {}: {}".format(fn, e))
                if deleted:
                    st.success("Deleted {} file(s).".format(deleted))
                st.rerun()
    with c2:
        if st.button("Delete ALL (guarded)"):
            st.session_state["delete_all_guard"] = True

    if st.session_state["delete_all_guard"]:
        st.warning("Type DELETE ALL to confirm full purge of data/ folder.")
        confirm = st.text_input("Type exactly: DELETE ALL")
        cL, cR = st.columns(2)
        with cL:
            if st.button("Confirm full delete"):
                if confirm.strip().upper() == "DELETE ALL":
                    errs = 0
                    for fn in list(_list_files()):
                        try:
                            os.remove(os.path.join(DATA_DIR, fn))
                        except Exception:
                            errs += 1
                    st.session_state["delete_all_guard"] = False
                    if errs == 0:
                        st.success("All files deleted.")
                    else:
                        st.error("Some files could not be deleted.")
                    st.rerun()
                else:
                    st.error("Confirmation text did not match.")
        with cR:
            if st.button("Cancel"):
                st.session_state["delete_all_guard"] = False
                st.rerun()

# ---------------- Inspect and Correlation ----------------
st.subheader("Inspect a file")
files = _list_files()
if not files:
    st.info("Nothing to inspect yet.")
    st.stop()

sel_file = st.selectbox("Choose a file", options=files, index=0)

excel_sheet = None
if sel_file.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(os.path.join(DATA_DIR, sel_file))
        excel_sheet = st.selectbox("Excel sheet", options=xls.sheet_names, index=0)
    except Exception as e:
        st.error("Could not read Excel sheets: {}".format(e))

# Load and preview
df = pd.DataFrame()
try:
    df = _load_df(os.path.join(DATA_DIR, sel_file), excel_sheet=excel_sheet)
except Exception as e:
    st.error("Failed to load {}: {}".format(sel_file, e))

if df.empty:
    st.info("No rows to display.")
    st.stop()

st.markdown("Preview (up to first 500 rows)")
st.dataframe(df.head(500), use_container_width=True, height=360)

# quick info
st.markdown("Summary")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Rows", len(df))
with c2: st.metric("Columns", len(df.columns))
with c3: st.metric("Numeric columns", int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)))

# dtypes and nulls
meta = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "nulls": [int(df[c].isna().sum()) for c in df.columns],
    "null_pct": [round(100.0*df[c].isna().mean(), 2) for c in df.columns],
})
st.dataframe(meta, use_container_width=True, height=min(300, 40 + 22*max(1, len(meta))))

st.divider()
st.subheader("Correlation matrix")
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.info("No numeric columns in this file.")
else:
    default_sel = num_cols[: min(8, len(num_cols))]
    sel_cols = st.multiselect("Select numeric metrics", options=num_cols, default=default_sel)
    method = st.selectbox("Method", options=["pearson", "spearman", "kendall"], index=0)
    if st.button("Compute correlation"):
        if len(sel_cols) < 2:
            st.warning("Select at least two columns.")
        else:
            sub = df[sel_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            corr = sub.corr(method=method)
            st.dataframe(corr, use_container_width=True)

            # Altair heatmap (no matplotlib dependency)
            if ALT_AVAILABLE:
                corr_reset = corr.reset_index().melt(id_vars="index")
                corr_reset.columns = ["X", "Y", "value"]
                chart = (
                    alt.Chart(corr_reset)
                    .mark_rect()
                    .encode(
                        x=alt.X("X:O", title=None),
                        y=alt.Y("Y:O", title=None),
                        color=alt.Color("value:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
                        tooltip=[alt.Tooltip("X:O"), alt.Tooltip("Y:O"), alt.Tooltip("value:Q", format=".2f")],
                    )
                    .properties(width=min(60*len(sel_cols), 900), height=min(60*len(sel_cols), 900), title="Correlation ({})".format(method))
                )
                labels = (
                    alt.Chart(corr_reset)
                    .mark_text(baseline="middle", fontSize=10)
                    .encode(
                        x="X:O",
                        y="Y:O",
                        text=alt.Text("value:Q", format=".2f"),
                        color=alt.condition(alt.datum.value < 0, alt.value("#111111"), alt.value("#111111")),
                    )
                )
                st.altair_chart(chart + labels, use_container_width=True)
            else:
                st.info("Altair is not available; showing table only.")

            # Download CSV
            csv_bytes = corr.to_csv(index=True).encode("utf-8")
            st.download_button(
                "Download correlation CSV",
                data=csv_bytes,
                file_name="{}_correlation.csv".format(os.path.splitext(sel_file)[0]),
                mime="text/csv",
            )
