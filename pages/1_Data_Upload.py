import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any
from core import io

st.title("📁 Data Upload")
st.caption("Upload CSV/XLSX files, review & set column types, then finalize. No previews until you select a file.")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def human_size(bytes_: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_ < 1024.0:
            return f"{bytes_:,.0f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:,.0f} TB"

# ---- Upload
uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv","xlsx"], accept_multiple_files=True)
if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join(DATA_DIR, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s) to `data/`")

# ---- File list (no auto-preview)
st.subheader("Files")
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv",".xlsx"))]
if not files:
    st.info("No files in `data/` yet.")
else:
    rows = []
    for f in sorted(files):
        p = os.path.join(DATA_DIR, f)
        try:
            stat = os.stat(p)
            rows.append({"File": f, "Type": "CSV" if f.lower().endswith(".csv") else "Excel",
                         "Size": human_size(stat.st_size)})
        except Exception:
            pass
    st.dataframe(rows, use_container_width=True)

# ---- Inspect & set types
st.subheader("Inspect & Set Column Types")

sel = st.selectbox("Select a file", [None] + sorted(files), index=0, format_func=lambda x: "— Select —" if x is None else x)
if sel is not None:
    path = os.path.join(DATA_DIR, sel)

    # read file
    if sel.lower().endswith(".xlsx"):
        try:
            xl = pd.ExcelFile(path)
            sheet = st.selectbox("Excel sheet", xl.sheet_names, index=0)
            df_raw = xl.parse(sheet)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")
            st.stop()
    else:
        try:
            df_raw = pd.read_csv(path, low_memory=False)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

    # infer types as suggestions
    st.write("**Set column types (prefilled by inference).**")
    type_options = ["integer","float","string","boolean","datetime","category"]
    if "schema_map" not in st.session_state:
        st.session_state["schema_map"] = {}
    if sel not in st.session_state["schema_map"]:
        st.session_state["schema_map"][sel] = {c: io.infer_dtype(df_raw[c]) for c in df_raw.columns}
    schema_map = st.session_state["schema_map"][sel]

    if "date_fmt" not in st.session_state:
        st.session_state["date_fmt"] = {}
    if sel not in st.session_state["date_fmt"]:
        st.session_state["date_fmt"][sel] = {}
    date_fmt = st.session_state["date_fmt"][sel]

    # UI for types + date formats
    cols_per_row = 2
    for i, c in enumerate(df_raw.columns):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        idx = i % cols_per_row
        with row[idx]:
            t = st.selectbox(f"{c}", type_options, index=type_options.index(schema_map.get(c, "string")), key=f"type_{sel}_{c}")
            schema_map[c] = t
            if t == "datetime":
                choice = st.selectbox(f"{c} • Date format", ["Auto","DD/MM/YYYY","MM/DD/YYYY","YYYY-MM-DD","Custom"], key=f"fmt_{sel}_{c}")
                custom = ""
                if choice == "Custom":
                    custom = st.text_input(f"{c} • Custom strftime (e.g., %d-%b-%Y)", key=f"custfmt_{sel}_{c}")
                date_fmt[c] = (choice, custom)

    # Raw quick stats (no coercion yet)
    with st.expander("Raw schema & stats"):
        sample_n = st.slider("Rows to preview", 50, 1000, 200, 50)
        st.dataframe(df_raw.head(sample_n), use_container_width=True)
        rows = []
        for c in df_raw.columns:
            s = df_raw[c]
            row = {"column": c, "dtype": str(s.dtype),
                   "non_null": int(s.notna().sum()),
                   "nulls": int(s.isna().sum()),
                   "unique": int(s.nunique(dropna=True))}
            if pd.api.types.is_numeric_dtype(s):
                s_num = pd.to_numeric(s, errors="coerce")
                row["mean"] = float(np.nanmean(s_num))
                row["median"] = float(np.nanmedian(s_num))
                try:
                    md = pd.to_numeric(pd.Series(s).mode(), errors="coerce").dropna()
                    row["mode"] = float(md.iloc[0]) if not md.empty else np.nan
                except Exception:
                    row["mode"] = np.nan
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---- Finalize upload: coerce + report + save
    st.markdown("### Finalize Upload")
    overwrite = st.checkbox("Overwrite original file", value=False)
    default_out = f"{os.path.splitext(sel)[0]}__typed.csv"
    out_name = st.text_input("Output filename (CSV)", value=default_out)

    if st.button("Apply Types & Save"):
        try:
            df_typed, report = io.coerce_with_report(df_raw, schema_map, date_fmt)
        except Exception as e:
            st.error(f"Type coercion failed: {e}")
            st.stop()

        # Show coercion report & block if issues unless user confirms
        st.markdown("**Type Coercion Report**")
        st.dataframe(report, use_container_width=True)
        problems = report[(report["coerced_nulls"] > 0) | (report["non_finite"] > 0)]
        if not problems.empty:
            st.error("Some columns produced NaNs/non-finite values during coercion. Review the report above.")
            proceed = st.checkbox("I understand the risks. Proceed anyway.")
            if not proceed:
                st.stop()

        # Save
        try:
            if overwrite and sel.lower().endswith(".csv"):
                df_typed.to_csv(path, index=False)
                saved_as = sel
            else:
                out_csv = out_name if out_name.strip().lower().endswith(".csv") else out_name + ".csv"
                df_typed.to_csv(os.path.join(DATA_DIR, out_csv), index=False)
                saved_as = out_csv
            st.success(f"Saved as `{saved_as}` in `data/`")
            st.dataframe(df_typed.head(min(sample_n, 200)), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to save: {e}")
