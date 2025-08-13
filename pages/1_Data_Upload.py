import streamlit as st
import pandas as pd
import os
from typing import Optional
from core import io

st.title("📁 Data Upload")
st.caption("Upload multiple CSV/XLSX files. No previews are shown unless you select a file below.")

os.makedirs("data", exist_ok=True)

# --- Upload UI at top
uploaded_files = st.file_uploader(
    "Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True
)
if uploaded_files:
    for uf in uploaded_files:
        save_path = os.path.join("data", uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} file(s) to `data/`")

# --- List files (no previews)
def _human_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024.0:
            return f"{bytes_:,.0f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:,.0f} TB"

files = [f for f in os.listdir("data") if f.lower().endswith((".csv", ".xlsx"))]
if not files:
    st.info("No files in `data/` yet.")
else:
    rows = []
    for f in sorted(files):
        p = os.path.join("data", f)
        try:
            stat = os.stat(p)
            rows.append({
                "File": f,
                "Type": "CSV" if f.lower().endswith(".csv") else "Excel",
                "Size": _human_size(stat.st_size)
            })
        except Exception:
            continue
    st.subheader("Files")
    st.dataframe(rows, use_container_width=True)

# --- Show details only after user selects a file
st.subheader("Inspect a File (optional)")
sel = st.selectbox(
    "Select a file to inspect",
    [None] + sorted(files),
    index=0,
    format_func=lambda x: "— None —" if x is None else x
)

if sel is not None:
    path = os.path.join("data", sel)
    try:
        df = io.load_csv_or_excel(path)
    except Exception as e:
        st.error(f"Failed to load `{sel}`: {e}")
        st.stop()

    dqr = io.data_quality_report(df)
    st.markdown(f"**Data Quality Report • {sel}**")
    if dqr.empty:
        st.success("No issues detected ✅")
    else:
        st.dataframe(dqr, use_container_width=True)

    st.markdown("**Schema**")
    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    st.dataframe(schema, use_container_width=True)

    st.markdown("**Preview (first 20 rows)**")
    st.dataframe(df.head(20), use_container_width=True)

# --- Optional helper: Build master
st.divider()
st.caption(
    "Optional helper: Build `data/master.csv` by merging a spend file and a sales file on a date column (e.g., Week)."
)
left, right = st.columns(2)
with left:
    spend_file = st.selectbox("Spend file", [None] + sorted(files), index=0, key="spend_file")
    spend_date_col = st.text_input("Spend date column", value="Week")
    spend_channel_col = st.text_input("Spend channel column", value="Channel")
    spend_value_col = st.text_input("Spend value column", value="Spend")
with right:
    sales_file = st.selectbox("Sales file", [None] + sorted(files), index=0, key="sales_file")
    sales_date_col = st.text_input("Sales date column", value="Week")
    sales_value_col = st.text_input("Sales target column", value="Sales")

if st.button("Build `data/master.csv`"):
    if not (spend_file and sales_file):
        st.warning("Please select both spend and sales files.")
    else:
        try:
            spend_df = io.load_csv_or_excel(os.path.join("data", spend_file))
            sales_df = io.load_csv_or_excel(os.path.join("data", sales_file))
            spend_df = spend_df.rename(columns={spend_date_col: "Week", spend_channel_col: "Channel", spend_value_col: "Spend"})[["Week","Channel","Spend"]]
            sales_df = sales_df.rename(columns={sales_date_col: "Week", sales_value_col: "Sales"})[["Week","Sales"]]
            master = io.merge_spend_sales(spend_df, sales_df, date_key="Week")
            master.to_csv("data/master.csv", index=False)
            st.success("Created `data/master.csv` ✅")
            st.dataframe(master.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to build master: {e}")
