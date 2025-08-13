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

files = [f for f in os.listdir("data") if f.lower().endswith((".csv",".xlsx"))]
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
sel = st.selectbox("Select a file to inspect", [No]()
