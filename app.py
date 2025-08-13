import streamlit as st
import os
import time
import json
from datetime import datetime

st.set_page_config(page_title="Marketing Mix Modeling Tool", layout="wide", page_icon="📊")

# --- Header (logo + title)
col1, col2 = st.columns([1, 6])
with col1:
    st.image("assets/logo.png", use_column_width=True)
with col2:
    st.title("Marketing Mix Modeling Tool")
    st.caption("Blue Matter • End-to-end MMM workflow")

# --- Helpers
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def _human_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024.0:
            return f"{bytes_:,.0f} {unit}"
        bytes_ /= 1024.0
    return f"{bytes_:,.0f} TB"

def _file_rows_cols(path: str):
    # We deliberately avoid reading files here to keep homepage snappy.
    # Detailed previews belong on the Data Upload page.
    return None, None

def _file_info_table():
    files = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.lower().endswith((".csv", ".xlsx")):
            p = os.path.join(DATA_DIR, f)
            try:
                stat = os.stat(p)
                files.append({
                    "File": f,
                    "Type": "CSV" if f.lower().endswith(".csv") else "Excel",
                    "Size": _human_size(stat.st_size),
                    "Last Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
            except Exception:
                continue
    return files

def _load_last_run():
    # Priority: session state, then JSON on disk, else None
    last = st.session_state.get("mmm_last_run")
    if last:
        return last
    meta_path = os.path.join(DATA_DIR, "last_run.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# --- Overview cards
files = _file_info_table()
last_run = _load_last_run()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Uploaded Files", len(files))
with c2:
    ds = (last_run or {}).get("dataset") or "None"
    st.metric("Last Model • Dataset", ds)
with c3:
    mt = (last_run or {}).get("model_type") or "None"
    st.metric("Last Model • Type", mt)

# --- Last run details (compact)
st.subheader("Last Model Run")
if not last_run:
    st.info("None")
else:
    left, right = st.columns([2, 1])
    with left:
        st.write(
            f"**Model:** {last_run.get('model_type','—')} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Dataset:** {last_run.get('dataset','—')} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**Target:** {last_run.get('target','—')}"
        )
        feats = last_run.get("features") or []
        st.write("**Features:** " + (", ".join(feats) if feats else "—"))
    with right:
        ts = last_run.get("timestamp")
        st.write(f"**Timestamp:** {ts or '—'}")
        metrics = last_run.get("metrics") or {}
        if metrics:
            st.caption("Key Metrics")
            st.json(metrics)

# --- Uploaded files (no previews here)
st.subheader("Uploaded Files")
if not files:
    st.info("No files found. Use the **Data Upload** page (sidebar) to upload CSV/XLSX.")
else:
    # Lightweight list only; no schema/preview on homepage
    st.dataframe(files, use_container_width=True)

st.divider()
st.caption("Use the left sidebar to navigate. This home screen intentionally avoids data previews and step buttons.")
