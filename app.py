# app.py
# v1.4.0  ASCII-only Home page for MMM app.
# - Lists files in data/
# - Shows latest saved model (by timestamp) and recent runs
# - Uses same results-root policy as Modeling/Results/Advanced pages
# - No buttons duplicating sidebar navigation
# - Small footer "Built by BLUE MATTER"

import os
import re
import glob
import json
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

PAGE_ID = "HOME_PAGE_v1_4_0"
st.title("Home")
st.caption("Page ID: {}".format(PAGE_ID))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Persisted banners shown across reruns if another page just saved something
if "last_saved_path" in st.session_state and st.session_state["last_saved_path"]:
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if "last_save_error" in st.session_state and st.session_state["last_save_error"]:
    st.error(st.session_state["last_save_error"])

# ---------------- Writable results root (reuse the same policy everywhere) ----------------
def _abs(p: str) -> str:
    return os.path.abspath(p)

CANDIDATE_RESULTS_ROOTS: List[str] = []
_env_dir = os.environ.get("MMM_RESULTS_DIR")
if _env_dir:
    CANDIDATE_RESULTS_ROOTS.append(_abs(_env_dir))
CANDIDATE_RESULTS_ROOTS.append(_abs(os.path.expanduser("~/.mmm_results")))
CANDIDATE_RESULTS_ROOTS.append(_abs("/tmp/mmm_results"))
CANDIDATE_RESULTS_ROOTS.append(_abs("results"))

def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False

def pick_writable_results_root() -> str:
    for root in CANDIDATE_RESULTS_ROOTS:
        if _ensure_dir(root):
            return root
    fb = _abs(os.path.expanduser("~/mmm_results_fallback"))
    _ensure_dir(fb)
    return fb

RESULTS_ROOT = pick_writable_results_root()

# ---------------- Utilities ----------------
def _human_size(bytes_: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return "{:,.0f} {}".format(size, u)
        size /= 1024.0

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:72]

def load_results_catalog(results_roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for root in results_roots:
        patt = os.path.join(root, "**", "*.json")
        files = sorted(glob.glob(patt, recursive=True), reverse=True)
        for jf in files:
            if jf in seen:
                continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    r = json.load(f)
                r["_path"] = jf
                ts = r.get("batch_ts")
                if ts:
                    try:
                        r["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                    except Exception:
                        r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                else:
                    r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                rows.append(r)
                seen.add(jf)
            except Exception:
                continue
    rows.sort(key=lambda x: x.get("_ts", datetime.min), reverse=True)
    return rows

def fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name", "(unnamed)")
    tp = r.get("type", "base")
    tgt = r.get("target", "?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
    return "{} | {} | {} | {}".format(nm, tp, tgt, when)

# ---------------- Layout ----------------
left, right = st.columns([2, 3], gap="large")

# Left: Data files inventory (no previews)
with left:
    st.subheader("Data files in data/")
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".xlsx"))]
    if not files:
        st.info("No files found. Use the Data Upload page (sidebar) to add CSV or Excel files.")
    else:
        rows = []
        for fn in sorted(files):
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(380, 40 + 38*max(1, len(rows))))

    st.caption("Tip: Use the sidebar to navigate to each step. This Home page only summarizes data and saved runs.")

# Right: Latest saved model and recent runs
with right:
    st.subheader("Latest saved model")
    catalog = load_results_catalog(CANDIDATE_RESULTS_ROOTS)
    if not catalog:
        st.info("No saved models yet. Go to the Modeling page, run a model, and click Save.")
    else:
        latest = catalog[0]
        st.metric("Last model name", latest.get("name", "(unnamed)"))
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Type", str(latest.get("type", "base")))
        with c2: st.metric("Target", str(latest.get("target", "?")))
        with c3:
            ts = latest.get("_ts")
            when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
            st.metric("Saved at", when)

        # show a compact table of recent runs
        st.markdown("Recent runs")
        recent_rows = []
        for r in catalog[:8]:
            recent_rows.append({
                "Name": r.get("name", ""),
                "Type": r.get("type", "base"),
                "Target": r.get("target", ""),
                "Saved at": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
            })
        st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, height=min(360, 40 + 38*max(1, len(recent_rows))))

st.divider()
st.caption("Built by BLUE MATTER")
