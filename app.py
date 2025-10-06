# app.py
# v1.7.0  Polished Home for MMM app (minimalist white + blue).
# - Lists files in data/
# - Shows latest saved model and recent runs
# - Subtle cards, accents using brand blue
# - Reuses results discovery logic used across pages
# - No duplicate navigation buttons
#
# Tip: If you know the exact Blue Matter hex, change BM_PRIMARY below.

import os
import re
import glob
import json
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Home", page_icon="📊", layout="wide")

# ---------- Brand palette (single source of truth) ----------
BM_PRIMARY = "#1EA7E0"  # Approx from Blue Matter logo; update if you have the official hex
BM_PRIMARY_SOFT = "#E8F6FC"
BM_DARK = "#0C2A3C"
BM_TEXT = "#1F2937"
BM_MUTED = "#6B7280"
CARD_BG = "#FFFFFF"
APP_BG = "#FFFFFF"

# ---------- CSS (minimalist, card-based) ----------
st.markdown(
    f"""
<style>
/* base */
html, body, [data-testid="stAppViewContainer"] {{
  background: {APP_BG};
  color: {BM_TEXT};
}}
/* hide default table row index */
[data-testid="stDataFrame"] .row_heading.level0, .blank {{ display: none; }}
/* hero */
.hero {{
  border: 1px solid #E5E7EB;
  background: linear-gradient(180deg,{BM_PRIMARY_SOFT} 0%, #FFFFFF 60%);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 18px;
}}
.hero-title {{
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0.2px;
  color: {BM_DARK};
  margin: 0 0 4px 0;
}}
.hero-sub {{
  font-size: 14px;
  color: {BM_MUTED};
  margin: 0;
}}
.badge {{
  display:inline-block;
  background:{BM_PRIMARY};
  color:white;
  padding:2px 10px;
  font-size:12px;
  border-radius:999px;
  margin-left:8px;
}}
/* cards */
.card {{
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 16px 16px;
  background: {CARD_BG};
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}}
.card h3 {{
  margin: 0 0 10px 0;
  font-size: 16px;
  color: {BM_DARK};
}}
.kpi {{
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  padding: 14px 16px;
  background: white;
  text-align: left;
}}
.kpi .label {{
  font-size: 12px;
  color: {BM_MUTED};
  margin: 0;
}}
.kpi .value {{
  font-size: 18px;
  font-weight: 700;
  color: {BM_DARK};
  margin: 2px 0 0 0;
}}
.small {{
  font-size: 12px;
  color: {BM_MUTED};
}}
.footer {{
  color: {BM_MUTED};
  font-size: 12px;
  text-align: right;
  padding-top: 8px;
}}
</style>
""",
    unsafe_allow_html=True,
)

PAGE_ID = "HOME_PAGE_v1_7_0"

# ---------- Shared results-root logic ----------
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

# ---------- Utilities ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def _human_size(bytes_: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, bytes_))
    for u in units:
        if size < 1024.0 or u == "TB":
            return "{:,.0f} {}".format(size, u)
        size /= 1024.0

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

# ---------- Hero ----------
st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Marketing Mix Modeling Studio <span class="badge">Home</span></div>
  <p class="hero-sub">Workspace to upload data, run models, view results, and optimize budgets. Page ID: {PAGE_ID}</p>
</div>
""",
    unsafe_allow_html=True,
)

# Show cross-page save banners if present
if st.session_state.get("last_saved_path"):
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

left, right = st.columns([2, 3], gap="large")

# ---------- Left: Data inventory ----------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Data files</h3>", unsafe_allow_html=True)

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv", ".xlsx"))]
    if not files:
        st.info("No files found. Use the Data Upload page (sidebar) to add CSV or Excel files.")
    else:
        rows = []
        total_sz = 0
        for fn in sorted(files):
            p = os.path.join(DATA_DIR, fn)
            try:
                stt = os.stat(p)
                total_sz += stt.st_size
                rows.append({
                    "File": fn,
                    "Type": "CSV" if fn.lower().endswith(".csv") else "Excel",
                    "Size": _human_size(stt.st_size),
                    "Modified": datetime.fromtimestamp(stt.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception:
                rows.append({"File": fn, "Type": "?", "Size": "?", "Modified": "?"})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="kpi"><p class="label">Files</p><p class="value">{}</p></div>'.format(len(files)), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="kpi"><p class="label">Total size</p><p class="value">{}</p></div>'.format(_human_size(total_sz)), unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=min(360, 40 + 38*max(1, len(rows))))
        st.caption("Tip: File previews and correlations live on the Data Upload page.")

    st.markdown("</div>", unsafe_allow_html=True)

    

# ---------- Right: Latest model + recent runs ----------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Latest saved model</h3>", unsafe_allow_html=True)

    catalog = load_results_catalog([RESULTS_ROOT] + CANDIDATE_RESULTS_ROOTS)
    if not catalog:
        st.info("No saved models yet. Go to the Modeling page, run a model, and click Save.")
    else:
        latest = catalog[0]
        ts = latest.get("_ts")
        when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown('<div class="kpi"><p class="label">Model</p><p class="value">{}</p></div>'.format(latest.get("name","(unnamed)")), unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="kpi"><p class="label">Type</p><p class="value">{}</p></div>'.format(latest.get("type","base")), unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="kpi"><p class="label">Target</p><p class="value">{}</p></div>'.format(latest.get("target","?")), unsafe_allow_html=True)

        st.markdown('<p class="small">Saved at {}</p>'.format(when), unsafe_allow_html=True)

        # Compact recent table
        st.markdown("<h3>Recent runs</h3>", unsafe_allow_html=True)
        recent_rows = []
        for r in catalog[:8]:
            recent_rows.append({
                "Name": r.get("name", ""),
                "Type": r.get("type", "base"),
                "Target": r.get("target", ""),
                "Saved at": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
            })
        st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, height=min(340, 40 + 38*max(1, len(recent_rows))))

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="footer">Built by BLUE MATTER • Primary accent {BM_PRIMARY} • Update in app.py if you have the official hex.</div>
""",
    unsafe_allow_html=True,
)
