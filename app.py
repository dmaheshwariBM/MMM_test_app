# streamlit_app.py
import streamlit as st
import pandas as pd
import os, io, json, glob, math
from datetime import datetime

# -------------------------
# Config / Paths
# -------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
BUDGETS_DIR = "budgets"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BUDGETS_DIR, exist_ok=True)

st.set_page_config(
    page_title="MMM Control Center",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Minimal style (clean, healthcare feel)
# -------------------------
st.markdown("""
<style>
:root {
  --brand:#0A6DFF;             /* primary */
  --brand-2:#0FB2B2;           /* accent */
  --ink:#0F172A;               /* text */
  --muted:#64748B;             /* secondary text */
  --bg:#F8FAFC;                /* page bg */
  --card:#FFFFFF;
  --border:#E2E8F0;
}
body { background: var(--bg); }
.block-container { padding-top: 1rem; }

/* Hero */
.hero {
  background: linear-gradient(135deg, rgba(10,109,255,0.08), rgba(15,178,178,0.08));
  border:1px solid var(--border);
  border-radius:16px; padding:22px 22px;
}
.hero-title{
  font-size: 1.55rem; font-weight: 700; color: var(--ink); margin: 2px 0 6px;
}
.hero-sub{
  color: var(--muted); font-size: 0.98rem; margin-top: 2px;
}
.builtby{
  color: var(--muted); font-size: 0.76rem; text-align: right; margin-top: 8px;
}

/* KPI cards */
.card {
  background: var(--card); border:1px solid var(--border);
  border-radius:14px; padding:18px; height:116px;
}
.card .label { color: var(--muted); font-size:0.85rem; }
.card .value { font-size:1.6rem; font-weight:700; color:var(--ink); line-height:1.2; }
.card .sub   { color:var(--muted); font-size:0.78rem; margin-top:4px; }

/* Section headers */
h3 { margin-top: 8px; }

/* Tables */
table { font-size: 0.95rem; }
.small-note { color: var(--muted); font-size: 0.85rem; }
.hr { height:1px; background:var(--border); margin: 8px 0 18px; }

/* File badges */
.badge {
  display:inline-block; padding:2px 8px; border-radius:999px;
  background:#EEF2FF; color:#3730A3; font-size:0.76rem; border:1px solid #E0E7FF;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Util
# -------------------------
def _human_size(n: int) -> str:
    if n is None: return "—"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0; i += 1
    return f"{f:,.1f} {units[i]}"

def _file_info(path: str) -> dict:
    try:
        stat = os.stat(path)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        ext = os.path.splitext(path)[1].lower()
        cols = "—"
        rows = "—"
        name = os.path.basename(path)
        if ext == ".csv":
            try:
                hdr = pd.read_csv(path, nrows=0)
                cols = f"{hdr.shape[1]:,}"
                try:
                    with open(path, "rb") as f:
                        lines = sum(1 for _ in f)
                    rows = f"{max(lines-1,0):,}"
                except Exception:
                    rows = "—"
            except Exception:
                pass
        elif ext in (".xls",".xlsx"):
            try:
                hdr = pd.read_excel(path, nrows=0)
                cols = f"{hdr.shape[1]:,}"
                rows = "—"
            except Exception:
                pass
        ftype = "CSV" if ext==".csv" else ("Excel" if ext in (".xls",".xlsx") else ext.upper().strip("."))
        return {
            "File": name,
            "Type": ftype,
            "Size": _human_size(stat.st_size),
            "Columns": cols,
            "Rows": rows,
            "Modified": mtime
        }
    except Exception:
        return {}

def _latest_result() -> dict|None:
    batches = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), reverse=True)
    for b in batches:
        js = sorted(glob.glob(os.path.join(b, "*.json")), reverse=True)
        for jf in js:
            try:
                with open(jf, "r") as f:
                    rec = json.load(f)
                if "name" in rec and "metrics" in rec:
                    return rec
            except Exception:
                continue
    return None

def _count_budget_packs() -> int:
    return len([d for d in glob.glob(os.path.join(BUDGETS_DIR, "*")) if os.path.isdir(d)])

# -------------------------
# HERO (no logo; subtle built-by note)
# -------------------------
st.markdown(
    "<div class='hero'>"
    "<div class='hero-title'>Marketing Mix Modeling Control Center</div>"
    "<div class='hero-sub'>Healthcare analytics & consulting—upload data, transform, model, "
    "compare, and optimize budgets with confidence.</div>"
    "<div class='builtby'>built by <strong>BLUE MATTER</strong></div>"
    "</div>",
    unsafe_allow_html=True
)

# -------------------------
# KPIs
# -------------------------
files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".csv",".xlsx"))])
datasets_count = len(files)

latest = _latest_result()
latest_model_name = latest["name"] if latest else "—"
latest_model_time = latest["batch_ts"] if latest else "—"
budget_pack_count = _count_budget_packs()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='card'><div class='label'>Datasets</div>"
                f"<div class='value'>{datasets_count:,}</div>"
                "<div class='sub'>CSV/XLSX in <code>data/</code></div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><div class='label'>Latest Model</div>"
                f"<div class='value'>{latest_model_name}</div>"
                f"<div class='sub'>Batch: {latest_model_time}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><div class='label'>Budget Packs</div>"
                f"<div class='value'>{budget_pack_count:,}</div>"
                "<div class='sub'>Saved scenarios in <code>budgets/</code></div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='card'><div class='label'>Environment</div>"
                "<div class='value'>Streamlit</div>"
                "<div class='sub'>Analytics Workspace</div></div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------------
# RECENT DATASETS (no previews)
# -------------------------
st.markdown("### 📁 Recent datasets")
if not files:
    st.info("No files in `data/` yet. Use **Data Upload** (sidebar) to add CSV/XLSX.")
else:
    rows = []
    for f in sorted(files, key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)), reverse=True)[:20]:
        rows.append(_file_info(os.path.join(DATA_DIR, f)))
    df_files = pd.DataFrame(rows)
    st.dataframe(df_files, use_container_width=True, height=min(420, 60 + 28*len(df_files)))
st.caption("Only metadata is shown here. Previews live in the **Data Upload** page after you select a file.")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------------
# LATEST MODELS (concise)
# -------------------------
st.markdown("### 📈 Latest models")
def _load_models(limit: int = 25) -> pd.DataFrame:
    recs = []
    batches = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), reverse=True)
    for b in batches:
        js = sorted(glob.glob(os.path.join(b, "*.json")), reverse=True)
        for jf in js:
            try:
                with open(jf, "r") as f:
                    r = json.load(f)
                m = r.get("metrics", {})
                recs.append({
                    "🕒 Batch": r.get("batch_ts",""),
                    "Name": r.get("name",""),
                    "Type": r.get("type",""),
                    "Target": r.get("target",""),
                    "R²": m.get("r2", None),
                    "RMSE": m.get("rmse", None),
                    "Base %": r.get("base_pct", None),
                    "Carryover %": r.get("carryover_pct", None),
                    "Dataset": r.get("dataset",""),
                })
                if len(recs) >= limit:
                    raise StopIteration
            except StopIteration:
                break
            except Exception:
                continue
        if len(recs) >= limit:
            break
    return pd.DataFrame(recs)

df_models = _load_models(limit=25)
if df_models.empty:
    st.info("No saved model runs yet. Build models in **Modeling** and they’ll appear here.")
else:
    for col in ["R²","RMSE","Base %","Carryover %"]:
        if col in df_models.columns:
            df_models[col] = pd.to_numeric(df_models[col], errors="coerce")
    st.dataframe(df_models, use_container_width=True, height=min(420, 60 + 28*len(df_models)))
st.caption("For comparison, open **Results**. To optimize, use **Budget Optimization**.")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------------
# LATEST BUDGET PACKS
# -------------------------
st.markdown("### 💰 Recent budget scenarios")
packs = [d for d in sorted(glob.glob(os.path.join(BUDGETS_DIR,"*")), reverse=True) if os.path.isdir(d)]
if not packs:
    st.info("No budget runs yet. Create one in **Budget Optimization**.")
else:
    rows = []
    for d in packs[:10]:
        ts = os.path.basename(d)
        meta_path = os.path.join(d, "meta.json")
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            scen = meta.get("scenario","").split(')')[0]
            models = ", ".join(meta.get("models", []))
            clipped = models[:80] + ("…" if len(models)>80 else "")
            rows.append({
                "🕒 Timestamp": ts,
                "Scenario": scen,
                "Models": clipped,
                "Channels": len(meta.get("channels", []))
            })
        except Exception:
            rows.append({
                "🕒 Timestamp": ts, "Scenario": "—", "Models": "—", "Channels": "—"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------
# Footer note (small, tasteful)
# -------------------------
st.markdown(
    "<div class='small-note' style='margin-top:10px; text-align:right;'>"
    "built by <strong>BLUE MATTEr</strong>"
    "</div>",
    unsafe_allow_html=True
)
