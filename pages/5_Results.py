# pages/6_Results.py
import os, json, glob, re
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

st.title("📊 Results — Compare & Select for Optimization")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Utils
# ---------------------------
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:72]

def _load_all_results() -> List[Dict[str, Any]]:
    rows = []
    batches = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), reverse=True)
    for b in batches:
        for jf in sorted(glob.glob(os.path.join(b, "*.json")), reverse=True):
            try:
                with open(jf, "r") as f:
                    r = json.load(f)
                # enrich
                r["_path"] = jf
                r["_batch"] = os.path.basename(b)
                # parse ts if possible
                ts = r.get("batch_ts") or os.path.basename(b)
                try:
                    r["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                except Exception:
                    r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                rows.append(r)
            except Exception:
                continue
    rows.sort(key=lambda x: x.get("_ts", datetime.min), reverse=True)
    return rows

def _short_type(t: str) -> str:
    return {
        "base": "base",
        "breakout": "breakout",
        "breakout_split": "breakout-split",
        "interaction": "interactions",
        "residual": "residual",
        "residual_reattribute": "resid-reattrib",
        "pathway_redistribute": "pathway",
        "breakout_overwrite": "breakout*",
        "interaction_overwrite": "interactions*",
        "residual_overwrite": "residual*",
    }.get(str(t), str(t) or "base")

def _fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name","(unnamed)")
    tp = _short_type(r.get("type","base"))
    ds = r.get("dataset","?")
    tgt = r.get("target","?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
    return f"{nm} • {tp} • {tgt} • {when}"

def _metrics_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {}) or {}
    return {
        "R²": m.get("r2", None),
        "Adj R²": m.get("adj_r2", None),
        "RMSE": m.get("rmse", None),
        "n": m.get("n", None),
        "p": m.get("p", None),
    }

def _decomp(r: Dict[str, Any]) -> Dict[str, Any]:
    d = r.get("decomp", {})
    # Backward-compat guard
    if not d or not isinstance(d, dict):
        # Minimal decomp if missing
        d = {
            "base_pct": np.nan,
            "carryover_pct": np.nan,
            "incremental_pct": np.nan,
            "impactable_pct": {}
        }
    d.setdefault("impactable_pct", {})
    return d

def _union_channels(models: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for r in models:
        imp = _decomp(r).get("impactable_pct", {})
        names.update(imp.keys())
    return sorted(names)

def _to_pct(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)

# ---------------------------
# Load catalog
# ---------------------------
catalog = _load_all_results()
if not catalog:
    st.info("No saved models yet. Build a model in **Modeling**, then come back here.")
    st.stop()

# Sidebar-like scroller of all runs
st.markdown("#### All runs (latest first)")
summary_rows = []
for r in catalog:
    summary_rows.append({
        "Name": r.get("name",""),
        "Type": _short_type(r.get("type","base")),
        "Dataset": r.get("dataset",""),
        "Target": r.get("target",""),
        "Saved at": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, height=min(400, 48*len(summary_rows)+40))

# ---------------------------
# Pick up to 5 models for comparison
# ---------------------------
labels = [_fmt_label(r) for r in catalog]
default_sel = labels[:2] if len(labels) >= 2 else labels[:1]
chosen = st.multiselect("Select up to 5 models to compare", options=labels, default=default_sel, max_selections=5)

if not chosen:
    st.stop()

models = [catalog[labels.index(lbl)] for lbl in chosen]
baseline_idx = 0
baseline = st.selectbox("Baseline for Δ calculations", options=chosen, index=baseline_idx)
baseline_rec = models[chosen.index(baseline)]

st.divider()

# ---------------------------
# Top-level metrics table
# ---------------------------
st.subheader("Fit metrics")
met = []
for lbl, r in zip(chosen, models):
    row = _metrics_row(r)
    row["Model"] = lbl
    met.append(row)
met_df = pd.DataFrame(met).set_index("Model")
st.dataframe(met_df, use_container_width=True)

# ---------------------------
# Decomposition summary (Base / Carryover / Incremental)
# ---------------------------
st.subheader("Decomposition summary")
decomp_rows = []
for lbl, r in zip(chosen, models):
    d = _decomp(r)
    decomp_rows.append({
        "Model": lbl,
        "Base %": _to_pct(d.get("base_pct")),
        "Carryover %": _to_pct(d.get("carryover_pct")),
        "Incremental %": _to_pct(d.get("incremental_pct")),
    })
decomp_df = pd.DataFrame(decomp_rows).set_index("Model")
st.dataframe(decomp_df, use_container_width=True)

# Quick chart (stacked feel via separate bars)
st.bar_chart(decomp_df)

# ---------------------------
# Impactable % by channel (aligned across models)
# ---------------------------
st.subheader("Impactable % by channel — aligned across models")

channels = _union_channels(models)
if not channels:
    st.info("No per-channel impactable breakdown available in selected models.")
else:
    # Build a matrix: rows = Channel, columns = Model, values = %
    mat = []
    for ch in channels:
        row = {"Channel": ch}
        for lbl, r in zip(chosen, models):
            imp = _decomp(r).get("impactable_pct", {})
            row[lbl] = _to_pct(imp.get(ch, 0.0))
        mat.append(row)
    impact_df = pd.DataFrame(mat).set_index("Channel")

    # Reorder columns to match user's chosen order
    impact_df = impact_df[[lbl for lbl in chosen]]

    st.dataframe(impact_df, use_container_width=True)

    # Per-channel chart vs baseline (first selected by default)
    st.markdown("**Δ vs baseline (selected above)**")
    base_col = baseline
    if base_col in impact_df.columns:
        deltas = impact_df.subtract(impact_df[base_col], axis=0)
        st.dataframe(deltas, use_container_width=True)
    else:
        st.caption("Baseline not in columns? (unexpected)")

# ---------------------------
# Download comparison
# ---------------------------
st.subheader("Export")
# Merge all tables into one CSV (wide)
export_parts = []
met_wide = met_df.reset_index().rename(columns={"index":"Model"})
decomp_wide = decomp_df.reset_index()
export_parts.append(("metrics", met_wide))
export_parts.append(("decomposition", decomp_wide))
if 'impact_df' in locals():
    export_parts.append(("impactable_by_channel", impact_df.reset_index()))

# Build a single CSV text
from io import StringIO
csv_buf = StringIO()
for name, dfp in export_parts:
    csv_buf.write(f"## {name}\n")
    dfp.to_csv(csv_buf, index=False)
    csv_buf.write("\n")
csv_bytes = csv_buf.getvalue().encode("utf-8")
st.download_button("Download comparison CSV", data=csv_bytes, file_name="mmm_results_comparison.csv", mime="text/csv")

# ---------------------------
# Hand off to Budget Optimization
# ---------------------------
st.subheader("Select for Budget Optimization")
st.caption("Choose any models here; they’ll be available on the Budget page as `budget_candidates`.")
to_budget = st.multiselect("Models to send", options=chosen, default=chosen[: min(3, len(chosen))])
if st.button("Send to Budget Optimization"):
    selected_records = [models[chosen.index(lbl)] for lbl in to_budget]
    # Keep only the light bits (avoid huge objects)
    skinny = []
    for r in selected_records:
        skinny.append({
            "name": r.get("name"),
            "type": r.get("type"),
            "dataset": r.get("dataset"),
            "target": r.get("target"),
            "metrics": r.get("metrics", {}),
            "decomp": r.get("decomp", {}),
            "features": r.get("features", []),
            "_path": r.get("_path", ""),
            "_ts": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
        })
    st.session_state["budget_candidates"] = skinny
    st.success(f"Sent {len(skinny)} model(s) to Budget Optimization.")
    st.caption("Open the **Budget Optimization** page to use them.")
