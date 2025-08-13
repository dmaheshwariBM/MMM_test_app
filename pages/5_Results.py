# pages/5_Results.py
import streamlit as st
import os, json, glob
import pandas as pd
from typing import List, Dict, Any

st.title("📦 Results — Model Runs")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Load all saved models ----------
def _load_all_results() -> List[Dict[str, Any]]:
    batches = sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), reverse=True)  # newest first
    runs: List[Dict[str, Any]] = []
    for bdir in batches:
        if not os.path.isdir(bdir):
            continue
        for jf in glob.glob(os.path.join(bdir, "*.json")):
            try:
                with open(jf, "r") as f:
                    rec = json.load(f)
                # minimal sanity
                if "name" in rec and "metrics" in rec and "target" in rec:
                    rec["_path"] = jf
                    runs.append(rec)
            except Exception:
                continue
    # Sort by batch_ts desc (newest first), then by name
    runs.sort(key=lambda r: (r.get("batch_ts",""), r.get("name","")), reverse=True)
    return runs

runs = _load_all_results()
if not runs:
    st.info("No saved results yet. Run some models in the Modeling page to populate this.")
    st.stop()

# ---------- Scrollable summary table ----------
def _short_feats(feats: List[str], max_show: int = 6) -> str:
    feats = feats or []
    return ", ".join(feats[:max_show]) + (" …" if len(feats) > max_show else "")

rows = []
for r in runs:
    m = r.get("metrics", {})
    rows.append({
        "🕒 Batch": r.get("batch_ts", ""),
        "Name": r.get("name", ""),
        "Type": r.get("type",""),
        "Dataset": r.get("dataset",""),
        "Target": r.get("target",""),
        "Features (preview)": _short_feats(r.get("features", [])),
        "R²": round(m.get("r2", float("nan")), 6) if m.get("r2") == m.get("r2") else None,
        "Adj R²": round(m.get("adj_r2", float("nan")), 6) if m.get("adj_r2") == m.get("adj_r2") else None,
        "RMSE": round(m.get("rmse", float("nan")), 6) if m.get("rmse") == m.get("rmse") else None,
        "Base %": round(r.get("base_pct", 0.0), 2),
        "Carryover %": round(r.get("carryover_pct", 0.0), 2),
        "_id": f"{r.get('batch_ts','')}|{r.get('name','')}"
    })
summary_df = pd.DataFrame(rows)

st.caption("Newest runs are at the top. Select models below to compare or send to budget optimization.")
st.dataframe(summary_df.drop(columns=["_id"]), use_container_width=True, height=min(520, 60 + 28*len(summary_df)))

# ---------- Selection widgets ----------
# Unique ID → index mapping for user selection
id_list = summary_df["_id"].tolist()
label_list = [
    f"{row['🕒 Batch']} • {row['Name']} • {row['Target']}"
    for _, row in summary_df.iterrows()
]

st.subheader("Select models")
c1, c2 = st.columns([2, 1])
with c1:
    compare_sel = st.multiselect(
        "✅ Pick up to 5 models to compare",
        options=list(range(len(id_list))),  # use idx to keep labels simple
        format_func=lambda i: label_list[i],
        max_selections=5,
        key="results_compare_sel"
    )
with c2:
    budget_sel = st.multiselect(
        "💰 Pick ANY models for Budget Optimization",
        options=list(range(len(id_list))),
        format_func=lambda i: label_list[i],
        key="results_budget_sel"
    )

st.divider()

# ---------- Comparison (brief, with charts) ----------
if compare_sel:
    st.subheader("📊 Comparison")
    picked = [runs[i] for i in compare_sel]

    # Build a compact metrics table
    comp_rows = []
    all_channels = set()
    for r in picked:
        ip = r.get("impactable_pct", {}) or {}
        all_channels.update(ip.keys())
    all_channels = sorted(all_channels)

    for r in picked:
        m = r.get("metrics", {})
        row = {
            "🕒 Batch": r.get("batch_ts",""),
            "Name": r.get("name",""),
            "Type": r.get("type",""),
            "Target": r.get("target",""),
            "R²": round(m.get("r2",float("nan")), 6) if m.get("r2")==m.get("r2") else None,
            "Adj R²": round(m.get("adj_r2",float("nan")), 6) if m.get("adj_r2")==m.get("adj_r2") else None,
            "RMSE": round(m.get("rmse",float("nan")), 6) if m.get("rmse")==m.get("rmse") else None,
            "Base %": round(r.get("base_pct",0.0), 2),
            "Carryover %": round(r.get("carryover_pct",0.0), 2),
        }
        # add aligned impactable columns (brief)
        ip = r.get("impactable_pct", {}) or {}
        for ch in all_channels:
            row[f"Impactable % • {ch}"] = round(float(ip.get(ch, float("nan"))), 2) if ch in ip else None
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True)

    # Compact charts: tabs, one bar per model for Impactable %
    st.markdown("**Impactable % (per model)**")
    tabs = st.tabs([f"{r.get('name','')} ({r.get('target','')})" for r in picked])
    for tab, r in zip(tabs, picked):
        with tab:
            ip = r.get("impactable_pct", {}) or {}
            if not ip:
                st.info("No impactable breakdown found.")
            else:
                ip_df = pd.DataFrame({"channel": list(ip.keys()), "Impactable %": list(ip.values())}).sort_values("Impactable %", ascending=False).set_index("channel")
                st.bar_chart(ip_df)

st.divider()

# ---------- Budget optimization selection ----------
st.subheader("💰 Send to Budget Optimization")
if st.button("Proceed with selected models"):
    chosen = [runs[i] for i in (st.session_state.get("results_budget_sel") or [])]
    if not chosen:
        st.warning("Please select at least one model for budget optimization.")
    else:
        # Store minimal info for the optimization page to consume
        st.session_state["selected_for_budget"] = [{
            "batch_ts": r.get("batch_ts",""),
            "name": r.get("name",""),
            "target": r.get("target",""),
            "type": r.get("type",""),
            "impactable_pct": r.get("impactable_pct", {}),
            "base_pct": r.get("base_pct", 0.0),
            "carryover_pct": r.get("carryover_pct", 0.0),
            "metrics": r.get("metrics", {}),
            "dataset": r.get("dataset",""),
            "features": r.get("features", []),
            "_path": r.get("_path",""),
        } for r in chosen]
        st.success(f"Stored {len(chosen)} model(s) for Budget Optimization. Open the Budget tab to continue.")
