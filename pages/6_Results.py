# pages/6_Results.py
# v2.0.0  ASCII-only. Compare and Inspect saved runs. Robust loader and exports.

import os
import re
import glob
import json
from io import StringIO
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "RESULTS_PAGE_v2_0_0"
st.title("Results")
st.caption("Page ID: {}".format(PAGE_ID))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------- Writable roots --------
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
st.caption("Scanning results under: {}".format(", ".join(CANDIDATE_RESULTS_ROOTS)))
st.caption("Using results root: {}".format(RESULTS_ROOT))

# Persisted banners
if "last_saved_path" in st.session_state and st.session_state["last_saved_path"]:
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if "last_save_error" in st.session_state and st.session_state["last_save_error"]:
    st.error(st.session_state["last_save_error"])

# -------- Utilities --------
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

def ensure_decomp(record: Dict[str, Any]) -> Dict[str, Any]:
    d = record.get("decomp")
    if isinstance(d, dict) and "impactable_pct" in d:
        # normalize/round
        base_pct = float(d.get("base_pct", 0.0))
        carry_pct = float(d.get("carryover_pct", 0.0))
        impact = dict(d.get("impactable_pct", {}))
        incr = float(sum(impact.values()))
        total = base_pct + carry_pct + incr
        if incr > 0 and abs(total - 100.0) > 0.05:
            target_incr = max(0.0, 100.0 - base_pct - carry_pct)
            scale = target_incr / incr if incr > 0 else 1.0
            for k in list(impact.keys()):
                impact[k] = float(impact[k]) * scale
            incr = float(sum(impact.values()))
        return {
            "base_pct": float(round(base_pct, 6)),
            "carryover_pct": float(round(carry_pct, 6)),
            "incremental_pct": float(round(incr, 6)),
            "impactable_pct": {k: float(round(v, 6)) for k, v in impact.items()},
        }

    # fallback: recompute from coef and features if dataset available
    features = record.get("features", []) or []
    coef = record.get("coef", {}) or {}
    dataset = record.get("dataset")
    yhat = record.get("yhat", [])
    if not dataset:
        return {"base_pct": float("nan"), "carryover_pct": 0.0, "incremental_pct": float("nan"), "impactable_pct": {}}
    path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(path):
        return {"base_pct": float("nan"), "carryover_pct": 0.0, "incremental_pct": float("nan"), "impactable_pct": {}}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"base_pct": float("nan"), "carryover_pct": 0.0, "incremental_pct": float("nan"), "impactable_pct": {}}

    # compute contributions
    n = len(df)
    contrib_sum: Dict[str, float] = {}
    if "const" in coef:
        contrib_sum["const"] = float(coef.get("const", 0.0)) * n
    for f in features:
        c = float(coef.get(f, 0.0))
        x = pd.to_numeric(df[f], errors="coerce").fillna(0.0) if f in df.columns else pd.Series([0.0] * n)
        s = float(np.sum(np.maximum(c * x.values, 0.0)))
        contrib_sum[f] = s

    tgt = record.get("target")
    total_from_y = float(pd.to_numeric(df[tgt], errors="coerce").fillna(0.0).sum()) if (tgt and tgt in df.columns) else 0.0
    total_from_yhat = float(np.nansum(np.asarray(yhat, float))) if yhat else 0.0
    total_from_contrib = float(sum(contrib_sum.values())) if contrib_sum else 0.0
    candidates = [t for t in (total_from_y, total_from_yhat, total_from_contrib) if t and t > 0]
    denom = candidates[0] if candidates else 1.0

    base_sum = float(contrib_sum.get("const", 0.0))
    base_pct = 100.0 * base_sum / denom

    impact_map: Dict[str, float] = {}
    for f, s in contrib_sum.items():
        if f == "const":
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        if s > 0:
            impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * s / denom

    incr_pct = float(sum(impact_map.values()))
    carry_pct = 0.0
    total = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total - 100.0) > 0.05:
        target_incr = max(0.0, 100.0 - base_pct - carry_pct)
        scale = target_incr / incr_pct if incr_pct > 0 else 1.0
        for k in list(impact_map.keys()):
            impact_map[k] = impact_map[k] * scale
        incr_pct = float(sum(impact_map.values()))

    return {
        "base_pct": float(round(base_pct, 6)),
        "carryover_pct": float(round(carry_pct, 6)),
        "incremental_pct": float(round(incr_pct, 6)),
        "impactable_pct": {k: float(round(v, 6)) for k, v in impact_map.items()},
    }

def metrics_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {}) or {}
    return {"R2": m.get("r2"), "AdjR2": m.get("adj_r2"), "RMSE": m.get("rmse"), "n": m.get("n"), "p": m.get("p")}

def fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name", "(unnamed)")
    tp = r.get("type", "base")
    tgt = r.get("target", "?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
    return "{} | {} | {} | {}".format(nm, tp, tgt, when)

# -------- Top actions --------
c1, c2, c3 = st.columns([1, 2, 3])
with c1:
    if st.button("Refresh list"):
        st.rerun()
with c2:
    show_diag = st.checkbox("Diagnostics", value=False)
with c3:
    st.caption("Roots: {}".format(", ".join(CANDIDATE_RESULTS_ROOTS)))

# -------- Load catalog --------
catalog = load_results_catalog(CANDIDATE_RESULTS_ROOTS)
if show_diag:
    st.info("Found {} saved JSON file(s).".format(len(catalog)))
    for r in catalog[:20]:
        st.text(" - {}".format(r.get("_path")))

if not catalog:
    st.info("No saved models yet. Go to Modeling, save a run, then refresh here.")
    st.stop()

# -------- Tabs --------
tab_cmp, tab_inspect = st.tabs(["Compare", "Inspect"])

# ===== Compare =====
with tab_cmp:
    st.subheader("All runs (latest first)")
    summary_rows = []
    for r in catalog:
        summary_rows.append({
            "Name": r.get("name", ""),
            "Type": (r.get("type") or "base"),
            "Target": r.get("target", ""),
            "Saved at": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, height=min(420, 48*max(1, len(summary_rows))+40))

    labels = [fmt_label(r) for r in catalog]
    default_sel = labels[:2] if len(labels) >= 2 else labels[:1]
    chosen = st.multiselect("Select up to 5 models", options=labels, default=default_sel, max_selections=5)
    if chosen:
        models = [catalog[labels.index(lbl)] for lbl in chosen]
        baseline = st.selectbox("Baseline for delta", options=chosen, index=0)
        st.divider()

        st.subheader("Fit metrics")
        met = []
        for lbl, r in zip(chosen, models):
            row = metrics_row(r)
            row["Model"] = lbl
            met.append(row)
        met_df = pd.DataFrame(met).set_index("Model")
        st.dataframe(met_df, use_container_width=True)

        st.subheader("Decomposition summary")
        decomp_map = {lbl: ensure_decomp(r) for lbl, r in zip(chosen, models)}
        decomp_rows = []
        for lbl in chosen:
            d = decomp_map[lbl]
            decomp_rows.append({"Model": lbl, "Base %": d["base_pct"], "Carryover %": d["carryover_pct"], "Incremental %": d["incremental_pct"]})
        decomp_df = pd.DataFrame(decomp_rows).set_index("Model")
        st.dataframe(decomp_df, use_container_width=True)
        st.bar_chart(decomp_df)

        st.subheader("Impactable % by channel (aligned)")
        channels = sorted({ch for d in decomp_map.values() for ch in d.get("impactable_pct", {}).keys()})
        if channels:
            mat = []
            for ch in channels:
                row = {"Channel": ch}
                for lbl in chosen:
                    row[lbl] = float(decomp_map[lbl]["impactable_pct"].get(ch, 0.0))
                mat.append(row)
            impact_df = pd.DataFrame(mat).set_index("Channel")
            st.dataframe(impact_df, use_container_width=True)

            st.markdown("Delta vs baseline")
            if baseline in impact_df.columns:
                deltas = impact_df.subtract(impact_df[baseline], axis=0)
                st.dataframe(deltas, use_container_width=True)

        st.subheader("Export")
        csv_buf = StringIO()
        csv_buf.write("## metrics\n"); met_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        csv_buf.write("## decomposition\n"); decomp_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        if "impact_df" in locals():
            csv_buf.write("## impactable_by_channel\n"); impact_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        st.download_button("Download comparison CSV", data=csv_bytes, file_name="mmm_results_comparison.csv", mime="text/csv")

        st.subheader("Send to Budget Optimization")
        to_budget = st.multiselect("Models to send", options=chosen, default=chosen[:min(3, len(chosen))])
        if st.button("Send"):
            skinny = []
            for lbl in to_budget:
                r = models[chosen.index(lbl)]
                d = decomp_map[lbl]
                skinny.append({
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "dataset": r.get("dataset"),
                    "target": r.get("target"),
                    "metrics": r.get("metrics", {}),
                    "decomp": d,
                    "features": r.get("features", []),
                    "_path": r.get("_path", ""),
                    "_ts": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
                })
            st.session_state["budget_candidates"] = skinny
            st.success("Sent {} model(s) to Budget Optimization.".format(len(skinny)))

# ===== Inspect =====
with tab_inspect:
    st.subheader("Pick a saved run")
    labels = [fmt_label(r) for r in catalog]
    i = st.selectbox("Saved model", options=list(range(len(labels))), format_func=lambda k: labels[k], index=0)
    r = catalog[i]
    st.write("Name: {} | Type: {} | Target: {} | Saved: {}".format(
        r.get("name"), r.get("type","base"), r.get("target"), r.get("_ts")))
    d = ensure_decomp(r)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base %", "{:.2f}%".format(d["base_pct"]))
    with c2: st.metric("Carryover %", "{:.2f}%".format(d["carryover_pct"]))
    with c3: st.metric("Incremental %", "{:.2f}%".format(d["incremental_pct"]))
    imp = d.get("impactable_pct", {})
    if imp:
        dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())}).set_index("Channel")
        st.bar_chart(dfv)
        st.dataframe(dfv, use_container_width=True)

    st.subheader("Raw JSON")
    st.code(json.dumps({k: v for k, v in r.items() if k not in ("_ts","_path")}, indent=2))
