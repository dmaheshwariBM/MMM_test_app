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
                r["_path"] = jf
                r["_batch"] = os.path.basename(b)
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

def _metrics_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {}) or {}
    return {"R²": m.get("r2"), "Adj R²": m.get("adj_r2"), "RMSE": m.get("rmse"), "n": m.get("n"), "p": m.get("p")}

def _intercept_key(coef: Dict[str, float]) -> str | None:
    for k in ("const", "Intercept", "intercept", "CONST", "const_", "_const", "beta0", "b0"):
        if k in coef: return k
    return None

def _to_num_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "const":
        return pd.Series(np.ones(len(df)), index=df.index, name="const")
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    if name.endswith("__tfm") and name[:-5] in df.columns:
        return pd.to_numeric(df[name[:-5]], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, name=name)

def _normalize_and_round(d: Dict[str, Any]) -> Dict[str, Any]:
    base_pct = float(d.get("base_pct", 0.0))
    carry_pct = float(d.get("carryover_pct", 0.0))
    impact_map: Dict[str, float] = dict(d.get("impactable_pct", {}))
    incr_pct = float(sum(impact_map.values()))
    total = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total - 100.0) > 0.05:
        target_incr = max(0.0, 100.0 - base_pct - carry_pct)
        scale = target_incr / incr_pct if incr_pct > 0 else 1.0
        for k in list(impact_map.keys()):
            impact_map[k] *= scale
        incr_pct = float(sum(impact_map.values()))
    base_pct = float(round(base_pct, 6))
    carry_pct = float(round(carry_pct, 6))
    impact_map = {k: float(round(v, 6)) for k, v in impact_map.items()}
    incr_pct = float(round(sum(impact_map.values()), 6))
    return {"base_pct": base_pct, "carryover_pct": carry_pct, "incremental_pct": incr_pct, "impactable_pct": impact_map}

def _ensure_decomp(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust decomp for display:
      Denominator: sum(target) → sum(yhat) → sum(contrib) → 1.0
      Detect intercept (const/Intercept/...) even if features omitted it.
      Normalize so Base+Carry+Incremental ≈ 100
    """
    d = record.get("decomp")
    if isinstance(d, dict) and "impactable_pct" in d:
        return _normalize_and_round(d)

    dataset = record.get("dataset")
    features = record.get("features", []) or []
    coef = record.get("coef", {}) or {}
    yhat = np.asarray(record.get("yhat", []), float)

    if not dataset:
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}
    path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(path):
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}

    # Build contributions (include intercept even if not in features)
    contrib_sum: Dict[str, float] = {}
    n = len(df)
    ik = _intercept_key(coef)
    if ik is not None:
        contrib_sum["const"] = float(coef.get(ik, 0.0)) * n

    for f in features:
        if f == "const":
            # already included via ik
            continue
        c = float(coef.get(f, 0.0))
        x = _to_num_series(df, f)
        contrib_sum[f] = float((c * x).clip(lower=0.0).sum())

    # Denominator
    total_from_y = None
    tgt = record.get("target")
    if tgt and tgt in df.columns:
        total_from_y = float(pd.to_numeric(df[tgt], errors="coerce").fillna(0.0).sum())
    total_from_yhat = float(np.nansum(yhat)) if yhat.size > 0 else 0.0
    total_from_contrib = float(sum(contrib_sum.values())) if contrib_sum else 0.0
    candidates = [t for t in (total_from_y, total_from_yhat, total_from_contrib) if t and t > 0]
    total_pred = candidates[0] if candidates else 1.0

    base_sum = float(contrib_sum.get("const", 0.0))
    base_pct = 100.0 * base_sum / total_pred

    impact_map: Dict[str, float] = {}
    for f, s in contrib_sum.items():
        if f == "const":
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        if s > 0:
            impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * s / total_pred

    return _normalize_and_round({"base_pct": base_pct, "carryover_pct": 0.0, "incremental_pct": float(sum(impact_map.values())), "impactable_pct": impact_map})

def _short_type(t: str) -> str:
    return {
        "base": "base",
        "breakout_split": "breakout",
        "residual_reattribute": "residual",
        "pathway_redistribute": "pathway",
    }.get(str(t), str(t) or "base")

def _fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name","(unnamed)")
    tp = _short_type(r.get("type","base"))
    tgt = r.get("target","?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
    return f"{nm} • {tp} • {tgt} • {when}"

# ---------------------------
# Load catalog
# ---------------------------
catalog = _load_all_results()
if not catalog:
    st.info("No saved models yet. Build a model in **Modeling**, then come back here.")
    st.stop()

st.markdown("#### All runs (latest first)")
summary_rows = []
for r in catalog:
    summary_rows.append({
        "Name": r.get("name",""),
        "Type": _short_type(r.get("type","base")),
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
baseline = st.selectbox("Baseline for Δ", options=chosen, index=0)
baseline_rec = models[chosen.index(baseline)]

st.divider()

# ---------------------------
# Fit metrics
# ---------------------------
st.subheader("Fit metrics")
met = []
for lbl, r in zip(chosen, models):
    row = _metrics_row(r); row["Model"] = lbl
    met.append(row)
met_df = pd.DataFrame(met).set_index("Model")
st.dataframe(met_df, use_container_width=True)

# ---------------------------
# Decomposition summary
# ---------------------------
st.subheader("Decomposition summary")
decomp_map = {lbl: _ensure_decomp(r) for lbl, r in zip(chosen, models)}
decomp_rows = []
for lbl in chosen:
    d = decomp_map[lbl]
    decomp_rows.append({"Model": lbl, "Base %": d["base_pct"], "Carryover %": d["carryover_pct"], "Incremental %": d["incremental_pct"]})
decomp_df = pd.DataFrame(decomp_rows).set_index("Model")
st.dataframe(decomp_df, use_container_width=True)
st.bar_chart(decomp_df)

# ---------------------------
# Impactable % by channel — aligned
# ---------------------------
st.subheader("Impactable % by channel — aligned")
channels = sorted({ch for d in decomp_map.values() for ch in d.get("impactable_pct", {}).keys()})
if not channels:
    st.info("No per-channel impactable breakdown in selected models.")
else:
    mat = []
    for ch in channels:
        row = {"Channel": ch}
        for lbl in chosen:
            row[lbl] = float(decomp_map[lbl]["impactable_pct"].get(ch, 0.0))
        mat.append(row)
    impact_df = pd.DataFrame(mat).set_index("Channel")
    st.dataframe(impact_df, use_container_width=True)

    st.markdown("**Δ vs baseline (selected above)**")
    if baseline in impact_df.columns:
        deltas = impact_df.subtract(impact_df[baseline], axis=0)
        st.dataframe(deltas, use_container_width=True)

# ---------------------------
# Export
# ---------------------------
st.subheader("Export")
from io import StringIO
csv_buf = StringIO()
csv_buf.write("## metrics\n"); met_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
csv_buf.write("## decomposition\n"); decomp_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
if 'impact_df' in locals():
    csv_buf.write("## impactable_by_channel\n"); impact_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
csv_bytes = csv_buf.getvalue().encode("utf-8")
st.download_button("Download comparison CSV", data=csv_bytes, file_name="mmm_results_comparison.csv", mime="text/csv")

# ---------------------------
# Handoff to Budget Optimization
# ---------------------------
st.subheader("Select for Budget Optimization")
to_budget = st.multiselect("Models to send", options=chosen, default=chosen[:min(3, len(chosen))])
if st.button("Send to Budget Optimization"):
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
    st.success(f"Sent {len(skinny)} model(s) to Budget Optimization.")
