# pages/5_Advanced_Models.py
import os, json, glob, re
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from core import advanced_models  # module with breakout/residual/pathway functions

st.title("🧠 Advanced Models — Breakout • Residual • Pathway")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def _load_results_catalog() -> List[Dict[str, Any]]:
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

def _load_dataset_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path)

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:64]

def _persist_record(model_name: str, target: str, dataset: str, record: Dict[str, Any]) -> str:
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, batch_ts)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{batch_ts}__{_safe(model_name)}__{_safe(target)}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump({**record,
                   "batch_ts": batch_ts,
                   "name": model_name,
                   "target": target,
                   "dataset": dataset}, f, indent=2)
    return out_dir

def _overwrite_base(base_name: str, new_record: Dict[str, Any]) -> None:
    for jf in glob.glob(os.path.join(RESULTS_DIR, "*", "*.json")):
        try:
            with open(jf, "r") as f:
                rec = json.load(f)
            if str(rec.get("name","")).strip().lower() == str(base_name).strip().lower():
                keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset")}
                with open(jf, "w") as g:
                    json.dump({**keep, **new_record}, g, indent=2)
                return
        except Exception:
            continue

def _render_decomp(decomp: Dict[str, Any]):
    if not decomp: 
        st.info("No decomposition available.")
        return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base %", f"{float(decomp.get('base_pct',0)):.1f}%")
    with c2: st.metric("Carryover %", f"{float(decomp.get('carryover_pct',0)):.1f}%")
    with c3: st.metric("Incremental %", f"{float(decomp.get('incremental_pct',0)):.1f}%")
    imp = decomp.get("impactable_pct", {}) or {}
    if imp:
        dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())})
        st.bar_chart(dfv.set_index("Channel"))

def _ensure_list_unique(seq: List[str]) -> List[str]:
    seen=set(); out=[]
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

# ---------- Fallback for missing helper on older modules ----------
def _fallback_ensure_decomp(record: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Minimal decomp if module helper is absent."""
    coef = record.get("coef", {}) or {}
    features = record.get("features", []) or []
    yhat = np.asarray(record.get("yhat", []), float)
    if yhat.size == 0:
        yhat = np.zeros(len(df), dtype=float)

    # Build contribution series per feature: coef * x
    contrib_sum = {}
    base_series = None
    for f in features:
        c = float(coef.get(f, 0.0))
        if f == "const":
            base_series = pd.Series(np.full(len(df), c), index=df.index)
            contrib_sum[f] = float(base_series.sum())
        else:
            if f in df.columns:
                x = pd.to_numeric(df[f], errors="coerce").fillna(0.0)
            elif f.endswith("__tfm") and f[:-5] in df.columns:
                x = pd.to_numeric(df[f[:-5]], errors="coerce").fillna(0.0)
            else:
                x = pd.Series(np.zeros(len(df)), index=df.index)
            s = (c * x).clip(lower=0.0).sum()
            contrib_sum[f] = float(s)

    total_pred = float(max(yhat.sum(), 1e-12))
    if total_pred <= 0:
        total_pred = float(sum(contrib_sum.values())) or 1.0

    base_pct = 100.0 * float(contrib_sum.get("const", 0.0)) / total_pred
    impact_map = {}
    for f, s in contrib_sum.items():
        if f == "const": 
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        if s > 0:
            impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * float(s) / total_pred

    carry_pct = 0.0
    incr_pct = max(0.0, 100.0 - base_pct - carry_pct)
    return {
        "base_pct": base_pct,
        "carryover_pct": carry_pct,
        "incremental_pct": incr_pct,
        "impactable_pct": impact_map
    }

# Use module helper if present; else fallback
ensure_decomp = getattr(advanced_models, "_ensure_decomp_from_record_or_recompute", _fallback_ensure_decomp)

# ---------------------------
# Pick base model
# ---------------------------
catalog = _load_results_catalog()
if not catalog:
    st.info("No saved models found. Build a base model in **Modeling** first.")
    st.stop()

base_labels = [f"{r.get('name','(unnamed)')}  —  ({r.get('dataset','?')} / target: {r.get('target','?')})" for r in catalog]
sel = st.selectbox("Base model to extend", options=base_labels, index=0)
base = catalog[base_labels.index(sel)]

# Load dataset
try:
    df = _load_dataset_csv(base["dataset"])
except Exception as e:
    st.error(f"Could not open dataset '{base['dataset']}'.")
    st.exception(e)
    st.stop()

# Prepare lists
features = [f for f in base.get("features", []) if f != "const"]
features_disp = [f[:-5] if f.endswith("__tfm") else f for f in features]
decomp = ensure_decomp(base, df)
impact_map = dict(decomp.get("impactable_pct", {}))
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

st.caption(f"Dataset: **{base['dataset']}** • Target: **{base.get('target','?')}**")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Breakout split", "➕ Residual re-attribution", "🧩 Pathway redistribution"])

# ===== Breakout split =====
with tab1:
    st.subheader("Breakout split (no intercept; sum preserved)")
    st.caption("Pick a base channel to split; then pick sub-metrics NOT in the base model. The parent impact is distributed across the chosen sub-metrics; totals stay the same.")

    parent = st.selectbox("Channel to split (from base model)", options=features_disp)
    base_set_disp = set(features_disp)
    candidates = [c for c in num_cols if (c not in base_set_disp and not c.startswith("_tfm_"))]
    subs = st.multiselect("Sub-metrics (not in base)", options=candidates)

    if st.button("Run breakout split", type="primary"):
        try:
            out = advanced_models.breakout_split(
                df=df,
                base_record=base,
                channel_to_split=parent,
                sub_metrics=_ensure_list_unique(subs),
            )
            st.success("Breakout computed.")
            st.write(pd.DataFrame([{"Parent": out["split_channel"], "Original %": out["original_channel_pct"]}]))
            alloc_tbl = pd.DataFrame({"Sub-metric": list(out["allocated"].keys()),
                                      "Allocated %": list(out["allocated"].values())})
            st.dataframe(alloc_tbl, use_container_width=True)

            new_decomp = advanced_models.apply_decomp_update(base, df, out)
            st.markdown("**Preview updated decomposition**")
            _render_decomp(new_decomp)

            c1, c2 = st.columns(2)
            with c1:
                name_new = st.text_input("Save as name", value=f"{base['name']}__breakout_{parent}")
                if st.button("Save as new result"):
                    where = _persist_record(name_new, base.get("target","?"), base["dataset"], {
                        **base,
                        "type": "breakout_split",
                        "decomp": new_decomp
                    })
                    st.success(f"Saved to `{where}`.")
            with c2:
                if st.button("Update base result"):
                    payload = {k: base.get(k) for k in ("metrics","coef","yhat","features")}
                    payload.update({"type": base.get("type","base"), "decomp": new_decomp})
                    _overwrite_base(base["name"], payload)
                    st.success("Base model updated.")
        except Exception as e:
            st.error("Breakout failed.")
            st.exception(e)

# ===== Residual re-attribution =====
with tab2:
    st.subheader("Residual re-attribution (allocate from Base % to new channels)")
    st.caption("Select extra channels NOT in the base model. A share of **Base %** will be allocated to them; Base % decreases by the same total; incremental % increases.")

    base_pct = float(decomp.get("base_pct", 0.0))
    st.info(f"Current Base %: **{base_pct:.1f}%**")

    base_set_disp = set(features_disp)
    extra = st.multiselect("Extra channels (not in base)", options=[c for c in num_cols if (c not in base_set_disp and not c.startswith("_tfm_"))])

    if st.button("Run residual re-attribution", type="primary"):
        try:
            out = advanced_models.residual_reattribute(
                df=df,
                base_record=base,
                extra_channels=_ensure_list_unique(extra)
            )
            st.success("Residual re-attribution computed.")
            st.write(pd.DataFrame([{"Base % before": out["base_pct_before"], "Base % after": out["base_pct_after"]}]))
            st.dataframe(pd.DataFrame({"Channel": list(out["allocated"].keys()),
                                       "From Base % (pp)": list(out["allocated"].values())}),
                         use_container_width=True)

            new_decomp = advanced_models.apply_decomp_update(base, df, out)
            st.markdown("**Preview updated decomposition**")
            _render_decomp(new_decomp)

            c1, c2 = st.columns(2)
            with c1:
                name_new = st.text_input("Save as name", value=f"{base['name']}__residual_reattrib")
                if st.button("Save as new result", key="save_resid"):
                    where = _persist_record(name_new, base.get("target","?"), base["dataset"], {
                        **base,
                        "type": "residual_reattribute",
                        "decomp": new_decomp
                    })
                    st.success(f"Saved to `{where}`.")
            with c2:
                if st.button("Update base result", key="upd_resid"):
                    payload = {k: base.get(k) for k in ("metrics","coef","yhat","features")}
                    payload.update({"type": base.get("type","base"), "decomp": new_decomp})
                    _overwrite_base(base["name"], payload)
                    st.success("Base model updated.")
        except Exception as e:
            st.error("Residual re-attribution failed.")
            st.exception(e)

# ===== Pathway redistribution =====
with tab3:
    st.subheader("Pathway redistribution (move impact from A to B)")
    st.caption("Pick **Channel A** (loses some share) and **Channel B** (gains it). Share is inferred from a single-factor fit of A’s contribution on B’s series; totals stay the same.")

    if not features_disp:
        st.warning("No channels found in base features.")
    else:
        A = st.selectbox("Channel A (from base)", options=features_disp, index=0, key="path_A")
        B_options = [c for c in features_disp if c != A]
        B = st.selectbox("Channel B (from base)", options=B_options, index=0, key="path_B")

        if st.button("Run pathway redistribution", type="primary"):
            try:
                out = advanced_models.pathway_redistribute(
                    df=df,
                    base_record=base,
                    channel_A=A,
                    channel_B=B
                )
                st.success("Pathway redistribution computed.")
                st.write(pd.DataFrame([{
                    "Share from A→B": out["share_from_A_to_B"],
                    "Moved (pp)": out["moved_pct_points"],
                    "A old": out["A_old"], "A new": out["A_new"],
                    "B old": out["B_old"], "B new": out["B_new"],
                }]))

                new_decomp = advanced_models.apply_decomp_update(base, df, out)
                st.markdown("**Preview updated decomposition**")
                _render_decomp(new_decomp)

                c1, c2 = st.columns(2)
                with c1:
                    name_new = st.text_input("Save as name", value=f"{base['name']}__pathway_{A}_to_{B}")
                    if st.button("Save as new result", key="save_path"):
                        where = _persist_record(name_new, base.get("target","?"), base["dataset"], {
                            **base,
                            "type": "pathway_redistribute",
                            "decomp": new_decomp
                        })
                        st.success(f"Saved to `{where}`.")
                with c2:
                    if st.button("Update base result", key="upd_path"):
                        payload = {k: base.get(k) for k in ("metrics","coef","yhat","features")}
                        payload.update({"type": base.get("type","base"), "decomp": new_decomp})
                        _overwrite_base(base["name"], payload)
                        st.success("Base model updated.")
            except Exception as e:
                st.error("Pathway redistribution failed.")
                st.exception(e)
