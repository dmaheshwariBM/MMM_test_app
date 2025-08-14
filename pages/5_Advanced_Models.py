# pages/5_Advanced_Models.py
import os, json, glob, re, importlib.util, pathlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "ADVANCED_MODELS_PAGE_v2_3_0"

st.title("🧠 Advanced Models — Breakout • Residual • Pathway")
st.caption(f"Page ID: `{PAGE_ID}`")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------
# Force-load *local* core/advanced_models.py by file path (avoids any 'core' name clash)
# ---------------------------------------------------------------------------------------
CORE_PATH = pathlib.Path("core/advanced_models.py").resolve()
if not CORE_PATH.exists():
    st.error(f"Expected file not found: {CORE_PATH}")
    st.stop()

_spec = importlib.util.spec_from_file_location("advanced_models_local", CORE_PATH)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)  # type: ignore
st.caption(f"Loaded advanced_models from: `{getattr(am, '__file__', 'unknown')}` (core version: {getattr(am, 'ADV_MODELS_VERSION', 'unknown')})")

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
    return pd.read_csv(os.path.join(DATA_DIR, name))

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

def _overwrite_base(base_name: str, new_decomp: Dict[str, Any]) -> None:
    """Replace only the decomp in the JSON with name==base_name."""
    for jf in glob.glob(os.path.join(RESULTS_DIR, "*", "*.json")):
        try:
            with open(jf, "r") as f:
                rec = json.load(f)
            if str(rec.get("name","")).strip().lower() == str(base_name).strip().lower():
                keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset","metrics","coef","yhat","features","type")}
                keep["decomp"] = new_decomp
                with open(jf, "w") as g:
                    json.dump(keep, g, indent=2)
                return
        except Exception:
            continue

def _render_decomp(decomp: Dict[str, Any]):
    if not decomp:
        st.info("No decomposition available."); return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base %", f"{float(decomp.get('base_pct',0)):.2f}%")
    with c2: st.metric("Carryover %", f"{float(decomp.get('carryover_pct',0)):.2f}%")
    with c3: st.metric("Incremental %", f"{float(decomp.get('incremental_pct',0)):.2f}%")
    imp = decomp.get("impactable_pct", {}) or {}
    if imp:
        dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())})
        st.bar_chart(dfv.set_index("Channel"))

# ---------------------------
# Guards
# ---------------------------
REQUIRED_FUNCS = [
    "breakout_split",
    "residual_reattribute",
    "pathway_redistribute",
    "apply_decomp_update",
    "_ensure_decomp_from_record_or_recompute",
]
missing = [fn for fn in REQUIRED_FUNCS if not hasattr(am, fn)]
if missing:
    st.error(f"`advanced_models.py` missing: {', '.join(missing)}\nLoaded from: {getattr(am, '__file__', 'unknown')}")
    st.stop()

# ---------------------------
# Pick base model
# ---------------------------
catalog = _load_results_catalog()
if not catalog:
    st.info("No saved models found. Build a base model in **Modeling** first.")
    st.stop()

base_labels = [f"{r.get('name','(unnamed)')} • {r.get('target','?')} • {r.get('_ts').strftime('%Y-%m-%d %H:%M:%S') if r.get('_ts') else ''}" for r in catalog]
sel = st.selectbox("Base model to extend", options=base_labels, index=0)
base = catalog[base_labels.index(sel)]

try:
    df = _load_dataset_csv(base["dataset"])
except Exception as e:
    st.error(f"Could not open dataset '{base['dataset']}'."); st.exception(e); st.stop()

features = [f for f in base.get("features", []) if f != "const"]
features_disp = [f[:-5] if f.endswith("__tfm") else f for f in features]
decomp0 = am._ensure_decomp_from_record_or_recompute(base, df)
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
base_set_disp = set(features_disp)

st.caption(f"Dataset: **{base['dataset']}** • Target: **{base.get('target','?')}**")
st.info(f"Base % (current): **{float(decomp0.get('base_pct',0)):.2f}%**")

# ---------------------------
# Tabs (Advanced Models only)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Breakout split", "➕ Residual (on Base)", "🧩 Pathway redistribution"])

# ===== Breakout split =====
with tab1:
    st.subheader("Breakout split")
    parent = st.selectbox("Channel to split (from base)", options=features_disp)
    candidates = [c for c in num_cols if (c not in base_set_disp and not c.startswith("_tfm_"))]
    subs = st.multiselect("Sub-metrics (not in base model)", options=candidates)

    if st.button("Run breakout split", type="primary"):
        try:
            out = am.breakout_split(df=df, base_record=base, channel_to_split=parent, sub_metrics=subs)
            new_decomp = am.apply_decomp_update(base, df, out)
            st.success("Breakout computed. Preview below:")
            _render_decomp(new_decomp)

            c1, c2 = st.columns(2)
            with c1:
                name_new = st.text_input("Save as name", value=f"{base['name']}__breakout_{parent}")
                if st.button("Save as new result"):
                    where = _persist_record(name_new, base.get("target","?"), base["dataset"], {**base, "type": "breakout_split", "decomp": new_decomp})
                    st.success(f"Saved to `{where}`.")
            with c2:
                if st.button("Update base result"):
                    _overwrite_base(base["name"], new_decomp)
                    st.success("Base model updated.")
        except Exception as e:
            st.error("Breakout failed.")
            st.exception(e)

# ===== Residual (on Base) =====
with tab2:
    st.subheader("Residual re-attribution (regress Base on new channels)")
    st.caption("We fit the **Base (intercept)** series on selected channels. The **fitted share of Base** is moved from Base% to those channels (split by fitted contributions).")

    extra = st.multiselect("Channels to explain Base (not in base model)", options=[c for c in num_cols if (c not in base_set_disp and not c.startswith("_tfm_"))])
    frac = st.slider("Apply what fraction of the *fitted* Base to reattribute?", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

    if st.button("Run residual re-attribution", type="primary"):
        try:
            out = am.residual_reattribute(df=df, base_record=base, extra_channels=extra, fraction=frac)
            new_decomp = am.apply_decomp_update(base, df, out)
            st.success(f"Residual computed. Fitted share of Base = {out['fitted_share_of_base']*100:.1f}%. Applied fraction = {out['fraction']*100:.0f}%.")
            st.write(pd.DataFrame([{
                "Base % before": out["base_pct_before"],
                "Base % after": out["base_pct_after"],
                "Fitted Base share (%)": out["fitted_share_of_base"] * 100.0
            }]))
            st.dataframe(pd.DataFrame({"Channel": list(out["allocated"].keys()),
                                       "Allocated (pp)": list(out["allocated"].values())}),
                         use_container_width=True)
            _render_decomp(new_decomp)

            c1, c2 = st.columns(2)
            with c1:
                name_new = st.text_input("Save as name", value=f"{base['name']}__residual_on_base")
                if st.button("Save as new result", key="save_resid"):
                    where = _persist_record(name_new, base.get("target","?"), base["dataset"], {**base, "type": "residual_reattribute", "decomp": new_decomp})
                    st.success(f"Saved to `{where}`.")
            with c2:
                if st.button("Update base result", key="upd_resid"):
                    _overwrite_base(base["name"], new_decomp)
                    st.success("Base model updated.")
        except Exception as e:
            st.error("Residual re-attribution failed.")
            st.exception(e)

# ===== Pathway redistribution =====
with tab3:
    st.subheader("Pathway redistribution")
    if not features_disp:
        st.warning("No channels found in base features.")
    else:
        A = st.selectbox("Channel A (loses some share)", options=features_disp, index=0, key="path_A")
        B = st.selectbox("Channel B (gains that share)", options=[c for c in features_disp if c != A], index=0, key="path_B")

        if st.button("Run pathway redistribution", type="primary"):
            try:
                out = am.pathway_redistribute(df=df, base_record=base, channel_A=A, channel_B=B)
                new_decomp = am.apply_decomp_update(base, df, out)
                st.success("Pathway redistribution computed. Preview below:")
                st.write(pd.DataFrame([{
                    "Share from A→B": out["share_from_A_to_B"],
                    "Moved (pp)": out["moved_pct_points"],
                    "A old": out["A_old"], "A new": out["A_new"],
                    "B old": out["B_old"], "B new": out["B_new"],
                }]))
                _render_decomp(new_decomp)

                c1, c2 = st.columns(2)
                with c1:
                    name_new = st.text_input("Save as name", value=f"{base['name']}__pathway_{A}_to_{B}")
                    if st.button("Save as new result", key="save_path"):
                        where = _persist_record(name_new, base.get("target","?"), base["dataset"], {**base, "type": "pathway_redistribute", "decomp": new_decomp})
                        st.success(f"Saved to `{where}`.")
                with c2:
                    if st.button("Update base result", key="upd_path"):
                        _overwrite_base(base["name"], new_decomp)
                        st.success("Base model updated.")
            except Exception as e:
                st.error("Pathway redistribution failed.")
                st.exception(e)
