# pages/5_Advanced_Models.py
import os, json, glob, re, pathlib, importlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "ADVANCED_MODELS_PAGE_v2_6_0"
st.title("🧠 Advanced Models — Breakout • Residual • Pathway")
st.caption(f"Page ID: `{PAGE_ID}`")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Writable results roots (same logic as Results page)
def _abs(p: str) -> str: return os.path.abspath(p)
CANDIDATE_RESULTS_ROOTS = [
    _abs(os.environ.get("MMM_RESULTS_DIR", "")) if os.environ.get("MMM_RESULTS_DIR") else None,
    _abs("results"),
    _abs("/tmp/mmm_results"),
]
CANDIDATE_RESULTS_ROOTS = [p for p in CANDIDATE_RESULTS_ROOTS if p]

def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        testf = os.path.join(path, ".write_test")
        with open(testf, "w") as f: f.write("ok")
        os.remove(testf)
        return True
    except Exception:
        return False

def pick_writable_results_root() -> str:
    for root in CANDIDATE_RESULTS_ROOTS:
        if _ensure_dir(root):
            return root
    fb = _abs("results_fallback"); _ensure_dir(fb); return fb

RESULTS_ROOT = pick_writable_results_root()
st.caption(f"Results root: `{RESULTS_ROOT}`")

# ---- Import core (pure python) ----
def _import_advanced_models():
    try:
        from core import advanced_models as am  # type: ignore
        return am
    except Exception:
        pass
    core_path = pathlib.Path("core/advanced_models.py").resolve()
    if not core_path.exists():
        st.error(f"Expected file not found: {core_path}"); st.stop()
    spec = importlib.util.spec_from_file_location("advanced_models_local", core_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    return module

am = _import_advanced_models()
st.caption(f"Loaded advanced_models from: `{getattr(am, '__file__', 'unknown')}` (core v {getattr(am, 'ADV_MODELS_VERSION', 'unknown')})")

REQUIRED_FUNCS = ["breakout_split","residual_reattribute","pathway_redistribute","apply_decomp_update","_ensure_decomp_from_record_or_recompute"]
missing = [fn for fn in REQUIRED_FUNCS if not hasattr(am, fn)]
if missing:
    st.error(f"`advanced_models` missing: {', '.join(missing)}"); st.stop()

# ───────── JSON-safe save helpers ─────────
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:64]

def _json_ready(obj: Any):
    import numpy as _np
    from datetime import datetime as _dt
    if obj is None or isinstance(obj, (bool, int, float, str)): return obj
    if isinstance(obj, _dt): return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, (list, tuple)): return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("_ts","_path"): continue
            out[str(k)] = _json_ready(v)
        return out
    if hasattr(obj, "tolist"):
        try: return obj.tolist()
        except Exception: pass
    try:
        if isinstance(obj, (_np.floating, _np.integer)): return obj.item()
    except Exception: pass
    try: return float(obj)
    except Exception: return str(obj)

def save_result_json(results_root: str, name: str, target: str, dataset: str, payload: Dict[str, Any]) -> str:
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, batch_ts)
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{batch_ts}__{_safe(name)}__{_safe(target)}.json"
    body = {"batch_ts": batch_ts, "name": name, "target": target, "dataset": dataset, **payload}
    body = _json_ready(body)
    full_path = os.path.join(out_dir, fname)
    with open(full_path, "w") as f:
        json.dump(body, f, indent=2)
    return full_path

# ---------------------------
# Helpers
# ---------------------------
def _load_results_catalog() -> List[Dict[str, Any]]:
    rows = []
    files = sorted(glob.glob(os.path.join(RESULTS_ROOT, "**", "*.json"), recursive=True), reverse=True)
    for jf in files:
        try:
            with open(jf, "r") as f: r = json.load(f)
            r["_path"] = jf
            ts = r.get("batch_ts")
            if ts:
                try: r["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                except Exception: r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
            else:
                r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
            rows.append(r)
        except Exception:
            continue
    rows.sort(key=lambda x: x.get("_ts", datetime.min), reverse=True)
    return rows

def _load_dataset_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, name))

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
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Breakout split", "➕ Residual (on Base)", "🧩 Pathway redistribution"])

# Breakout
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
                    payload = {**base, "type": "breakout_split", "decomp": new_decomp}
                    saved_file = save_result_json(RESULTS_ROOT, name_new, base.get("target","?"), base["dataset"], payload)
                    st.success(f"Saved: `{saved_file}`"); st.rerun()
            with c2:
                if st.button("Update base result"):
                    for jf in glob.glob(os.path.join(RESULTS_ROOT, "**", "*.json"), recursive=True):
                        try:
                            with open(jf, "r") as f: rec = json.load(f)
                            if str(rec.get("name","")).strip().lower() == str(base["name"]).strip().lower():
                                keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset","metrics","coef","yhat","features","type")}
                                keep["decomp"] = new_decomp
                                with open(jf, "w") as g: json.dump(_json_ready(keep), g, indent=2)
                                st.success("Base model updated."); st.rerun()
                        except Exception:
                            continue
        except Exception as e:
            st.error("Breakout failed."); st.exception(e)

# Residual
with tab2:
    st.subheader("Residual re-attribution (regress Base on new channels)")
    st.caption("Move a fitted share of Base% to selected channels, split by fitted contributions.")

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
            st.dataframe(pd.DataFrame({"Channel": list(out["allocated"].keys()), "Allocated (pp)": list(out["allocated"].values())}), use_container_width=True)
            _render_decomp(new_decomp)

            c1, c2 = st.columns(2)
            with c1:
                name_new = st.text_input("Save as name", value=f"{base['name']}__residual_on_base")
                if st.button("Save as new result", key="save_resid"):
                    payload = {**base, "type": "residual_reattribute", "decomp": new_decomp}
                    saved_file = save_result_json(RESULTS_ROOT, name_new, base.get("target","?"), base["dataset"], payload)
                    st.success(f"Saved: `{saved_file}`"); st.rerun()
            with c2:
                if st.button("Update base result", key="upd_resid"):
                    for jf in glob.glob(os.path.join(RESULTS_ROOT, "**", "*.json"), recursive=True):
                        try:
                            with open(jf, "r") as f: rec = json.load(f)
                            if str(rec.get("name","")).strip().lower() == str(base["name"]).strip().lower():
                                keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset","metrics","coef","yhat","features","type")}
                                keep["decomp"] = new_decomp
                                with open(jf, "w") as g: json.dump(_json_ready(keep), g, indent=2)
                                st.success("Base model updated."); st.rerun()
                        except Exception:
                            continue
        except Exception as e:
            st.error("Residual re-attribution failed."); st.exception(e)

# Pathway
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
                        payload = {**base, "type": "pathway_redistribute", "decomp": new_decomp}
                        saved_file = save_result_json(RESULTS_ROOT, name_new, base.get("target","?"), base["dataset"], payload)
                        st.success(f"Saved: `{saved_file}`"); st.rerun()
                with c2:
                    if st.button("Update base result", key="upd_path"):
                        for jf in glob.glob(os.path.join(RESULTS_ROOT, "**", "*.json"), recursive=True):
                            try:
                                with open(jf, "r") as f: rec = json.load(f)
                                if str(rec.get("name","")).strip().lower() == str(base["name"]).strip().lower():
                                    keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset","metrics","coef","yhat","features","type")}
                                    keep["decomp"] = new_decomp
                                    with open(jf, "w") as g: json.dump(_json_ready(keep), g, indent=2)
                                    st.success("Base model updated."); st.rerun()
                            except Exception:
                                continue
            except Exception as e:
                st.error("Pathway redistribution failed."); st.exception(e)
