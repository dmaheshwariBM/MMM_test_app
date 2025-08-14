# pages/5_Advanced_Models.py
# v2.2.0  ASCII-only. Advanced adjustments with robust preview and save.

import os
import re
import glob
import json
import importlib.util
import pathlib
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "ADVANCED_MODELS_PAGE_v2_2_0"
st.title("Advanced Models")
st.caption("Page ID: {}".format(PAGE_ID))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Writable results root (same policy everywhere) ----------------
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
st.caption("Saving under: {}".format(RESULTS_ROOT))

# Persisted banners for last save status
if "last_saved_path" in st.session_state and st.session_state["last_saved_path"]:
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if "last_save_error" in st.session_state and st.session_state["last_save_error"]:
    st.error(st.session_state["last_save_error"])

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:72]

def _save_result_json(results_root: str, name: str, target: str, dataset: str, payload: Dict[str, Any]) -> str:
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, batch_ts)
    os.makedirs(out_dir, exist_ok=True)
    fname = "{}__{}__{}.json".format(batch_ts, _safe(name), _safe(target))
    full_path = os.path.join(out_dir, fname)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return full_path

# ---------------- Load Results Catalog (same shape as Results page) ----------------
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

# ---------------- Import advanced core ----------------
def _import_advanced_models():
    try:
        from core import advanced_models as am  # type: ignore
        return am
    except Exception:
        pass
    # fallback: try local file path
    core_path = pathlib.Path("core/advanced_models.py").resolve()
    if not core_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("advanced_models_local", core_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    return module

am = _import_advanced_models()
if am is None:
    st.error("core/advanced_models.py not found or failed to import.")
    st.stop()

# Require functions
required = ["breakout_split", "residual_reattribute", "pathway_redistribute", "apply_decomp_update", "_ensure_decomp_from_record_or_recompute"]
missing = [fn for fn in required if not hasattr(am, fn)]
if missing:
    st.error("advanced_models is missing: {}".format(", ".join(missing)))
    st.stop()

# ---------------- Read saved models ----------------
catalog = load_results_catalog(CANDIDATE_RESULTS_ROOTS)
if not catalog:
    st.info("No saved base models. Please run and save a model in the Modeling page first.")
    st.stop()

def fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name", "(unnamed)")
    tp = r.get("type", "base")
    tgt = r.get("target", "?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "-"
    return "{} | {} | {} | {}".format(nm, tp, tgt, when)

labels = [fmt_label(r) for r in catalog]
idx = st.selectbox("Pick base model", options=list(range(len(labels))), format_func=lambda k: labels[k], index=0)
base = catalog[idx]

# load dataset for base
dataset = base.get("dataset")
if not dataset or not os.path.exists(os.path.join(DATA_DIR, dataset)):
    st.error("Dataset '{}' not found in data/".format(dataset))
    st.stop()

df_base = pd.read_csv(os.path.join(DATA_DIR, dataset))

# compute starting decomp (normalized)
decomp0 = am._ensure_decomp_from_record_or_recompute(base, df_base)

# feature sets for UI
features = [f for f in base.get("features", []) if f != "const"]
features_disp = [f[:-5] if f.endswith("__tfm") else f for f in features]
base_set = set(features_disp)
numeric_cols = [c for c in df_base.columns if pd.api.types.is_numeric_dtype(df_base[c])]
not_in_base = [c for c in numeric_cols if (c not in base_set and not c.startswith("_tfm_"))]

st.divider()
cB, cR, cP = st.columns(3)

with cB:
    st.subheader("Breakout split")
    parent = st.selectbox("Parent channel", options=sorted(list(base_set)) or [""], key="adv_parent")
    subs = st.multiselect("Sub-metrics", options=sorted(not_in_base), key="adv_subs")
    add_break = st.button("Add Breakout")

with cR:
    st.subheader("Residual on Base")
    extras = st.multiselect("Explain Base with", options=sorted(not_in_base), key="adv_extras")
    frac = st.slider("Fraction of fitted Base", 0.1, 1.0, 1.0, 0.1, key="adv_frac")
    add_resid = st.button("Add Residual")

with cP:
    st.subheader("Pathway")
    A = st.selectbox("From (A)", options=sorted(list(base_set)) or [""], key="adv_A")
    B = st.selectbox("To (B)", options=[c for c in sorted(list(base_set)) if c != st.session_state.get("adv_A")] or [""], key="adv_B")
    add_path = st.button("Add Pathway")

# pipeline state
OP_KEY = "advanced_pipeline_ops"
if OP_KEY not in st.session_state:
    st.session_state[OP_KEY] = []

if add_break:
    if parent and subs:
        st.session_state[OP_KEY].append({"type": "breakout_split", "parent": parent, "subs": list(subs)})

if add_resid:
    if extras:
        st.session_state[OP_KEY].append({"type": "residual_reattribute", "extras": list(extras), "fraction": float(frac)})

if add_path:
    if A and B and A != B:
        st.session_state[OP_KEY].append({"type": "pathway_redistribute", "A": A, "B": B})

st.subheader("Pipeline")
if not st.session_state[OP_KEY]:
    st.info("No operations added yet.")
else:
    pretty = []
    for i, op in enumerate(st.session_state[OP_KEY], 1):
        if op["type"] == "breakout_split":
            pretty.append({"#": i, "Type": "breakout", "Parent": op["parent"], "Subs": ", ".join(op["subs"])})
        elif op["type"] == "residual_reattribute":
            pretty.append({"#": i, "Type": "residual", "Extras": ", ".join(op["extras"]), "Fraction": op["fraction"]})
        else:
            pretty.append({"#": i, "Type": "pathway", "A_to_B": "{}->{}".format(op["A"], op["B"])})
    st.dataframe(pd.DataFrame(pretty), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Clear all"):
            st.session_state[OP_KEY] = []
            st.rerun()
    with c2:
        rm_idx = st.number_input("Remove step #", min_value=1, max_value=len(st.session_state[OP_KEY]), value=1, step=1)
    with c3:
        if st.button("Remove step"):
            st.session_state[OP_KEY].pop(int(rm_idx) - 1)
            st.rerun()

st.divider()
if st.button("Preview"):
    try:
        current = dict(base)
        cur_decomp = am._ensure_decomp_from_record_or_recompute(current, df_base)
        current["decomp"] = cur_decomp

        for op in st.session_state[OP_KEY]:
            if op["type"] == "breakout_split":
                out = am.breakout_split(df=df_base, base_record=current, channel_to_split=op["parent"], sub_metrics=op["subs"])
            elif op["type"] == "residual_reattribute":
                out = am.residual_reattribute(df=df_base, base_record=current, extra_channels=op["extras"], fraction=float(op["fraction"]))
            else:
                out = am.pathway_redistribute(df=df_base, base_record=current, channel_A=op["A"], channel_B=op["B"])
            cur_decomp = am.apply_decomp_update(current, df_base, out)
            current["decomp"] = cur_decomp

        st.session_state["adv_preview_decomp"] = cur_decomp
        st.success("Preview ready.")
        # Show preview
        d = cur_decomp
        c1a, c2a, c3a = st.columns(3)
        with c1a: st.metric("Base %", "{:.2f}%".format(d["base_pct"]))
        with c2a: st.metric("Carryover %", "{:.2f}%".format(d["carryover_pct"]))
        with c3a: st.metric("Incremental %", "{:.2f}%".format(d["incremental_pct"]))
        imp = d.get("impactable_pct", {})
        if imp:
            dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())}).set_index("Channel")
            st.bar_chart(dfv)
            st.dataframe(dfv, use_container_width=True)
    except Exception as e:
        st.session_state["adv_preview_decomp"] = None
        st.error("Preview failed: {}".format(e))

st.divider()
st.subheader("Save / Send")
save_name = st.text_input("Save as name", value="{}_composite".format(base.get("name", "base")), key="adv_save_name")

cL, cM = st.columns(2)
with cL:
    if st.button("Save composed model"):
        try:
            if not st.session_state.get("adv_preview_decomp"):
                st.warning("Click Preview first to generate the composed result.")
            else:
                payload = {
                    "batch_ts": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "name": save_name.strip() or "{}_composite".format(base.get("name","base")),
                    "type": "composite",
                    "dataset": base.get("dataset"),
                    "target": base.get("target"),
                    "metrics": base.get("metrics", {}),
                    "coef": base.get("coef", {}),
                    "yhat": base.get("yhat", []),
                    "features": base.get("features", []),
                    "decomp": st.session_state["adv_preview_decomp"],
                    "pipeline": st.session_state.get(OP_KEY, []),
                }
                # IMPORTANT: use same layout as Modeling/Results saver
                # Inject top-level merge to match Results loader
                name = payload["name"]; target = payload["target"]; dataset = payload["dataset"]
                body = {"batch_ts": payload["batch_ts"], "name": name, "target": target, "dataset": dataset}
                body.update({k: v for k, v in payload.items() if k not in ("batch_ts","name","target","dataset")})

                path = _save_result_json(RESULTS_ROOT, name, target, dataset, body)
                st.session_state["last_saved_path"] = path
                st.session_state["last_save_error"] = ""
                st.success("Saved: {}".format(path))
        except Exception as e:
            st.session_state["last_saved_path"] = ""
            st.session_state["last_save_error"] = "Save failed: {}".format(e)
            st.error(st.session_state["last_save_error"])

with cM:
    if st.button("Send preview to Budget Optimization"):
        try:
            if not st.session_state.get("adv_preview_decomp"):
                st.warning("Click Preview first.")
            else:
                preview_skinny = [{
                    "name": "{}_composite_preview".format(base.get("name", "base")),
                    "type": "composite",
                    "dataset": base.get("dataset"),
                    "target": base.get("target"),
                    "metrics": base.get("metrics", {}),
                    "decomp": st.session_state["adv_preview_decomp"],
                    "features": base.get("features", []),
                    "_path": "(preview)",
                    "_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }]
                st.session_state["budget_candidates"] = preview_skinny
                st.success("Preview sent to Budget Optimization.")
        except Exception as e:
            st.error("Send failed: {}".format(e))

st.divider()
if st.button("Reload catalog"):
    st.rerun()
