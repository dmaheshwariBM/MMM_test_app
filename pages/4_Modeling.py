# pages/4_Modeling.py
# v2.3.0  ASCII-only. Queue multiple model specs, run, and robustly save JSON.

import os
import re
import glob
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core import modeling

PAGE_ID = "MODELING_PAGE_v2_3_0"
st.title("Modeling")
st.caption("Page ID: {}".format(PAGE_ID))

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Writable results root (same policy as Results/Advanced) ----------
def _abs(p: str) -> str:
    return os.path.abspath(p)

CANDIDATE_RESULTS_ROOTS: List[str] = []
_env = os.environ.get("MMM_RESULTS_DIR")
if _env:
    CANDIDATE_RESULTS_ROOTS.append(_abs(_env))
CANDIDATE_RESULTS_ROOTS.append(_abs(os.path.expanduser("~/.mmm_results")))
CANDIDATE_RESULTS_ROOTS.append(_abs("/tmp/mmm_results"))
CANDIDATE_RESULTS_ROOTS.append(_abs("results"))

def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        testf = os.path.join(path, ".write_test")
        with open(testf, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(testf)
        return True
    except Exception:
        return False

def _pick_results_root() -> str:
    for root in CANDIDATE_RESULTS_ROOTS:
        if _ensure_dir(root):
            return root
    fb = _abs(os.path.expanduser("~/mmm_results_fallback"))
    _ensure_dir(fb)
    return fb

RESULTS_ROOT = _pick_results_root()
st.caption("Saving models under: {}".format(RESULTS_ROOT))

# Persisted banners for last save status
if "last_saved_path" in st.session_state and st.session_state["last_saved_path"]:
    st.success("Saved: {}".format(st.session_state["last_saved_path"]))
if "last_save_error" in st.session_state and st.session_state["last_save_error"]:
    st.error(st.session_state["last_save_error"])

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:72]

def _json_ready(obj: Any):
    import numpy as _np
    from datetime import datetime as _dt
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, _dt):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("_ts", "_path"):
                continue
            out[str(k)] = _json_ready(v)
        return out
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    try:
        if isinstance(obj, (_np.floating, _np.integer)):
            return obj.item()
    except Exception:
        pass
    try:
        return float(obj)
    except Exception:
        return str(obj)

def _save_result_json(results_root: str, name: str, target: str, dataset: str, payload: Dict[str, Any]) -> str:
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_root, batch_ts)
    os.makedirs(out_dir, exist_ok=True)
    fname = "{}__{}__{}.json".format(batch_ts, _safe(name), _safe(target))
    body = {"batch_ts": batch_ts, "name": name, "target": target, "dataset": dataset}
    body.update(payload)
    body = _json_ready(body)
    full_path = os.path.join(out_dir, fname)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return full_path

# ---------- Data selection ----------
files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
if not files:
    st.info("No CSV files in data/. Please upload data first.")
    st.stop()

dataset = st.selectbox("Dataset (CSV in data/)", options=sorted(files), index=0)
df = pd.read_csv(os.path.join(DATA_DIR, dataset))

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns found in this dataset.")
    st.stop()

target = st.selectbox("Target column", options=numeric_cols, index=0)
feature_choices = [c for c in numeric_cols if c != target]

st.markdown("Add one or more model specs to a queue, then run all and save.")
m_name = st.text_input("Model name", value="base_model_1")
X_cols = st.multiselect("Independent variables", options=feature_choices)
force_nonneg = st.checkbox("Force negative estimates to 0", value=True)
train_start, train_end = st.slider(
    "Train window (row indices)", min_value=0, max_value=max(1, len(df)-1),
    value=(0, max(1, len(df)-1)), step=1
)

# Queue state
if "model_specs" not in st.session_state:
    st.session_state["model_specs"] = []  # list of dicts
if "model_results" not in st.session_state:
    st.session_state["model_results"] = []  # parallel results

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("Add to queue"):
        if not m_name.strip():
            st.warning("Please provide a model name.")
        elif not X_cols:
            st.warning("Select at least one independent variable.")
        else:
            spec = {
                "name": m_name.strip(),
                "dataset": dataset,
                "target": target,
                "features": list(X_cols),
                "force_nonnegative": bool(force_nonneg),
                "train_window": [int(train_start), int(train_end)],
            }
            # prevent duplicates by name
            names_in_queue = {s["name"] for s in st.session_state["model_specs"]}
            if spec["name"] in names_in_queue:
                st.warning("A spec with this model name already exists in the queue.")
            else:
                st.session_state["model_specs"].append(spec)
with c2:
    if st.button("Clear queue"):
        st.session_state["model_specs"] = []
        st.session_state["model_results"] = []
with c3:
    rm_idx = st.number_input("Remove index", min_value=1, max_value=max(1, len(st.session_state["model_specs"])), value=1, step=1)
with c4:
    if st.button("Remove item"):
        i = int(rm_idx) - 1
        if 0 <= i < len(st.session_state["model_specs"]):
            st.session_state["model_specs"].pop(i)
            if i < len(st.session_state["model_results"]):
                st.session_state["model_results"].pop(i)

# Show queue
if st.session_state["model_specs"]:
    st.subheader("Queued model specs")
    st.dataframe(pd.DataFrame(st.session_state["model_specs"]), use_container_width=True)

# ---------- Run models ----------
def _run_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    ds = spec["dataset"]
    tgt = spec["target"]
    feats = spec["features"]
    fneg = spec["force_nonnegative"]
    w0, w1 = spec["train_window"]
    df_local = pd.read_csv(os.path.join(DATA_DIR, ds))
    df_local = df_local.iloc[w0:w1+1].copy()
    X_df = modeling.build_design(df_local, feats)
    y = pd.to_numeric(df_local[tgt], errors="coerce").fillna(0.0)
    coef, metrics, yhat = modeling.ols_model(X_df, y, force_nonnegative=fneg)
    # standard payload expected by Results page
    res = {
        "name": spec["name"],
        "type": "base",
        "dataset": ds,
        "target": tgt,
        "features": feats,            # do not include "const" here
        "coef": coef,                 # includes "const" inside dict
        "metrics": metrics,
        "yhat": yhat,
    }
    return res

st.divider()
if st.button("Run all models in queue", type="primary"):
    if not st.session_state["model_specs"]:
        st.warning("Queue is empty.")
    else:
        results = []
        for spec in st.session_state["model_specs"]:
            try:
                results.append(_run_one(spec))
            except Exception as e:
                results.append({"name": spec["name"], "error": str(e)})
        st.session_state["model_results"] = results

# Show results summary
if st.session_state["model_results"]:
    st.subheader("Results summary")
    rows = []
    for r in st.session_state["model_results"]:
        if "error" in r:
            rows.append({"name": r["name"], "status": "ERROR: {}".format(r["error"])})
        else:
            m = r.get("metrics", {})
            rows.append({
                "name": r.get("name"),
                "target": r.get("target"),
                "vars": ",".join(r.get("features", [])),
                "r2": m.get("r2"),
                "adj_r2": m.get("adj_r2"),
                "rmse": m.get("rmse"),
                "n": m.get("n"),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Optional: inspect a single model detail
    idx = st.number_input("Inspect result index", min_value=1, max_value=len(st.session_state["model_results"]), value=1, step=1)
    res = st.session_state["model_results"][int(idx)-1]
    if "error" in res:
        st.error("This spec failed: {}".format(res["error"]))
    else:
        st.markdown("Details for: **{}**".format(res["name"]))
        st.json({
            "name": res["name"],
            "target": res["target"],
            "features": res["features"],
            "metrics": res["metrics"],
            "coef": res["coef"],
        })

    st.divider()
    cL, cM, cR = st.columns(3)
    with cL:
        if st.button("Save selected"):
            try:
                res = st.session_state["model_results"][int(idx)-1]
                if "error" in res:
                    st.error("Cannot save errored spec.")
                else:
                    path = _save_result_json(RESULTS_ROOT, res["name"], res["target"], res["dataset"], res)
                    st.session_state["last_saved_path"] = path
                    st.session_state["last_save_error"] = ""
                    st.success("Saved: {}".format(path))
            except Exception as e:
                st.session_state["last_saved_path"] = ""
                st.session_state["last_save_error"] = "Save failed: {}".format(e)
                st.error(st.session_state["last_save_error"])
    with cM:
        if st.button("Save ALL"):
            ok = 0
            try:
                for res in st.session_state["model_results"]:
                    if "error" in res:
                        continue
                    path = _save_result_json(RESULTS_ROOT, res["name"], res["target"], res["dataset"], res)
                    st.session_state["last_saved_path"] = path
                    st.session_state["last_save_error"] = ""
                    ok += 1
                st.success("Saved {} model(s). Last: {}".format(ok, st.session_state.get("last_saved_path","")))
            except Exception as e:
                st.session_state["last_save_error"] = "Save failed: {}".format(e)
                st.error(st.session_state["last_save_error"])
    with cR:
        if st.button("Send selected to Budget Optimization"):
            try:
                res = st.session_state["model_results"][int(idx)-1]
                if "error" in res:
                    st.error("Cannot send errored spec.")
                else:
                    skinny = [{
                        "name": res.get("name"),
                        "type": res.get("type"),
                        "dataset": res.get("dataset"),
                        "target": res.get("target"),
                        "metrics": res.get("metrics", {}),
                        "decomp": {},  # Results page will derive if missing
                        "features": res.get("features", []),
                        "_path": "(session)",
                        "_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }]
                    st.session_state["budget_candidates"] = skinny
                    st.success("Sent to Budget Optimization.")
            except Exception as e:
                st.error("Send failed: {}".format(e))
