# pages/5_Advanced_Models.py
import os, json, glob, re, itertools
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from core import advanced_models

st.title("🧠 Advanced Models — Breakout • Pathway (Interactions) • Residual")

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
                rows.append({
                    "batch_dir": os.path.basename(b),
                    "path": jf,
                    "name": r.get("name",""),
                    "type": r.get("type","base"),
                    "dataset": r.get("dataset",""),
                    "target": r.get("target",""),
                    "features": r.get("features", []),
                    "metrics": r.get("metrics", {}),
                })
            except Exception:
                continue
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
            if str(rec.get("name","")).strip().lower() == base_name.strip().lower():
                keep = {k: rec.get(k) for k in ("batch_ts","name","target","dataset")}
                with open(jf, "w") as g:
                    json.dump({**keep, **new_record}, g, indent=2)
                return
        except Exception:
            continue

def _render_decomp_summary(decomp: Dict[str, Any]):
    if not decomp: return
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base %", f"{decomp.get('base_pct',0):.1f}%")
    with c2: st.metric("Carryover %", f"{decomp.get('carryover_pct',0):.1f}%")
    with c3: st.metric("Incremental %", f"{decomp.get('incremental_pct',0):.1f}%")
    imp = decomp.get("impactable_pct", {})
    if imp:
        dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())})
        st.bar_chart(dfv.set_index("Channel"))

# ---------------------------
# Pick base model to extend
# ---------------------------
cat = _load_results_catalog()
if not cat:
    st.info("No saved models found. Build a base model in **Modeling** first.")
    st.stop()

base_options = [f"{r['name']}  —  ({r['dataset']}  /  target: {r['target']})" for r in cat]
sel = st.selectbox("Base model to extend", options=base_options, index=0)
base = cat[base_options.index(sel)]

# Load base data
try:
    df = _load_dataset_csv(base["dataset"])
except Exception as e:
    st.error(f"Could not open dataset '{base['dataset']}' referenced by base model.")
    st.exception(e)
    st.stop()

target = base["target"]
features = [c for c in base["features"] if c in df.columns]

st.caption(f"Using dataset: **{base['dataset']}**  •  Target: **{target}**  •  Base features: {', '.join(features)}")

# ---------------------------
# Tabs per advanced type
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Breakout model", "🧩 Pathway (Interactions)", "➕ Residual model"])

# ===== Breakout =====
with tab1:
    st.subheader("Breakout (grouped) models")
    cat_like = [c for c in df.columns if (df[c].dtype == "object" or df[c].nunique() <= 50)]
    group_col = st.selectbox("Group by", options=cat_like or list(df.columns))
    min_n = st.number_input("Minimum rows per group", min_value=10, value=30, step=5)
    add_const = st.checkbox("Add intercept", value=True)
    force_nn = st.checkbox("Force non-negative coefficients", value=False,
                           help="If enabled, constrained fit zeroes negative coefs.")
    run_break = st.button("Run breakout models", type="primary")

    if run_break:
        try:
            out = advanced_models.run_breakout_models(
                df, target, features, group_col,
                dataset_csv_name=base["dataset"],
                min_group_n=int(min_n),
                add_const=add_const,
                force_nonnegative=force_nn
            )
            # summarize
            rows = []
            for g, rec in out["results"].items():
                m = rec["metrics"]; d = rec.get("decomp", {})
                rows.append({
                    "Group": g, "n": m.get("n"), "R²": m.get("r2"),
                    "Base %": d.get("base_pct"), "Carryover %": d.get("carryover_pct"),
                    "Incremental %": d.get("incremental_pct"), "RMSE": m.get("rmse")
                })
            if rows:
                st.dataframe(pd.DataFrame(rows).sort_values("R²", ascending=False), use_container_width=True)
            # quick inspect decomp of top group
            if rows:
                best_g = max(out["results"].keys(), key=lambda gg: out["results"][gg]["metrics"].get("r2", -1))
                st.markdown(f"**Top group by R²:** {best_g}")
                _render_decomp_summary(out["results"][best_g].get("decomp", {}))

            # save pack
            name_new = st.text_input("Name for breakout pack", value=f"{base['name']}__breakout__{group_col}")
            if st.button("Save breakout results"):
                where = _persist_record(name_new, target, base["dataset"], {
                    "type": "breakout",
                    "group_col": group_col,
                    "base_name": base["name"],
                    "base_features": features,
                    "results_per_group": out["results"],
                    "metrics": {"note": "see per-group"},
                })
                st.success(f"Saved breakout pack in `{where}`.")

            # overwrite base with chosen group
            st.markdown("—")
            st.caption("Optionally overwrite the base model with one group's fit.")
            if rows:
                pick = st.selectbox("Choose group to overwrite base with", options=list(out["results"].keys()))
                if st.button("Overwrite base model with chosen group"):
                    rec = out["results"][str(pick)]
                    overwrite_payload = {
                        "type": "breakout_overwrite",
                        "metrics": rec["metrics"],
                        "coef": rec["coef"],
                        "yhat": rec["yhat"],
                        "features": rec["features"],
                        "decomp": rec.get("decomp", {})
                    }
                    _overwrite_base(base["name"], overwrite_payload)
                    st.success(f"Base model **{base['name']}** overwritten using group **{pick}**.")
        except Exception as e:
            st.error("Breakout modeling failed.")
            st.exception(e)

# ===== Pathway / Interactions =====
with tab2:
    st.subheader("Pathway / Interaction model")
    st.caption("Add interaction (product) terms between drivers to capture synergies or cannibalization.")
    cols = [c for c in features]
    k = st.number_input("Suggest top-K interactions by |corr(product, target)|", min_value=0, value=5, step=1)
    suggested = []
    if st.button("Suggest interactions"):
        try:
            suggested = advanced_models.suggest_interactions_by_corr(df, target, cols, top_k=int(k))
            st.info("Suggested pairs: " + (", ".join([f"{a}×{b}" for a,b in suggested]) if suggested else "None"))
        except Exception as e:
            st.warning(f"Suggestion failed: {e}")

    all_pairs = list(itertools.combinations(cols, 2))
    def _fmt_pair(p): return f"{p[0]} × {p[1]}"
    label_pairs = [_fmt_pair(p) for p in all_pairs]
    preselect = [label_pairs[all_pairs.index(p)] for p in suggested] if suggested else []
    chosen_labels = st.multiselect("Choose interaction pairs", options=label_pairs, default=preselect)
    chosen_pairs = [all_pairs[label_pairs.index(lbl)] for lbl in chosen_labels] if chosen_labels else []

    add_const_ix = st.checkbox("Add intercept", value=True)
    force_nn_ix = st.checkbox("Force non-negative coefficients", value=False)
    run_ix = st.button("Run interaction model", type="primary")

    if run_ix:
        try:
            out = advanced_models.run_interaction_model(
                df, target, features, chosen_pairs,
                dataset_csv_name=base["dataset"],
                add_const=add_const_ix, force_nonnegative=force_nn_ix
            )
            m = out["metrics"]
            st.write(pd.DataFrame([m]))
            _render_decomp_summary(out.get("decomp", {}))
            st.write("Interaction terms added:", [f"{a}×{b}" for a,b in chosen_pairs])

            name_new = st.text_input("Name for interaction model", value=f"{base['name']}__interactions_{len(chosen_pairs)}")
            if st.button("Save interaction model"):
                payload = {
                    "type": "interaction",
                    "metrics": m,
                    "coef": out["coef"],
                    "yhat": out["yhat"],
                    "features": out["features"],
                    "interaction_pairs": out["interaction_pairs"],
                    "decomp": out.get("decomp", {}),
                    "base_name": base["name"],
                }
                where = _persist_record(name_new, target, base["dataset"], payload)
                st.success(f"Saved interaction model to `{where}`.")
            if st.button("Overwrite base model with this interaction model"):
                overwrite_payload = {
                    "type": "interaction_overwrite",
                    "metrics": m,
                    "coef": out["coef"],
                    "yhat": out["yhat"],
                    "features": out["features"],
                    "decomp": out.get("decomp", {}),
                }
                _overwrite_base(base["name"], overwrite_payload)
                st.success(f"Base model **{base['name']}** overwritten with interaction model.")
        except Exception as e:
            st.error("Interaction modeling failed.")
            st.exception(e)

# ===== Residual =====
with tab3:
    st.subheader("Residual (two-stage) model")
    st.caption("Stage 1 uses the base features; Stage 2 models residuals with additional features and optional AR lags.")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    extra_candidates = [c for c in num_cols if c not in set(features+[target])]
    resid_feats = st.multiselect("Residual-stage features", options=extra_candidates, default=[])
    ar_lags = st.number_input("AR lags of residuals (0–6)", min_value=0, max_value=6, value=1, step=1)
    force_nn1 = st.checkbox("Force non-negative (Stage 1)", value=False)
    force_nn2 = st.checkbox("Force non-negative (Stage 2)", value=False)
    run_resid = st.button("Run residual model", type="primary")

    if run_resid:
        try:
            out = advanced_models.run_residual_model(
                df, target,
                base_features=features,
                residual_features=resid_feats,
                dataset_csv_name=base["dataset"],
                ar_lags=int(ar_lags),
                force_nonnegative_base=force_nn1,
                force_nonnegative_resid=force_nn2,
            )
            m = out["metrics"]
            st.write(pd.DataFrame([m]))
            _render_decomp_summary(out.get("decomp", {}))
            st.json({"stage1_metrics": out["stage1"]["metrics"], "stage2_metrics": out["stage2"]["metrics"]})

            name_new = st.text_input("Name for residual model", value=f"{base['name']}__residual_aug")
            if st.button("Save residual model"):
                payload = {
                    "type": "residual",
                    "metrics": m,
                    "yhat": out["yhat"],
                    "stage1": out["stage1"],
                    "stage2": out["stage2"],
                    "features_stage1": out["features_stage1"],
                    "features_stage2": out["features_stage2"],
                    "decomp": out.get("decomp", {}),
                    "base_name": base["name"],
                }
                where = _persist_record(name_new, target, base["dataset"], payload)
                st.success(f"Saved residual model to `{where}`.")
            if st.button("Overwrite base model with this residual model"):
                overwrite_payload = {
                    "type": "residual_overwrite",
                    "metrics": m,
                    "coef": out["stage1"]["coef"],  # keep stage1 coefs as 'base' view
                    "yhat": out["yhat"],
                    "features": out["features_stage1"],
                    "decomp": out.get("decomp", {}),
                }
                _overwrite_base(base["name"], overwrite_payload)
                st.success(f"Base model **{base['name']}** overwritten with residual-augmented predictions.")
        except Exception as e:
            st.error("Residual modeling failed.")
            st.exception(e)
