# pages/4_Modeling.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import json
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from core import modeling

st.title("📈 Modeling — Batch Runner (NNLS option)")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Helpers ----------
def _list_csvs() -> List[str]:
    if not os.path.isdir(DATA_DIR): return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False

def _load_transforms_meta(dataset_csv: str) -> Optional[Dict[str, Any]]:
    """Load transforms_<dataset_stem>.json if present (for Carryover %)."""
    stem = os.path.splitext(dataset_csv)[0]
    meta_path = os.path.join(DATA_DIR, f"transforms_{stem}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ---------- Data pick ----------
files = _list_csvs()
if not files:
    st.info("No CSV files in `data/`. Save a dataset from Transformations first.")
    st.stop()

default_name = st.session_state.get("mmm_current_dataset")
ds_index = files.index(default_name) if default_name in files else 0
dataset = st.selectbox("Dataset (CSV)", files, index=ds_index)
df = pd.read_csv(os.path.join(DATA_DIR, dataset))
st.caption(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# load transforms meta for carryover %
transforms_meta = _load_transforms_meta(dataset)

# usable columns
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
default_target = st.session_state.get("mmm_target")
target_default_idx = numeric_cols.index(default_target) if default_target in numeric_cols else (
    numeric_cols.index("Sales") if "Sales" in numeric_cols else 0
)

# ---------- Global options ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    add_const = st.checkbox("Add intercept (const)", value=True)
with c2:
    compute_vif = st.checkbox("Compute VIF", value=True, help="Check multicollinearity among features.")
with c3:
    force_nonneg = st.checkbox("Force negative estimates to 0", value=True,
                               help="When ON: use NNLS (if available) or clamp to ≥0; metrics come from constrained predictions.")
with c4:
    have_sklearn = _has_sklearn()
    st.caption(("scikit-learn detected ✓" if have_sklearn else "scikit-learn not available — OLS only"))

MODEL_TYPES = ["OLS"] + (["Ridge", "Lasso"] if have_sklearn else [])

# ---------- Session state ----------
if "model_queue" not in st.session_state:
    st.session_state["model_queue"] = []
if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}

# ---------- Add Model (form) ----------
st.subheader("Configure models")
with st.expander("➕ Add a model", expanded=False):
    with st.form("add_model_form", clear_on_submit=False):
        model_name = st.text_input("Model name", value=f"Model {len(st.session_state['model_queue'])+1}")
        target = st.selectbox("Target (numeric)", numeric_cols, index=target_default_idx, key="target_add")

        tfm_cols = [c for c in df.columns if c.endswith("__tfm")]
        feature_candidates = [c for c in numeric_cols if c != target]
        default_feats = tfm_cols if tfm_cols else feature_candidates
        features = st.multiselect("Feature columns", options=feature_candidates, default=default_feats)

        mtype = st.selectbox("Model type", MODEL_TYPES, index=0)
        alpha = None
        if mtype in ("Ridge", "Lasso"):
            alpha = st.number_input("Regularization strength (alpha)", min_value=0.0, value=1.0, step=0.1)

        submitted = st.form_submit_button("Add to queue")
        if submitted:
            if not features:
                st.error("Select at least one feature.")
            elif len(st.session_state["model_queue"]) >= 10:
                st.error("You can add up to 10 models.")
            else:
                spec = {
                    "id": f"{len(st.session_state['model_queue'])+1}_{datetime.now().strftime('%H%M%S')}",
                    "name": model_name.strip() or f"Model {len(st.session_state['model_queue'])+1}",
                    "dataset": dataset,
                    "target": target,
                    "features": features,
                    "type": mtype,
                    "alpha": float(alpha) if alpha is not None else None,
                    "add_const": bool(add_const),
                    "compute_vif": bool(compute_vif),
                }
                st.session_state["model_queue"].append(spec)
                st.success(f"Added **{spec['name']}** with {len(features)} feature(s).")

# ---------- Queue table & controls ----------
if not st.session_state["model_queue"]:
    st.info("No models in the queue yet. Add at least one model above.")
else:
    st.write("### Models to run")
    queue_rows = []
    for i, spec in enumerate(st.session_state["model_queue"], start=1):
        queue_rows.append({
            "Order": i, "Name": spec["name"], "Type": spec["type"], "Target": spec["target"],
            "#Features": len(spec["features"]),
            "Features": ", ".join(spec["features"][:8]) + (" …" if len(spec["features"]) > 8 else "")
        })
    st.dataframe(pd.DataFrame(queue_rows), use_container_width=True, height=min(420, 60 + 28*len(queue_rows)))

    cols = st.columns(len(st.session_state["model_queue"]) + 1)
    with cols[0]:
        if st.button("🗑️ Clear all"):
            st.session_state["model_queue"] = []
            st.session_state["model_results"] = {}
            st.toast("Cleared all queued models.")
    for idx, spec in enumerate(st.session_state["model_queue"], start=1):
        with cols[idx]:
            if st.button(f"Delete “{spec['name']}”", key=f"del_{spec['id']}"):
                st.session_state["model_queue"].pop(idx-1)
                st.session_state["model_results"].pop(spec["id"], None)
                st.toast(f"Deleted {spec['name']}")
                st.rerun()

# ---------- Run all models ----------
st.divider()
if st.button("🚀 Run all models", disabled=not st.session_state["model_queue"]):
    st.session_state["model_results"] = {}
    try:
        for spec in st.session_state["model_queue"]:
            # Prepare
            X_df, y, _ = modeling.prepare_xy(df, spec["target"], spec["features"], fillna=0.0)

            # Fit
            mtype = spec["type"]
            if mtype == "OLS":
                res = modeling.ols_model(
                    X_df, y,
                    add_constant=spec["add_const"],
                    compute_vif_flag=spec["compute_vif"],
                    force_nonnegative=force_nonneg
                )
            else:
                try:
                    if mtype == "Ridge":
                        res = modeling.ridge_model(
                            X_df, y, alpha=spec["alpha"] or 1.0,
                            add_constant=spec["add_const"],
                            compute_vif_flag=spec["compute_vif"],
                            force_nonnegative=force_nonneg
                        )
                    elif mtype == "Lasso":
                        res = modeling.lasso_model(
                            X_df, y, alpha=spec["alpha"] or 1.0,
                            add_constant=spec["add_const"],
                            compute_vif_flag=spec["compute_vif"],
                            force_nonnegative=force_nonneg
                        )
                    else:
                        res = modeling.ols_model(
                            X_df, y,
                            add_constant=spec["add_const"],
                            compute_vif_flag=spec["compute_vif"],
                            force_nonnegative=force_nonneg
                        )
                except Exception:
                    # Safe fallback → OLS
                    res = modeling.ols_model(
                        X_df, y,
                        add_constant=spec["add_const"],
                        compute_vif_flag=spec["compute_vif"],
                        force_nonnegative=force_nonneg
                    )

            # Decomposition (Base %, Carryover %, Impactable % by channel)
            decomp = modeling.impact_decomposition(
                y=y, yhat=res["yhat"], coef=res["coef"], X_df=X_df,
                add_constant=spec["add_const"], transforms_meta=_load_transforms_meta(spec["dataset"])
            )

            st.session_state["model_results"][spec["id"]] = {
                "spec": spec, "result": res, "decomp": decomp,
                "X_df": X_df, "y": y,
            }
        st.success(f"Ran {len(st.session_state['model_results'])} model(s).")
    except Exception as e:
        st.error(f"Batch run failed: {e}")
        st.exception(e)
        st.code(traceback.format_exc())

# ---------- Summary grid ----------
if st.session_state["model_results"]:
    st.subheader("📊 Model comparison")

    # union of all channels across models (exclude const)
    all_channels = set()
    for _, blob in st.session_state["model_results"].items():
        all_channels.update(blob["decomp"]["impactable_pct"].index.tolist())
    all_channels = sorted(all_channels)

    # build summary rows
    rows = []
    for _, blob in st.session_state["model_results"].items():
        spec = blob["spec"]; res = blob["result"]; dc = blob["decomp"]
        m = res["metrics"]
        row = {
            "Name": spec["name"], "Type": spec["type"],
            "R²": round(m["r2"], 6) if m["r2"]==m["r2"] else None,
            "Adj R²": round(m["adj_r2"], 6) if m["adj_r2"]==m["adj_r2"] else None,
            "RMSE": round(m["rmse"], 6) if m["rmse"]==m["rmse"] else None,
            "Base %": round(dc["base_pct"], 4),
            "Carryover %": round(dc["carryover_pct"], 4),
        }
        ip = dc["impactable_pct"]  # sums to 100 across channels
        for ch in all_channels:
            row[f"Impactable % • {ch}"] = round(float(ip.get(ch, np.nan)), 4) if ch in ip.index else np.nan
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True)

    with io.BytesIO() as buf:
        summary_df.to_csv(buf, index=False)
        st.download_button("⬇️ Download comparison CSV", data=buf.getvalue(), file_name="model_comparison.csv", mime="text/csv")

    st.divider()

    # ---------- Per-model details ----------
    st.subheader("🔎 Per-model details")
    for _, blob in st.session_state["model_results"].items():
        spec = blob["spec"]; res = blob["result"]; dc = blob["decomp"]

        with st.expander(f"Details — {spec['name']} ({spec['type']})", expanded=False):
            m = res["metrics"]
            st.write({
                "R²": m["r2"], "Adj R²": m["adj_r2"], "RMSE": m["rmse"],
                "MAE": m["mae"], "MAPE": m["mape"], "AIC": m["aic"], "BIC": m["bic"],
                "n": m["n"], "p": m["p"], "df_resid": m["df_resid"]
            })

            coef_df = pd.concat([res["coef"], res["stderr"], res["tvalues"], res["pvalues"]], axis=1)
            coef_df.columns = ["coef", "std_err", "t", "p_value"]
            if force_nonneg:
                st.info("Non-negative constraint is ON: std_err / t / p_value are not shown (NaN).")

            st.markdown("**Coefficients**")
            st.dataframe(coef_df.style.format(precision=6), use_container_width=True)

            if res.get("vif") is not None:
                st.markdown("**VIF**")
                st.dataframe(res["vif"].rename("VIF").to_frame(), use_container_width=True)

            preds = pd.DataFrame({"y": blob["y"], "yhat": res["yhat"], "residual": res["residuals"]})
            st.markdown("**Diagnostics**")
            st.line_chart(res["residuals"])
            st.scatter_chart(preds.rename(columns={"y": "Actual", "yhat": "Predicted"})[["Actual", "Predicted"]])

            # Decomposition blocks
            st.markdown("**Decomposition**")
            st.write({
                "Base % (intercept)": round(dc["base_pct"], 4),
                "Incremental % (channels)": round(dc["incremental_pct"], 4),
                "Carryover % (of total, est.)": round(dc["carryover_pct"], 4),
            })
            st.markdown("**Impactable % by channel (sums to 100%)**")
            st.dataframe(dc["impactable_pct"].to_frame(), use_container_width=True)

            # Export to Excel (per-model)
            with io.BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    coef_df.to_excel(writer, sheet_name="Coefficients")
                    pd.DataFrame([m]).to_excel(writer, sheet_name="Metrics", index=False)
                    preds.to_excel(writer, sheet_name="Predictions", index=False)
                    if res.get("vif") is not None:
                        res["vif"].rename("VIF").to_frame().to_excel(writer, sheet_name="VIF")
                    # decomposition
                    pd.DataFrame([{
                        "Base %": dc["base_pct"],
                        "Incremental %": dc["incremental_pct"],
                        "Carryover % (of total)": dc["carryover_pct"],
                        "Denominator (base+inc)": dc["denominator_total"],
                    }]).to_excel(writer, sheet_name="Decomposition", index=False)
                    dc["impactable_pct"].rename("Impactable %").to_frame().to_excel(writer, sheet_name="ImpactablePct")
                st.download_button(
                    label=f"⬇️ Download {spec['name']} Excel",
                    data=buffer.getvalue(),
                    file_name=f"{spec['name'].replace(' ','_')}_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
