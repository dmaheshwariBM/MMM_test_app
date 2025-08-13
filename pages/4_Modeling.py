import streamlit as st
import pandas as pd
import numpy as np
import os, json
from datetime import datetime
from core import modeling, io

st.title("🧠 Modeling")
os.makedirs("data", exist_ok=True)

all_csv = [f for f in os.listdir("data") if f.lower().endswith(".csv")]
typed_first = sorted([f for f in all_csv if "__typed" in f]) + [f for f in all_csv if "__typed" not in f]
file_choice = st.selectbox("Dataset (CSV)", [None] + typed_first, index=0)
if not file_choice:
    st.info("Pick a typed dataset (e.g., `__typed.csv`) or a clean master CSV.")
    st.stop()

df = pd.read_csv(os.path.join("data", file_choice), low_memory=False)
st.dataframe(df.head(), use_container_width=True)

target = st.selectbox("Target variable", df.columns, index=(list(df.columns).index("Sales") if "Sales" in df.columns else 0))
candidates = [c for c in df.columns if c != target]
features = st.multiselect("Features", candidates, default=[c for c in candidates if c.lower() not in ("week","date")])

if not features:
    st.warning("Select at least one feature.")
    st.stop()

# Validate readiness
ok, errors, warnings, X, y = io.validate_for_modeling(df, target, features)
if errors:
    st.error("Modeling cannot proceed due to:")
    for e in errors:
        st.write(f"• {e}")
    st.stop()
if warnings:
    st.warning("Notes:")
    for w in warnings:
        st.write(f"• {w}")

model_type = st.selectbox("Model", ["OLS","Ridge (CV)","Lasso (CV)"])

if model_type == "OLS":
    model, metrics, yhat = modeling.ols_model(X, y)
elif model_type == "Ridge (CV)":
    model, metrics, yhat = modeling.ridge_model_cv(X, y)
else:
    model, metrics, yhat = modeling.lasso_model_cv(X, y)

st.subheader("Metrics")
st.json(metrics)

st.subheader("Coefficients & Contributions")
contrib = modeling.contributions(model, X)
st.dataframe(contrib, use_container_width=True)

st.subheader("Residuals")
st.line_chart((y - yhat).reset_index(drop=True))

# Persist artifacts + last run
st.session_state["mmm_model"] = model
st.session_state["mmm_X_cols"] = list(X.columns)
st.session_state["mmm_contrib"] = contrib
st.session_state["mmm_dataset"] = file_choice

last_run = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_type": model_type,
    "dataset": file_choice,
    "target": target,
    "features": list(X.columns),
    "metrics": metrics
}
st.session_state["mmm_last_run"] = last_run
try:
    with open(os.path.join("data","last_run.json"), "w") as f:
        json.dump(last_run, f, indent=2)
except Exception as e:
    st.warning(f"Could not persist last run metadata: {e}")

st.success("Model trained successfully. You can proceed to Results or Optimizer.")
