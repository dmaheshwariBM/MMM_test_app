import streamlit as st
import pandas as pd
import numpy as np
import os
from core import optimize

st.title("🧭 Budget Optimization")

if "mmm_model" not in st.session_state or "mmm_X_cols" not in st.session_state:
    st.info("Train a model first on the Modeling page.")
    st.stop()

dataset = st.session_state.get("mmm_dataset")
if not dataset or not os.path.exists(os.path.join("data", dataset)):
    st.error("The dataset used for the model is missing. Re-run Modeling.")
    st.stop()

df = pd.read_csv(os.path.join("data", dataset), low_memory=False)
X_cols = st.session_state["mmm_X_cols"]

st.caption("Assume Hill-type response per selected channels. Tune params and optimize.")
channels = st.multiselect("Channels to optimize", X_cols, default=X_cols[:min(4, len(X_cols))])
if not channels:
    st.warning("Select at least one channel.")
    st.stop()

total_budget = st.number_input("Total budget to allocate", value=200000.0, step=5000.0, min_value=0.0)
if total_budget <= 0:
    st.error("Total budget must be positive.")
    st.stop()

vmax = {c: st.number_input(f"{c} vmax (impact cap)", value=10000.0, step=1000.0) for c in channels}
k = {c: st.number_input(f"{c} k (curvature)", value=1.0, step=0.1) for c in channels}
theta = {c: st.number_input(f"{c} theta (half-sat)", value=50000.0, step=1000.0) for c in channels}
minmax = {c: (0.0, total_budget) for c in channels}

if st.button("Optimize"):
    try:
        alloc, val, res = optimize.optimize_budget_hill(channels, vmax, k, theta, total_budget, minmax)
        out = pd.DataFrame({"Channel": channels, "Spend": alloc})
        st.subheader("Optimized Allocation")
        st.dataframe(out, use_container_width=True)
        st.write(f"Expected KPI (objective): {val:.2f}")
    except Exception as e:
        st.error(f"Optimization failed: {e}")
