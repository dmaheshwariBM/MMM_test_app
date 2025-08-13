import streamlit as st
import pandas as pd
import numpy as np
import os
from core import optimize

st.title("🧭 Budget Optimization")

if "mmm_model" not in st.session_state:
    st.info("Train a model first on the Modeling page.")
else:
    dataset = st.session_state.get("mmm_dataset","master.csv")
    df = pd.read_csv(os.path.join("data", dataset))
    X_cols = st.session_state["mmm_X_cols"]

    st.caption("Assume Hill-type response per selected channels (toy params).")
    channels = st.multiselect("Channels to optimize", X_cols, default=X_cols[:min(4,len(X_cols))])
    total_budget = st.number_input("Total budget to allocate", value=200000.0, step=5000.0)

    vmax = {c: st.number_input(f"{c} vmax (impact cap)", value=10000.0, step=1000.0) for c in channels}
    k = {c: st.number_input(f"{c} k (curvature)", value=1.0, step=0.1) for c in channels}
    theta = {c: st.number_input(f"{c} theta (half-sat)", value=50000.0, step=1000.0) for c in channels}
    minmax = {c: (0.0, total_budget) for c in channels}

    if st.button("Optimize"):
        alloc, val, res = optimize.optimize_budget_hill(channels, vmax, k, theta, total_budget, minmax)
        st.subheader("Optimized Allocation")
        out = pd.DataFrame({"Channel": channels, "Spend": alloc})
        st.dataframe(out, use_container_width=True)
        st.write(f"Expected KPI (objective): {val:.2f}")
