import streamlit as st
import pandas as pd
import numpy as np
import os
from core import curves

st.title("📊 Results")

if "mmm_model" not in st.session_state:
    st.info("Train a model in the Modeling page first.")
else:
    dataset = st.session_state.get("mmm_dataset","master.csv")
    df = pd.read_csv(os.path.join("data", dataset))
    X_cols = st.session_state["mmm_X_cols"]
    model = st.session_state["mmm_model"]
    st.caption(f"Dataset: {dataset}; Features: {', '.join(X_cols)}")

    # Contributions table from Modeling page (already computed)
    contrib = st.session_state.get("mmm_contrib")
    if contrib is not None:
        st.subheader("Channel/Feature Contributions")
        st.dataframe(contrib, use_container_width=True)

    # Response curves for any selected numeric feature
    num_cols = [c for c in X_cols if pd.api.types.is_numeric_dtype(df[c])]
    sel = st.selectbox("Select feature for response curve", num_cols)

    # Get its coefficient (0 if not present)
    coef_map = {c:float(model.coef_[i]) for i,c in enumerate(X_cols)} if hasattr(model,"coef_") else {}
    coef = coef_map.get(sel, 0.0)
    curve_df = curves.linear_curve(df[sel], coef, intercept=float(getattr(model, 'intercept_', 0.0)))
    st.line_chart(curve_df.set_index("Spend"))
