import streamlit as st
import pandas as pd
import os
from core import curves

st.title("📊 Results")

if "mmm_model" not in st.session_state or "mmm_X_cols" not in st.session_state:
    st.info("Train a model in the Modeling page first.")
    st.stop()

dataset = st.session_state.get("mmm_dataset")
if not dataset or not os.path.exists(os.path.join("data", dataset)):
    st.error("The dataset used for the model is missing. Re-run Modeling.")
    st.stop()

df = pd.read_csv(os.path.join("data", dataset), low_memory=False)
X_cols = st.session_state["mmm_X_cols"]
model = st.session_state["mmm_model"]

st.caption(f"Dataset: {dataset}")
st.subheader("Feature Contributions (from model page)")
contrib = st.session_state.get("mmm_contrib")
if contrib is None or contrib.empty:
    st.info("No contribution table found; re-run Modeling.")
else:
    st.dataframe(contrib, use_container_width=True)

num_cols = [c for c in X_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.info("No numeric features available for response curves.")
    st.stop()

sel = st.selectbox("Feature for response curve", num_cols)
coef_map = {c: float(model.coef_[i]) for i, c in enumerate(X_cols)} if hasattr(model, "coef_") else {}
coef = coef_map.get(sel, 0.0)
curve_df = curves.linear_curve(df[sel], coef, intercept=float(getattr(model, "intercept_", 0.0)))
st.line_chart(curve_df.set_index("Spend"))
