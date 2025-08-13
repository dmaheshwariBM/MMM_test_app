import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from core import modeling

st.title("🧠 Modeling")

os.makedirs("data", exist_ok=True)

files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.isdir("data") else []
file_choice = st.selectbox("Dataset (CSV)", files, index=files.index("master.csv") if "master.csv" in files else 0 if files else None)

if file_choice:
    df = pd.read_csv(os.path.join("data", file_choice))
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox("Target variable", df.columns, index=list(df.columns).index("Sales") if "Sales" in df.columns else 0)
    feature_candidates = [c for c in df.columns if c != target]
    features = st.multiselect("Features", feature_candidates, default=[c for c in feature_candidates if c.lower() not in ("week","date")])

    if features:
        X = df[features].select_dtypes(include=[np.number]).fillna(0)
        y = df[target].astype(float)

        model_type = st.selectbox("Model", ["OLS","Ridge (CV)","Lasso (CV)"])

        if model_type=="OLS":
            model, metrics, yhat = modeling.ols_model(X,y)
        elif model_type=="Ridge (CV)":
            model, metrics, yhat = modeling.ridge_model_cv(X,y)
        else:
            model, metrics, yhat = modeling.lasso_model_cv(X,y)

        st.subheader("Metrics")
        st.json(metrics)

        st.subheader("Coefficients & Contributions")
        contrib = modeling.contributions(model, X)
        st.dataframe(contrib, use_container_width=True)

        st.subheader("VIF (multicollinearity)")
        try:
            vif = modeling.compute_vif(X)
            st.dataframe(vif, use_container_width=True)
        except Exception as e:
            st.info(f"VIF not available: {e}")

        st.subheader("Residuals (actual - predicted)")
        resid = y - yhat
        st.line_chart(resid.reset_index(drop=True))

        # Save artifacts for other pages
        st.session_state["mmm_model"] = model
        st.session_state["mmm_X_cols"] = list(X.columns)
        st.session_state["mmm_contrib"] = contrib
        st.session_state["mmm_dataset"] = file_choice

        # --- NEW: write "last run" metadata for homepage
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
            with open(os.path.join("data", "last_run.json"), "w") as f:
                json.dump(last_run, f, indent=2)
        except Exception as e:
            st.warning(f"Could not persist last run metadata: {e}")

        st.success("Model artifacts + last run metadata stored. Check the **Home** screen.")
