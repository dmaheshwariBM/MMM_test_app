import streamlit as st
import pandas as pd
import numpy as np
import os
from core import transforms

st.title("🔧 Transformations")
os.makedirs("data", exist_ok=True)

files = [f for f in os.listdir("data") if f.lower().endswith(".csv")]
file_choice = st.selectbox("Select CSV to transform", [None] + sorted(files), index=0)
if not file_choice:
    st.info("Select a CSV produced by the Upload page (preferably a `__typed.csv`).")
    st.stop()

path = os.path.join("data", file_choice)
try:
    df = pd.read_csv(path, low_memory=False)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

st.dataframe(df.head(), use_container_width=True)

col = st.selectbox("Column", df.columns)
method = st.selectbox("Method", ["Lag","Adstock","Saturation","Log","Scale"])

df_out = df.copy()
if method == "Lag":
    k = st.slider("Lag periods", 1, 12, 1)
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.error(f"Column '{col}' must be numeric for Lag.")
        st.stop()
    df_out[f"{col}_lag{str(k)}"] = transforms.lag(df[col], k)

elif method == "Adstock":
    a = st.slider("Alpha", 0.0, 0.95, 0.5, 0.05)
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.error(f"Column '{col}' must be numeric for Adstock.")
        st.stop()
    df_out[f"{col}_adstock{str(a)}"] = transforms.adstock(df[col], a)

elif method == "Saturation":
    k = st.slider("k", 0.1, 3.0, 1.0, 0.1)
    theta = st.slider("theta", 0.1, 5.0, 1.0, 0.1)
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.error(f"Column '{col}' must be numeric for Saturation.")
        st.stop()
    df_out[f"{col}_sat"] = transforms.saturation(df[col], k, theta)

elif method == "Log":
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.error(f"Column '{col}' must be numeric for Log.")
        st.stop()
    df_out[f"{col}_log"] = transforms.log_transform(df[col])

elif method == "Scale":
    if not pd.api.types.is_numeric_dtype(df[col]):
        st.error(f"Column '{col}' must be numeric for Scale.")
        st.stop()
    df_out[f"{col}_scale"] = transforms.scale(df[col])

st.subheader("Preview")
st.dataframe(df_out.tail(), use_container_width=True)

save_as = st.text_input("Save as (CSV)", value=f"{os.path.splitext(file_choice)[0]}__tfm.csv")
if st.button("Save"):
    try:
        out = save_as if save_as.lower().endswith(".csv") else save_as + ".csv"
        df_out.to_csv(os.path.join("data", out), index=False)
        st.success(f"Saved to data/{out}")
    except Exception as e:
        st.error(f"Failed to save: {e}")
