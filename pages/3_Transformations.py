import streamlit as st
import pandas as pd
import os
from core import transforms

st.title("🔧 Transformations")

files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.isdir("data") else []
file_choice = st.selectbox("Select CSV to transform", files)
if file_choice:
    path = os.path.join("data", file_choice)
    df = pd.read_csv(path)
    st.dataframe(df.head(), use_container_width=True)
    col = st.selectbox("Column", df.columns)
    method = st.selectbox("Method", ["Lag","Adstock","Saturation","Log","Scale"])
    if method=="Lag":
        k = st.slider("Lag periods",1,12,1)
        df[col+"_lag"] = transforms.lag(df[col], k)
    elif method=="Adstock":
        a = st.slider("Alpha",0.0,0.9,0.5,0.05)
        df[col+"_adstock"] = transforms.adstock(df[col], a)
    elif method=="Saturation":
        k = st.slider("k",0.1,2.5,1.0,0.1)
        theta = st.slider("theta",0.1,2.5,1.0,0.1)
        df[col+"_sat"] = transforms.saturation(df[col], k, theta)
    elif method=="Log":
        df[col+"_log"] = transforms.log_transform(df[col])
    else:
        df[col+"_scale"] = transforms.scale(df[col])
    st.dataframe(df.tail(), use_container_width=True)
    if st.button("Save back"):
        df.to_csv(path, index=False)
        st.success(f"Saved transformed data to {path}")
