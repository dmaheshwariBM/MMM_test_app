import streamlit as st
import pandas as pd
import os
from core import export

st.title("⬇   Export")

files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.isdir("data") else []
file_choice = st.selectbox("Pick dataset to export", files, index=files.index("master.csv") if "master.csv" in files else 0 if files else None)

if file_choice:
    df = pd.read_csv(os.path.join("data", file_choice))
    if st.button("Build and download Excel pack"):
        # Build sheets (extend as needed)
        export.export_to_excel("mmm_results.xlsx", Data=df)
        st.success("Created mmm_results.xlsx in app working directory.")
        st.download_button("Download mmm_results.xlsx", data=open("mmm_results.xlsx","rb").read(), file_name="mmm_results.xlsx")
