import streamlit as st
import pandas as pd
import os
from core import io

st.title("📁 Data Upload")
os.makedirs("data", exist_ok=True)

uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv","xlsx"], accept_multiple_files=True)
if uploaded_files:
    for uf in uploaded_files:
        path = os.path.join("data", uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} files to data/")

st.subheader("Data Quality Reports")
files = [f for f in os.listdir("data") if f.endswith(('.csv','.xlsx'))]
for f in files:
    st.markdown(f"**{f}**")
    df = io.load_csv_or_excel(os.path.join("data", f))
    rep = io.data_quality_report(df)
    if rep.empty:
        st.success("No issues detected")
    else:
        st.dataframe(rep, use_container_width=True)
    st.dataframe(df.head(), use_container_width=True)

# Optional quick merge for sample data
st.divider()
st.caption("Quick merge (optional): If you have spend (Channel,Spend) and sales (Sales) by a common date column (e.g., Week), you can generate a merged master CSV.")
if st.button("Build sample master from sample_spend.csv + sample_sales.csv"):
    spend = io.load_csv_or_excel("data/sample_spend.csv")
    sales = io.load_csv_or_excel("data/sample_sales.csv")
    master = io.merge_spend_sales(spend, sales, date_key="Week")
    master.to_csv("data/master.csv", index=False)
    st.success("Created data/master.csv")
    st.dataframe(master.head(), use_container_width=True)
