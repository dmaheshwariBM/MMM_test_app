import streamlit as st
import duckdb
import pandas as pd
import os

st.title("🧮 SQL Query & Checks")
con = duckdb.connect(database=':memory:')

# Load all CSV/Excel from data folder
if os.path.isdir("data"):
    for f in os.listdir("data"):
        name = os.path.splitext(f)[0]
        path = os.path.join("data", f)
        if f.endswith(".csv"):
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")
        elif f.endswith(".xlsx"):
            df = pd.read_excel(path)
            con.register(name, df)

st.caption("Tables loaded from data/: " + ", ".join([t[0] for t in con.execute("SHOW TABLES").fetchall()]))

query = st.text_area("Write SQL query:")
if st.button("Run Query"):
    try:
        result = con.execute(query).fetchdf()
        st.dataframe(result, use_container_width=True)
        # Optionally save as master
        if st.checkbox("Save as data/master.csv"):
            result.to_csv("data/master.csv", index=False)
            st.success("Saved to data/master.csv")
    except Exception as e:
        st.error(str(e))
