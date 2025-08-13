import streamlit as st

st.set_page_config(page_title="MMM Tool", layout="wide", page_icon="📊")

col1, col2 = st.columns([1,6])
with col1:
    st.image("assets/logo.png", use_column_width=True)
with col2:
    st.title("Marketing Mix Modeling Tool")
    st.caption("Blue Matter • End-to-end MMM workflow")

st.markdown(
    '''
**Navigation (left sidebar):**
1. 📁 Data Upload
2. 🧮 SQL Query & Checks
3. 🔧 Transformations
4. 🧠 Modeling
5. 📊 Results
6. 🧭 Budget Optimization
7. ⬇️ Export
'''
)
