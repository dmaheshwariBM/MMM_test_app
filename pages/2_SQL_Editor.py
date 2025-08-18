# pages/2_SQL_Query.py
# SQL over files in data/ + save results back to data/ as CSV/XLSX.

from __future__ import annotations
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

PAGE_ID = "SQL_QUERY_v2_1_0"
st.title("🧰 SQL Query")
st.caption(f"Page ID: `{PAGE_ID}` — Query files in `data/` with SQL, and save results as new datasets.")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------ helpers ------------------------------
def _safe_table_name(name: str) -> str:
    # lower, alnum + underscore only
    base = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip()).strip("_").lower()
    if not base:
        base = "t"
    # cannot start with digit in SQLite identifiers if quoted; but we'll not quote—so make safe:
    if base[0].isdigit():
        base = "t_" + base
    return base

def _unique_name(n: str, taken: set) -> str:
    if n not in taken:
        return n
    i = 2
    while f"{n}_{i}" in taken:
        i += 1
    return f"{n}_{i}"

def _list_files() -> List[Path]:
    return sorted([p for p in DATA_DIR.glob("**/*") if p.is_file() and p.suffix.lower() in (".csv", ".xlsx")])

@st.cache_data(show_spinner=False)
def load_tables_from_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[Tuple[str, str]]]]:
    """
    Returns: (tables, schema)
      tables: {table_name: df}
      schema: {table_name: [(col, dtype), ...]}
    """
    files = _list_files()
    tables: Dict[str, pd.DataFrame] = {}
    taken = set()

    for p in files:
        fname = p.stem  # without extension
        if p.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(p)
            except Exception:
                # try with engine fallback
                df = pd.read_csv(p, low_memory=False)
            t = _unique_name(_safe_table_name(fname), taken); taken.add(t)
            tables[t] = df
        elif p.suffix.lower() == ".xlsx":
            try:
                x = pd.ExcelFile(p)
                for sheet in x.sheet_names:
                    df = pd.read_excel(p, sheet_name=sheet)
                    t = _safe_table_name(f"{fname}__{sheet}")
                    t = _unique_name(t, taken); taken.add(t)
                    tables[t] = df
            except Exception:
                # fallback: read first sheet only
                df = pd.read_excel(p)
                t = _unique_name(_safe_table_name(fname), taken); taken.add(t)
                tables[t] = df

    # build schema
    schema: Dict[str, List[Tuple[str, str]]] = {}
    for t, df in tables.items():
        schema[t] = [(c, str(df[c].dtype)) for c in df.columns.tolist()]
    return tables, schema

def build_sqlite_with_tables(tables: Dict[str, pd.DataFrame]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for t, df in tables.items():
        # sanitize column names for SQLite friendliness
        df2 = df.copy()
        df2.columns = [re.sub(r"\s+", "_", str(c)) for c in df2.columns]
        df2.to_sql(t, conn, index=False, if_exists="replace")
    return conn

def run_query(conn: sqlite3.Connection, q: str) -> pd.DataFrame:
    # Multiple statements? Execute all, return last SELECT result if any.
    # Simple split by ; while being pragmatic:
    stmts = [s.strip() for s in q.split(";") if s.strip()]
    last_df = None
    with conn:
        cur = conn.cursor()
        for i, s in enumerate(stmts):
            try:
                cur.execute(s)
                # if this statement is a SELECT, fetch into df
                if s.lower().lstrip().startswith("select") or s.lower().lstrip().startswith("with"):
                    cols = [d[0] for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    last_df = pd.DataFrame(rows, columns=cols)
            except sqlite3.Error as e:
                raise RuntimeError(f"SQL error in statement {i+1}: {e}\n\n{ s }")
    if last_df is None:
        # no SELECT result; return empty df with a note
        return pd.DataFrame()
    return last_df

def _save_df(df: pd.DataFrame, filename: str, fmt: str, sheet: str = "Sheet1", overwrite: bool = False) -> Path:
    # enforce filename safety and extension
    filename = filename.strip()
    if not filename:
        raise ValueError("Please enter a file name.")
    filename = re.sub(r"[^\w\-.]+", "_", filename)
    # add extension if missing
    ext = {"CSV": ".csv", "XLSX": ".xlsx"}[fmt]
    if not filename.lower().endswith(ext.lower()):
        filename += ext
    path = DATA_DIR / filename
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path.name} (uncheck 'Overwrite' to keep it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "CSV":
        df.to_csv(path, index=False)
    else:
        # don't pin engine; let pandas choose (usually openpyxl)
        df.to_excel(path, index=False, sheet_name=sheet)
    return path

# ------------------------------ UI: data & schema ------------------------------
st.subheader("Available tables")
tables, schema = load_tables_from_data()
if not tables:
    st.info("No files in `data/` yet. Upload files on the **Data Upload** page.")
else:
    st.caption("Each CSV and Excel sheet in `data/` is loaded as a separate table.")
    with st.expander("Browse tables & columns", expanded=False):
        table_list = list(tables.keys())
        sel_table = st.selectbox("Pick a table to inspect columns", options=table_list, key="sql_table_select")
        if sel_table:
            cols = schema[sel_table]
            st.write(f"**{sel_table}** — {len(cols)} columns")
            st.dataframe(pd.DataFrame(cols, columns=["column", "dtype"]), use_container_width=True)
            # Convenience chips: paste column names
            with st.popover("Copy column list"):
                st.code(", ".join([c for c, _ in cols]), language="text")

    with st.expander("All table names", expanded=False):
        st.code("\n".join(list(tables.keys())), language="text")

# Refresh tables
c1, c2 = st.columns([1,3])
with c1:
    if st.button("🔄 Refresh file list"):
        load_tables_from_data.clear()
        st.experimental_rerun()
with c2:
    st.caption("If you've added/removed files in `data/`, refresh to rebuild the SQL catalog.")

st.divider()

# ------------------------------ UI: query editor ------------------------------
st.subheader("SQL editor")
default_example = ""
if tables:
    first = list(tables.keys())[0]
    default_example = f"SELECT * FROM {first} LIMIT 100;"
query = st.text_area("Write SQL (SQLite flavor). You can join across any tables listed above.", height=200, value=default_example)

# Quick insert helpers
if tables:
    helper_col1, helper_col2, helper_col3 = st.columns([2,2,3])
    with helper_col1:
        pick_for_cols = st.selectbox("Insert columns from…", options=["(choose)"] + list(tables.keys()), key="sql_cols_from")
    with helper_col2:
        if st.button("Insert SELECT skeleton"):
            if pick_for_cols and pick_for_cols != "(choose)":
                colnames = [c for c, _ in schema[pick_for_cols]]
                skeleton = f"SELECT {', '.join(colnames[:5])}{', ...' if len(colnames)>5 else ''}\nFROM {pick_for_cols}\nLIMIT 100;"
                st.session_state["sql_skeleton"] = skeleton
    with helper_col3:
        if "sql_skeleton" in st.session_state:
            st.code(st.session_state["sql_skeleton"], language="sql")

# ------------------------------ Run query ------------------------------
run = st.button("▶️ Run query", type="primary", use_container_width=True)
result_df = None
if run:
    if not query.strip():
        st.warning("Please enter a SQL query.")
    else:
        try:
            conn = build_sqlite_with_tables(tables)
            result_df = run_query(conn, query)
            conn.close()
            if result_df is None or result_df.empty:
                st.info("Query executed. No rows returned (or non-SELECT statements only).")
            else:
                st.success(f"Query returned {len(result_df):,} rows × {result_df.shape[1]} columns.")
                st.dataframe(result_df.head(1000), use_container_width=True)
                st.caption("Preview shows up to the first 1000 rows.")
                st.session_state["last_query_df"] = result_df
        except Exception as e:
            st.error(f"{e}")

# ------------------------------ Save result ------------------------------
st.divider()
st.subheader("Save result as a new dataset")

if "last_query_df" not in st.session_state or (isinstance(st.session_state["last_query_df"], pd.DataFrame) and st.session_state["last_query_df"].empty):
    st.caption("Run a SELECT query to enable saving.")
else:
    save_df = st.session_state["last_query_df"]
    col_a, col_b, col_c, col_d = st.columns([2,1,1,1])
    with col_a:
        fname = st.text_input("File name", value="master_table.csv", help="Add .csv or .xlsx (we’ll add it if missing).")
    with col_b:
        fmt = st.selectbox("Format", options=["CSV","XLSX"], index=0)
    with col_c:
        overwrite = st.checkbox("Overwrite if exists", value=False)
    with col_d:
        sheet = st.text_input("Sheet (XLSX)", value="Sheet1")

    csave1, csave2 = st.columns([1,1])
    with csave1:
        if st.button("💾 Save to data/", type="primary"):
            try:
                path = _save_df(save_df, fname, fmt, sheet=sheet, overwrite=overwrite)
                # Optional: cache for immediate use by other pages
                st.session_state.setdefault("DATA_CACHE", {})
                st.session_state["DATA_CACHE"][Path(path).name] = save_df
                st.success(f"Saved: `{path}`")
                st.session_state["last_saved_path"] = str(path)
            except Exception as e:
                st.error(f"Save failed: {e}")
    with csave2:
        # one-click download too
        if fmt == "CSV":
            st.download_button("⬇️ Download result", data=save_df.to_csv(index=False), file_name=(fname if fname.endswith(".csv") else f"{fname}.csv"), mime="text/csv")
        else:
            from io import BytesIO
            bio = BytesIO()
            with pd.ExcelWriter(bio) as writer:
                save_df.to_excel(writer, index=False, sheet_name=sheet)
            bio.seek(0)
            st.download_button("⬇️ Download result", data=bio.getvalue(), file_name=(fname if fname.endswith(".xlsx") else f"{fname}.xlsx"), mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Saved datasets appear under `data/` and will be visible across the app (e.g., Data Upload, Transformations, Modeling).")
