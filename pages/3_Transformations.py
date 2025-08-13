import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from core import transforms, io

st.title("🔧 Transformations")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# Helpers
# =========================
TRANS_CHOICES = ["None", "Log", "NegExp", "NegExp+Cannibalization"]
ORDER_CHOICES = ["Transform → Finite Adstock(K, α)", "Finite Adstock(K, α) → Transform"]
NUMERIC_REQUIRED = {"Log", "NegExp", "NegExp+Cannibalization"}

def _list_csvs() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])

def _ensure_datetime(s: pd.Series, colname: str) -> pd.Series:
    s2 = pd.to_datetime(s, errors="coerce")
    if s2.isna().any():
        bad = s.loc[s2.isna()].astype(str).unique()[:10]
        raise ValueError(
            f"Time column `{colname}` could not be parsed for some rows. "
            f"Examples of bad values: {list(bad)}. Fix on the Upload page."
        )
    return s2

def _default_time_col(df: pd.DataFrame) -> Optional[str]:
    cand = io.detect_date_col(df)
    return cand or ("Month" if "Month" in df.columns else None)

def _suggest_transform(s: pd.Series) -> str:
    s_num = pd.to_numeric(s, errors="coerce")
    negatives = (s_num < 0).sum()
    nonneg = negatives == 0
    zeros = (s_num == 0).sum()
    frac_zero = zeros / max(len(s_num), 1)
    skew = float(pd.Series(s_num).dropna().skew()) if s_num.notna().any() else 0.0
    if not nonneg:
        return "None"
    if frac_zero >= 0.20:
        return "NegExp"
    if skew > 1.0:
        return "Log"
    return "None"

def _suggest_beta(s: pd.Series) -> float:
    s_num = pd.to_numeric(s, errors="coerce")
    s_pos = s_num[s_num > 0]
    if s_pos.empty or np.median(s_pos) == 0:
        return 0.01
    beta = np.log(2.0) / float(np.median(s_pos))
    return float(np.clip(beta, 0.0005, 1.0))

def _initial_table(metrics: List[str], df: pd.DataFrame) -> pd.DataFrame:
    sug = [ _suggest_transform(df[m]) for m in metrics ]
    return pd.DataFrame({
        "metric": metrics,
        "suggested": sug,
        "transform": sug,
        "lag_months": [0] * len(metrics),       # K
        "adstock_alpha": [0.0] * len(metrics),  # α
        "order": ["Transform → Finite Adstock(K, α)"] * len(metrics),
    })

def _merge_config(persisted: pd.DataFrame, metrics: List[str], df: pd.DataFrame) -> pd.DataFrame:
    persisted = persisted.copy()
    keep = persisted[persisted["metric"].isin(metrics)]
    missing = [m for m in metrics if m not in keep["metric"].tolist()]
    if missing:
        keep = pd.concat([keep, _initial_table(missing, df)], ignore_index=True)
    keep["lag_months"] = keep["lag_months"].fillna(0).astype(int)
    keep["adstock_alpha"] = keep["adstock_alpha"].fillna(0.0).astype(float)
    keep["transform"] = keep["transform"].fillna("None").astype(str)
    if "order" not in keep.columns:
        keep["order"] = "Transform → Finite Adstock(K, α)"
    keep["order"] = keep["order"].astype(str)
    return keep

def _coerce_numeric_report(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    report = {}
    for c in cols:
        before = int(df[c].isna().sum()) if c in df.columns else 0
        coerced = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype=float)
        after = int(coerced.isna().sum()) if c in df.columns else 0
        report[c] = max(0, after - before)
    return report

def _apply_pipeline(df: pd.DataFrame,
                    cfg: pd.DataFrame,
                    beta_map: Dict[str, float],
                    id_col: Optional[str],
                    seg_col: Optional[str],
                    time_col: Optional[str],
                    gamma: float) -> pd.DataFrame:
    out = df.copy()
    group_cols = [c for c in [id_col, seg_col] if c]

    # Need ordering if K>0 or α>0
    needs_order = (cfg["lag_months"].astype(int) > 0).any() or (cfg["adstock_alpha"].astype(float) > 0).any()
    if needs_order:
        if not time_col or time_col not in out.columns:
            raise ValueError("Finite Adstock requires a valid Time column. Please select one.")
        out[time_col] = _ensure_datetime(out[time_col], time_col)
        sort_keys = group_cols + [time_col]
        out = out.sort_values(sort_keys)

    # Cannibal pool (for NegExp+Cannibalization)
    cannibal_metrics = cfg.loc[cfg["transform"].isin(["NegExp", "NegExp+Cannibalization"]), "metric"].tolist()
    numeric_pool = out.copy()
    for c in cannibal_metrics:
        numeric_pool[c] = pd.to_numeric(numeric_pool[c], errors="coerce").fillna(0)
    pool_sum = numeric_pool[cannibal_metrics].sum(axis=1) if cannibal_metrics else pd.Series(0.0, index=out.index)
    pool_norm = pool_sum / (np.median(pool_sum[pool_sum > 0]) if (pool_sum > 0).any() else 1.0)

    for _, row in cfg.iterrows():
        m = row["metric"]
        base = row["transform"]
        K = int(row.get("lag_months", 0))
        a = float(row.get("adstock_alpha", 0.0))
        order = str(row.get("order", ORDER_CHOICES[0]))

        if m not in out.columns:
            raise ValueError(f"Metric `{m}` not found in dataset.")
        s_raw = out[m].copy()

        # Helpers to apply finite adstock over groups
        def apply_finite(x: pd.Series) -> pd.Series:
            return transforms.adstock_finite(x, alpha=a, K=K)

        def do_base(x: pd.Series) -> pd.Series:
            if base in NUMERIC_REQUIRED:
                x = pd.to_numeric(x, errors="coerce").fillna(0)
            if base == "Log":
                return transforms.log_transform(x)
            elif base == "NegExp":
                beta = float(beta_map.get(m, _suggest_beta(out[m])))
                return transforms.negexp(x, beta=beta)
            elif base == "NegExp+Cannibalization":
                beta = float(beta_map.get(m, _suggest_beta(out[m])))
                return transforms.negexp_cannibal(x, beta=beta, pool=pool_norm, gamma=float(gamma))
            else:
                return x

        if order == "Finite Adstock(K, α) → Transform":
            # First finite adstock on RAW
            if (K > 0) or (a > 0):
                if group_cols:
                    s_eff = out.groupby(group_cols, group_keys=False).apply(
                        lambda g: apply_finite(s_raw.loc[g.index])
                    ).reset_index(level=list(range(len(group_cols))), drop=True)
                else:
                    s_eff = apply_finite(s_raw)
            else:
                s_eff = s_raw
            s_t = do_base(s_eff)
        else:
            # Transform first, then finite adstock
            s_base = do_base(s_raw)
            if (K > 0) or (a > 0):
                if group_cols:
                    s_t = out.groupby(group_cols, group_keys=False).apply(
                        lambda g: apply_finite(s_base.loc[g.index])
                    ).reset_index(level=list(range(len(group_cols))), drop=True)
                else:
                    s_t = apply_finite(s_base)
            else:
                s_t = s_base

        out[f"{m}__tfm"] = s_t

    if needs_order:
        out = out.sort_index()

    return out

def _curve_preview_df(x: np.ndarray, transform: str, beta: float, gamma: float) -> pd.DataFrame:
    x = np.maximum(x, 0.0)
    if transform == "Log":
        y = np.log1p(x)
        df = pd.DataFrame({"x": x, "Log(x)": y})
    elif transform == "NegExp":
        y = 1.0 - np.exp(-beta * x)
        df = pd.DataFrame({"x": x, "NegExp": y})
    elif transform == "NegExp+Cannibalization":
        y_base = 1.0 - np.exp(-beta * x)
        y_cann = y_base * np.exp(-gamma * 1.0)  # pool=1 for intuition
        df = pd.DataFrame({"x": x, "NegExp (base)": y_base, "With Cannibalization": y_cann})
    else:
        df = pd.DataFrame({"x": x, "None": x})
    return df.set_index("x")

# =========================
# Dataset & keys
# =========================
files = _list_csvs()
if not files:
    st.info("No CSV files in `data/`. Upload and type your data first.")
    st.stop()

dataset = st.selectbox("Dataset (CSV)", files, index=(files.index("master.csv") if "master.csv" in files else 0))
df0 = pd.read_csv(os.path.join(DATA_DIR, dataset))
st.caption(f"Rows: {len(df0):,}  |  Columns: {len(df0.columns)}")

cols_all = list(df0.columns)
target_col = st.selectbox("Target variable (excluded from transforms)", [None] + cols_all, index=(cols_all.index("Sales")+1 if "Sales" in cols_all else 0))
time_col_default = _default_time_col(df0)
time_col = st.selectbox("Time column", [None] + cols_all, index=(cols_all.index(time_col_default)+1 if time_col_default in cols_all else 0))
id_col = st.selectbox("ID column (e.g., HCP_ID)", [None] + cols_all, index=(cols_all.index("HCP_ID")+1 if "HCP_ID" in cols_all else 0))
seg_col = st.selectbox("Segment column (optional)", [None] + cols_all, index=0)

# Time window filter
df = df0.copy()
if time_col:
    try:
        dt = pd.to_datetime(df[time_col], errors="coerce")
        min_dt, max_dt = dt.min(), dt.max()
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("Start date", value=min_dt.date() if pd.notna(min_dt) else None)
        with c2:
            end_date = st.date_input("End date", value=max_dt.date() if pd.notna(max_dt) else None)
        if start_date and end_date:
            mask = (dt >= pd.to_datetime(start_date)) & (dt <= pd.to_datetime(end_date))
            df = df.loc[mask].copy()
            if df.empty:
                st.warning("Selected date window has no rows after filtering.")
    except Exception as e:
        st.warning(f"Time filter not applied: {e}")

# =========================
# Metrics table (stateful)
# =========================
exclude = {c for c in [target_col, time_col, id_col, seg_col] if c}
metric_candidates = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

if not metric_candidates:
    st.error("No numeric metrics available for transformation after exclusions. Check your selections.")
    st.stop()

# Persistent per-dataset state
if "tfm_state" not in st.session_state:
    st.session_state["tfm_state"] = {}

state = st.session_state["tfm_state"].get(dataset)
if state is None:
    state = {
        "cfg": _initial_table(metric_candidates, df),
        "beta_map": {m: _suggest_beta(df[m]) for m in metric_candidates},
        "needs_gamma": False,
    }
    st.session_state["tfm_state"][dataset] = state
else:
    state["cfg"] = _merge_config(state["cfg"], metric_candidates, df)

cfg_df = state["cfg"]

st.subheader("Configure per-metric transforms")
st.caption("Set transform, **Lag K** and **Adstock α**. Finite adstock computes:  x_t + α x_{t-1} + α² x_{t-2} + … + α^K x_{t-K}.")

cfg_df = st.data_editor(
    cfg_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "metric": st.column_config.Column("Metric", disabled=True, help="Input column to transform"),
        "suggested": st.column_config.Column("Suggested", disabled=True, help="Heuristic based on zeros, skewness, and sign"),
        "transform": st.column_config.SelectboxColumn(
            "Transform",
            options=TRANS_CHOICES,
            required=True,
            help="None / Log / Negative Exponential / NegExp+Cannibalization"
        ),
        "lag_months": st.column_config.NumberColumn(
            "Lag K (months)", min_value=0, max_value=24, step=1,
            help="K in the finite adstock sum."
        ),
        "adstock_alpha": st.column_config.NumberColumn(
            "Adstock α", min_value=0.0, max_value=0.99, step=0.01,
            help="Weight decay per lag: α^i on x_{t-i}"
        ),
        "order": st.column_config.SelectboxColumn(
            "Order",
            options=ORDER_CHOICES,
            help="Apply base transform before finite adstock, or after."
        ),
    },
    key=f"tfm_table_editor_{dataset}",
)

# Persist
cfg_df["lag_months"] = cfg_df["lag_months"].fillna(0).astype(int)
cfg_df["adstock_alpha"] = cfg_df["adstock_alpha"].fillna(0.0).astype(float)
cfg_df["transform"] = cfg_df["transform"].fillna("None").astype(str)
cfg_df["order"] = cfg_df["order"].fillna(ORDER_CHOICES[0]).astype(str)
state["cfg"] = cfg_df
st.session_state["tfm_state"][dataset] = state

# ---------- Conditional parameter UI ----------
needs_beta_metrics = cfg_df.loc[cfg_df["transform"].isin(["NegExp", "NegExp+Cannibalization"]), "metric"].tolist()
if needs_beta_metrics:
    st.markdown("**β (per-metric) for Negative Exponential**")
    cols = st.columns(min(3, len(needs_beta_metrics)))
    for i, m in enumerate(needs_beta_metrics):
        with cols[i % len(cols)]:
            default_beta = state["beta_map"].get(m, _suggest_beta(df[m]))
            state["beta_map"][m] = st.slider(
                f"{m} • β", 0.0005, 1.0, float(default_beta), 0.0005, key=f"beta_{dataset}_{m}"
            )
            x50 = np.log(2.0) / max(state["beta_map"][m], 1e-9)
            st.caption(f"x₅₀ ≈ {x50:,.2f}")

needs_gamma = cfg_df["transform"].eq("NegExp+Cannibalization").any()
state["needs_gamma"] = bool(needs_gamma)
gamma = 0.3
if needs_gamma:
    gamma = st.slider(
        "Cannibalization strength γ (global)",
        0.0, 2.0, 0.3, 0.05,
        help="(1 − e^{−βx}) · e^{−γ·pool}, with pool the normalized sum of other NegExp metrics."
    )

# Coercion report
need_numeric_cols = cfg_df.loc[cfg_df["transform"].isin(list(NUMERIC_REQUIRED)), "metric"].tolist()
if need_numeric_cols:
    report = _coerce_numeric_report(df, need_numeric_cols)
    with st.expander("Coercion report (values that would become NaN if coerced to numeric):", expanded=False):
        st.json(report)

# ---------- Guide ----------
with st.expander("📘 Transform Guide (math & intuition)", expanded=False):
    st.markdown(r"""
**Finite Adstocked Lag (this app):**  
\[
\text{Effective}_t = \sum_{i=0}^{K} \alpha^{i} \, x_{t-i}
\]
- **K** = Lag months, **α** = Adstock. Example: K=2, α=0.5 → \(x_t + 0.5x_{t-1} + 0.25x_{t-2}\).

**Log:** \( y = \ln(1 + x) \)  
**NegExp:** \( y = 1 - e^{-\beta x} \)  (choose β via \(x_{50}=\ln 2/\beta\))  
**NegExp+Cannibalization:** \( y = (1 - e^{-\beta x}) \cdot e^{-\gamma \cdot \text{pool}} \)
""")

# ---------- Curve previews (sparklines) ----------
with st.expander("📈 Curve previews (by metric)", expanded=False):
    selectable = [m for m in cfg_df["metric"].tolist() if cfg_df.loc[cfg_df["metric"]==m, "transform"].iloc[0] != "None"]
    if selectable:
        pick = st.multiselect("Select metrics to preview", selectable, default=selectable[:6])
        for m in pick:
            t = cfg_df.loc[cfg_df["metric"]==m, "transform"].iloc[0]
            beta = state["beta_map"].get(m, _suggest_beta(df[m]))
            s = pd.to_numeric(df[m], errors="coerce").fillna(0)
            x_max = max(1.0, float(np.nanpercentile(s, 95)))
            x = np.linspace(0.0, x_max*1.2, 120)
            chart_df = _curve_preview_df(x, t, beta=beta, gamma=(gamma if needs_gamma else 0.0))
            st.caption(f"**{m}** — Transform: {t}")
            st.line_chart(chart_df)
    else:
        st.info("Select a transform other than 'None' to see a curve preview.")

st.divider()

# =========================
# Preview + Save
# =========================
c1, c2 = st.columns(2)
with c1:
    preview_n = st.slider("Rows to preview", min_value=50, max_value=1000, value=200, step=50)
with c2:
    out_name_default = f"{os.path.splitext(dataset)[0]}__tfm.csv"
    out_name = st.text_input("Output filename (CSV)", value=out_name_default)

def _apply_all():
    # Need time column if K>0 or α>0
    if ((cfg_df["lag_months"].astype(int) > 0).any() or (cfg_df["adstock_alpha"].astype(float) > 0).any()) and not time_col:
        raise ValueError("Finite Adstock selected but no Time column chosen.")
    return _apply_pipeline(
        df=df,
        cfg=cfg_df,
        beta_map=state["beta_map"],
        id_col=id_col,
        seg_col=seg_col,
        time_col=time_col,
        gamma=(gamma if state["needs_gamma"] else 0.0),
    )

left, right = st.columns(2)

with left:
    if st.button("👀 Preview transformed dataset"):
        try:
            df_prev = _apply_all()
            added = [c for c in df_prev.columns if c.endswith("__tfm")]
            st.success(f"Added {len(added)} new column(s).")
            st.dataframe(df_prev.head(preview_n), use_container_width=True)
        except Exception as e:
            st.error(f"Preview failed: {e}")

with right:
    if st.button("💾 Save transformed dataset"):
        try:
            df_out = _apply_all()
            out_csv = out_name if out_name.strip().lower().endswith(".csv") else out_name + ".csv"
            save_path = os.path.join(DATA_DIR, out_csv)
            df_out.to_csv(save_path, index=False)

            meta = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": dataset,
                "time_col": time_col,
                "id_col": id_col,
                "seg_col": seg_col,
                "target_col": target_col,
                "beta_map": state["beta_map"],
                "config": cfg_df.to_dict(orient="records"),
                "gamma": (gamma if state["needs_gamma"] else 0.0),
                "output": out_csv,
            }
            with open(os.path.join(DATA_DIR, f"transforms_{os.path.splitext(dataset)[0]}.json"), "w") as f:
                json.dump(meta, f, indent=2)

            st.session_state["mmm_current_dataset"] = out_csv
            st.session_state["mmm_target"] = target_col
            st.success(f"Saved: `{out_csv}` in `data/`. Ready for Modeling.")
        except Exception as e:
            st.error(f"Save failed: {e}")
