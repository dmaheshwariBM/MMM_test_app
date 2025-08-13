# pages/6_Budget_Optimization.py
import streamlit as st
import pandas as pd
import numpy as np
import os, io, json, glob, zipfile, re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

st.title("💰 Budget Optimization")

DATA_DIR = "data"
RESULTS_DIR = "results"
BUDGETS_DIR = "budgets"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BUDGETS_DIR, exist_ok=True)

# ---------------------------
# Utility: load saved models (from results/)
# ---------------------------
def _load_all_results() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for bdir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*")), reverse=True):
        if not os.path.isdir(bdir):
            continue
        for jf in glob.glob(os.path.join(bdir, "*.json")):
            try:
                with open(jf, "r") as f:
                    rec = json.load(f)
                if "name" in rec and "metrics" in rec and "target" in rec:
                    rec["_path"] = jf
                    runs.append(rec)
            except Exception:
                continue
    # newest first by batch_ts then name
    runs.sort(key=lambda r: (r.get("batch_ts",""), r.get("name","")), reverse=True)
    return runs

def _from_session_or_results() -> List[Dict[str, Any]]:
    sess = st.session_state.get("selected_for_budget")
    if sess:
        # try to enrich from the saved JSON path if available
        enriched = []
        for s in sess:
            p = s.get("_path")
            if p and os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        enriched.append(json.load(f))
                    continue
                except Exception:
                    pass
            enriched.append(s)
        return enriched
    # Else let user pick from all saved results
    all_runs = _load_all_results()
    if not all_runs:
        st.info("No saved models found. Please run models and add them via the Results page.")
        return []
    labels = [f"{r.get('batch_ts','')} • {r.get('name','')} • {r.get('target','')}" for r in all_runs]
    sel = st.multiselect("Select models for optimization", options=list(range(len(all_runs))),
                         format_func=lambda i: labels[i])
    return [all_runs[i] for i in sel]

# ---------------------------
# Response curves (concave) f(u) and f'(u)
# u = controllable 'units' (spend / unit_cost)
# ---------------------------
def f_linear(u, k):       # k ignored
    return u
def fp_linear(u, k):
    return np.ones_like(u)

def f_log(u, k):          # k>0
    k = max(float(k), 1e-9)
    return np.log1p(k * u)
def fp_log(u, k):
    k = max(float(k), 1e-9)
    return k / (1.0 + k * u)

def f_negexp(u, k):       # k>0
    k = max(float(k), 1e-9)
    return 1.0 - np.exp(-k * u)
def fp_negexp(u, k):
    k = max(float(k), 1e-9)
    return k * np.exp(-k * u)

CURVES = {
    "Linear":  (f_linear,  fp_linear, 0.0),
    "Log":     (f_log,     fp_log,    0.01),
    "NegExp":  (f_negexp,  fp_negexp, 0.01),
}

# ---------------------------
# Transform meta → curve suggestion
# ---------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def _load_transform_meta_for_dataset(dataset_csv: str) -> Optional[Dict[str, Any]]:
    """
    Look for data/transforms_<stem>.json where <stem> matches dataset_csv without extension.
    """
    if not dataset_csv:
        return None
    stem = os.path.splitext(os.path.basename(dataset_csv))[0]
    path = os.path.join(DATA_DIR, f"transforms_{stem}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _row_curve_from_meta_row(row: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """
    Accepts flexible keys:
      transform / transformation / trans_type in {'linear','log','negexp','negative_exponential', ...}
      k / beta / lambda for the shape parameter
    """
    # find transform name
    tname = None
    for key in ("transform", "transformation", "trans_type", "type"):
        if key in row and row[key]:
            tname = str(row[key]).strip().lower()
            break
    if not tname:
        return None

    # map to one of our curves
    if tname in ("linear",):
        cname = "Linear"
    elif tname in ("log", "logarithm", "logarithmic"):
        cname = "Log"
    elif tname in ("negexp", "negative_exponential", "negativeexp", "nege", "neg_exp"):
        cname = "NegExp"
    else:
        # unknown; skip
        return None

    # get k parameter
    kval = None
    for kkey in ("k", "beta", "lambda", "param", "shape"):
        if kkey in row and row[kkey] is not None:
            try:
                kval = float(row[kkey])
                break
            except Exception:
                pass
    if kval is None:
        kval = CURVES[cname][2]

    return (cname, float(kval))

def _suggest_curve_for_channel(channel_name: str, metas: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    """
    channel_name may be 'Metric__tfm'. We'll try to match against 'metric' field in meta rows.
    """
    # try to strip "__tfm"
    base = channel_name.replace("__tfm", "")
    nchan = _norm(channel_name)
    nbase = _norm(base)
    for meta in metas:
        config = meta.get("config") or meta.get("rows") or meta.get("data") or []
        if isinstance(config, dict):
            config = list(config.values())
        if not isinstance(config, list):
            continue
        for row in config:
            metric_name = str(row.get("metric", row.get("name", row.get("feature", ""))))
            if not metric_name:
                continue
            nmetric = _norm(metric_name)
            if nmetric in (nchan, nbase):
                # found a matching metric row
                cur = _row_curve_from_meta_row(row)
                if cur:
                    return cur
    return None

# ---------------------------
# Marginal ROI & contribution
# ---------------------------
def mroi(beta: float, spend: float, cost: float, curve: str, k: float, y_value: float) -> float:
    """Marginal ROI at 'spend' for a single channel."""
    if beta <= 0 or cost <= 0:
        return 0.0
    u = max(spend, 0.0) / cost
    _, fp, _ = CURVES[curve]
    return float(beta * fp(u, k) * (y_value / cost))

def contrib(beta: float, spend: float, cost: float, curve: str, k: float, y_value: float) -> float:
    """Total incremental contribution (revenue) from this channel at 'spend'."""
    if beta <= 0 or cost <= 0:
        return 0.0
    u = max(spend, 0.0) / cost
    f, _, _ = CURVES[curve]
    return float(beta * f(u, k) * y_value)

# ---------------------------
# Optimizers
# ---------------------------
def optimize_target_budget(
    betas: Dict[str, float],
    costs: Dict[str, float],
    curves: Dict[str, Tuple[str, float]],
    total_budget: float,
    mins: Dict[str, float],
    maxs: Dict[str, float],
    fixed: Dict[str, Optional[float]],
    y_value: float,
    step_ct: int = 1000
) -> Dict[str, float]:
    """
    Greedy water-filling on concave curves: allocate budget in small steps to channel with highest MROI.
    Respects min/max and fixed spends.
    """
    chs = list(betas.keys())
    spend = {ch: 0.0 for ch in chs}

    # Initialize to fixed/min
    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])
        else:
            spend[ch] = max(0.0, float(mins.get(ch, 0.0)))
    used = sum(spend.values())
    B = max(0.0, float(total_budget))
    if used > B:
        # Scale non-fixed mins to fit
        fixed_sum = sum(float(fixed[ch]) for ch in chs if fixed.get(ch) is not None)
        if fixed_sum > B:
            # infeasible → keep fixed, zero others
            for ch in chs:
                if fixed.get(ch) is None:
                    spend[ch] = 0.0
            return spend
        free = B - fixed_sum
        nf = [ch for ch in chs if fixed.get(ch) is None]
        s = sum(spend[ch] for ch in nf)
        if s > 0:
            for ch in nf:
                spend[ch] = spend[ch] * free / s
        used = sum(spend.values())

    # Greedy allocate
    remaining = max(0.0, B - used)
    if step_ct <= 0:
        step_ct = 1000
    step = remaining / step_ct if remaining > 0 else 0.0

    for _ in range(step_ct):
        if remaining <= 1e-12 or step <= 0:
            break
        best_ch, best_m = None, -1.0
        for ch in chs:
            if fixed.get(ch) is not None:
                continue
            if spend[ch] + step > maxs.get(ch, float("inf")):
                continue
            mr = mroi(betas[ch], spend[ch], costs[ch], curves[ch][0], curves[ch][1], y_value)
            if mr > best_m:
                best_m, best_ch = mr, ch
        if best_ch is None:
            break
        spend[best_ch] += step
        remaining -= step

    # Final clamps and normalization
    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])
        spend[ch] = max(mins.get(ch, 0.0), min(spend[ch], maxs.get(ch, float("inf"))))

    # Normalize non-fixed to meet total B
    fixed_sum = sum(spend[ch] for ch in chs if fixed.get(ch) is not None)
    target_nf = max(B - fixed_sum, 0.0)
    nf = [ch for ch in chs if fixed.get(ch) is None]
    cur_nf = sum(spend[ch] for ch in nf)
    if cur_nf > 0:
        for ch in nf:
            spend[ch] = spend[ch] * target_nf / cur_nf
    return spend

def optimize_equalize_mroi_to_one(
    betas: Dict[str, float],
    costs: Dict[str, float],
    curves: Dict[str, Tuple[str, float]],
    start_spend: Dict[str, float],
    mins: Dict[str, float],
    maxs: Dict[str, float],
    fixed: Dict[str, Optional[float]],
    y_value: float,
    iters: int = 4000,
    delta: float = 0.001
) -> Dict[str, float]:
    """
    Reallocate given total spend to move MROIs toward 1 subject to constraints.
    """
    chs = list(betas.keys())
    spend = {ch: float(start_spend.get(ch, 0.0)) for ch in chs}

    # enforce fixed/min/max initially
    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])
        spend[ch] = max(mins.get(ch, 0.0), min(spend[ch], maxs.get(ch, float("inf"))))

    total = sum(spend.values())
    if total <= 0:
        return spend

    for _ in range(iters):
        mrs = {ch: mroi(betas[ch], spend[ch], costs[ch], curves[ch][0], curves[ch][1], y_value) for ch in chs}
        receivers = [ch for ch in chs if fixed.get(ch) is None and spend[ch] < maxs.get(ch, float("inf"))]
        donors = [ch for ch in chs if fixed.get(ch) is None and spend[ch] > mins.get(ch, 0.0)]
        if not receivers or not donors:
            break
        best_recv = max(receivers, key=lambda c: mrs[c])
        worst_donor = min(donors, key=lambda c: mrs[c])
        if mrs[best_recv] <= 1.0 and mrs[worst_donor] >= 1.0:
            break
        if mrs[best_recv] <= mrs[worst_donor] + 1e-9:
            break
        move = min(delta * total,
                   maxs.get(best_recv, float("inf")) - spend[best_recv],
                   spend[worst_donor] - mins.get(worst_donor, 0.0))
        if move <= 1e-12:
            break
        spend[best_recv] += move
        spend[worst_donor] -= move

    # clamp + preserve total
    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])
        spend[ch] = max(mins.get(ch, 0.0), min(spend[ch], maxs.get(ch, float("inf"))))

    nf = [ch for ch in chs if fixed.get(ch) is None]
    fixed_sum = sum(spend[ch] for ch in chs if fixed.get(ch) is not None)
    rem_target = max(total - fixed_sum, 0.0)
    cur_nf = sum(spend[ch] for ch in nf)
    if cur_nf > 0:
        for ch in nf:
            spend[ch] = spend[ch] * rem_target / cur_nf
    return spend

def optimize_profit_subject_to_mroi_ge_1(
    betas: Dict[str, float],
    costs: Dict[str, float],
    curves: Dict[str, Tuple[str, float]],
    mins: Dict[str, float],
    maxs: Dict[str, float],
    fixed: Dict[str, Optional[float]],
    y_value: float,
    step_ct: int = 1200
) -> Dict[str, float]:
    """
    No global cap. Allocate greedily to any channel with MROI > 1 until all MROIs <= 1 or channel hits max.
    """
    chs = list(betas.keys())
    spend = {ch: max(0.0, float(mins.get(ch, 0.0))) for ch in chs}
    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])

    avg_range = np.mean([maxs.get(ch, 0.0) - mins.get(ch, 0.0) for ch in chs if np.isfinite(maxs.get(ch, 0.0))] or [1.0])
    step = max(avg_range / step_ct, 1e-6)

    for _ in range(step_ct):
        best_ch, best_m = None, 1.0
        for ch in chs:
            if fixed.get(ch) is not None:
                continue
            if spend[ch] + step > maxs.get(ch, float("inf")):
                continue
            mr = mroi(betas[ch], spend[ch], costs[ch], curves[ch][0], curves[ch][1], y_value)
            if mr > best_m:
                best_m, best_ch = mr, ch
        if best_ch is None:
            break
        spend[best_ch] += step

    for ch in chs:
        if fixed.get(ch) is not None:
            spend[ch] = float(fixed[ch])
        spend[ch] = max(mins.get(ch, 0.0), min(spend[ch], maxs.get(ch, float("inf"))))
    return spend

# ---------------------------
# Export helpers
# ---------------------------
def _excel_bytes(sheets: Dict[str, pd.DataFrame]) -> Optional[bytes]:
    try:
        import xlsxwriter  # noqa: F401
        with io.BytesIO() as buf:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            return buf.getvalue()
    except Exception:
        pass
    try:
        import openpyxl  # noqa: F401
        with io.BytesIO() as buf:
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                for name, df in sheets.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
            return buf.getvalue()
    except Exception:
        return None

def _zip_csv_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    with io.BytesIO() as zip_buf:
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, df in sheets.items():
                zf.writestr(f"{name}.csv", df.to_csv(index=False).encode("utf-8"))
        return zip_buf.getvalue()

def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:64]

# ---------------------------
# UI — Select models & scenario
# ---------------------------
models = _from_session_or_results()
if not models:
    st.stop()

st.subheader("Scenario")
scenario = st.radio(
    "Choose one scenario to run:",
    options=[
        "1) Target total budget (with channel constraints)",
        "2) Reallocate current total to MROI ≈ 1 (breakeven)",
        "3) Profit maximization with manual constraints (MROI ≥ 1)"
    ],
    index=0
)

st.caption("Tip: ‘Y-unit value’ converts predicted Y into revenue. Keep 1.0 if Y is already revenue.")
y_value = st.number_input("Y-unit value (revenue per 1 Y unit)", min_value=0.0, value=1.0, step=0.1)

# ---------------------------
# Gather Transform metas for selected models (for auto-fill)
# ---------------------------
dataset_to_meta: Dict[str, Any] = {}
for m in models:
    ds = m.get("dataset", "")
    if ds and ds not in dataset_to_meta:
        meta = _load_transform_meta_for_dataset(ds)
        if meta:
            dataset_to_meta[ds] = meta

# Build a list of metas to search (order: as selected)
metas_list = [dataset_to_meta[d] for d in dataset_to_meta] if dataset_to_meta else []

# ---------------------------
# Build per-channel config table (costs, current, min/max/fixed, curve)
# Channels = union across selected models' coef keys minus 'const'
# ---------------------------
all_channels = sorted({ch for m in models for ch in (list((m.get("coef", {}) or {}).keys())) if ch != "const"})
if not all_channels:
    st.error("No channels found in selected models (no coefficients?).")
    st.stop()

st.subheader("Channel settings")

# Auto-fill suggestions from Transform meta
suggestions: Dict[str, Tuple[str, float]] = {}
for ch in all_channels:
    sug = _suggest_curve_for_channel(ch, metas_list) if metas_list else None
    if sug:
        suggestions[ch] = sug

with st.expander("Edit channel settings", expanded=True):
    defaults = []
    for ch in all_channels:
        # default curve (NegExp) possibly overridden by suggestion
        default_curve = "NegExp"
        default_k = CURVES["NegExp"][2]
        if ch in suggestions:
            default_curve, default_k = suggestions[ch]

        defaults.append({
            "channel": ch,
            "unit_cost": 1.0,
            "current_spend": 0.0,
            "min_spend": 0.0,
            "max_spend": float("inf"),
            "fixed_spend": None,
            "curve": default_curve,
            "curve_k": float(default_k),
        })
    cfg_df = pd.DataFrame(defaults)
    cfg_df = st.data_editor(
        cfg_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "channel": st.column_config.TextColumn("Channel", disabled=True),
            "unit_cost": st.column_config.NumberColumn("Unit cost", help="Spend per controllable unit"),
            "current_spend": st.column_config.NumberColumn("Current spend (baseline)"),
            "min_spend": st.column_config.NumberColumn("Min spend"),
            "max_spend": st.column_config.NumberColumn("Max spend (use large number for ∞)"),
            "fixed_spend": st.column_config.NumberColumn("Fixed spend (overrides min/max)", help="Leave empty for None"),
            "curve": st.column_config.SelectboxColumn("Response curve", options=list(CURVES.keys())),
            "curve_k": st.column_config.NumberColumn("Curve parameter k", help="Log/NegExp shape (>0). For Linear ignored."),
        }
    )
    if suggestions:
        st.caption("Auto-filled curve suggestions loaded from Transformations metadata.")

# Optional global min/max share (only applies to scenario 1)
global_min_share = None
global_max_share = None
if scenario.startswith("1)"):
    st.subheader("Global share constraints (optional)")
    c1, c2, c3 = st.columns(3)
    with c1:
        total_budget = st.number_input("Target total budget", min_value=0.0, value=100000.0, step=1000.0)
    with c2:
        gmin = st.number_input("Global min share per channel (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        global_min_share = gmin / 100.0 if gmin > 0 else None
    with c3:
        gmax = st.number_input("Global max share per channel (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
        global_max_share = gmax / 100.0 if gmax < 100.0 else None

# ---------------------------
# Run scenario
# ---------------------------
run_clicked = st.button("🚀 Run optimization")
if not run_clicked:
    st.stop()

# Build per-channel dictionaries
costs = {row["channel"]: float(row["unit_cost"] or 0.0) for _, row in cfg_df.iterrows()}
cur_spend = {row["channel"]: float(row["current_spend"] or 0.0) for _, row in cfg_df.iterrows()}
mins = {row["channel"]: float(row["min_spend"] or 0.0) for _, row in cfg_df.iterrows()}
maxs = {row["channel"]: float(row["max_spend"] if np.isfinite(row["max_spend"]) else 1e18) for _, row in cfg_df.iterrows()}
fixed = {row["channel"]: (float(row["fixed_spend"]) if pd.notna(row["fixed_spend"]) else None) for _, row in cfg_df.iterrows()}
curves = {row["channel"]: (row["curve"], float(row["curve_k"])) for _, row in cfg_df.iterrows()}

# Apply global share constraints (scenario 1 only)
if scenario.startswith("1)") and (global_min_share is not None or global_max_share is not None):
    for ch in all_channels:
        if global_min_share is not None:
            mins[ch] = max(mins[ch], global_min_share * float(total_budget))
        if global_max_share is not None:
            maxs[ch] = min(maxs[ch], global_max_share * float(total_budget))

# ---------------------------
# Run for each selected model
# ---------------------------
batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(BUDGETS_DIR, batch_ts)
os.makedirs(out_dir, exist_ok=True)

results_tables: List[Tuple[str, pd.DataFrame]] = []
summary_rows = []

for model in models:
    name = model.get("name","Model")
    target = model.get("target","Y")
    coef = model.get("coef", {}) or {}
    # reduce to channels we have config for
    betas = {ch: float(coef.get(ch, 0.0)) for ch in all_channels if ch != "const" and ch in coef}

    # Guards: drop non-positive betas (no spend)
    betas = {ch: b for ch, b in betas.items() if b > 0}

    if not betas:
        st.warning(f"Skipping {name}: no positive coefficients found.")
        continue

    # Run scenario
    if scenario.startswith("1)"):
        spend = optimize_target_budget(
            betas, costs, curves, total_budget=total_budget,
            mins=mins, maxs=maxs, fixed=fixed, y_value=y_value, step_ct=1200
        )
    elif scenario.startswith("2)"):
        total_current = sum(cur_spend[ch] for ch in all_channels)
        if total_current <= 0:
            st.warning(f"{name}: current total spend is 0. Set 'current_spend' first.")
            continue
        spend = optimize_equalize_mroi_to_one(
            betas, costs, curves, start_spend=cur_spend,
            mins=mins, maxs=maxs, fixed=fixed, y_value=y_value,
            iters=5000, delta=0.002
        )
    else:  # scenario 3
        spend = optimize_profit_subject_to_mroi_ge_1(
            betas, costs, curves, mins=mins, maxs=maxs, fixed=fixed, y_value=y_value, step_ct=3000
        )

    # Build per-channel table with KPIs
    rows = []
    total_spend = 0.0
    total_contrib = 0.0
    for ch in all_channels:
        s = float(spend.get(ch, 0.0))
        c = float(costs[ch])
        cv, kv = curves[ch]
        b = float(betas.get(ch, 0.0))
        mr = mroi(b, s, c, cv, kv, y_value) if b > 0 else 0.0
        rev = contrib(b, s, c, cv, kv, y_value) if b > 0 else 0.0
        roi = (rev / s) if s > 0 else np.nan
        rows.append({
            "Channel": ch, "Spend": s, "Unit cost": c, "Curve": cv, "k": kv,
            "Beta": b, "MROI": mr, "ROI": roi, "Revenue (contrib)": rev
        })
        total_spend += s
        total_contrib += rev

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("Spend", ascending=False)

    st.subheader(f"📌 {name} — {target}")
    st.dataframe(df_out, use_container_width=True)
    st.bar_chart(df_out.set_index("Channel")["Spend"])

    # Save per-model CSV
    results_tables.append((f"{name}_{target}", df_out))

    # Summary row
    summary_rows.append({
        "Model": name, "Target": target, "Scenario": scenario.split(')')[0],
        "Total spend": total_spend, "Total revenue": total_contrib,
        "Total ROI": (total_contrib / total_spend) if total_spend > 0 else np.nan
    })

# ---------------------------
# Export & persist scenario pack
# ---------------------------
if results_tables:
    st.divider()
    st.subheader("📦 Download & Saved Runs")

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    # Write JSON+CSVs to budgets/<timestamp>/
    pack_meta = {
        "timestamp": batch_ts,
        "scenario": scenario,
        "y_value": y_value,
        "channels": all_channels,
        "config": {
            "costs": costs, "current_spend": cur_spend,
            "mins": mins, "maxs": maxs, "fixed": fixed,
            "curves": curves,
            **({"total_budget": float(total_budget)} if scenario.startswith("1)") else {})
        },
        "models": [m.get("name","Model") for m in models],
        "transform_sources": list(dataset_to_meta.keys())
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(pack_meta, f, indent=2)

    # Save CSVs
    for name, df_tab in results_tables:
        df_tab.to_csv(os.path.join(out_dir, f"{_safe(name)}.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Build Excel (if possible) or ZIP of CSVs
    sheets = {name[:31]: df for name, df in results_tables}
    sheets["Summary"] = summary_df

    def _excel_bytes_local(sheets: Dict[str, pd.DataFrame]) -> Optional[bytes]:
        try:
            import xlsxwriter  # noqa: F401
            with io.BytesIO() as buf:
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    for nm, df in sheets.items():
                        df.to_excel(writer, sheet_name=nm[:31], index=False)
                return buf.getvalue()
        except Exception:
            pass
        try:
            import openpyxl  # noqa: F401
            with io.BytesIO() as buf:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    for nm, df in sheets.items():
                        df.to_excel(writer, sheet_name=nm[:31], index=False)
                return buf.getvalue()
        except Exception:
            return None

    xbytes = _excel_bytes_local(sheets)
    if xbytes:
        st.download_button("⬇️ Download Excel (all models)", xbytes, file_name=f"budget_{batch_ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        with io.BytesIO() as zip_buf:
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for name, df in sheets.items():
                    zf.writestr(f"{_safe(name)}.csv", df.to_csv(index=False).encode("utf-8"))
            st.download_button("⬇️ Download ZIP (CSVs)", zip_buf.getvalue(), file_name=f"budget_{batch_ts}.zip",
                               mime="application/zip")

    st.success(f"Saved scenario pack to {out_dir}")
else:
    st.info("No outputs produced. Check that at least one selected model had positive coefficients.")
