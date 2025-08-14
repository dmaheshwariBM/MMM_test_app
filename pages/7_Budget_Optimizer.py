# pages/7_Budget_Optimizer.py
# Standalone budget optimizer + response curves (no core imports).
# v1.1.0  — Correct mROI via analytic derivatives + revenue/profit toggle.

from __future__ import annotations
import os, glob, json, math
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "BUDGET_OPTIMIZER_STANDALONE_v1_1_0"
st.title("Budget Optimization")
st.caption(f"Page ID: {PAGE_ID}")

# -----------------------------------------------
# Cross-page banners
# -----------------------------------------------
if st.session_state.get("last_saved_path"):
    st.success(f"Saved: {st.session_state['last_saved_path']}")
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# -----------------------------------------------
# Results-root discovery
# -----------------------------------------------
def _abs(p: str) -> str: return os.path.abspath(p)

def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f: f.write("ok")
        os.remove(probe); return True
    except Exception:
        return False

def pick_writable_results_root() -> str:
    prefs: List[str] = []
    env_dir = os.environ.get("MMM_RESULTS_DIR")
    if env_dir: prefs.append(_abs(env_dir))
    prefs += [_abs(os.path.expanduser("~/.mmm_results")),_abs("/tmp/mmm_results"),_abs("results")]
    for root in prefs:
        if _ensure_dir(root): return root
    fb = _abs(os.path.expanduser("~/mmm_results_fallback")); _ensure_dir(fb); return fb

RESULTS_ROOT = pick_writable_results_root()

# -----------------------------------------------
# Catalog helpers (load saved models)
# -----------------------------------------------
def load_models_catalog(results_roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []; seen = set()
    for root in results_roots:
        patt = os.path.join(root, "**", "*.json")
        for jf in sorted(glob.glob(patt, recursive=True)):
            if jf in seen: continue
            try:
                with open(jf, "r", encoding="utf-8") as f: rec = json.load(f)
                if not isinstance(rec, dict): continue
                # Index only "model-like" JSONs
                if ("coefficients" in rec) or ("coef" in rec) or ("impact_shares" in rec):
                    rec["_path"] = jf
                    ts = rec.get("batch_ts")
                    try:
                        rec["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else datetime.fromtimestamp(os.path.getmtime(jf))
                    except Exception:
                        rec["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                    rows.append(rec); seen.add(jf)
            except Exception:
                continue
    rows.sort(key=lambda r: r.get("_ts"), reverse=True)
    return rows

def get_channels(model: Dict[str, Any]) -> List[str]:
    ch = model.get("channels") or model.get("features") or list((model.get("coefficients") or model.get("coef") or {}).keys())
    return [c for c in ch if str(c).lower() not in ("intercept","_intercept","(intercept)","base")]

def get_transform_map(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return model.get("transform_config") or model.get("transforms") or {}

def get_current_spend(model: Dict[str, Any]) -> Dict[str, float]:
    base = model.get("current_spend") or model.get("baseline_spend") or {}
    return {k: float(base.get(k, 0.0)) for k in get_channels(model)}

def get_impact_shares(model: Dict[str, Any]) -> Dict[str, float]:
    inc = model.get("impact_shares") or model.get("decomposition") or {}
    clean = {k: float(v) for k, v in inc.items() if str(k).lower() not in ("base","intercept")}
    s = sum(abs(v) for v in clean.values()) or 1.0
    return {k: max(0.0, float(v))/s for k, v in clean.items()}

def get_segments(model: Dict[str, Any]) -> List[str]:
    segs = model.get("segments") or model.get("segment_values") or []
    return list(segs) if isinstance(segs, list) else []

# -----------------------------------------------
# Response curves + analytic derivatives
# -----------------------------------------------
def _negexp(sp, k=0.01, beta=1.0):
    sp = max(0.0, float(sp)); k = max(0.0, float(k)); beta = max(0.0, float(beta))
    return beta * (1.0 - math.exp(-k * sp))

def _negexp_deriv(sp, k=0.01, beta=1.0):
    sp = max(0.0, float(sp)); k = max(0.0, float(k)); beta = max(0.0, float(beta))
    return beta * k * math.exp(-k * sp)

def _log1p(sp, k=1.0):
    sp = max(0.0, float(sp)); k = max(1e-12, float(k))
    return math.log1p(k * sp)

def _log1p_deriv(sp, k=1.0):
    sp = max(0.0, float(sp)); k = max(1e-12, float(k))
    return k / (1.0 + k * sp)

def build_channel_response_and_deriv(model: Dict[str, Any], channel: str):
    """
    Returns (f, fprime), both mapping spend->effect (relative units).
    Scales both by impact share to preserve modeled relative contributions.
    """
    tfm = (get_transform_map(model) or {}).get(channel, {})
    ttype = str(tfm.get("transform", "negexp")).lower()
    k = float(tfm.get("k", 0.01 if ttype.startswith("negexp") else 1.0))
    beta = float(tfm.get("beta", 1.0))

    scale = float(get_impact_shares(model).get(channel, 1.0))

    if ttype == "log":
        def f(sp):  return scale * _log1p(sp, k=k)
        def fp(sp): return scale * _log1p_deriv(sp, k=k)
    else:  # treat negexp & negexp_cann similarly for stand-alone curves
        def f(sp):  return scale * _negexp(sp, k=k, beta=beta)
        def fp(sp): return scale * _negexp_deriv(sp, k=k, beta=beta)

    # Fallback: numeric symmetric derivative if needed elsewhere
    def fp_numeric(sp, h=1e-3):
        return (f(sp + h) - f(max(0.0, sp - h))) / (2*h)
    return f, fp, fp_numeric

# -----------------------------------------------
# Load catalog & pick model (preselect from Results)
# -----------------------------------------------
CANDIDATES = [RESULTS_ROOT]
catalog = load_models_catalog(CANDIDATES)
if not catalog:
    st.info("No saved models found. Run a model on the Modeling/Results page and click Save."); st.stop()

preselect_path = None
if st.session_state.get("optimizer_model_path") and os.path.exists(str(st.session_state["optimizer_model_path"])):
    preselect_path = st.session_state["optimizer_model_path"]
elif st.session_state.get("last_saved_path") and str(st.session_state["last_saved_path"]).endswith(".json") and os.path.exists(str(st.session_state["last_saved_path"])):
    preselect_path = st.session_state["last_saved_path"]

labels = [f"{m.get('name','(unnamed)')} | {m.get('type','base')} | {m.get('target','?')} | {m.get('_ts')}" for m in catalog]
default_idx = 0
if preselect_path:
    for i, r in enumerate(catalog):
        if r.get("_path") == preselect_path:
            default_idx = i; break

model_idx = st.selectbox("Model to optimize", options=list(range(len(catalog))), format_func=lambda i: labels[i], index=default_idx)
model = catalog[model_idx]
model_path = model.get("_path","")
st.caption(f"Loaded model: {model.get('name','(unnamed)')} • {model.get('type','?')} • path: {model_path}")

channels = get_channels(model)
if not channels:
    st.error("This saved model has no channels/features. Please save a valid model."); st.stop()

# Optional segment support
segments = get_segments(model)
segment = st.selectbox("Segment (optional)", options=["(all)"] + segments, index=0) if segments else None
if segment == "(all)": segment = None

# -----------------------------------------------
# Constraints table
# -----------------------------------------------
st.subheader("Channels & constraints")

baseline_spend = get_current_spend(model)
rows = []
for ch in channels:
    rows.append({
        "channel": ch,
        "start_spend": float(baseline_spend.get(ch, 0.0)),
        "min": 0.0,
        "max": float(baseline_spend.get(ch, 0.0)) * 3.0 if baseline_spend else 1e9,
        "lock": False,
        "step": max(1.0, round(float(baseline_spend.get(ch, 0.0)) * 0.02, 2)) if baseline_spend else 1.0,
    })
df_constraints = pd.DataFrame(rows)

df_constraints = st.data_editor(
    df_constraints, key="opt_constraints", use_container_width=True, num_rows="fixed",
    column_config={
        "channel": st.column_config.TextColumn("Channel", disabled=True, help="Feature name from the saved model"),
        "start_spend": st.column_config.NumberColumn("Starting spend", step=1.0, min_value=0.0),
        "min": st.column_config.NumberColumn("Min", step=1.0, min_value=0.0),
        "max": st.column_config.NumberColumn("Max", step=1.0, min_value=0.0),
        "lock": st.column_config.CheckboxColumn("Lock"),
        "step": st.column_config.NumberColumn("Step", step=1.0, min_value=0.01),
    }
)

# Build response + derivative functions for each channel
FUNCS: Dict[str, Any] = {}
DERIVS: Dict[str, Any] = {}
DERIVS_NUM: Dict[str, Any] = {}
for ch in channels:
    f, fp, fp_num = build_channel_response_and_deriv(model, ch)
    FUNCS[ch] = f; DERIVS[ch] = fp; DERIVS_NUM[ch] = fp_num  # keep numeric fallback

# -----------------------------------------------
# Economic settings for mROI
# -----------------------------------------------
st.subheader("Economic settings (for mROI)")

c1, c2, c3 = st.columns(3)
with c1:
    unit_cost = st.number_input("Unit cost (per spend unit)", min_value=0.0, value=1.0, step=0.1,
                                help="Your currency per unit of spend (e.g., $1 per spend unit).")
with c2:
    unit_value = st.number_input("Outcome value per unit", min_value=0.0, value=1.0, step=0.1,
                                 help="Value of 1 outcome unit (e.g., $ revenue per sale).")
with c3:
    roi_mode = st.selectbox("mROI mode", options=["Revenue mROI", "Profit mROI"], index=0,
                            help="Revenue mROI ignores margin; Profit mROI multiplies by gross margin %.")

margin = 1.0
if roi_mode == "Profit mROI":
    margin = st.slider("Gross margin %", min_value=0, max_value=100, value=50, step=1,
                       help="Applied only for Profit mROI.") / 100.0

val_factor = unit_value * (margin if roi_mode == "Profit mROI" else 1.0)

# -----------------------------------------------
# Scenarios
# -----------------------------------------------
st.subheader("Scenario")
scenario = st.selectbox(
    "Choose scenario",
    options=[
        "1) Maximize response at target total budget",
        "2) Reallocate current total to maximize profit (mROI ≥ 1)",
        "3) Maximize profit with custom mROI floor"
    ],
    index=0
)

current_total = float(df_constraints["start_spend"].sum())
d1, d2, d3 = st.columns(3)
with d1: st.metric("Current total (table)", f"{current_total:,.2f}")
with d2: floor = st.number_input("mROI floor", min_value=0.0, value=1.0, step=0.1,
                                 help="Only allocate where marginal ROI ≥ this value (scenarios 2 & 3).")
with d3:
    target_budget = st.number_input("Target total budget", min_value=0.0,
                                    value=current_total if current_total > 0 else 100.0, step=1.0,
                                    help="Used only in scenario 1.")

# -----------------------------------------------
# Helpers: mROI at spend and greedy allocators
# -----------------------------------------------
def mroi_at(channel: str, spend: float) -> float:
    """
    Marginal ROI at spend s:
      mROI(s) = f'(s) * unit_value * (margin if profit) / unit_cost
    Uses analytic derivative; falls back to numeric symmetric derivative if needed.
    """
    try:
        m = DERIVS[channel](float(spend))
    except Exception:
        m = DERIVS_NUM[channel](float(spend))
    denom = unit_cost if unit_cost > 0 else 1e-12
    return (m * val_factor) / denom

def greedy_max_response(funcs: Dict[str, Any], start: Dict[str, float], min_b: Dict[str, float],
                        max_b: Dict[str, float], total_budget: float, step: float,
                        locks: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
    ch = list(funcs.keys()); step = float(max(1e-9, step))
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    if locks:
        for c, is_locked in locks.items():
            if is_locked:
                v = float(start.get(c, 0.0)); spend[c] = v
                min_b[c] = v; max_b[c] = v
    current_total = sum(spend.values()); target_total = float(total_budget)
    def mg(c, s): return funcs[c](s + step) - funcs[c](s)
    if current_total < target_total:
        iters = int((target_total - current_total) / step + 0.5)
        for _ in range(max(0, iters)):
            best_c, best_gain = None, -1e18
            for c in ch:
                if spend[c] + step <= max_b.get(c, 1e18) + 1e-12:
                    g = mg(c, spend[c])
                    if g > best_gain: best_gain, best_c = g, c
            if best_c is None: break
            spend[best_c] += step
    elif current_total > target_total:
        iters = int((current_total - target_total) / step + 0.5)
        for _ in range(max(0, iters)):
            best_c, best_loss = None, 1e18
            for c in ch:
                if spend[c] - step >= min_b.get(c, 0.0) - 1e-12:
                    loss = funcs[c](spend[c]) - funcs[c](spend[c] - step)
                    if loss < best_loss: best_loss, best_c = loss, c
            if best_c is None: break
            spend[best_c] -= step
    for c in ch:
        spend[c] = min(max(spend[c], min_b.get(c, 0.0)), max_b.get(c, 1e18))
    return spend

def greedy_profit_with_floor(start: Dict[str, float], min_b: Dict[str, float], max_b: Dict[str, float],
                             mroi_floor: float, step: float, locks: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
    """
    Profit objective with analytic mROI:
      add increments where mROI(s) >= floor, until no further qualifying increments.
    """
    ch = list(FUNCS.keys()); step = float(max(1e-9, step))
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    if locks:
        for c, is_locked in locks.items():
            if is_locked:
                v = float(start.get(c, 0.0)); spend[c] = v; min_b[c] = v; max_b[c] = v

    improved = True
    while improved:
        improved = False
        best_c, best_mroi = None, mroi_floor
        for c in ch:
            if spend[c] + step <= max_b.get(c, 1e18) + 1e-12:
                r = mroi_at(c, spend[c])  # analytic marginal ROI
                if r >= mroi_floor and r > best_mroi + 1e-12:
                    best_mroi, best_c = r, c
        if best_c is not None:
            spend[best_c] += step
            improved = True

    for c in ch:
        spend[c] = min(max(spend[c], min_b.get(c, 0.0)), max_b.get(c, 1e-18))
    return spend

# -----------------------------------------------
# Run optimization
# -----------------------------------------------
if st.button("Run optimization", type="primary", use_container_width=True):
    try:
        start = {r["channel"]: float(r["start_spend"]) for _, r in df_constraints.iterrows()}
        min_b = {r["channel"]: float(r["min"]) for _, r in df_constraints.iterrows()}
        max_b = {r["channel"]: float(r["max"]) for _, r in df_constraints.iterrows()}
        locks = {r["channel"]: bool(r["lock"]) for _, r in df_constraints.iterrows()}
        steps = {r["channel"]: float(max(r["step"], 1e-9)) for _, r in df_constraints.iterrows()}
        global_step = min(steps.values()) if steps else 1.0

        if scenario.startswith("1)"):
            result_spend = greedy_max_response(FUNCS, start, min_b, max_b, total_budget=float(target_budget), step=global_step, locks=locks)
        elif scenario.startswith("2)"):
            result_spend = greedy_profit_with_floor(start, min_b, max_b, mroi_floor=1.0, step=global_step, locks=locks)
        else:
            result_spend = greedy_profit_with_floor(start, min_b, max_b, mroi_floor=float(floor), step=global_step, locks=locks)

        # Summaries
        result_df = pd.DataFrame({
            "channel": list(result_spend.keys()),
            "optimized_spend": [result_spend[c] for c in result_spend],
            "start_spend": [start.get(c, 0.0) for c in result_spend],
        })
        result_df["delta"] = result_df["optimized_spend"] - result_df["start_spend"]

        # Effects and mROI near the optimized point
        eff_start = {c: FUNCS[c](start.get(c, 0.0)) for c in result_spend}
        eff_opt   = {c: FUNCS[c](result_spend[c]) for c in result_spend}
        result_df["effect_start"] = result_df["channel"].map(lambda c: eff_start[c])
        result_df["effect_opt"]   = result_df["channel"].map(lambda c: eff_opt[c])
        result_df["effect_delta"] = result_df["effect_opt"] - result_df["effect_start"]
        result_df["mROI_opt"]     = result_df.apply(lambda r: mroi_at(r["channel"], r["optimized_spend"]), axis=1)

        st.success("Optimization complete.")
        st.dataframe(result_df.sort_values("optimized_spend", ascending=False), use_container_width=True)

        st.session_state["opt_last_result"] = {
            "model_path": model_path,
            "model_name": model.get("name", ""),
            "scenario": scenario,
            "unit_cost": unit_cost,
            "unit_value": unit_value,
            "roi_mode": roi_mode,
            "gross_margin": margin,
            "mroi_floor": float(floor),
            "target_budget": float(target_budget) if scenario.startswith("1)") else None,
            "allocation": result_spend,
            "constraints_table": df_constraints.to_dict(orient="list"),
            "result_table": result_df.to_dict(orient="records"),
        }

    except Exception as e:
        st.error(f"Optimization failed: {e}")

# -----------------------------------------------
# Save optimization (JSON + CSV)
# -----------------------------------------------
st.divider()
st.subheader("Save optimization")

opt_state = st.session_state.get("opt_last_result")
default_name = f"{(opt_state or {}).get('model_name','')}_{(opt_state or {}).get('scenario','').split(')')[0]}".strip("_").replace(" ", "_")
scenario_name = st.text_input("Scenario name", value=default_name or "optimization_run")

if st.button("Save scenario"):
    if not opt_state:
        st.warning("Run an optimization first.")
    else:
        try:
            out_dir = os.path.join(RESULTS_ROOT, "optimizer"); os.makedirs(out_dir, exist_ok=True)
            base = scenario_name if scenario_name else "optimization_run"
            json_path = os.path.join(out_dir, base + ".json")
            csv_path  = os.path.join(out_dir, base + ".csv")

            with open(json_path, "w", encoding="utf-8") as f: json.dump(opt_state, f, ensure_ascii=False, indent=2)
            pd.DataFrame(opt_state["result_table"]).to_csv(csv_path, index=False)

            st.session_state["last_saved_path"] = json_path
            st.session_state["last_save_error"] = ""
            st.success(f"Saved: {json_path}")
        except Exception as e:
            st.session_state["last_saved_path"] = ""
            st.session_state["last_save_error"] = f"Save failed: {e}"
            st.error(st.session_state["last_save_error"])

# -----------------------------------------------
# Response curves
# -----------------------------------------------
st.divider()
st.subheader("Response curves")

try:
    import altair as alt
    ALT = True
except Exception:
    ALT = False

curve_channels = st.multiselect("Select channels", options=channels, default=channels[:min(5, len(channels))])

max_hint = {str(r["channel"]): float(r["max"]) for _, r in df_constraints.iterrows()}
x_points = st.number_input("Curve horizon (max spend per channel)", min_value=10.0,
                           value=max(100.0, max(max_hint.values()) if max_hint else 100.0), step=10.0)
n_points = st.slider("Points per curve", min_value=20, max_value=200, value=100)

seg_opts = get_segments(model)
seg_choice = st.selectbox("Segment (optional)", options=["(all)"] + seg_opts, index=0) if seg_opts else None
if seg_choice == "(all)": seg_choice = None

if st.button("Generate curves"):
    try:
        rows = []
        for ch in curve_channels:
            f, fp, _ = build_channel_response_and_deriv(model, ch)
            xs = np.linspace(0.0, float(x_points), int(n_points))
            ys = [f(x) for x in xs]
            mrois = [(fp(x) * val_factor) / (unit_cost if unit_cost > 0 else 1e-12) for x in xs]
            for x, y, r in zip(xs, ys, mrois):
                rows.append({"channel": ch, "spend": float(x), "effect": float(y), "mROI": float(r), "segment": seg_choice or "(all)"})
        curve_df = pd.DataFrame(rows)
        if ALT:
            chart = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    x=alt.X("spend:Q", title="Spend"),
                    y=alt.Y("effect:Q", title="Predicted effect (relative units)"),
                    color="channel:N",
                    tooltip=["channel:N", alt.Tooltip("spend:Q", format=".1f"),
                             alt.Tooltip("effect:Q", format=".3f"), alt.Tooltip("mROI:Q", format=".2f")]
                )
                .properties(width="container", height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(curve_df.pivot(index="spend", columns="channel", values="effect"))
        st.dataframe(curve_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Curve generation failed: {e}")
