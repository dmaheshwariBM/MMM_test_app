# pages/7_Budget_Optimizer.py
# Budget Optimizer — Revenue-only, with ROI column + flexible RC max controls
# v2.8.0

from __future__ import annotations
import os, glob, json, math
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "BUDGET_OPTIMIZER_v2_8_0"
st.title("💸 Budget Optimization")
st.caption(f"Page ID: {PAGE_ID} — Blue Sky • Historical Optimized • Target Budget (Revenue only)")

# Banners from other pages
if st.session_state.get("last_saved_path"):
    st.success(f"Saved: {st.session_state['last_saved_path']}")
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# ---------- results root ----------
def _abs(p: str) -> str: return os.path.abspath(p)
def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w", encoding="utf-8") as f: f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False

def pick_writable_results_root() -> str:
    prefs: List[str] = []
    env_dir = os.environ.get("MMM_RESULTS_DIR")
    if env_dir: prefs.append(_abs(env_dir))
    prefs += [_abs(os.path.expanduser("~/.mmm_results")),
              _abs("/tmp/mmm_results"),
              _abs("results")]
    for root in prefs:
        if _ensure_dir(root): return root
    fb = _abs(os.path.expanduser("~/mmm_results_fallback"))
    _ensure_dir(fb); return fb

RESULTS_ROOT = pick_writable_results_root()

# ---------- catalog ----------
def load_models_catalog(results_roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []; seen = set()
    for root in results_roots:
        patt = os.path.join(root, "**", "*.json")
        for jf in sorted(glob.glob(patt, recursive=True)):
            if jf in seen: continue
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if not isinstance(rec, dict): 
                    continue
                if ("coefficients" in rec) or ("coef" in rec) or ("impact_shares" in rec) or (rec.get("type") in {"base","composite","advanced"}):
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

# ---------- response curves ----------
def _negexp(sp, k=0.01, beta=1.0):
    sp = max(0.0, float(sp)); k = max(0.0, float(k)); beta = max(0.0, float(beta))
    return beta * (1.0 - math.exp(-k * sp))

def _negexp_deriv(sp, k=0.01, beta=1.0):
    sp = max(0.0, float(sp)); k = max(0.0, float(k)); beta = max(0.0, float(beta))
    return beta * k * math.exp(-k * sp)

def _log1p(sp, k=1.0):
    sp = max(0.0, float(sp)); k = max(1e-9, float(k))
    return math.log1p(k * sp)

def _log1p_deriv(sp, k=1.0):
    sp = max(0.0, float(sp)); k = max(1e-9, float(k))
    return k / (1.0 + k * sp)

def _auto_k_log(s_ref: float) -> float:
    # make log curve show curvature over observed scale
    s_ref = max(1.0, float(s_ref))
    return 1.0 / s_ref

def _auto_k_negexp(s50: float) -> float:
    # 50% saturation at s50
    s50 = max(1.0, float(s50))
    return math.log(2.0) / s50

def build_channel_response_and_deriv(model: Dict[str, Any], channel: str,
                                     start_spend_ref: float,
                                     max_spend_ref: float,
                                     auto_calibrate: bool = True):
    tfm = (get_transform_map(model) or {}).get(channel, {})
    ttype = str(tfm.get("transform", "negexp")).lower()
    k_in = tfm.get("k", None)
    beta_in = tfm.get("beta", None)

    s_ref = max(start_spend_ref, 1.0)
    if ttype == "log":
        k = float(k_in) if k_in is not None and float(k_in) > 0 else (_auto_k_log(s_ref) if auto_calibrate else 1.0)
        beta = 1.0
        def base_f(sp):  return _log1p(sp, k=k)
        def base_fp(sp): return _log1p_deriv(sp, k=k)
    else:
        k = float(k_in) if k_in is not None and float(k_in) > 0 else (_auto_k_negexp(s_ref) if auto_calibrate else 0.01)
        beta = float(beta_in) if beta_in is not None and float(beta_in) > 0 else 1.0
        def base_f(sp):  return _negexp(sp, k=k, beta=beta)
        def base_fp(sp): return _negexp_deriv(sp, k=k, beta=beta)

    scale = float(get_impact_shares(model).get(channel, 1.0))
    def f(sp):  return scale * base_f(sp)
    def fp(sp): return scale * base_fp(sp)

    def fp_numeric(sp, h=1e-3): return (f(sp + h) - f(max(0.0, sp - h))) / (2*h)
    return f, fp, fp_numeric, {"k": k, "beta": beta, "scale": scale, "type": ttype}

# ---------- load model ----------
CANDIDATES = [RESULTS_ROOT]
catalog = load_models_catalog(CANDIDATES)
if not catalog:
    st.info("No saved models found. Run/save a model first in Modeling/Results."); st.stop()

preselect_path = st.session_state.get("optimizer_model_path") or st.session_state.get("last_saved_path")
pre_idx = 0
if preselect_path:
    for i, m in enumerate(catalog):
        if m.get("_path") == preselect_path: pre_idx = i; break

labels = [f"{m.get('name','(unnamed)')} | {m.get('type','base')} | {m.get('target','?')} | {m.get('_ts')}" for m in catalog]
mid = st.selectbox("Model to optimize", options=list(range(len(catalog))), format_func=lambda i: labels[i], index=pre_idx)
model = catalog[mid]
model_path = model.get("_path","")
st.caption(f"Loaded: {model.get('name','(unnamed)')} • {model.get('type','?')} • `{model_path}`")

channels = get_channels(model)
if not channels:
    st.error("Saved model has no channels/features."); st.stop()

# Optional segment (if present)
segments = get_segments(model)
seg = st.selectbox("Segment (optional)", options=["(all)"] + segments, index=0) if segments else None
if seg == "(all)": seg = None

# ---------- constraints ----------
st.subheader("Channels & constraints")

baseline_spend = get_current_spend(model)
rows = []
for ch in channels:
    base_sp = float(baseline_spend.get(ch, 0.0))
    rows.append({
        "channel": ch,
        "start_spend": base_sp,
        "min": 0.0,
        "max": (base_sp * 3.0) if base_sp > 0 else 1e9,
        "lock_hist": False,
        "unit_cost": 1.0,
        "step": max(1.0, round(base_sp * 0.02, 2)) if base_sp > 0 else 1.0,
        "target_spend": None,
        "target_lock": False
    })
df_constraints = pd.DataFrame(rows)

df_constraints = st.data_editor(
    df_constraints, key="opt_constraints", use_container_width=True, num_rows="fixed",
    column_config={
        "channel":     st.column_config.TextColumn("Channel", disabled=True),
        "start_spend": st.column_config.NumberColumn("Starting spend", step=1.0, min_value=0.0),
        "min":         st.column_config.NumberColumn("Min", step=1.0, min_value=0.0),
        "max":         st.column_config.NumberColumn("Max", step=1.0, min_value=0.0),
        "lock_hist":   st.column_config.CheckboxColumn("Keep historical"),
        "unit_cost":   st.column_config.NumberColumn("Unit cost", step=0.1, min_value=0.0),
        "step":        st.column_config.NumberColumn("Granularity step", step=0.1, min_value=1e-6),
        "target_spend":st.column_config.NumberColumn("Target spend (scenario 3)", step=1.0, min_value=0.0),
        "target_lock": st.column_config.CheckboxColumn("Lock to target (scenario 3)"),
    }
)

AUTO_CAL = st.checkbox("Auto-calibrate curvature (suggested)", value=True,
                       help="Adjust k (and beta for negexp) from historical scale so curves behave sensibly.")

# Build functions per channel
FUNCS: Dict[str, Any] = {}
DERIVS: Dict[str, Any] = {}
DERIVS_NUM: Dict[str, Any] = {}
META: Dict[str, Dict[str, Any]] = {}

for _, r in df_constraints.iterrows():
    ch = r["channel"]
    start_ref = float(r["start_spend"])
    max_ref   = float(r["max"])
    f, fp, fp_num, meta = build_channel_response_and_deriv(
        model, ch, start_spend_ref=start_ref, max_spend_ref=max_ref, auto_calibrate=AUTO_CAL
    )
    FUNCS[ch] = f; DERIVS[ch] = fp; DERIVS_NUM[ch] = fp_num; META[ch] = meta

# ---------- economics (Revenue only) ----------
st.subheader("Revenue settings")
unit_value = st.number_input("Outcome value per unit (revenue)", min_value=0.0, value=1.0, step=0.1,
                             help="Revenue generated by one outcome unit (e.g., per Rx).")
val_factor = unit_value  # revenue-only

UNIT_COST = {r["channel"]: float(r["unit_cost"]) for _, r in df_constraints.iterrows()}

# ---------- scenarios ----------
st.subheader("Scenario")
scenario = st.selectbox(
    "Choose scenario",
    options=[
        "1) Blue Sky / Revenue Maximization (mROI = 1 rule)",
        "2) Historical Optimized (reallocate, same total)",
        "3) Target Budget Optimization (overall target; optional per-channel targets)"
    ],
    index=0
)

current_total = float(df_constraints["start_spend"].sum())
d1, d2, d3 = st.columns(3)
with d1: st.metric("Current total (table)", f"{current_total:,.2f}")
with d2: floor = st.number_input("mROI floor (for Scenario 1 & 3)", min_value=0.0, value=1.0, step=0.1)
with d3: target_budget = st.number_input("Target total (Scenario 3)", min_value=0.0,
                                         value=current_total if current_total > 0 else 100.0, step=1.0)

# ---------- helpers ----------
def mroi_at(channel: str, s: float) -> float:
    """Marginal Revenue ROI at spend s: mROI(s) = f'(s) * unit_value / unit_cost_channel"""
    try: deriv = DERIVS[channel](float(s))
    except Exception: deriv = DERIVS_NUM[channel](float(s))
    denom = UNIT_COST.get(channel, 1.0)
    denom = denom if denom > 0 else 1e-9
    return (deriv * val_factor) / denom

def _mgain(func, s: float, h: float) -> float:
    # marginal gain in EFFECT units for +h (finite diff)
    if h <= 0: return -1e18
    try:
        return func(s + h) - func(s)
    except Exception:
        return (func(s + h) - func(max(0.0, s - h))) / (2*h)

def greedy_equalize_mroi(start: Dict[str, float], min_b: Dict[str, float], max_b: Dict[str, float],
                         locks: Dict[str, bool], steps: Dict[str, float], mroi_floor: float = 1.0) -> Dict[str, float]:
    ch = list(FUNCS.keys())
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    for c, is_locked in locks.items():
        if is_locked:
            v = float(start.get(c, 0.0))
            spend[c] = v; min_b[c] = v; max_b[c] = v

    guard = 0
    while guard < 2_000_000:
        guard += 1
        best_c, best_mroi = None, mroi_floor
        for c in ch:
            s = spend[c]
            if s + steps[c] <= max_b.get(c, 1e18) + 1e-12:
                r = mroi_at(c, s)
                if r >= mroi_floor and r > best_mroi + 1e-12:
                    best_mroi, best_c = r, c
        if best_c is None: break
        spend[best_c] += steps[best_c]

    for c in ch:
        spend[c] = min(max(spend[c], min_b.get(c, 0.0)), max_b.get(c, 1e18))
    return spend

def reallocate_equal_total(funcs: Dict[str, Any], start: Dict[str, float],
                           min_b: Dict[str, float], max_b: Dict[str, float],
                           steps: Dict[str, float], locks: Dict[str, bool],
                           max_iter: int = 500000, tol: float = 1e-12) -> Dict[str, float]:
    """
    Pairwise transfer with fixed total: move delta from lowest marginal gain to highest marginal gain.
    Optimizes EFFECT under a fixed budget (same as revenue ranking since unit_value is constant).
    """
    ch = list(funcs.keys())
    s = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    for c, is_locked in locks.items():
        if is_locked:
            v = float(start.get(c, 0.0))
            s[c] = v; min_b[c] = v; max_b[c] = v

    def mg(c):  # marginal gain per channel for +step
        h = steps[c]
        if s[c] + h > max_b.get(c, 1e18) + 1e-12: return -1e18
        return _mgain(funcs[c], s[c], h)

    def mloss(c):  # loss if we remove step
        h = steps[c]
        if s[c] - h < min_b.get(c, 0.0) - 1e-12: return 1e18
        return _mgain(funcs[c], s[c] - h, h)

    it = 0
    while it < max_iter:
        it += 1
        gains = {c: mg(c) for c in ch}
        losses = {c: mloss(c) for c in ch}

        c_add = max(gains, key=gains.get)
        c_rem = min(losses, key=losses.get)

        best_gain = gains[c_add]
        worst_loss = losses[c_rem]

        if best_gain - worst_loss > tol:
            delta = min(steps[c_add], steps[c_rem])
            if s[c_rem] - delta >= min_b.get(c_rem, 0.0) - 1e-12 and s[c_add] + delta <= max_b.get(c_add, 1e18) + 1e-12:
                s[c_rem] -= delta
                s[c_add] += delta
            else:
                break
        else:
            break
    return s

def waterfill_to_target(funcs: Dict[str, Any], start: Dict[str, float], min_b: Dict[str, float],
                        max_b: Dict[str, float], total_budget: float, steps: Dict[str, float],
                        locks: Dict[str, bool], max_iter: int = 500000) -> Dict[str, float]:
    """Allocate to max EFFECT (equals revenue ranking) under budget target."""
    ch = list(funcs.keys())
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    for c, is_locked in locks.items():
        if is_locked:
            v = float(start.get(c, 0.0))
            spend[c] = v; min_b[c] = v; max_b[c] = v

    cur_total = sum(spend.values()); tgt_total = float(total_budget)

    def mg(c, s, h): return _mgain(funcs[c], s, h)

    it = 0
    while cur_total + 1e-9 < tgt_total and it < max_iter:
        it += 1
        best_c, best_gain = None, -1e18
        for c in ch:
            h = steps[c]
            if spend[c] + h <= max_b.get(c, 1e18) + 1e-12:
                g = mg(c, spend[c], h)
                if g > best_gain: best_gain, best_c = g, c
        if best_c is None: break
        spend[best_c] += steps[best_c]
        cur_total += steps[best_c]

    it = 0
    while cur_total - 1e-9 > tgt_total and it < max_iter:
        it += 1
        worst_c, worst_loss = None, 1e18
        for c in ch:
            h = steps[c]
            if spend[c] - h >= min_b.get(c, 0.0) - 1e-12:
                loss = mg(c, spend[c]-h, h)
                if loss < worst_loss: worst_loss, worst_c = loss, c
        if worst_c is None: break
        spend[worst_c] -= steps[worst_c]
        cur_total -= steps[worst_c]

    for c in ch:
        spend[c] = min(max(spend[c], min_b.get(c, 0.0)), max_b.get(c, 1e18))
    return spend

# ---------- run ----------
if st.button("Run optimization", type="primary", use_container_width=True):
    try:
        start = {r["channel"]: float(r["start_spend"]) for _, r in df_constraints.iterrows()}
        min_b = {r["channel"]: float(r["min"]) for _, r in df_constraints.iterrows()}
        max_b = {r["channel"]: float(r["max"]) for _, r in df_constraints.iterrows()}
        locks  = {r["channel"]: bool(r["lock_hist"]) for _, r in df_constraints.iterrows()}
        steps  = {r["channel"]: float(max(r["step"], 1e-9)) for _, r in df_constraints.iterrows()}
        targets= {r["channel"]: (None if r["target_spend"] is None else float(r["target_spend"])) for _, r in df_constraints.iterrows()}
        tlocks = {r["channel"]: bool(r["target_lock"]) for _, r in df_constraints.iterrows()}

        # Scenario 3: enforce channel targets
        if scenario.startswith("3)"):
            for c in channels:
                if tlocks[c] and targets[c] is not None:
                    v = float(targets[c]); min_b[c] = v; max_b[c] = v

        if scenario.startswith("1)"):
            result_spend = greedy_equalize_mroi(start, min_b, max_b, locks, steps, mroi_floor=float(floor))
        elif scenario.startswith("2)"):
            result_spend = reallocate_equal_total(FUNCS, start, min_b, max_b, steps, locks)
        else:
            result_spend = waterfill_to_target(FUNCS, start, min_b, max_b, float(target_budget), steps, locks)

        # Summaries & diagnostics
        result_df = pd.DataFrame({
            "channel": list(result_spend.keys()),
            "optimized_spend": [result_spend[c] for c in result_spend],
            "start_spend": [start.get(c, 0.0) for c in result_spend],
            "unit_cost": [UNIT_COST.get(c, 1.0) for c in result_spend],
            "k": [META[c]["k"] for c in result_spend],
            "beta": [META[c]["beta"] for c in result_spend],
            "transform": [META[c]["type"] for c in result_spend],
        })
        result_df["delta"] = result_df["optimized_spend"] - result_df["start_spend"]

        eff_start = {c: FUNCS[c](start.get(c, 0.0)) for c in result_spend}
        eff_opt   = {c: FUNCS[c](result_spend[c]) for c in result_spend}
        result_df["effect_start"] = result_df["channel"].map(lambda c: eff_start[c])
        result_df["effect_opt"]   = result_df["channel"].map(lambda c: eff_opt[c])
        result_df["effect_delta"] = result_df["effect_opt"] - result_df["effect_start"]

        # mROI at start & opt (revenue)
        result_df["mROI_at_opt"] = result_df.apply(lambda r: mroi_at(r["channel"], r["optimized_spend"]), axis=1)
        result_df["mROI_at_start"] = result_df.apply(lambda r: mroi_at(r["channel"], r["start_spend"]), axis=1)

        # ROI (average ROI at optimized point) = Revenue / Spend
        def _avg_roi(row):
            cost = row["optimized_spend"] * max(row["unit_cost"], 1e-12)
            rev  = row["effect_opt"] * val_factor
            return float(rev / cost) if cost > 0 else float("nan")
        result_df["ROI_at_opt"] = result_df.apply(_avg_roi, axis=1)

        st.success("Optimization complete.")
        st.dataframe(result_df.sort_values("optimized_spend", ascending=False), use_container_width=True)

        # Store state
        st.session_state["opt_last_result"] = {
            "model_path": model_path,
            "model_name": model.get("name", ""),
            "scenario": scenario,
            "unit_value": unit_value,
            "mroi_floor": float(floor),
            "target_budget": float(target_budget) if scenario.startswith("3)") else sum(start.values()) if scenario.startswith("2)") else None,
            "allocation": {c: float(result_spend[c]) for c in result_spend},
            "constraints_table": df_constraints.to_dict(orient="list"),
            "result_table": result_df.to_dict(orient="records"),
        }

    except Exception as e:
        st.error(f"Optimization failed: {e}")

# ---------- save ----------
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

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(opt_state, f, ensure_ascii=False, indent=2)
            pd.DataFrame(opt_state["result_table"]).to_csv(csv_path, index=False)

            st.session_state["last_saved_path"] = json_path
            st.session_state["last_save_error"] = ""
            st.success(f"Saved: {json_path}")
        except Exception as e:
            st.session_state["last_saved_path"] = ""
            st.session_state["last_save_error"] = f"Save failed: {e}"
            st.error(st.session_state["last_save_error"])

# ---------- response curves ----------
st.divider()
st.subheader("Response curves")

try:
    import altair as alt
    ALT = True
except Exception:
    ALT = False

curve_channels = st.multiselect("Select channels", options=channels, default=channels[:min(5, len(channels))])
n_points = st.slider("Points per curve", min_value=50, max_value=400, value=150)

# Default per-channel max = constraint max
constraint_max_by_ch = {r["channel"]: float(r["max"]) for _, r in df_constraints.iterrows()}

# Horizon mode controls
curve_mode = st.radio(
    "Curve horizon",
    options=["Use constraint max per channel", "Use one global max", "Customize per channel"],
    index=0, horizontal=True
)

global_max = None
custom_max_df = None

if curve_mode == "Use one global max":
    global_max = st.number_input("Global curve max (applies to all selected channels)", min_value=10.0, value=float(max(100.0, max(constraint_max_by_ch.values() or [100.0]))), step=10.0)

elif curve_mode == "Customize per channel":
    # Build or reuse editable table
    if "curve_max_table" not in st.session_state:
        st.session_state["curve_max_table"] = pd.DataFrame({
            "channel": list(constraint_max_by_ch.keys()),
            "curve_max": [constraint_max_by_ch[c] for c in constraint_max_by_ch]
        })
    with st.expander("Per-channel curve max (editable)", expanded=True):
        custom_max_df = st.data_editor(
            st.session_state["curve_max_table"],
            key="curve_max_editor",
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "channel": st.column_config.TextColumn("Channel", disabled=True),
                "curve_max": st.column_config.NumberColumn("Curve max", min_value=0.0, step=10.0)
            }
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset to constraint max"):
                st.session_state["curve_max_table"]["curve_max"] = st.session_state["curve_max_table"]["channel"].map(constraint_max_by_ch).values
                st.experimental_rerun()
        with c2:
            st.caption("Edit the max spend horizon for each channel's RC plot.")

if st.button("Generate curves"):
    try:
        rows = []
        for ch in curve_channels:
            f = FUNCS[ch]; fp = DERIVS[ch]
            # Resolve x_max per channel based on chosen mode
            if curve_mode == "Use one global max" and global_max is not None:
                x_max = float(global_max)
            elif curve_mode == "Customize per channel" and custom_max_df is not None:
                # Pull from edited table; fallback to constraint max if missing
                tbl = custom_max_df.set_index("channel")["curve_max"].to_dict()
                x_max = float(tbl.get(ch, constraint_max_by_ch.get(ch, 100.0)))
            else:
                x_max = float(constraint_max_by_ch.get(ch, 100.0))

            x_max = max(10.0, x_max)
            xs = np.linspace(0.0, x_max, int(n_points))
            ys = [f(x) for x in xs]
            mrois = []
            denom = UNIT_COST.get(ch, 1.0); denom = denom if denom > 0 else 1e-9
            for x in xs:
                try:
                    der = fp(x)
                except Exception:
                    der = (f(x+1e-3)-f(max(0.0, x-1e-3)))/2e-3
                mrois.append((der * val_factor) / denom)
            for x, y, r in zip(xs, ys, mrois):
                rows.append({"channel": ch, "spend": float(x), "effect": float(y), "mROI": float(r)})
        curve_df = pd.DataFrame(rows)
        if curve_df.empty:
            st.info("No channels selected.")
        else:
            if ALT:
                e_chart = (
                    alt.Chart(curve_df)
                    .mark_line()
                    .encode(
                        x=alt.X("spend:Q", title="Spend"),
                        y=alt.Y("effect:Q", title="Predicted effect (relative units)"),
                        color="channel:N",
                        tooltip=["channel:N", alt.Tooltip("spend:Q", format=".1f"),
                                 alt.Tooltip("effect:Q", format=".3f"), alt.Tooltip("mROI:Q", format=".2f")]
                    )
                    .properties(width="container", height=360, title="Response curves")
                )
                r_chart = (
                    alt.Chart(curve_df)
                    .mark_line(strokeDash=[4,2])
                    .encode(
                        x=alt.X("spend:Q", title="Spend"),
                        y=alt.Y("mROI:Q", title="Marginal ROI (revenue)"),
                        color="channel:N",
                        tooltip=["channel:N", alt.Tooltip("spend:Q", format=".1f"),
                                 alt.Tooltip("mROI:Q", format=".2f")]
                    )
                    .properties(width="container", height=240, title="Marginal ROI curves")
                )
                st.altair_chart(e_chart, use_container_width=True)
                st.altair_chart(r_chart, use_container_width=True)
            else:
                st.line_chart(curve_df.pivot(index="spend", columns="channel", values="effect"))
            st.dataframe(curve_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Curve generation failed: {e}")
