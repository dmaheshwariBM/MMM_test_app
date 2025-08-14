# pages/7_Budget_Optimizer.py
# v2.2.1  Budget optimizer + response curves (Altair), preselects last/forwarded model.

import os, json
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

# Graceful import with a friendly message if the core file is missing
try:
    from core import optimizer as BO
except Exception as e:
    st.error("Could not import core/optimizer.py. Please add that file (and core/__init__.py).")
    st.stop()

PAGE_ID = "BUDGET_OPTIMIZER_v2_2_1"
st.title("Budget Optimization")
st.caption(f"Page ID: {PAGE_ID}")

# Cross-page banners
if st.session_state.get("last_saved_path"):
    st.success(f"Saved: {st.session_state['last_saved_path']}")
if st.session_state.get("last_save_error"):
    st.error(st.session_state["last_save_error"])

# ---------------- Results-root & catalog ----------------
RESULTS_ROOT = BO.pick_writable_results_root()
CANDIDATES = [RESULTS_ROOT]

# Allow Results page to pass us a specific model path
preselect_path = None
if st.session_state.get("optimizer_model_path") and os.path.exists(st.session_state["optimizer_model_path"]):
    preselect_path = st.session_state["optimizer_model_path"]
elif st.session_state.get("last_saved_path") and str(st.session_state["last_saved_path"]).endswith(".json") and os.path.exists(st.session_state["last_saved_path"]):
    preselect_path = st.session_state["last_saved_path"]

catalog = BO.load_models_catalog(CANDIDATES)
if not catalog:
    st.info("No saved models found. Run a model on the Modeling page and click Save.")
    st.stop()

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

channels = BO.get_channels(model)
if not channels:
    st.error("This saved model has no channels/features. Please save a valid model."); st.stop()

# Segments (optional)
segments = BO.get_segments(model)
segment = st.selectbox("Segment (optional)", options=["(all)"] + segments, index=0) if segments else None
if segment == "(all)": segment = None

st.subheader("Channels & constraints")

baseline_spend = BO.get_current_spend(model)

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
    df_constraints,
    key="opt_constraints",
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "channel": st.column_config.TextColumn("Channel", disabled=True),
        "start_spend": st.column_config.NumberColumn("Starting spend", step=1.0, min_value=0.0),
        "min": st.column_config.NumberColumn("Min", step=1.0, min_value=0.0),
        "max": st.column_config.NumberColumn("Max", step=1.0, min_value=0.0),
        "lock": st.column_config.CheckboxColumn("Lock"),
        "step": st.column_config.NumberColumn("Step", step=1.0, min_value=0.01),
    }
)

# Response functions per channel
funcs: Dict[str, Any] = {ch: BO.build_channel_response_func(model, ch) for ch in channels}

st.subheader("Scenario")
scenario = st.selectbox(
    "Choose scenario",
    options=[
        "1) Maximize response at target total budget",
        "2) Reallocate current total to maximize profit (mROI ≥ 1)",
        "3) Maximize profit with custom constraints"
    ],
    index=0
)

current_total = float(df_constraints["start_spend"].sum())
c1, c2, c3 = st.columns(3)
with c1: st.metric("Current total (table)", f"{current_total:,.2f}")
with c2: unit_cost = st.number_input("Unit cost", min_value=0.0, value=1.0, step=0.1)
with c3: floor = st.number_input("mROI floor", min_value=0.0, value=1.0, step=0.1)

target_budget = None
if scenario.startswith("1)"):
    target_budget = st.number_input("Target total budget", min_value=0.0, value=current_total if current_total > 0 else 100.0, step=1.0)

if st.button("Run optimization", type="primary", use_container_width=True):
    try:
        start = {r["channel"]: float(r["start_spend"]) for _, r in df_constraints.iterrows()}
        min_b = {r["channel"]: float(r["min"]) for _, r in df_constraints.iterrows()}
        max_b = {r["channel"]: float(r["max"]) for _, r in df_constraints.iterrows()}
        locks = {r["channel"]: bool(r["lock"]) for _, r in df_constraints.iterrows()}
        steps = {r["channel"]: float(max(r["step"], 1e-9)) for _, r in df_constraints.iterrows()}
        global_step = min(steps.values()) if steps else 1.0

        if scenario.startswith("1)"):
            result_spend = BO.greedy_max_response(funcs, start, min_b, max_b, total_budget=float(target_budget), step=global_step, locks=locks)
        elif scenario.startswith("2)"):
            result_spend = BO.greedy_profit_with_floor(funcs, start, min_b, max_b, unit_cost=unit_cost, mroi_floor=floor, step=global_step, locks=locks)
        else:
            result_spend = BO.greedy_profit_with_floor(funcs, start, min_b, max_b, unit_cost=unit_cost, mroi_floor=floor, step=global_step, locks=locks)

        result_df = pd.DataFrame({
            "channel": list(result_spend.keys()),
            "optimized_spend": [result_spend[c] for c in result_spend],
            "start_spend": [start.get(c, 0.0) for c in result_spend],
        })
        result_df["delta"] = result_df["optimized_spend"] - result_df["start_spend"]

        eps = global_step
        def mroi_at(c, s): return (funcs[c](s) - funcs[c](max(0.0, s - eps))) / (unit_cost * eps + 1e-12)

        eff_start = {c: funcs[c](start.get(c, 0.0)) for c in result_spend}
        eff_opt   = {c: funcs[c](result_spend[c]) for c in result_spend}
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
            "mroi_floor": floor,
            "target_budget": target_budget,
            "allocation": result_spend,
            "constraints_table": df_constraints.to_dict(orient="list"),
            "result_table": result_df.to_dict(orient="records"),
        }
    except Exception as e:
        st.error(f"Optimization failed: {e}")

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
            out_dir = os.path.join(BO.pick_writable_results_root(), "optimizer"); os.makedirs(out_dir, exist_ok=True)
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

# ---------------- Response Curves ----------------
st.divider()
st.subheader("Response curves")

try:
    import altair as alt
    ALT = True
except Exception:
    ALT = False

curve_channels = st.multiselect("Select channels", options=channels, default=channels[:min(5, len(channels))])
# derive a sensible max x from constraints
max_hint = {str(r["channel"]): float(r["max"]) for _, r in df_constraints.iterrows()}
x_points = st.number_input("Curve horizon (max spend per channel)", min_value=10.0,
                           value=max(100.0, max(max_hint.values()) if max_hint else 100.0), step=10.0)
n_points = st.slider("Points per curve", min_value=20, max_value=200, value=100)

seg_opts = BO.get_segments(model)
segment = st.selectbox("Segment (optional)", options=["(all)"] + seg_opts, index=0) if seg_opts else None
if segment == "(all)": segment = None

if st.button("Generate curves"):
    try:
        rows = []
        for ch in curve_channels:
            f = BO.build_channel_response_func(model, ch)
            xs = np.linspace(0.0, float(x_points), int(n_points))
            ys = [f(x) for x in xs]
            for x, y in zip(xs, ys):
                rows.append({"channel": ch, "spend": float(x), "effect": float(y), "segment": segment or "(all)"})
        curve_df = pd.DataFrame(rows)
        if ALT:
            chart = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    x=alt.X("spend:Q", title="Spend"),
                    y=alt.Y("effect:Q", title="Predicted effect (relative units)"),
                    color="channel:N",
                    tooltip=["channel:N", alt.Tooltip("spend:Q", format=".1f"), alt.Tooltip("effect:Q", format=".3f")]
                )
                .properties(width="container", height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(curve_df.pivot(index="spend", columns="channel", values="effect"))
        st.dataframe(curve_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Curve generation failed: {e}")
