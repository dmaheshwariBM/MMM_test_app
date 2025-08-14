# pages/6_Results.py
import os, re, glob, json, pathlib, importlib
from io import StringIO
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

PAGE_ID = "RESULTS_PAGE_STANDALONE_v1_4_0"

st.title("📊 Results — Compare, Inspect & Compose")
st.caption(f"Page ID: `{PAGE_ID}` • Standalone (no core/results.py needed)")

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# Utilities (standalone)
# ───────────────────────────────────────────────────────────────────────────────
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:72]

def _intercept_key(coef: Dict[str, float]) -> str | None:
    for k in ("const", "Intercept", "intercept", "CONST", "const_", "_const", "beta0", "b0"):
        if k in coef:
            return k
    return None

def _to_num_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "const":
        return pd.Series(np.ones(len(df)), index=df.index, name="const")
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    if name.endswith("__tfm") and name[:-5] in df.columns:
        return pd.to_numeric(df[name[:-5]], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, name=name)

def load_results_catalog(results_dir: str = RESULTS_DIR) -> List[Dict[str, Any]]:
    rows = []
    batches = sorted(glob.glob(os.path.join(results_dir, "*")), reverse=True)
    for b in batches:
        for jf in sorted(glob.glob(os.path.join(b, "*.json")), reverse=True):
            try:
                with open(jf, "r") as f:
                    r = json.load(f)
                r["_path"] = jf
                r["_batch"] = os.path.basename(b)
                ts = r.get("batch_ts") or os.path.basename(b)
                try:
                    r["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                except Exception:
                    r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                rows.append(r)
            except Exception:
                continue
    rows.sort(key=lambda x: x.get("_ts", datetime.min), reverse=True)
    return rows

def normalize_and_round(d: Dict[str, Any]) -> Dict[str, Any]:
    base_pct = float(d.get("base_pct", 0.0))
    carry_pct = float(d.get("carryover_pct", 0.0))
    impact_map: Dict[str, float] = dict(d.get("impactable_pct", {}))
    incr_pct = float(sum(impact_map.values()))
    total = base_pct + carry_pct + incr_pct
    if incr_pct > 0 and abs(total - 100.0) > 0.05:
        target_incr = max(0.0, 100.0 - base_pct - carry_pct)
        scale = target_incr / incr_pct if incr_pct > 0 else 1.0
        for k in list(impact_map.keys()):
            impact_map[k] *= scale
        incr_pct = float(sum(impact_map.values()))
    base_pct = float(round(base_pct, 6))
    carry_pct = float(round(carry_pct, 6))
    impact_map = {k: float(round(v, 6)) for k, v in impact_map.items()}
    incr_pct = float(round(sum(impact_map.values()), 6))
    return {"base_pct": base_pct, "carryover_pct": carry_pct, "incremental_pct": incr_pct, "impactable_pct": impact_map}

def ensure_decomp(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Robust decomp for display:
      Denominator: sum(target) → sum(yhat) → sum(contrib) → 1.0
      Detect intercept (const/Intercept/...) even if features omitted it.
      Normalize so Base+Carry+Incremental ≈ 100
    """
    d = record.get("decomp")
    if isinstance(d, dict) and "impactable_pct" in d:
        return normalize_and_round(d)

    dataset = record.get("dataset")
    features = record.get("features", []) or []
    coef = record.get("coef", {}) or {}
    yhat = np.asarray(record.get("yhat", []), float)

    if not dataset:
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}
    path = os.path.join(DATA_DIR, dataset)
    if not os.path.exists(path):
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {"base_pct": np.nan, "carryover_pct": 0.0, "incremental_pct": np.nan, "impactable_pct": {}}

    # Build contributions (include intercept even if not in features)
    contrib_sum: Dict[str, float] = {}
    n = len(df)
    ik = _intercept_key(coef)
    if ik is not None:
        contrib_sum["const"] = float(coef.get(ik, 0.0)) * n

    for f in features:
        if f == "const":
            continue
        c = float(coef.get(f, 0.0))
        x = _to_num_series(df, f)
        contrib_sum[f] = float((c * x).clip(lower=0.0).sum())

    # Denominator
    total_from_y = None
    tgt = record.get("target")
    if tgt and tgt in df.columns:
        total_from_y = float(pd.to_numeric(df[tgt], errors="coerce").fillna(0.0).sum())
    total_from_yhat = float(np.nansum(yhat)) if yhat.size > 0 else 0.0
    total_from_contrib = float(sum(contrib_sum.values())) if contrib_sum else 0.0
    candidates = [t for t in (total_from_y, total_from_yhat, total_from_contrib) if t and t > 0]
    total_pred = candidates[0] if candidates else 1.0

    base_sum = float(contrib_sum.get("const", 0.0))
    base_pct = 100.0 * base_sum / total_pred

    impact_map: Dict[str, float] = {}
    for f, s in contrib_sum.items():
        if f == "const":
            continue
        disp = f[:-5] if f.endswith("__tfm") else f
        if s > 0:
            impact_map[disp] = impact_map.get(disp, 0.0) + 100.0 * s / total_pred

    return normalize_and_round({
        "base_pct": base_pct,
        "carryover_pct": 0.0,
        "incremental_pct": float(sum(impact_map.values())),
        "impactable_pct": impact_map
    })

def metrics_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {}) or {}
    return {"R²": m.get("r2"), "Adj R²": m.get("adj_r2"), "RMSE": m.get("rmse"), "n": m.get("n"), "p": m.get("p")}

def _short_type(t: str) -> str:
    return {
        "base": "base",
        "breakout_split": "breakout",
        "residual_reattribute": "residual",
        "pathway_redistribute": "pathway",
        "composite": "composite",
    }.get(str(t), str(t) or "base")

def fmt_label(r: Dict[str, Any]) -> str:
    nm = r.get("name","(unnamed)")
    tp = _short_type(r.get("type","base"))
    tgt = r.get("target","?")
    ts = r.get("_ts")
    when = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
    return f"{nm} • {tp} • {tgt} • {when}"

def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, name))

# ───────────────────────────────────────────────────────────────────────────────
# Import advanced models core (for the composer). Falls back to file path.
# ───────────────────────────────────────────────────────────────────────────────
def _import_advanced_models():
    try:
        from core import advanced_models as am  # type: ignore
        return am
    except Exception:
        pass
    core_path = pathlib.Path("core/advanced_models.py").resolve()
    if not core_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("advanced_models_local", core_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore
    return module

am = _import_advanced_models()
if am is None:
    st.warning("Advanced builder is disabled because `core/advanced_models.py` was not found.")
else:
    st.caption(f"Advanced core loaded from: `{getattr(am, '__file__', 'unknown')}` (v {getattr(am, 'ADV_MODELS_VERSION','?')})")

# ───────────────────────────────────────────────────────────────────────────────
# Top actions: refresh, diagnostics
# ───────────────────────────────────────────────────────────────────────────────
cols_top = st.columns([1,1,4])
with cols_top[0]:
    if st.button("🔄 Refresh list"):
        st.rerun()
with cols_top[1]:
    show_diag = st.toggle("Diagnostics", value=False, help="Show raw files found under /results")

# ───────────────────────────────────────────────────────────────────────────────
# Load catalog
# ───────────────────────────────────────────────────────────────────────────────
catalog = load_results_catalog(RESULTS_DIR)
if show_diag:
    st.info(f"Found {len(catalog)} saved JSONs under `{RESULTS_DIR}/**/*.json`")
    for r in catalog[:15]:
        st.write("•", r.get("_path"))

if not catalog:
    st.info("No saved models yet. Build & save a model (Modeling or Advanced Models), then come back here.")
    st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# Tabs: Compare | Inspect | Compose adjustments
# ───────────────────────────────────────────────────────────────────────────────
tab_cmp, tab_inspect, tab_compose = st.tabs(["📊 Compare", "🔎 Inspect one model", "🧩 Compose adjustments"])

# ╭───────────────────────────── Compare ─────────────────────────────╮
with tab_cmp:
    st.markdown("#### All runs (latest first)")
    summary_rows = []
    for r in catalog:
        summary_rows.append({
            "Name": r.get("name",""),
            "Type": (r.get("type") or "base"),
            "Target": r.get("target",""),
            "Saved at": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, height=min(400, 48*len(summary_rows)+40))

    labels = [fmt_label(r) for r in catalog]
    default_sel = labels[:2] if len(labels) >= 2 else labels[:1]
    chosen = st.multiselect("Select up to 5 models to compare", options=labels, default=default_sel, max_selections=5)
    if chosen:
        models = [catalog[labels.index(lbl)] for lbl in chosen]
        baseline = st.selectbox("Baseline for Δ", options=chosen, index=0)
        st.divider()

        st.subheader("Fit metrics")
        met = []
        for lbl, r in zip(chosen, models):
            row = metrics_row(r); row["Model"] = lbl
            met.append(row)
        met_df = pd.DataFrame(met).set_index("Model")
        st.dataframe(met_df, use_container_width=True)

        st.subheader("Decomposition summary")
        decomp_map = {lbl: ensure_decomp(r) for lbl, r in zip(chosen, models)}
        decomp_rows = []
        for lbl in chosen:
            d = decomp_map[lbl]
            decomp_rows.append({"Model": lbl, "Base %": d["base_pct"], "Carryover %": d["carryover_pct"], "Incremental %": d["incremental_pct"]})
        decomp_df = pd.DataFrame(decomp_rows).set_index("Model")
        st.dataframe(decomp_df, use_container_width=True)
        st.bar_chart(decomp_df)

        st.subheader("Impactable % by channel — aligned")
        channels = sorted({ch for d in decomp_map.values() for ch in d.get("impactable_pct", {}).keys()})
        if channels:
            mat = []
            for ch in channels:
                row = {"Channel": ch}
                for lbl in chosen:
                    row[lbl] = float(decomp_map[lbl]["impactable_pct"].get(ch, 0.0))
                mat.append(row)
            impact_df = pd.DataFrame(mat).set_index("Channel")
            st.dataframe(impact_df, use_container_width=True)

            st.markdown("**Δ vs baseline (selected above)**")
            if baseline in impact_df.columns:
                deltas = impact_df.subtract(impact_df[baseline], axis=0)
                st.dataframe(deltas, use_container_width=True)

        st.subheader("Export")
        csv_buf = StringIO()
        csv_buf.write("## metrics\n"); met_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        csv_buf.write("## decomposition\n"); decomp_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        if 'impact_df' in locals():
            csv_buf.write("## impactable_by_channel\n"); impact_df.reset_index().to_csv(csv_buf, index=False); csv_buf.write("\n")
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        st.download_button("Download comparison CSV", data=csv_bytes, file_name="mmm_results_comparison.csv", mime="text/csv")

        # NEW: Send to Budget Optimization
        st.subheader("Select for Budget Optimization")
        to_budget = st.multiselect("Models to send", options=chosen, default=chosen[:min(3, len(chosen))])
        if st.button("Send to Budget Optimization"):
            skinny = []
            for lbl in to_budget:
                r = models[chosen.index(lbl)]
                d = decomp_map[lbl]
                skinny.append({
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "dataset": r.get("dataset"),
                    "target": r.get("target"),
                    "metrics": r.get("metrics", {}),
                    "decomp": d,
                    "features": r.get("features", []),
                    "_path": r.get("_path", ""),
                    "_ts": r.get("_ts").strftime("%Y-%m-%d %H:%M:%S") if r.get("_ts") else "",
                })
            st.session_state["budget_candidates"] = skinny
            st.success(f"Sent {len(skinny)} model(s) to Budget Optimization.")

# ╭────────────────────────── Inspect one ────────────────────────────╮
with tab_inspect:
    st.markdown("#### Pick a saved run")
    labels = [fmt_label(r) for r in catalog]
    i = st.selectbox("Saved model", options=list(range(len(labels))), format_func=lambda k: labels[k], index=0)
    r = catalog[i]
    st.write(f"**Name**: {r.get('name')}  •  **Type**: {_short_type(r.get('type','base'))}  •  **Target**: {r.get('target')}  •  **Saved**: {r.get('_ts')}")
    d = ensure_decomp(r)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Base %", f"{d['base_pct']:.2f}%")
    with c2: st.metric("Carryover %", f"{d['carryover_pct']:.2f}%")
    with c3: st.metric("Incremental %", f"{d['incremental_pct']:.2f}%")
    imp = d.get("impactable_pct", {})
    if imp:
        dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())}).set_index("Channel")
        st.bar_chart(dfv)
        st.dataframe(dfv, use_container_width=True)

# ╭─────────────────────── Compose adjustments ───────────────────────╮
with tab_compose:
    if am is None:
        st.warning("Install/restore `core/advanced_models.py` to use the composition workbench.")
        st.stop()

    labels = [fmt_label(r) for r in catalog]
    k = st.selectbox("Base model to start from", options=list(range(len(labels))), format_func=lambda idx: labels[idx], index=0)
    base = catalog[k]
    st.caption(f"Dataset: **{base.get('dataset','?')}** • Target: **{base.get('target','?')}**")

    # session: pipeline of operations
    ops_key = "compose_ops"
    if ops_key not in st.session_state:
        st.session_state[ops_key] = []

    # load dataset + derive channel/feature lists
    try:
        df_base = _load_csv(base["dataset"])
    except Exception as e:
        st.error(f"Could not open dataset '{base['dataset']}'."); st.exception(e); st.stop()

    d0 = am._ensure_decomp_from_record_or_recompute(base, df_base)
    features = [f for f in base.get("features", []) if f != "const"]
    features_disp = [f[:-5] if f.endswith("__tfm") else f for f in features]
    base_set = set(features_disp)
    numeric_cols = [c for c in df_base.columns if pd.api.types.is_numeric_dtype(df_base[c])]
    not_in_base = [c for c in numeric_cols if (c not in base_set and not c.startswith("_tfm_"))]

    st.markdown("##### Add operations")
    cB, cR, cP = st.columns(3)

    # Breakout builder
    with cB:
        st.write("**Breakout split**")
        parent = st.selectbox("Parent channel", options=features_disp, key="co_parent")
        subs = st.multiselect("Sub-metrics", options=not_in_base, key="co_subs")
        if st.button("➕ Add Breakout"):
            st.session_state[ops_key].append({"type":"breakout_split","parent": parent,"subs": subs})

    # Residual builder
    with cR:
        st.write("**Residual (on Base)**")
        extras = st.multiselect("Explain Base with", options=not_in_base, key="co_extras")
        frac = st.slider("Fraction of fitted Base", 0.1, 1.0, 1.0, 0.1, key="co_frac")
        if st.button("➕ Add Residual"):
            st.session_state[ops_key].append({"type":"residual_reattribute","extras": extras,"fraction": float(frac)})

    # Pathway builder
    with cP:
        st.write("**Pathway**")
        A = st.selectbox("From (A)", options=features_disp, key="co_A")
        B = st.selectbox("To (B)", options=[c for c in features_disp if c != A], key="co_B")
        if st.button("➕ Add Pathway"):
            st.session_state[ops_key].append({"type":"pathway_redistribute","A": A,"B": B})

    st.divider()

    # Pipeline table
    st.markdown("##### Pipeline")
    if not st.session_state[ops_key]:
        st.info("No operations yet. Add Breakout / Residual / Pathway above.")
    else:
        pretty = []
        for idx, op in enumerate(st.session_state[ops_key], 1):
            if op["type"]=="breakout_split":
                pretty.append({"#": idx, "Type": "breakout", "Parent": op["parent"], "Sub-metrics": ", ".join(op["subs"])})
            elif op["type"]=="residual_reattribute":
                pretty.append({"#": idx, "Type": "residual", "Extras": ", ".join(op["extras"]), "Fraction": op["fraction"]})
            else:
                pretty.append({"#": idx, "Type": "pathway", "A→B": f"{op['A']}→{op['B']}"})
        st.dataframe(pd.DataFrame(pretty), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⟲ Clear all"):
                st.session_state[ops_key] = []
                st.rerun()
        with c2:
            rm_idx = st.number_input("Remove step #", min_value=1, max_value=len(st.session_state[ops_key]), value=1, step=1)
        with c3:
            if st.button("🗑 Remove step"):
                st.session_state[ops_key].pop(int(rm_idx)-1)
                st.rerun()

    st.divider()
    # Preview apply pipeline
    if st.button("▶ Preview composed result", type="primary"):
        try:
            # start with base record; carry forward decomp after each apply
            rec = dict(base)
            current_decomp = am._ensure_decomp_from_record_or_recompute(rec, df_base)
            rec["decomp"] = current_decomp

            for op in st.session_state[ops_key]:
                if op["type"] == "breakout_split":
                    out = am.breakout_split(df=df_base, base_record=rec, channel_to_split=op["parent"], sub_metrics=op["subs"])
                elif op["type"] == "residual_reattribute":
                    out = am.residual_reattribute(df=df_base, base_record=rec, extra_channels=op["extras"], fraction=float(op["fraction"]))
                else:
                    out = am.pathway_redistribute(df=df_base, base_record=rec, channel_A=op["A"], channel_B=op["B"])
                current_decomp = am.apply_decomp_update(rec, df_base, out)
                rec["decomp"] = current_decomp  # so subsequent steps use the updated decomp

            st.success("Composed result preview")
            d = current_decomp
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Base %", f"{d['base_pct']:.2f}%")
            with c2: st.metric("Carryover %", f"{d['carryover_pct']:.2f}%")
            with c3: st.metric("Incremental %", f"{d['incremental_pct']:.2f}%")
            imp = d.get("impactable_pct", {})
            if imp:
                dfv = pd.DataFrame({"Channel": list(imp.keys()), "Impactable %": list(imp.values())}).set_index("Channel")
                st.bar_chart(dfv)
                st.dataframe(dfv, use_container_width=True)

            st.markdown("##### Save / Update / Send to Budget")
            cL, cM, cR = st.columns(3)
            with cL:
                nm = st.text_input("Save as name", value=f"{base.get('name','base')}__composite")
                if st.button("💾 Save as new result"):
                    # persist a new JSON next to others
                    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = os.path.join(RESULTS_DIR, batch_ts)
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"{batch_ts}__{_safe(nm)}__{_safe(base.get('target','?'))}.json"
                    # store the pipeline for auditability
                    payload = {**base}
                    payload.update({
                        "batch_ts": batch_ts,
                        "name": nm,
                        "target": base.get("target","?"),
                        "dataset": base.get("dataset","?"),
                        "type": "composite",
                        "pipeline": st.session_state[ops_key],
                        "decomp": current_decomp
                    })
                    with open(os.path.join(out_dir, fname), "w") as f:
                        json.dump(payload, f, indent=2)
                    st.success(f"Saved to `{out_dir}`")
                    st.rerun()  # <-- ensure the new composite appears in lists immediately
            with cM:
                # Send preview directly to Budget Optimization without saving
                if st.button("📤 Send PREVIEW to Budget Optimization"):
                    preview_skinny = [{
                        "name": f"{base.get('name','base')}__composite_preview",
                        "type": "composite",
                        "dataset": base.get("dataset"),
                        "target": base.get("target"),
                        "metrics": base.get("metrics", {}),
                        "decomp": current_decomp,
                        "features": base.get("features", []),
                        "_path": "(preview)",
                        "_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }]
                    st.session_state["budget_candidates"] = preview_skinny
                    st.success("Preview sent to Budget Optimization.")
            with cR:
                if st.button("✏ Update base result"):
                    # overwrite the selected base's decomp only
                    for jf in glob.glob(os.path.join(RESULTS_DIR, "*", "*.json")):
                        try:
                            with open(jf, "r") as f:
                                rec0 = json.load(f)
                            if str(rec0.get("name","")).strip().lower() == str(base.get("name","")).strip().lower():
                                keep = {k: rec0.get(k) for k in ("batch_ts","name","target","dataset","metrics","coef","yhat","features","type")}
                                keep["decomp"] = current_decomp
                                with open(jf, "w") as g:
                                    json.dump(keep, g, indent=2)
                                st.success("Base model updated.")
                                st.rerun()
                                break
                        except Exception:
                            continue
        except Exception as e:
            st.error("Composition failed.")
            st.exception(e)
