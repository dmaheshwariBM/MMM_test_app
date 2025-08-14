# core/optimizer.py
# v1.2.1  Loads saved model JSONs, builds response functions, runs discrete optimizations (no SciPy).

from __future__ import annotations
import os, glob, json, math
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# ------------ results root discovery ------------
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
    prefs = []
    env_dir = os.environ.get("MMM_RESULTS_DIR")
    if env_dir: prefs.append(_abs(env_dir))
    prefs += [_abs(os.path.expanduser("~/.mmm_results")),_abs("/tmp/mmm_results"),_abs("results")]
    for root in prefs:
        if _ensure_dir(root): return root
    fb = _abs(os.path.expanduser("~/mmm_results_fallback")); _ensure_dir(fb); return fb

# ------------ catalog helpers ------------
def load_models_catalog(results_roots: List[str]) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for root in results_roots:
        for jf in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            if jf in seen: continue
            try:
                with open(jf, "r", encoding="utf-8") as f: r = json.load(f)
                if not isinstance(r, dict): continue
                if ("coefficients" in r) or ("coef" in r) or ("impact_shares" in r):
                    r["_path"] = jf
                    ts = r.get("batch_ts")
                    try:
                        r["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else datetime.fromtimestamp(os.path.getmtime(jf))
                    except Exception:
                        r["_ts"] = datetime.fromtimestamp(os.path.getmtime(jf))
                    rows.append(r); seen.add(jf)
            except Exception:
                continue
    rows.sort(key=lambda x: x.get("_ts"), reverse=True)
    return rows

def load_model_record(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

# ------------ safe getters ------------
def get_coeffs(model: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in (model.get("coefficients") or model.get("coef") or {}).items()}

def get_channels(model: Dict[str, Any]) -> List[str]:
    ch = model.get("channels") or model.get("features") or list(get_coeffs(model).keys())
    return [c for c in ch if str(c).lower() not in ("intercept","_intercept","(intercept)","base")]

def get_transform_map(model: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return model.get("transform_config") or model.get("transforms") or {}

def get_current_spend(model: Dict[str, Any]) -> Dict[str, float]:
    cur = model.get("current_spend") or model.get("baseline_spend") or {}
    return {k: float(cur.get(k, 0.0)) for k in get_channels(model)}

def get_impact_shares(model: Dict[str, Any]) -> Dict[str, float]:
    inc = model.get("impact_shares") or model.get("decomposition") or {}
    clean = {k: float(v) for k, v in inc.items() if str(k).lower() not in ("base","intercept")}
    s = sum(abs(v) for v in clean.values()) or 1.0
    return {k: max(0.0, float(v))/s for k, v in clean.items()}

def get_segments(model: Dict[str, Any]) -> List[str]:
    segs = model.get("segments") or model.get("segment_values") or []
    return list(segs) if isinstance(segs, list) else []

# ------------ response functions ------------
def _negexp(sp, k=0.01, beta=1.0):
    sp = max(0.0, float(sp)); k = max(0.0, float(k)); beta = max(0.0, float(beta))
    return beta * (1.0 - math.exp(-k * sp))

def _log1p(sp, k=1.0):
    sp = max(0.0, float(sp)); k = max(1e-12, float(k))
    return math.log1p(k * sp)

def build_channel_response_func(model: Dict[str, Any], channel: str):
    tfm = (get_transform_map(model) or {}).get(channel, {})
    ttype = str(tfm.get("transform","negexp")).lower()
    k = float(tfm.get("k", 0.01 if ttype.startswith("negexp") else 1.0))
    beta = float(tfm.get("beta", 1.0))
    shares = get_impact_shares(model)
    scale = float(shares.get(channel, 1.0))  # keep relative shape sensible

    if ttype == "log":
        def f(sp): return scale * _log1p(sp, k=k)
    else:
        def f(sp): return scale * _negexp(sp, k=k, beta=beta)
    return f

# ------------ discrete greedy optimizers (no external deps) ------------
def greedy_max_response(funcs: Dict[str, Any], start: Dict[str, float], min_b: Dict[str, float],
                        max_b: Dict[str, float], total_budget: float, step: float = 1.0,
                        locks: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
    ch = list(funcs.keys()); step = float(max(1e-9, step))
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    if locks:
        for c, is_locked in locks.items():
            if is_locked:
                v = float(start.get(c, 0.0))
                spend[c] = v; min_b[c] = v; max_b[c] = v
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

def greedy_profit_with_floor(funcs: Dict[str, Any], start: Dict[str, float], min_b: Dict[str, float],
                             max_b: Dict[str, float], unit_cost: float = 1.0, mroi_floor: float = 1.0,
                             step: float = 1.0, locks: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
    ch = list(funcs.keys()); step = float(max(1e-9, step))
    spend = {c: float(max(min_b.get(c, 0.0), start.get(c, 0.0))) for c in ch}
    if locks:
        for c, is_locked in locks.items():
            if is_locked:
                v = float(start.get(c, 0.0))
                spend[c] = v; min_b[c] = v; max_b[c] = v

    def mroi(c, s):
        gain = funcs[c](s + step) - funcs[c](s)
        return gain / (unit_cost * step + 1e-12)

    improved = True
    while improved:
        improved = False
        best_c, best_mroi = None, mroi_floor
        for c in ch:
            if spend[c] + step <= max_b.get(c, 1e18) + 1e-12:
                r = mroi(c, spend[c])
                if r >= mroi_floor and r > best_mroi + 1e-12:
                    best_mroi, best_c = r, c
        if best_c is not None:
            spend[best_c] += step
            improved = True

    for c in ch:
        spend[c] = min(max(spend[c], min_b.get(c, 0.0)), max_b.get(c, 1e18))
    return spend
