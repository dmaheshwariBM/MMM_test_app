# pages/8_Export_Import.py
# Workspace backup & restore with per-model download + session catalog refresh.
# Compatible with base, advanced/composite, residual, breakout, pathway, and optimizer saves.

from __future__ import annotations
import os, io, json, zipfile, shutil, glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st

PAGE_ID = "EXPORT_IMPORT_v2_0_0"
st.title("Export & Import")
st.caption(f"Page ID: {PAGE_ID}")

# ---------- Where things live ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _abs(p: str) -> str: return os.path.abspath(p)

def _ensure_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def pick_writable_results_root() -> Path:
    prefs: List[Path] = []
    env_dir = os.environ.get("MMM_RESULTS_DIR")
    if env_dir: prefs.append(Path(_abs(env_dir)))
    prefs += [
        Path(_abs(os.path.expanduser("~/.mmm_results"))),
        Path(_abs("/tmp/mmm_results")),
        Path(_abs("results")),
    ]
    for root in prefs:
        if _ensure_dir(root):
            return root
    fb = Path(_abs(os.path.expanduser("~/mmm_results_fallback")))
    _ensure_dir(fb)
    return fb

RESULTS_ROOT = pick_writable_results_root()

st.info(f"Active workspace folders:\n- data/: `{DATA_DIR}`\n- results/: `{RESULTS_ROOT}`")

# ---------- Helpers ----------
def _walk_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file():
            if p.name in (".write_test",):
                continue
            files.append(p)
    return files

def _human_size(num: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0; n = float(num)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0; i += 1
    return f"{n:,.1f} {units[i]}"

def _make_manifest(data_files: List[Path], results_files: List[Path]) -> Dict[str, Any]:
    return {
        "tool": "MMM Tool",
        "kind": "workspace",
        "version": "2.0.0",
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {"data": len(data_files), "results": len(results_files)},
        "notes": "Paths inside zip are relative to roots: 'data/' and 'results/'.",
    }

def _zip_workspace() -> Tuple[bytes, str, Dict[str, Any]]:
    data_files = _walk_files(DATA_DIR)
    results_files = _walk_files(RESULTS_ROOT)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"mmm_workspace_{ts}.zip"

    manifest = _make_manifest(data_files, results_files)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # add manifest
        zf.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        # add data/
        for p in data_files:
            rel = p.relative_to(DATA_DIR)
            zf.write(p, arcname=str(Path("data") / rel))
        # add results/
        for p in results_files:
            rel = p.relative_to(RESULTS_ROOT)
            zf.write(p, arcname=str(Path("results") / rel))
    buf.seek(0)
    return buf.read(), fname, manifest

def _safe_dest(base: Path, arcname: str) -> Path:
    """Prevent path traversal. Returns a destination inside 'base' or raises ValueError."""
    dest = (base / arcname).resolve()
    if not str(dest).startswith(str(base.resolve())):
        raise ValueError(f"Blocked unsafe path: {arcname}")
    return dest

def _extract_workspace(zip_bytes: bytes, mode: str, overwrite: bool) -> Dict[str, Any]:
    """
    mode: 'merge' or 'wipe'
    overwrite: True -> overwrite files on conflict, False -> keep existing
    """
    if mode == "wipe":
        # wipe data/ and results/ content (keep dirs)
        for root in (DATA_DIR, RESULTS_ROOT):
            if root.exists():
                for p in list(root.glob("*")):
                    if p.is_file():
                        p.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(p, ignore_errors=True)

    stats = {"restored": 0, "skipped": 0, "errors": 0, "notes": []}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not any(n.startswith("data/") for n in names) and not any(n.startswith("results/") for n in names):
            raise ValueError("Zip does not contain 'data/' or 'results/' folders.")

        for info in zf.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            if not (name.startswith("data/") or name.startswith("results/") or name == "MANIFEST.json"):
                stats["skipped"] += 1
                stats["notes"].append(f"Skipped outside root: {name}")
                continue
            if name == "MANIFEST.json":
                continue

            if name.startswith("data/"):
                rel = Path(name).relative_to("data")
                base = DATA_DIR
            else:
                rel = Path(name).relative_to("results")
                base = RESULTS_ROOT

            try:
                dest = _safe_dest(base, rel.as_posix())
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue
                with zf.open(info, "r") as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                stats["restored"] += 1
            except Exception as e:
                stats["errors"] += 1
                stats["notes"].append(f"{name}: {e}")
                continue

    return stats

# ---------- Model catalog helpers (broad/robust) ----------
def _is_model_record(rec: Any) -> bool:
    if not isinstance(rec, dict):
        return False
    if any(k in rec for k in ("coefficients","coef","impact_shares")):
        return True
    if str(rec.get("type","")).lower() in {"base","composite","advanced","residual","breakout","pathway","optimizer"}:
        return True
    # many pages save a 'name' + 'target' + 'features' or 'channels'
    if rec.get("name") and (rec.get("features") or rec.get("channels")):
        return True
    return False

def _load_json_safely(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_models_catalog(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    patt = str(results_root / "**" / "*.json")
    for jf in sorted(glob.glob(patt, recursive=True)):
        p = Path(jf)
        rec = _load_json_safely(p)
        if not _is_model_record(rec):
            continue
        meta: Dict[str, Any] = dict(rec)
        meta["_path"] = str(p)
        # timestamp
        ts = rec.get("batch_ts")
        try:
            meta["_ts"] = datetime.strptime(ts, "%Y%m%d_%H%M%S") if ts else datetime.fromtimestamp(p.stat().st_mtime)
        except Exception:
            meta["_ts"] = datetime.fromtimestamp(p.stat().st_mtime)
        rows.append(meta)
    rows.sort(key=lambda x: x.get("_ts"), reverse=True)
    return rows

def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# ---------- Preload dataframes after import (optional) ----------
def preload_dataframes_to_memory() -> Dict[str, Dict[str, Any]]:
    """
    Loads CSV/XLSX under data/ into st.session_state['DATA_CACHE'] as {filename: df}
    Also returns a catalog with summary stats.
    """
    st.session_state.setdefault("DATA_CACHE", {})
    cat: Dict[str, Dict[str, Any]] = {}
    for p in sorted(DATA_DIR.rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        try:
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
            elif p.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(p)
            else:
                continue
            st.session_state["DATA_CACHE"][name] = df
            cat[name] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
        except Exception as e:
            cat[name] = {"error": str(e)}
    st.session_state["data_catalog"] = cat
    st.session_state["data_files_loaded"] = True
    return cat

def refresh_models_catalog_session():
    """Re-scan results root and refresh session keys used by other pages."""
    models = load_models_catalog(RESULTS_ROOT)
    st.session_state["models_catalog"] = models
    if models:
        # advertise latest model to other pages (e.g., Budget Optimizer)
        st.session_state["optimizer_model_path"] = models[0].get("_path")
        st.session_state["last_saved_path"] = models[0].get("_path")
    return models

# ==========================================================
# UI: 1) Export
# ==========================================================
st.subheader("Export workspace")
c1, c2 = st.columns([2,1])
with c1:
    dfiles = _walk_files(DATA_DIR)
    rfiles = _walk_files(RESULTS_ROOT)
    st.write(f"**data/**: {len(dfiles)} files • {_human_size(sum(p.stat().st_size for p in dfiles) if dfiles else 0)}")
    st.write(f"**results/**: {len(rfiles)} files • {_human_size(sum(p.stat().st_size for p in rfiles) if rfiles else 0)}")
    with st.expander("Preview file list", expanded=False):
        if not dfiles and not rfiles:
            st.caption("No files to export yet.")
        else:
            st.caption("data/")
            for p in sorted(dfiles): st.write(f"- {p.relative_to(DATA_DIR)}")
            st.caption("results/")
            for p in sorted(rfiles): st.write(f"- {p.relative_to(RESULTS_ROOT)}")

with c2:
    if st.button("Create ZIP", type="primary", use_container_width=True):
        try:
            zip_bytes, fname, manifest = _zip_workspace()
            st.session_state["export_zip_bytes"] = zip_bytes
            st.session_state["export_zip_name"] = fname
            st.session_state["export_manifest"] = manifest
            st.success(f"Workspace packaged: {fname}")
        except Exception as e:
            st.error(f"Export failed: {e}")

if "export_zip_bytes" in st.session_state:
    st.download_button(
        "Download workspace ZIP",
        data=st.session_state["export_zip_bytes"],
        file_name=st.session_state.get("export_zip_name","mmm_workspace.zip"),
        mime="application/zip",
        use_container_width=True,
    )
    with st.expander("Manifest", expanded=False):
        st.json(st.session_state.get("export_manifest", {}))

st.divider()

# ==========================================================
# UI: 2) Import
# ==========================================================
st.subheader("Import workspace")

zip_file = st.file_uploader("Upload a workspace ZIP (created by this app)", type=["zip"])
imode = st.radio("Restore mode", options=["merge", "wipe"], horizontal=True,
                 help="• merge: keep existing files and add/overwrite\n• wipe: clear data/ and results/ first")
overwrite = st.checkbox("Overwrite files on name conflict (merge mode)", value=True)
preload = st.checkbox("Preload imported data files into memory (DATA_CACHE)", value=False,
                      help="Loads CSV/XLSX from data/ into session for immediate use on other pages.")

if st.button("Restore", type="secondary", use_container_width=True, disabled=zip_file is None):
    if not zip_file:
        st.warning("Please choose a ZIP file first.")
    else:
        try:
            b = zip_file.read()
            stats = _extract_workspace(b, mode=imode, overwrite=overwrite)
            # After restore, refresh models catalog so other pages see new models immediately
            models = refresh_models_catalog_session()
            # Optionally preload dataframes
            data_cat = {}
            if preload:
                data_cat = preload_dataframes_to_memory()

            st.success(f"Restore complete — restored: {stats['restored']}, skipped: {stats['skipped']}, errors: {stats['errors']}")
            with st.expander("Restored models (latest first)", expanded=True):
                if models:
                    for m in models[:50]:  # show up to 50
                        st.write(f"- **{m.get('name','(unnamed)')}** • {m.get('type','?')} • target: {m.get('target','?')} • {m.get('_ts')}")
                else:
                    st.caption("No model JSONs found under results/ after import.")
            if preload:
                with st.expander("Preloaded dataframes", expanded=False):
                    if data_cat:
                        for k, v in data_cat.items():
                            if "error" in v: st.write(f"• {k}: error — {v['error']}")
                            else: st.write(f"• {k}: {v['rows']} rows × {v['cols']} cols")
                    else:
                        st.caption("No CSV/XLSX found to preload.")
            st.session_state["last_save_error"] = ""
        except Exception as e:
            st.session_state["last_save_error"] = f"Import failed: {e}"
            st.error(st.session_state["last_save_error"])

st.caption("Tip: On Streamlit Cloud the local disk is ephemeral. Export after runs you want to keep, and re-import later.")

st.divider()

# ==========================================================
# UI: 3) Download individual model results + send to optimizer
# ==========================================================
st.subheader("Download individual model results")

# Build (or reuse) catalog
session_models = st.session_state.get("models_catalog")
if session_models is None:
    session_models = refresh_models_catalog_session()

query = st.text_input("Filter by name/type/target (optional)", value="")
filtered = []
for m in session_models:
    line = f"{m.get('name','')} {m.get('type','')} {m.get('target','')}".lower()
    if query.lower() in line:
        filtered.append(m)

if not filtered:
    st.info("No saved models found (or filter removed them).")
else:
    for m in filtered[:100]:  # show up to 100
        p = Path(str(m.get("_path")))
        stem = p.with_suffix("")
        csv_candidate = stem.with_suffix(".csv")
        label = f"**{m.get('name','(unnamed)')}**  •  {m.get('type','?')}  •  target: {m.get('target','?')}  •  {m.get('_ts')}"
        with st.expander(label, expanded=False):
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                try:
                    st.download_button("Download JSON", data=_read_bytes(p), file_name=p.name, mime="application/json")
                except Exception as e:
                    st.caption(f"JSON not readable: {e}")
            with col2:
                if csv_candidate.exists():
                    try:
                        st.download_button("Download CSV", data=_read_bytes(csv_candidate), file_name=csv_candidate.name, mime="text/csv")
                    except Exception as e:
                        st.caption(f"CSV not readable: {e}")
                else:
                    st.caption("No CSV sibling")
            with col3:
                if st.button("Send to Optimizer", key=f"sendopt_{p}"):
                    st.session_state["optimizer_model_path"] = str(p)
                    st.success("Sent to Budget Optimizer. Open the Budget page to use it.")
            with col4:
                if st.button("Mark as latest", key=f"marklatest_{p}"):
                    st.session_state["last_saved_path"] = str(p)
                    st.success("Marked as latest for other pages.")

# ----------------------------------------------------------
# END
# ----------------------------------------------------------
