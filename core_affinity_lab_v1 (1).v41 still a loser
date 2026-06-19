#!/usr/bin/env python3
"""
Core Affinity Lab v41 — Entropy Signature Refiner
Core-first grouped seed refinement with global canonical signature registry and seed accounting.

Purpose:
- Stop trying to separate 120 cores inside broad trait groups.
- Refine/multiply trait signatures first using entropy/information gain.
- Preserve every seed row with actual core/member attached.
- Re-key child groups globally by canonical signature so A+B+C from anywhere merges logically.
- Lab only. No daily-play logic.
"""
from __future__ import annotations

import gc as gc_module
import io
import math
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v41"
BUILD_MARKER = "BUILD: core_affinity_lab_v41_ENTROPY_SIGNATURE_REFINER__2026-06-19"
DEPLOY_FILENAME_NOTE = "For Streamlit Cloud deployment, this file may be renamed to: core_affinity_lab_v1 (1).py"

st.set_page_config(page_title="Core Affinity Lab v41", layout="wide")
st.title("Core Affinity Lab v41 — Entropy Signature Refiner")
st.caption(BUILD_MARKER)
st.info(
    "Core-first grouped seed refinement. It refines broad trait groups into narrower canonical signatures before separator mining. "
    "Every seed keeps row_id, actual core, actual member, parent group, child signature, and accounting status."
)

# -------------------------
# Input helpers
# -------------------------

def _bytes(upload) -> bytes:
    if upload is None:
        return b""
    if hasattr(upload, "getvalue"):
        v = upload.getvalue()
        return v if isinstance(v, bytes) else str(v).encode("utf-8")
    if hasattr(upload, "read"):
        try:
            upload.seek(0)
        except Exception:
            pass
        v = upload.read()
        return v if isinstance(v, bytes) else str(v).encode("utf-8")
    return bytes(upload)


def read_ledger_from_upload(upload) -> pd.DataFrame:
    raw = _bytes(upload)
    if not raw:
        return pd.DataFrame()
    name = str(getattr(upload, "name", "uploaded")).lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            target = None
            for n in names:
                if n.endswith("aabc_seed_group_ledger.csv"):
                    target = n; break
            if target is None:
                for n in names:
                    if "ledger" in n.lower() and n.endswith(".csv"):
                        target = n; break
            if target is None:
                raise ValueError("ZIP did not contain aabc_seed_group_ledger.csv or another ledger CSV.")
            return pd.read_csv(z.open(target), dtype=str)
    text = raw.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), dtype=str)


def normalize_ledger(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().fillna("")
    # normalize required columns
    required = ["row_id", "winner_core", "winner_member"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Ledger missing required columns: {missing}")
    out["row_id"] = out["row_id"].astype(str)
    out["winner_core"] = out["winner_core"].astype(str).str.zfill(3)
    out["winner_member"] = out["winner_member"].astype(str).str.zfill(4)
    # force all object columns to string to avoid Arrow mixed-type errors
    for c in out.columns:
        out[c] = out[c].astype(str)
    return out


def canonicalize_traits(traits: Iterable[str]) -> str:
    clean = []
    for t in traits:
        t = str(t).strip()
        if t and t.lower() != "nan":
            clean.append(t)
    return "|".join(sorted(set(clean)))


def split_signature(sig: str) -> List[str]:
    return [x.strip() for x in str(sig).split("|") if x.strip()]

# -------------------------
# Trait atom builder
# -------------------------

BASE_ATOM_COLS = [
    "seed_sum_bucket", "seed_parity_pattern", "seed_highlow_pattern", "seed_structure",
    "seed_spread_bucket", "seed_mirror_signature", "seed_mirror_count",
    "seed_even_count", "seed_odd_count", "seed_high_count", "seed_low_count",
    "seed_unique", "seed_maxrep",
]


def atomize_row(row: pd.Series) -> Set[str]:
    atoms: Set[str] = set()
    for c in BASE_ATOM_COLS:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                atoms.add(f"{c}={v}")
    # present/missing digit atoms
    pres = str(row.get("seed_present_digits", ""))
    miss = str(row.get("seed_missing_digits", ""))
    for d in re.findall(r"\d", pres):
        atoms.add(f"has{d}=1")
    for d in re.findall(r"\d", miss):
        atoms.add(f"no{d}=1")
    # pair atoms from pair signature like pairs_03_04_05
    ps = str(row.get("seed_pair_signature", ""))
    for p in re.findall(r"(?<!\d)(\d{2})(?!\d)", ps):
        atoms.add(f"pair_{p}=1")
    # positional atoms from group_positional
    gp = str(row.get("group_positional", ""))
    for part in gp.split("|"):
        part = part.strip()
        if part:
            atoms.add(f"pos:{part}")
    # original grouping fields can also be atoms, but not as a whole massive signature
    for c in ["group_distribution", "group_digitset"]:
        v = str(row.get(c, "")).strip()
        if v:
            for part in v.split("|"):
                part = part.strip()
                if part:
                    atoms.add(f"{c}:{part}")
    return atoms


def attach_atoms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_atom_set"] = [atomize_row(r) for _, r in out.iterrows()]
    return out

# -------------------------
# Entropy / refinement engine
# -------------------------

def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    e = 0.0
    for n in counts.values():
        if n <= 0: continue
        p = n / total
        e -= p * math.log2(p)
    return e


def dominance(counts: Counter) -> Tuple[str, int, float]:
    total = sum(counts.values())
    if total <= 0:
        return "", 0, 0.0
    k, n = counts.most_common(1)[0]
    return str(k), int(n), float(n / total)


def counts_for_idxs(df: pd.DataFrame, idxs: List[int], target_col: str) -> Counter:
    return Counter(df.iloc[idxs][target_col].astype(str).tolist())


def candidate_atoms(df: pd.DataFrame, idxs: List[int], used: Set[str], top_n: int) -> List[Tuple[str, int]]:
    c = Counter()
    atom_sets = df["_atom_set"].tolist()
    for i in idxs:
        for a in atom_sets[i]:
            if a not in used and not a.startswith("NOT:"):
                c[a] += 1
    return c.most_common(top_n)


def evaluate_split(df: pd.DataFrame, idxs: List[int], atom: str, target_col: str, min_child_size: int) -> Dict:
    atom_sets = df["_atom_set"].tolist()
    yes = [i for i in idxs if atom in atom_sets[i]]
    no = [i for i in idxs if atom not in atom_sets[i]]
    if len(yes) < min_child_size or len(no) < min_child_size:
        return {"valid": False}
    parent_counts = counts_for_idxs(df, idxs, target_col)
    yes_counts = counts_for_idxs(df, yes, target_col)
    no_counts = counts_for_idxs(df, no, target_col)
    parent_e = entropy(parent_counts)
    weighted_e = (len(yes)/len(idxs))*entropy(yes_counts) + (len(no)/len(idxs))*entropy(no_counts)
    gain = parent_e - weighted_e
    ydom, yhits, ypct = dominance(yes_counts)
    ndom, nhits, npct = dominance(no_counts)
    pdom, phits, ppct = dominance(parent_counts)
    improvement = max(ypct, npct) - ppct
    return {
        "valid": True, "atom": atom, "yes": yes, "no": no,
        "parent_entropy": parent_e, "weighted_entropy": weighted_e, "info_gain": gain,
        "parent_dominant": pdom, "parent_dom_hits": phits, "parent_dom_pct": ppct,
        "yes_dominant": ydom, "yes_dom_hits": yhits, "yes_dom_pct": ypct,
        "no_dominant": ndom, "no_dom_hits": nhits, "no_dom_pct": npct,
        "yes_size": len(yes), "no_size": len(no), "dominance_improvement": improvement,
    }

@dataclass
class RefineCfg:
    target_col: str
    group_col: str
    dominance_threshold: float
    min_group_size: int
    min_child_size: int
    max_depth: int
    candidate_pool: int
    min_info_gain: float
    max_splits_per_group: int


def refine_one_group(df: pd.DataFrame, group_value: str, group_idxs: List[int], cfg: RefineCfg, progress_log: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return leaf lineage rows, split audit rows, profile rows."""
    leaf_rows: List[Dict] = []
    split_rows: List[Dict] = []
    profile_rows: List[Dict] = []
    split_counter = 0
    base_traits = split_signature(group_value)

    stack = [(group_idxs, tuple(base_traits), 0, "ROOT")]
    while stack:
        idxs, sig_traits, depth, parent_id = stack.pop()
        counts = counts_for_idxs(df, idxs, cfg.target_col)
        dom, dom_hits, dom_pct = dominance(counts)
        node_sig = canonicalize_traits(sig_traits)
        node_id = f"{abs(hash((group_value, node_sig, len(idxs)))) % 10**12:012d}"
        status = ""
        if len(idxs) < cfg.min_group_size:
            status = "LOW_SAMPLE"
        elif dom_pct >= cfg.dominance_threshold:
            status = "PROFILED_DOMINANT"
        elif depth >= cfg.max_depth:
            status = "SPARE_REVISIT_MAX_DEPTH"
        elif split_counter >= cfg.max_splits_per_group:
            status = "SPARE_REVISIT_SPLIT_CAP"
        else:
            # choose entropy-reducing atom
            used = set(sig_traits)
            best = None
            for atom, support in candidate_atoms(df, idxs, used, cfg.candidate_pool):
                ev = evaluate_split(df, idxs, atom, cfg.target_col, cfg.min_child_size)
                if not ev.get("valid"):
                    continue
                # favor entropy gain, then dominance improvement, then balanced useful child sizes
                score = ev["info_gain"] * 100.0 + ev["dominance_improvement"] * 10.0 + min(ev["yes_size"], ev["no_size"]) * 0.001
                ev["score"] = score
                if best is None or score > best["score"]:
                    best = ev
            if best is not None and (best["info_gain"] >= cfg.min_info_gain or best["dominance_improvement"] > 0):
                split_counter += 1
                atom = best["atom"]
                yes_sig = tuple(list(sig_traits) + [atom])
                no_sig = tuple(list(sig_traits) + [f"NOT:{atom}"])
                split_rows.append({
                    "original_group": group_value,
                    "parent_node_id": parent_id,
                    "node_id": node_id,
                    "depth": depth,
                    "separator_atom": atom,
                    "parent_size": len(idxs),
                    "parent_distinct_targets": len(counts),
                    "parent_dominant_target": best["parent_dominant"],
                    "parent_dominant_hits": best["parent_dom_hits"],
                    "parent_dominance_pct": round(best["parent_dom_pct"], 4),
                    "parent_entropy": round(best["parent_entropy"], 6),
                    "yes_size": best["yes_size"],
                    "yes_dominant_target": best["yes_dominant"],
                    "yes_dom_hits": best["yes_dom_hits"],
                    "yes_dominance_pct": round(best["yes_dom_pct"], 4),
                    "no_size": best["no_size"],
                    "no_dominant_target": best["no_dominant"],
                    "no_dom_hits": best["no_dom_hits"],
                    "no_dominance_pct": round(best["no_dom_pct"], 4),
                    "info_gain": round(best["info_gain"], 6),
                    "dominance_improvement": round(best["dominance_improvement"], 6),
                    "yes_signature": canonicalize_traits(yes_sig),
                    "no_signature": canonicalize_traits(no_sig),
                })
                # process children
                stack.append((best["no"], no_sig, depth+1, node_id))
                stack.append((best["yes"], yes_sig, depth+1, node_id))
                continue
            else:
                status = "SPARE_REVISIT_NO_GAIN"

        # leaf reached
        profile_rows.append({
            "original_group": group_value,
            "leaf_node_id": node_id,
            "refined_signature": node_sig,
            "depth": depth,
            "leaf_size": len(idxs),
            "distinct_targets": len(counts),
            "dominant_target": dom,
            "dominant_hits": dom_hits,
            "dominance_pct": round(dom_pct, 4),
            "entropy": round(entropy(counts), 6),
            "status": status,
            "target_distribution": ";".join(f"{k}:{v}" for k, v in counts.most_common(20)),
        })
        for i in idxs:
            r = df.iloc[i]
            leaf_rows.append({
                "row_id": r.get("row_id", ""),
                "original_group": group_value,
                "refined_signature": node_sig,
                "leaf_node_id": node_id,
                "status": status,
                "assigned_target": dom if status == "PROFILED_DOMINANT" else "",
                "actual_core": r.get("winner_core", ""),
                "actual_member": r.get("winner_member", ""),
                "Date": r.get("Date", ""),
                "StreamKey": r.get("StreamKey", ""),
                "SeedResult": r.get("SeedResult", ""),
                "Result4": r.get("Result4", ""),
            })
    return leaf_rows, split_rows, profile_rows


def zip_outputs(frames: Dict[str, pd.DataFrame], texts: Dict[str, str] | None = None) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(name, df.fillna("").astype(str).to_csv(index=False))
        for name, txt in (texts or {}).items():
            z.writestr(name, str(txt))
    bio.seek(0)
    return bio.getvalue()

# -------------------------
# UI
# -------------------------

upload = st.file_uploader("Upload trait grouping audit ZIP or aabc_seed_group_ledger.csv", type=["zip", "csv"])

with st.sidebar:
    st.header("v41 settings")
    stage = st.selectbox("Stage", [
        "1 - Audit ledger only",
        "2 - Core entropy signature refinement",
        "3 - Member entropy signature refinement",
    ])
    group_col = st.selectbox("Trait group column", [
        "group_primary", "group_separator_profile", "group_distribution", "group_digitset", "group_positional"
    ], index=0)
    start_group = st.number_input("Start group #", min_value=1, max_value=100000, value=1, step=1)
    groups_to_process = st.slider("Groups to process", 10, 2000, 100, 10)
    dominance_threshold = st.slider("Dominance threshold", 0.40, 1.00, 0.60, 0.05)
    min_group_size = st.slider("Min group size", 2, 50, 5, 1)
    min_child_size = st.slider("Min child size after split", 1, 25, 1, 1)
    max_depth = st.slider("Max refinement depth", 1, 8, 4, 1)
    candidate_pool = st.slider("Candidate atoms per node", 20, 500, 150, 10)
    min_info_gain = st.slider("Min information gain", 0.0, 1.0, 0.005, 0.005)
    max_splits_per_group = st.slider("Max splits per original group", 1, 100, 25, 1)
    st.caption("v41 refines signatures first. It does not force broad groups to become dominant in one step.")

if not upload:
    st.stop()

if "v41_package" not in st.session_state:
    st.session_state["v41_package"] = None
    st.session_state["v41_manifest"] = None
    st.session_state["v41_filename"] = None

if st.button("Run selected v41 stage and prepare ZIP", type="primary"):
    t0 = time.time()
    progress = st.progress(0.0)
    status_box = st.empty()
    try:
        status_box.info("Loading ledger...")
        ledger = normalize_ledger(read_ledger_from_upload(upload))
        if ledger.empty:
            st.error("Ledger is empty.")
            st.stop()
        if group_col not in ledger.columns:
            raise ValueError(f"Selected group column {group_col} not found in ledger.")

        summary_rows = [
            {"metric":"app_version", "value":APP_VERSION},
            {"metric":"build_marker", "value":BUILD_MARKER},
            {"metric":"stage", "value":stage},
            {"metric":"group_col", "value":group_col},
            {"metric":"ledger_rows", "value":len(ledger)},
            {"metric":"unique_row_ids", "value":ledger["row_id"].nunique()},
            {"metric":"distinct_cores", "value":ledger["winner_core"].nunique()},
            {"metric":"distinct_members", "value":ledger["winner_member"].nunique()},
            {"metric":"settings", "value":f"start={start_group};groups={groups_to_process};dom={dominance_threshold};min_group={min_group_size};min_child={min_child_size};depth={max_depth};pool={candidate_pool};gain={min_info_gain};split_cap={max_splits_per_group}"},
        ]

        group_audit = ledger.groupby(group_col).agg(
            rows=("row_id", "count"),
            unique_rows=("row_id", "nunique"),
            distinct_cores=("winner_core", "nunique"),
            distinct_members=("winner_member", "nunique"),
        ).reset_index().sort_values(["rows", "distinct_cores"], ascending=[False, False]).reset_index(drop=True)
        group_audit.insert(0, "group_number", range(1, len(group_audit)+1))

        frames: Dict[str, pd.DataFrame] = {
            "summary_basic.csv": pd.DataFrame(summary_rows),
            "ledger_group_column_audit.csv": group_audit,
            "seed_count_by_core.csv": ledger.groupby("winner_core").size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False),
            "seed_count_by_core_member.csv": ledger.groupby(["winner_core", "winner_member"]).size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False),
        }

        if stage.startswith("1"):
            status_box.info("Preparing Stage 1 audit package...")
            end_audit = pd.DataFrame([
                {"check":"row_count", "value":len(ledger)},
                {"check":"unique_row_ids", "value":ledger["row_id"].nunique()},
                {"check":"duplicate_row_ids", "value":len(ledger)-ledger["row_id"].nunique()},
                {"check":"core_count_sum", "value":int(ledger.groupby("winner_core").size().sum())},
                {"check":"member_count_sum", "value":int(ledger.groupby(["winner_core", "winner_member"]).size().sum())},
                {"check":"accounting_balanced", "value":str(len(ledger)==ledger["row_id"].nunique()).upper()},
            ])
            frames["end_audit_seed_accounting.csv"] = end_audit
            progress.progress(1.0)
        else:
            target_col = "winner_core" if stage.startswith("2") else "winner_member"
            status_box.info("Attaching trait atoms...")
            ledger = attach_atoms(ledger)
            # select groups by sorted audit order
            selected = group_audit.iloc[int(start_group)-1:int(start_group)-1+int(groups_to_process)][group_col].astype(str).tolist()
            cfg = RefineCfg(
                target_col=target_col,
                group_col=group_col,
                dominance_threshold=float(dominance_threshold),
                min_group_size=int(min_group_size),
                min_child_size=int(min_child_size),
                max_depth=int(max_depth),
                candidate_pool=int(candidate_pool),
                min_info_gain=float(min_info_gain),
                max_splits_per_group=int(max_splits_per_group),
            )
            all_lineage: List[Dict] = []
            all_splits: List[Dict] = []
            all_profiles: List[Dict] = []
            for gi, gv in enumerate(selected, start=1):
                status_box.info(f"Refining {target_col}: group {gi}/{len(selected)} — size {int(group_audit.loc[group_audit[group_col].astype(str)==str(gv),'rows'].iloc[0]) if (group_audit[group_col].astype(str)==str(gv)).any() else '?'}")
                idxs = ledger.index[ledger[group_col].astype(str).eq(str(gv))].tolist()
                lineage, splits, profiles = refine_one_group(ledger, str(gv), idxs, cfg, [])
                all_lineage.extend(lineage); all_splits.extend(splits); all_profiles.extend(profiles)
                if gi % 5 == 0:
                    gc_module.collect()
                progress.progress(min(1.0, gi/max(1, len(selected))))
            lineage_df = pd.DataFrame(all_lineage)
            split_df = pd.DataFrame(all_splits)
            profile_df = pd.DataFrame(all_profiles)
            # global signature registry: merge same canonical signatures from anywhere
            if not lineage_df.empty:
                registry = lineage_df.groupby("refined_signature").agg(
                    seed_rows=("row_id", "count"),
                    unique_rows=("row_id", "nunique"),
                    distinct_actual_cores=("actual_core", "nunique"),
                    distinct_actual_members=("actual_member", "nunique"),
                    statuses=("status", lambda s: ";".join(sorted(set(map(str,s))))),
                ).reset_index().sort_values("seed_rows", ascending=False)
                # add dominant target after global re-key
                dom_rows=[]
                for sig, g in lineage_df.groupby("refined_signature"):
                    counts = Counter(g["actual_core" if target_col=="winner_core" else "actual_member"].astype(str).tolist())
                    dom, hits, pct = dominance(counts)
                    dom_rows.append({"refined_signature":sig, "global_dominant_target":dom, "global_dom_hits":hits, "global_dom_pct":round(pct,4), "global_target_distribution":";".join(f"{k}:{v}" for k,v in counts.most_common(20))})
                registry = registry.merge(pd.DataFrame(dom_rows), on="refined_signature", how="left")
            else:
                registry = pd.DataFrame()
            assigned_count = int((lineage_df["status"].eq("PROFILED_DOMINANT")).sum()) if not lineage_df.empty else 0
            missing_rows = 0
            duplicate_assignments = 0
            if not lineage_df.empty:
                expected = set(ledger.loc[ledger[group_col].astype(str).isin(selected), "row_id"].astype(str))
                got = lineage_df["row_id"].astype(str).tolist()
                missing_rows = len(expected - set(got))
                duplicate_assignments = len(got) - len(set(got))
            end_audit = pd.DataFrame([
                {"check":"processed_groups", "value":len(selected)},
                {"check":"processed_seed_rows", "value":len(lineage_df)},
                {"check":"profiled_dominant_rows", "value":assigned_count},
                {"check":"spare_or_low_sample_rows", "value":int(len(lineage_df)-assigned_count)},
                {"check":"split_events", "value":len(split_df)},
                {"check":"leaf_profiles", "value":len(profile_df)},
                {"check":"global_registry_signatures", "value":len(registry)},
                {"check":"missing_rows", "value":missing_rows},
                {"check":"duplicate_row_assignments", "value":duplicate_assignments},
                {"check":"accounting_balanced", "value":str(missing_rows==0 and duplicate_assignments==0).upper()},
            ])
            frames.update({
                "refined_seed_lineage.csv": lineage_df,
                "signature_split_audit.csv": split_df,
                "refined_signature_profiles.csv": profile_df,
                "global_signature_registry.csv": registry,
                "end_audit_seed_accounting.csv": end_audit,
            })
            summary_rows.extend([
                {"metric":"processed_groups", "value":len(selected)},
                {"metric":"processed_seed_rows", "value":len(lineage_df)},
                {"metric":"split_events", "value":len(split_df)},
                {"metric":"leaf_profiles", "value":len(profile_df)},
                {"metric":"accounting_balanced", "value":str(missing_rows==0 and duplicate_assignments==0).upper()},
            ])
            frames["summary_basic.csv"] = pd.DataFrame(summary_rows)
            # explicitly delete heavy ledger atom sets after output frames materialized
            del ledger
            gc_module.collect()

        readme = f"{BUILD_MARKER}\n{DEPLOY_FILENAME_NOTE}\nStage: {stage}\nThis app refines group signatures by entropy/information gain and globally re-keys identical child signatures. Lab only.\n"
        package = zip_outputs(frames, {"README_v41.txt": readme})
        filename = f"core_affinity_lab_v41_{stage.split()[0]}_{group_col}_start_{int(start_group)}.zip"
        st.session_state["v41_package"] = package
        st.session_state["v41_filename"] = filename
        st.session_state["v41_manifest"] = pd.DataFrame([
            {"file": k, "rows": (len(v) if isinstance(v, pd.DataFrame) else "") } for k, v in frames.items()
        ])
        del frames, package
        gc_module.collect()
        status_box.success(f"v41 package ready in {time.time()-t0:.2f}s")
    except Exception as e:
        st.error("v41 stage failed. Full traceback below.")
        st.exception(e)

if st.session_state.get("v41_manifest") is not None:
    st.subheader("Frozen package manifest")
    st.dataframe(st.session_state["v41_manifest"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download v41 stage outputs ZIP",
        st.session_state["v41_package"],
        file_name=st.session_state["v41_filename"],
        mime="application/zip",
    )
