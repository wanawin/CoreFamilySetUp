#!/usr/bin/env python3
"""
Core Affinity Lab v38 — Trait Group Split Separator Engine

Purpose:
- Read the grouped seed ledger from the trait-grouping audit ZIP/CSV.
- Keep every seed row attached to actual core/member.
- Build core-level groups first, then optional member-level groups inside core.
- Split collision groups only when a separator trait creates a dominant child bucket.
- Send unresolved/low-sample leftovers to SPARE/LOW_SAMPLE with lineage preserved.
- Export complete accounting so no seed disappears.

Lab only. This does not create a daily play playlist.
"""
from __future__ import annotations

import gc
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd
import streamlit as st

APP_VERSION = "v38"
BUILD_MARKER = "BUILD: core_affinity_lab_v38_TRAIT_GROUP_SPLIT_SEPARATOR_ENGINE__2026-06-18"
DEPLOY_FILENAME_NOTE = "For Streamlit Cloud deployment, this file may be renamed to: core_affinity_lab_v1 (1).py"

st.set_page_config(page_title="Core Affinity Lab v38", layout="wide")
st.title("Core Affinity Lab v38 — Trait Group Split Separator Engine")
st.caption(BUILD_MARKER)
st.info(
    "Group first, split only collision groups, preserve seed lineage. "
    "Core separators first; member separators inside core second. Discovery only — not production play logic."
)

LEDGER_FILE = "aabc_seed_group_ledger.csv"
GROUP_COLS = [
    "group_primary",
    "group_distribution",
    "group_digitset",
    "group_positional",
    "group_separator_profile",
]

# -------------------------
# Safe IO
# -------------------------
def _upload_bytes(upload) -> bytes:
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


def _safe_str_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prevent Arrow mixed-type display/export failures."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.fillna("")
    for c in out.columns:
        out[c] = out[c].astype(str)
    return out


def read_ledger(upload) -> pd.DataFrame:
    raw = _upload_bytes(upload)
    if not raw:
        return pd.DataFrame()
    name = str(getattr(upload, "name", "uploaded")).lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
            names = z.namelist()
            target = LEDGER_FILE if LEDGER_FILE in names else None
            if target is None:
                # fallback: first ledger-looking csv
                for n in names:
                    if n.lower().endswith(".csv") and "ledger" in n.lower():
                        target = n
                        break
            if target is None:
                raise ValueError("Could not find aabc_seed_group_ledger.csv or another ledger CSV inside ZIP.")
            with z.open(target) as f:
                df = pd.read_csv(f, dtype=str)
    else:
        text = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), dtype=str)

    df = df.fillna("")
    if "row_id" not in df.columns:
        df.insert(0, "row_id", [str(i + 1) for i in range(len(df))])
    df["row_id"] = df["row_id"].astype(str)
    # normalize core/member as zero-padded-ish sorted strings where possible
    for col in ["winner_core", "winner_member", "SeedResult", "Result4"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def zip_frames(frames: Dict[str, pd.DataFrame], texts: Optional[Dict[str, str]] = None) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(name, _safe_str_df(df).to_csv(index=False))
        for name, txt in (texts or {}).items():
            z.writestr(name, str(txt))
    bio.seek(0)
    return bio.getvalue()

# -------------------------
# Trait candidates
# -------------------------
def _tokenize_pipe_field(prefix: str, value: str) -> List[str]:
    value = str(value or "")
    if not value:
        return []
    toks = [x for x in value.split("|") if x]
    return [f"{prefix}:{x}" for x in toks]


def row_candidate_traits(row: pd.Series) -> List[str]:
    """Generate separator atoms from the ledger row.

    Uses ledger columns only; does not recompute deep raw traits. This is intentional:
    v38 mines inside pre-formed groups and keeps the candidate set bounded.
    """
    traits: List[str] = []
    scalar_cols = [
        "seed_sum", "seed_spread", "seed_parity_pattern", "seed_highlow_pattern",
        "seed_structure", "seed_unique", "seed_maxrep", "seed_sum_bucket",
        "seed_spread_bucket", "seed_even_count", "seed_odd_count",
        "seed_high_count", "seed_low_count", "seed_mirror_count",
        "seed_mirror_signature", "seed_present_digits", "seed_missing_digits",
    ]
    for c in scalar_cols:
        if c in row.index:
            v = str(row.get(c, ""))
            if v != "":
                traits.append(f"{c}={v}")
    if "seed_pair_signature" in row.index:
        traits += _tokenize_pipe_field("seed_pair_signature", str(row.get("seed_pair_signature", "")))
    if "group_positional" in row.index:
        traits += _tokenize_pipe_field("pos", str(row.get("group_positional", "")))
    # digit inclusion/exclusion atoms from present/missing strings
    present = str(row.get("seed_present_digits", ""))
    missing = str(row.get("seed_missing_digits", ""))
    for d in "0123456789":
        if d in present:
            traits.append(f"has{d}=1")
        if d in missing:
            traits.append(f"no{d}=1")
    return sorted(set(traits))


@st.cache_data(show_spinner=False)
def build_trait_map_from_csv(csv_text: str) -> Dict[str, List[str]]:
    df = pd.read_csv(io.StringIO(csv_text), dtype=str).fillna("")
    mp: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        mp[str(r["row_id"])] = row_candidate_traits(r)
    return mp

# -------------------------
# Group splitting engine
# -------------------------
@dataclass
class SplitConfig:
    level: str  # core or member
    group_col: str
    dominance_threshold: float
    min_group_size: int
    min_child_hits: int
    min_child_pct: float
    max_depth: int
    max_groups: int
    start_group: int
    candidate_pool: int
    max_children_per_group: int
    core_filter: str = ""


def _target_col(level: str) -> str:
    return "winner_core" if level == "core" else "winner_member"


def _group_scope(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    d = df.copy()
    if cfg.level == "member" and cfg.core_filter.strip():
        d = d[d["winner_core"].astype(str).eq(cfg.core_filter.strip())].copy()
    return d


def _target_counts(rows: pd.DataFrame, target_col: str) -> Tuple[str, int, int, float, str]:
    c = Counter(rows[target_col].astype(str).tolist())
    total = len(rows)
    if not c:
        return "", 0, 0, 0.0, ""
    dom, cnt = c.most_common(1)[0]
    pct = cnt / total if total else 0.0
    counts_str = ";".join(f"{k}:{v}" for k, v in c.most_common())
    return dom, cnt, len(c), pct, counts_str


def _counter_traits(df: pd.DataFrame, row_ids: Iterable[str], trait_map: Dict[str, List[str]]) -> Counter:
    c = Counter()
    for rid in row_ids:
        c.update(trait_map.get(str(rid), []))
    return c


def _score_child(df: pd.DataFrame, row_ids: List[str], trait: str, trait_map: Dict[str, List[str]], target_col: str) -> Optional[Dict[str, object]]:
    child_ids = [rid for rid in row_ids if trait in trait_map.get(str(rid), [])]
    if not child_ids:
        return None
    child = df[df["row_id"].astype(str).isin(set(child_ids))]
    dom, cnt, distinct, pct, counts_str = _target_counts(child, target_col)
    return {
        "trait": trait,
        "child_ids": set(child_ids),
        "child_size": len(child_ids),
        "dominant_target": dom,
        "dominant_count": cnt,
        "distinct_targets": distinct,
        "dominant_pct": pct,
        "target_counts": counts_str,
    }


def _best_separator(df: pd.DataFrame, row_ids: List[str], trait_map: Dict[str, List[str]], target_col: str, cfg: SplitConfig) -> Tuple[Optional[Dict[str, object]], pd.DataFrame]:
    counts = _counter_traits(df, row_ids, trait_map)
    audit_rows = []
    best = None
    best_score = -1.0
    for trait, support in counts.most_common(cfg.candidate_pool):
        stat = _score_child(df, row_ids, trait, trait_map, target_col)
        if stat is None:
            continue
        child_size = int(stat["child_size"])
        dom_count = int(stat["dominant_count"])
        dom_pct = float(stat["dominant_pct"])
        # Separator must create a meaningful child. For tiny groups, 4/5 is acceptable if threshold allows it.
        passes = (child_size >= cfg.min_child_hits and dom_count >= cfg.min_child_hits and dom_pct >= cfg.min_child_pct and child_size < len(row_ids))
        # Prefer high dominance, then more target hits, then smaller cleaner child.
        score = (dom_pct * 1000.0) + (dom_count * 10.0) - (child_size * 0.01)
        audit_rows.append({
            "candidate_trait": trait,
            "support_in_group": support,
            "child_size": child_size,
            "dominant_target": stat["dominant_target"],
            "dominant_count": dom_count,
            "dominant_pct": round(dom_pct, 4),
            "distinct_targets": stat["distinct_targets"],
            "target_counts": stat["target_counts"],
            "passes_separator_gate": passes,
            "score": round(score, 4),
        })
        if passes and score > best_score:
            best_score = score
            best = stat
    return best, pd.DataFrame(audit_rows)


def process_one_group(df: pd.DataFrame, group_key: str, group_df: pd.DataFrame, trait_map: Dict[str, List[str]], cfg: SplitConfig, group_idx: int) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Returns profiles, lineage, spares, candidate audit."""
    target_col = _target_col(cfg.level)
    profiles: List[Dict] = []
    lineage: List[Dict] = []
    spares: List[Dict] = []
    audits: List[Dict] = []

    root_ids = [str(x) for x in group_df["row_id"].astype(str).tolist()]
    queue = [("root", root_ids, 0, "")]
    child_num = 0

    while queue:
        node_name, row_ids, depth, sep_stack = queue.pop(0)
        node_df = group_df[group_df["row_id"].astype(str).isin(set(row_ids))]
        dom, cnt, distinct, pct, counts_str = _target_counts(node_df, target_col)
        node_id = f"{cfg.level.upper()}_{cfg.group_col}_{group_idx:05d}_{node_name}"

        if len(row_ids) < cfg.min_group_size:
            status = "LOW_SAMPLE"
        elif pct >= cfg.dominance_threshold:
            status = "PROFILED_DOMINANT"
        elif depth >= cfg.max_depth:
            status = "SPARE_REVISIT_MAX_DEPTH"
        else:
            status = "NEEDS_SPLIT"

        if status == "PROFILED_DOMINANT":
            profiles.append({
                "level": cfg.level,
                "group_type": cfg.group_col,
                "original_trait_group": group_key,
                "child_group_id": node_id,
                "parent_group_id": "" if node_name == "root" else f"{cfg.level.upper()}_{cfg.group_col}_{group_idx:05d}",
                "separator_stack": sep_stack,
                "total_seed_rows": len(row_ids),
                "dominant_target": dom,
                "dominant_count": cnt,
                "dominant_pct": round(pct, 4),
                "distinct_targets": distinct,
                "target_counts": counts_str,
                "status": status,
            })
            for rid in row_ids:
                lineage.append({"row_id": rid, "level": cfg.level, "group_type": cfg.group_col, "original_trait_group": group_key, "assigned_child_group_id": node_id, "assigned_target": dom, "separator_stack": sep_stack, "status": status})
            continue

        if status.startswith("LOW_SAMPLE") or status.startswith("SPARE"):
            spares.append({
                "level": cfg.level,
                "group_type": cfg.group_col,
                "original_trait_group": group_key,
                "child_group_id": node_id,
                "separator_stack": sep_stack,
                "total_seed_rows": len(row_ids),
                "dominant_target": dom,
                "dominant_count": cnt,
                "dominant_pct": round(pct, 4),
                "distinct_targets": distinct,
                "target_counts": counts_str,
                "status": status,
            })
            for rid in row_ids:
                lineage.append({"row_id": rid, "level": cfg.level, "group_type": cfg.group_col, "original_trait_group": group_key, "assigned_child_group_id": node_id, "assigned_target": dom, "separator_stack": sep_stack, "status": status})
            continue

        # Try to split collision group.
        best, audit_df = _best_separator(node_df, row_ids, trait_map, target_col, cfg)
        if not audit_df.empty:
            audit_df = audit_df.head(100)
            for _, ar in audit_df.iterrows():
                d = ar.to_dict()
                d.update({
                    "level": cfg.level,
                    "group_type": cfg.group_col,
                    "original_trait_group": group_key,
                    "node_id": node_id,
                    "node_size": len(row_ids),
                    "node_target_counts": counts_str,
                    "depth": depth,
                    "current_separator_stack": sep_stack,
                })
                audits.append(d)
        if best is None:
            status = "SPARE_REVISIT_NO_SEPARATOR"
            spares.append({
                "level": cfg.level,
                "group_type": cfg.group_col,
                "original_trait_group": group_key,
                "child_group_id": node_id,
                "separator_stack": sep_stack,
                "total_seed_rows": len(row_ids),
                "dominant_target": dom,
                "dominant_count": cnt,
                "dominant_pct": round(pct, 4),
                "distinct_targets": distinct,
                "target_counts": counts_str,
                "status": status,
            })
            for rid in row_ids:
                lineage.append({"row_id": rid, "level": cfg.level, "group_type": cfg.group_col, "original_trait_group": group_key, "assigned_child_group_id": node_id, "assigned_target": dom, "separator_stack": sep_stack, "status": status})
            continue

        sep = str(best["trait"])
        child_ids = set(best["child_ids"])
        rem_ids = [rid for rid in row_ids if rid not in child_ids]
        child_ids_list = [rid for rid in row_ids if rid in child_ids]
        child_num += 1
        new_stack = sep if not sep_stack else sep_stack + " && " + sep
        # Add child first and remainder second. Limit children/recursion by config.
        queue.insert(0, (f"D{depth+1}_A{child_num:02d}", child_ids_list, depth + 1, new_stack))
        if rem_ids:
            queue.insert(1, (f"D{depth+1}_R{child_num:02d}", rem_ids, depth + 1, sep_stack + f" && NOT({sep})" if sep_stack else f"NOT({sep})"))
        if child_num >= cfg.max_children_per_group:
            # Any remaining queued nodes are sent to spare to prevent runaway recursion.
            for qnode, qids, qdepth, qstack in queue:
                qdf = group_df[group_df["row_id"].astype(str).isin(set(qids))]
                qdom, qcnt, qdistinct, qpct, qcounts = _target_counts(qdf, target_col)
                qid = f"{cfg.level.upper()}_{cfg.group_col}_{group_idx:05d}_{qnode}"
                spares.append({
                    "level": cfg.level, "group_type": cfg.group_col, "original_trait_group": group_key,
                    "child_group_id": qid, "separator_stack": qstack, "total_seed_rows": len(qids),
                    "dominant_target": qdom, "dominant_count": qcnt, "dominant_pct": round(qpct,4),
                    "distinct_targets": qdistinct, "target_counts": qcounts, "status": "SPARE_REVISIT_CHILD_CAP",
                })
                for rid in qids:
                    lineage.append({"row_id": rid, "level": cfg.level, "group_type": cfg.group_col, "original_trait_group": group_key, "assigned_child_group_id": qid, "assigned_target": qdom, "separator_stack": qstack, "status": "SPARE_REVISIT_CHILD_CAP"})
            queue.clear()
            break
    return profiles, lineage, spares, audits


def run_split_engine(ledger: pd.DataFrame, cfg: SplitConfig, progress=None, status=None) -> Dict[str, pd.DataFrame]:
    if ledger.empty:
        return {"profiles": pd.DataFrame(), "lineage": pd.DataFrame(), "spares": pd.DataFrame(), "candidate_audit": pd.DataFrame(), "group_audit": pd.DataFrame(), "end_audit": pd.DataFrame()}
    if cfg.group_col not in ledger.columns:
        raise ValueError(f"Group column not found: {cfg.group_col}")
    if _target_col(cfg.level) not in ledger.columns:
        raise ValueError(f"Target column not found for level={cfg.level}")

    scoped = _group_scope(ledger, cfg).copy()
    if scoped.empty:
        raise ValueError("No rows remain after scope/core filter.")
    csv_text = scoped.to_csv(index=False)
    trait_map = build_trait_map_from_csv(csv_text)

    # Build group list sorted by size descending; process chunk by group number.
    groups = []
    for key, g in scoped.groupby(cfg.group_col, dropna=False):
        key = str(key)
        if key == "":
            key = "<EMPTY_GROUP>"
        groups.append((key, len(g)))
    groups.sort(key=lambda x: (-x[1], x[0]))
    start = max(1, int(cfg.start_group))
    selected_keys = [k for k, _ in groups[start-1:start-1+cfg.max_groups]]

    all_profiles: List[Dict] = []
    all_lineage: List[Dict] = []
    all_spares: List[Dict] = []
    all_audits: List[Dict] = []
    group_audit_rows: List[Dict] = []
    total = max(1, len(selected_keys))

    for idx, key in enumerate(selected_keys, start=1):
        if progress is not None:
            progress.progress((idx-1)/total)
        if status is not None:
            status.info(f"Processing {cfg.level} group {idx}/{total}: {cfg.group_col} = {key[:90]}")
        gdf = scoped[scoped[cfg.group_col].astype(str).eq(key if key != "<EMPTY_GROUP>" else "")].copy()
        if gdf.empty and key == "<EMPTY_GROUP>":
            gdf = scoped[scoped[cfg.group_col].astype(str).eq("")].copy()
        dom, cnt, distinct, pct, counts_str = _target_counts(gdf, _target_col(cfg.level))
        group_audit_rows.append({
            "level": cfg.level,
            "group_type": cfg.group_col,
            "group_process_index": start + idx - 1,
            "trait_group": key,
            "total_seed_rows": len(gdf),
            "dominant_target": dom,
            "dominant_count": cnt,
            "dominant_pct": round(pct,4),
            "distinct_targets": distinct,
            "target_counts": counts_str,
            "initial_status": "PROFILED_DOMINANT" if len(gdf)>=cfg.min_group_size and pct>=cfg.dominance_threshold else "LOW_SAMPLE" if len(gdf)<cfg.min_group_size else "NEEDS_SPLIT",
        })
        p, l, s, a = process_one_group(scoped, key, gdf, trait_map, cfg, start + idx - 1)
        all_profiles.extend(p); all_lineage.extend(l); all_spares.extend(s); all_audits.extend(a)

    if progress is not None:
        progress.progress(1.0)
    if status is not None:
        status.success(f"Finished {cfg.level} split engine for {len(selected_keys)} groups.")

    profiles = pd.DataFrame(all_profiles)
    lineage = pd.DataFrame(all_lineage)
    spares = pd.DataFrame(all_spares)
    cand = pd.DataFrame(all_audits)
    group_audit = pd.DataFrame(group_audit_rows)

    # End accounting: exactly one lineage assignment for each processed row.
    processed_ids = set()
    for key in selected_keys:
        ids = scoped[scoped[cfg.group_col].astype(str).eq(key if key != "<EMPTY_GROUP>" else "")]["row_id"].astype(str).tolist()
        processed_ids.update(ids)
    lineage_ids = set(lineage["row_id"].astype(str).tolist()) if not lineage.empty else set()
    dup_lineage = int(lineage["row_id"].duplicated().sum()) if not lineage.empty and "row_id" in lineage.columns else 0
    status_counts = lineage["status"].value_counts().reset_index() if not lineage.empty and "status" in lineage.columns else pd.DataFrame(columns=["status","count"])
    if not status_counts.empty:
        status_counts.columns = ["status", "seed_rows"]
    end_audit = pd.DataFrame([
        {"metric": "app_version", "value": APP_VERSION},
        {"metric": "level", "value": cfg.level},
        {"metric": "group_col", "value": cfg.group_col},
        {"metric": "groups_processed", "value": len(selected_keys)},
        {"metric": "processed_seed_rows", "value": len(processed_ids)},
        {"metric": "lineage_seed_rows", "value": len(lineage)},
        {"metric": "unique_lineage_seed_rows", "value": len(lineage_ids)},
        {"metric": "missing_from_lineage", "value": len(processed_ids - lineage_ids)},
        {"metric": "extra_lineage_rows", "value": len(lineage_ids - processed_ids)},
        {"metric": "duplicate_lineage_row_assignments", "value": dup_lineage},
        {"metric": "accounting_balanced", "value": (len(processed_ids - lineage_ids)==0 and len(lineage_ids - processed_ids)==0 and dup_lineage==0)},
    ])
    return {
        "profiles": profiles,
        "lineage": lineage,
        "spares": spares,
        "candidate_audit": cand,
        "group_audit": group_audit,
        "status_counts": status_counts,
        "end_audit": end_audit,
    }

# -------------------------
# Streamlit UI
# -------------------------
if "v38_package" not in st.session_state:
    st.session_state["v38_package"] = None
if "v38_manifest" not in st.session_state:
    st.session_state["v38_manifest"] = None

upload = st.file_uploader("Upload trait grouping audit ZIP or aabc_seed_group_ledger.csv", type=["zip", "csv"])

with st.sidebar:
    st.header("v38 settings")
    stage = st.selectbox("Stage", [
        "1 - Audit ledger only",
        "2 - Core trait-group split separators",
        "3 - Member trait-group split separators inside one core",
    ])
    group_col = st.selectbox("Trait group column", GROUP_COLS, index=4)
    dominance = st.slider("Dominance threshold to accept child group", 0.50, 1.00, 0.80, 0.05)
    min_group_size = st.slider("Min group size", 1, 50, 5, 1)
    min_child_hits = st.slider("Min child target hits", 1, 25, 3, 1)
    min_child_pct = st.slider("Min child dominance pct", 0.50, 1.00, 0.80, 0.05)
    max_depth = st.slider("Max split depth", 1, 5, 2, 1)
    candidate_pool = st.slider("Candidate trait pool per group", 10, 300, 80, 10)
    start_group = st.number_input("Start group #", min_value=1, max_value=100000, value=1, step=1)
    max_groups = st.slider("Groups to process this run", 10, 1000, 100, 10)
    max_children = st.slider("Max child splits per original group", 1, 20, 6, 1)
    core_filter = st.text_input("Member stage only: core filter, e.g. 025 or 389", value="")
    st.divider()
    st.caption("For tiny groups: min size=5 and child hits=3 allows a 4-of-5 split to be accepted if dominance threshold is met.")

if not upload:
    st.stop()

try:
    with st.spinner("Reading grouped seed ledger..."):
        ledger = read_ledger(upload)
    if ledger.empty:
        st.error("Ledger is empty.")
        st.stop()

    basic = pd.DataFrame([
        {"metric":"ledger_rows", "value": len(ledger)},
        {"metric":"distinct_row_ids", "value": ledger["row_id"].astype(str).nunique()},
        {"metric":"distinct_cores", "value": ledger["winner_core"].astype(str).nunique() if "winner_core" in ledger.columns else "missing"},
        {"metric":"distinct_members", "value": ledger["winner_member"].astype(str).nunique() if "winner_member" in ledger.columns else "missing"},
        {"metric":"distinct_streams", "value": ledger["StreamKey"].astype(str).nunique() if "StreamKey" in ledger.columns else "missing"},
        {"metric":"build_marker", "value": BUILD_MARKER},
    ])
    st.success(f"Loaded ledger: {len(ledger):,} rows, {ledger['row_id'].astype(str).nunique():,} unique row IDs.")

    run = st.button("Run selected v38 stage and prepare frozen ZIP", type="primary")
    if run:
        t0 = time.time()
        frames: Dict[str, pd.DataFrame] = {"summary_basic.csv": basic}
        texts: Dict[str, str] = {"README_v38.txt": f"{BUILD_MARKER}\n{DEPLOY_FILENAME_NOTE}\nStage: {stage}\nDiscovery lab only. Every processed seed row is assigned exactly once in seed_lineage_audit.csv.\n"}
        if stage.startswith("1"):
            group_summary_rows = []
            for gc in GROUP_COLS:
                if gc in ledger.columns:
                    gs = ledger.groupby(gc, dropna=False).size().reset_index(name="seed_rows")
                    group_summary_rows.append({"group_col": gc, "distinct_groups": len(gs), "largest_group": int(gs["seed_rows"].max()) if not gs.empty else 0, "total_seed_rows": int(gs["seed_rows"].sum()) if not gs.empty else 0})
                    frames[f"audit_{gc}_group_sizes_top500.csv"] = gs.sort_values("seed_rows", ascending=False).head(500)
            frames["ledger_group_column_audit.csv"] = pd.DataFrame(group_summary_rows)
            frames["seed_count_by_core.csv"] = ledger.groupby("winner_core").size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False) if "winner_core" in ledger.columns else pd.DataFrame()
            frames["seed_count_by_core_member.csv"] = ledger.groupby(["winner_core","winner_member"]).size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False) if {"winner_core","winner_member"}.issubset(ledger.columns) else pd.DataFrame()
            manifest = {
                "stage": stage,
                "ledger_rows": int(len(ledger)),
                "files": list(frames.keys()),
                "elapsed_seconds": round(time.time()-t0, 3),
            }
        else:
            level = "core" if stage.startswith("2") else "member"
            cfg = SplitConfig(
                level=level,
                group_col=group_col,
                dominance_threshold=float(dominance),
                min_group_size=int(min_group_size),
                min_child_hits=int(min_child_hits),
                min_child_pct=float(min_child_pct),
                max_depth=int(max_depth),
                max_groups=int(max_groups),
                start_group=int(start_group),
                candidate_pool=int(candidate_pool),
                max_children_per_group=int(max_children),
                core_filter=str(core_filter),
            )
            prog = st.progress(0.0)
            stat = st.empty()
            result = run_split_engine(ledger, cfg, progress=prog, status=stat)
            frames.update({
                f"{level}_split_profiles.csv": result["profiles"],
                f"{level}_seed_lineage_audit.csv": result["lineage"],
                f"{level}_spare_or_low_sample_groups.csv": result["spares"],
                f"{level}_separator_candidate_audit.csv": result["candidate_audit"].head(20000) if not result["candidate_audit"].empty else result["candidate_audit"],
                f"{level}_processed_group_audit.csv": result["group_audit"],
                f"{level}_status_counts.csv": result["status_counts"],
                f"{level}_end_accounting_audit.csv": result["end_audit"],
            })
            manifest = {
                "stage": stage,
                "level": level,
                "group_col": group_col,
                "start_group": int(start_group),
                "groups_processed_requested": int(max_groups),
                "profiles_found": int(len(result["profiles"])),
                "spare_groups": int(len(result["spares"])),
                "lineage_rows": int(len(result["lineage"])),
                "candidate_audit_rows_exported": int(min(len(result["candidate_audit"]), 20000)) if not result["candidate_audit"].empty else 0,
                "accounting_balanced": str(result["end_audit"].set_index("metric").loc["accounting_balanced", "value"]) if not result["end_audit"].empty else "unknown",
                "elapsed_seconds": round(time.time()-t0, 3),
                "files": list(frames.keys()),
            }
        package = zip_frames(frames, texts)
        # aggressively drop big frames before holding package
        zip_mb = len(package) / (1024*1024)
        del frames
        gc.collect()
        st.session_state["v38_package"] = package
        st.session_state["v38_manifest"] = manifest
        st.success(f"Frozen v38 ZIP prepared. Size: {zip_mb:.2f} MB. Elapsed: {manifest['elapsed_seconds']} sec.")

    if st.session_state.get("v38_manifest"):
        st.subheader("Frozen package manifest")
        st.json(st.session_state["v38_manifest"])
    if st.session_state.get("v38_package"):
        st.download_button(
            "Download v38 grouped separator outputs ZIP",
            st.session_state["v38_package"],
            file_name=f"core_affinity_lab_v38_{stage.split()[0]}_{group_col}_start_{int(start_group)}.zip",
            mime="application/zip",
        )

except Exception as e:
    st.error("v38 failed. Full exception below.")
    st.exception(e)
