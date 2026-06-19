#!/usr/bin/env python3
"""
Core Affinity Lab v40 — Pairwise Tournament Separator + Canonical Signature Registry

Purpose:
- Read the v37/v39 trait-grouping seed ledger.
- Separate cores FIRST using pairwise tournament mining inside trait groups.
- Split groups when a separator covers a child bucket.
- Canonicalize every child signature globally so A+B+C is the same group no matter where it was created.
- Preserve row_id / seed / winner_core / winner_member through every stage.
- Export end accounting proving no seed rows were lost.

Lab only. No daily playlist logic.
"""
from __future__ import annotations

import gc as py_gc
import io
import itertools
import re
import time
import zipfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v40"
BUILD_MARKER = "BUILD: core_affinity_lab_v40_PAIRWISE_TOURNAMENT_SIGNATURE_REGISTRY__2026-06-19"
DEPLOY_FILENAME_NOTE = "For Streamlit Cloud deployment, this file may be renamed to: core_affinity_lab_v1 (1).py"

st.set_page_config(page_title="Core Affinity Lab v40", layout="wide")
st.title("Core Affinity Lab v40 — Pairwise Tournament Separator + Signature Registry")
st.caption(BUILD_MARKER)
st.info(
    "Core-first grouped-seed separator engine. Pairwise tournament inside collision groups, canonical signature registry, "
    "child-group splitting, spare buckets, and end-to-end seed accounting."
)

# -------------------------
# IO helpers
# -------------------------

def safe_str_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(str).fillna("")
    return out


def read_zip_ledger(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()
    raw = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
        names = z.namelist()
        # Prefer the original grouping ledger.
        candidates = [n for n in names if n.endswith("aabc_seed_group_ledger.csv")]
        if not candidates:
            candidates = [n for n in names if "ledger" in n.lower() and n.lower().endswith(".csv")]
        if not candidates:
            raise ValueError("No aabc_seed_group_ledger.csv or ledger CSV found inside ZIP.")
        with z.open(candidates[0]) as f:
            return pd.read_csv(f, dtype=str)


def zip_frames(frames: Dict[str, pd.DataFrame], texts: Dict[str, str] | None = None) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(name, safe_str_df(df).to_csv(index=False))
        for name, txt in (texts or {}).items():
            z.writestr(name, str(txt))
    bio.seek(0)
    return bio.getvalue()


def canonical_signature(base_sig: str, extra_traits: Iterable[str] | None = None) -> str:
    atoms = []
    for part in str(base_sig).split("|"):
        p = part.strip()
        if p:
            atoms.append(p)
    if extra_traits:
        for t in extra_traits:
            t = str(t).strip()
            if t:
                atoms.append(t)
    # Canonical: dedupe + sort, so A+B+C matches everywhere.
    return "|".join(sorted(set(atoms)))

# -------------------------
# Trait generation from ledger row
# -------------------------

def row_traits(row: pd.Series) -> Set[str]:
    traits: Set[str] = set()

    # Basic scalar seed traits from ledger columns.
    direct_cols = [
        "seed_sum", "seed_spread", "seed_parity_pattern", "seed_highlow_pattern", "seed_structure",
        "seed_unique", "seed_maxrep", "seed_sum_bucket", "seed_spread_bucket", "seed_even_count",
        "seed_odd_count", "seed_high_count", "seed_low_count", "seed_mirror_count", "seed_mirror_signature",
        "seed_pair_signature",
    ]
    for c in direct_cols:
        if c in row and pd.notna(row[c]) and str(row[c]) != "":
            traits.add(f"{c}={row[c]}")

    # Present / missing digit atoms.
    present = str(row.get("seed_present_digits", ""))
    missing = str(row.get("seed_missing_digits", ""))
    for d in "0123456789":
        if d in present:
            traits.add(f"has{d}=1")
            traits.add(f"cnt{d}>0")
        if d in missing:
            traits.add(f"no{d}=1")
            traits.add(f"cnt{d}=0")

    # Pair signature atoms like pairs_03_04_05
    pair_sig = str(row.get("seed_pair_signature", ""))
    for m in re.findall(r"\d{2}", pair_sig):
        traits.add(f"has_pair_{m}=1")
    # Mirror atoms.
    mir = str(row.get("seed_mirror_signature", ""))
    if mir and mir != "nan":
        traits.add(f"mirror_sig={mir}")
        for m in re.findall(r"\d{2}", mir):
            traits.add(f"mirror_{m}=1")

    # Positional group atoms from group_positional field.
    pos = str(row.get("group_positional", ""))
    for atom in pos.split("|"):
        atom = atom.strip()
        if atom:
            traits.add(atom)

    # Atoms from each group column; useful because old miners used stacked profile conditions.
    for gc_col in ["group_primary", "group_distribution", "group_digitset", "group_separator_profile"]:
        val = str(row.get(gc_col, ""))
        for atom in val.split("|"):
            atom = atom.strip()
            if atom:
                traits.add(atom)

    return traits


@st.cache_data(show_spinner=False)
def add_trait_sets(df_csv: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(df_csv), dtype=str)
    if "row_id" not in df.columns:
        df["row_id"] = [str(i + 1) for i in range(len(df))]
    df["row_id"] = df["row_id"].astype(str)
    if "winner_core" not in df.columns:
        raise ValueError("Ledger must include winner_core column.")
    if "winner_member" not in df.columns:
        raise ValueError("Ledger must include winner_member column.")
    df["trait_set"] = [row_traits(r) for _, r in df.iterrows()]
    return df

# -------------------------
# Mining logic
# -------------------------

def core_counts(df: pd.DataFrame) -> Counter:
    return Counter(df["winner_core"].astype(str))


def dominance(counts: Counter) -> Tuple[str, int, float]:
    total = sum(counts.values())
    if not total:
        return "", 0, 0.0
    target, n = counts.most_common(1)[0]
    return target, n, n / total


def trait_counts_for_rows(rows: pd.DataFrame) -> Counter:
    c = Counter()
    for ts in rows["trait_set"]:
        c.update(ts)
    return c


def score_pairwise_trait(group_df: pd.DataFrame, pair_df: pd.DataFrame, core_a: str, core_b: str, trait: str) -> dict | None:
    # Candidate child is rows inside whole group matching trait.
    child = group_df[group_df["trait_set"].map(lambda s: trait in s)]
    if child.empty:
        return None
    pair_child = pair_df[pair_df["trait_set"].map(lambda s: trait in s)]
    if pair_child.empty:
        return None
    ca = int((pair_child["winner_core"].astype(str) == core_a).sum())
    cb = int((pair_child["winner_core"].astype(str) == core_b).sum())
    pair_support = ca + cb
    if pair_support == 0:
        return None
    favored = core_a if ca >= cb else core_b
    fav_hits = max(ca, cb)
    other_hits = min(ca, cb)
    pair_win_rate = fav_hits / pair_support
    pair_gap = (fav_hits - other_hits) / pair_support

    all_counts = core_counts(child)
    dom_core, dom_hits, dom_pct = dominance(all_counts)
    return {
        "trait": trait,
        "favored_core_pair": favored,
        "pair_support": pair_support,
        "pair_favored_hits": fav_hits,
        "pair_other_hits": other_hits,
        "pair_win_rate": pair_win_rate,
        "pair_gap": pair_gap,
        "child_total_rows": len(child),
        "child_distinct_cores": len(all_counts),
        "child_dominant_core": dom_core,
        "child_dominant_hits": dom_hits,
        "child_dominance_pct": dom_pct,
        "child_row_ids": set(child["row_id"].astype(str)),
        "score": (pair_win_rate * 100.0) + (pair_gap * 25.0) + min(pair_support, 25) + (dom_pct * 10.0),
    }


def mine_group_pairwise(
    group_df: pd.DataFrame,
    base_signature: str,
    cfg: dict,
    group_ordinal: int,
    progress_note: str = "",
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Tournament-style mining for one group.

    Returns profiles, rules, lineage rows, spare rows.
    """
    active = group_df.copy()
    profiles: List[dict] = []
    rules: List[dict] = []
    lineage: List[dict] = []
    spare: List[dict] = []

    base_sig = canonical_signature(base_signature)
    round_num = 0
    while round_num < int(cfg["max_rounds_per_group"]):
        round_num += 1
        if len(active) < int(cfg["min_group_size"]):
            break
        counts = core_counts(active)
        dom_core, dom_hits, dom_pct = dominance(counts)
        if dom_pct >= float(cfg["accept_dominance_pct"]):
            child_sig = base_sig
            profiles.append({
                "group_ordinal": group_ordinal,
                "round_num": round_num,
                "profile_type": "DOMINANT_BASE_OR_REMAINDER",
                "signature": child_sig,
                "assigned_core": dom_core,
                "assigned_hits": dom_hits,
                "total_rows": len(active),
                "dominance_pct": round(dom_pct, 4),
                "distinct_cores": len(counts),
                "separator_trait": "",
                "status": "PROFILED_DOMINANT",
            })
            for _, r in active.iterrows():
                lineage.append({
                    "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                    "original_signature": base_sig, "child_signature": child_sig,
                    "separator_trait": "", "assigned_core": dom_core,
                    "status": "PROFILED_DOMINANT",
                })
            active = active.iloc[0:0].copy()
            break

        if len(counts) < 2:
            break
        top2 = counts.most_common(2)
        core_a, n_a = top2[0]
        core_b, n_b = top2[1]
        pair_df = active[active["winner_core"].astype(str).isin([core_a, core_b])]
        if len(pair_df) < int(cfg["min_pair_support"]):
            break

        # Only mine traits from pair rows. This mirrors old pairwise miners and avoids 120-way explosion.
        tcounts = trait_counts_for_rows(pair_df)
        candidate_traits = [t for t, n in tcounts.most_common(int(cfg["candidate_pool"])) if n >= int(cfg["min_pair_support"])]
        best = None
        audit_local = []
        for t in candidate_traits:
            sc = score_pairwise_trait(active, pair_df, core_a, core_b, t)
            if not sc:
                continue
            audit_local.append(sc)
            if sc["pair_support"] < int(cfg["min_pair_support"]):
                continue
            if sc["pair_win_rate"] < float(cfg["min_pair_win_rate"]):
                continue
            if sc["pair_gap"] < float(cfg["min_pair_gap"]):
                continue
            if sc["child_total_rows"] < int(cfg["min_child_rows"]):
                continue
            if best is None or sc["score"] > best["score"]:
                best = sc

        # Save top audits even when no accepted split.
        for rank, sc in enumerate(sorted(audit_local, key=lambda x: x["score"], reverse=True)[:int(cfg["audit_top_n_per_group"])], start=1):
            rules.append({
                "group_ordinal": group_ordinal,
                "round_num": round_num,
                "rule_rank_in_round": rank,
                "base_signature": base_sig,
                "core_a": core_a,
                "core_b": core_b,
                "separator_trait": sc["trait"],
                "favored_core_pair": sc["favored_core_pair"],
                "pair_support": sc["pair_support"],
                "pair_favored_hits": sc["pair_favored_hits"],
                "pair_other_hits": sc["pair_other_hits"],
                "pair_win_rate": round(sc["pair_win_rate"], 4),
                "pair_gap": round(sc["pair_gap"], 4),
                "child_total_rows": sc["child_total_rows"],
                "child_distinct_cores": sc["child_distinct_cores"],
                "child_dominant_core": sc["child_dominant_core"],
                "child_dominant_hits": sc["child_dominant_hits"],
                "child_dominance_pct": round(sc["child_dominance_pct"], 4),
                "accepted_split": bool(best is not None and sc["trait"] == best["trait"]),
                "score": round(sc["score"], 4),
                "status": "ACCEPTED_SPLIT" if (best is not None and sc["trait"] == best["trait"]) else "AUDIT_CANDIDATE",
            })

        if best is None:
            break

        trait = best["trait"]
        child_mask = active["trait_set"].map(lambda s: trait in s)
        child = active[child_mask].copy()
        remainder = active[~child_mask].copy()
        child_sig = canonical_signature(base_sig, [trait])
        child_counts = core_counts(child)
        child_dom_core, child_dom_hits, child_dom_pct = dominance(child_counts)
        status = "SEPARATED_CHILD" if child_dom_pct >= float(cfg["accept_dominance_pct"]) else "CHILD_NEEDS_FUTURE_SEPARATOR"

        profiles.append({
            "group_ordinal": group_ordinal,
            "round_num": round_num,
            "profile_type": "PAIRWISE_SPLIT_CHILD",
            "signature": child_sig,
            "assigned_core": child_dom_core,
            "assigned_hits": child_dom_hits,
            "total_rows": len(child),
            "dominance_pct": round(child_dom_pct, 4),
            "distinct_cores": len(child_counts),
            "separator_trait": trait,
            "pair_core_a": core_a,
            "pair_core_b": core_b,
            "pair_win_rate": round(best["pair_win_rate"], 4),
            "pair_gap": round(best["pair_gap"], 4),
            "status": status,
        })
        for _, r in child.iterrows():
            lineage.append({
                "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                "original_signature": base_sig, "child_signature": child_sig,
                "separator_trait": trait, "assigned_core": child_dom_core,
                "status": status,
            })
        active = remainder

    # Anything left goes to spare; still assigned actual core/member and signature.
    if not active.empty:
        counts = core_counts(active)
        dom_core, dom_hits, dom_pct = dominance(counts)
        for _, r in active.iterrows():
            spare.append({
                "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                "original_signature": base_sig,
                "current_signature": base_sig,
                "dominant_core_in_spare": dom_core,
                "dominance_pct_in_spare": round(dom_pct, 4),
                "distinct_cores_in_spare": len(counts),
                "status": "SPARE_REVISIT_NO_PAIRWISE_SEPARATOR",
            })
            lineage.append({
                "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                "original_signature": base_sig, "child_signature": base_sig,
                "separator_trait": "", "assigned_core": "",
                "status": "SPARE_REVISIT_NO_PAIRWISE_SEPARATOR",
            })

    return profiles, rules, lineage, spare


def build_signature_registry(lineage_df: pd.DataFrame) -> pd.DataFrame:
    if lineage_df.empty:
        return pd.DataFrame()
    g = lineage_df.groupby("child_signature", dropna=False).agg(
        rows=("row_id", "count"),
        distinct_actual_cores=("winner_core", "nunique"),
        actual_cores=("winner_core", lambda x: ",".join(sorted(set(map(str, x))))[:1000]),
        statuses=("status", lambda x: ",".join(sorted(set(map(str, x))))),
    ).reset_index()
    return g.sort_values(["rows", "distinct_actual_cores"], ascending=[False, False])


def end_accounting(original: pd.DataFrame, lineage: pd.DataFrame, spare: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    orig_ids = set(original["row_id"].astype(str))
    lin_ids = set(lineage["row_id"].astype(str)) if not lineage.empty else set()
    return pd.DataFrame([
        {"metric": "original_rows_processed", "value": len(original)},
        {"metric": "unique_original_row_ids", "value": len(orig_ids)},
        {"metric": "lineage_rows", "value": len(lineage)},
        {"metric": "unique_lineage_row_ids", "value": len(lin_ids)},
        {"metric": "missing_row_ids", "value": len(orig_ids - lin_ids)},
        {"metric": "extra_row_ids", "value": len(lin_ids - orig_ids)},
        {"metric": "duplicate_lineage_assignments", "value": max(0, len(lineage) - len(lin_ids)) if not lineage.empty else 0},
        {"metric": "spare_rows", "value": len(spare)},
        {"metric": "profile_rows", "value": len(profiles)},
        {"metric": "accounting_balanced", "value": str(orig_ids == lin_ids and (len(lineage)==len(lin_ids))).upper()},
    ])

# -------------------------
# UI
# -------------------------

upload = st.file_uploader("Upload trait_grouping_seed_accounting_audit ZIP", type=["zip"])

with st.sidebar:
    st.header("v40 settings")
    stage = st.selectbox("Stage", [
        "1 - Audit ledger only",
        "2 - Core pairwise tournament separators",
    ])
    group_col = st.selectbox("Trait group column", [
        "group_primary", "group_distribution", "group_digitset", "group_positional", "group_separator_profile"
    ], index=0)
    start_group = st.number_input("Start group #", min_value=1, value=1, step=1)
    groups_to_process = st.slider("Groups to process", 10, 1000, 100, 10)
    sort_groups_by_size = st.checkbox("Sort groups by size descending", value=True)
    st.divider()
    min_group_size = st.slider("Min group size", 2, 50, 5, 1)
    accept_dominance_pct = st.slider("Accept dominance pct", 0.40, 0.95, 0.60, 0.05)
    max_rounds_per_group = st.slider("Max tournament rounds per group", 1, 20, 6, 1)
    min_pair_support = st.slider("Min pair support", 2, 30, 3, 1)
    min_child_rows = st.slider("Min child rows", 2, 50, 3, 1)
    min_pair_win_rate = st.slider("Min pair win-rate", 0.50, 0.95, 0.60, 0.05)
    min_pair_gap = st.slider("Min pair gap", 0.00, 0.80, 0.15, 0.05)
    candidate_pool = st.slider("Candidate trait pool", 20, 300, 120, 10)
    audit_top_n_per_group = st.slider("Audit candidates per round", 1, 50, 10, 1)

if not upload:
    st.stop()

try:
    with st.spinner("Loading grouped seed ledger..."):
        raw = read_zip_ledger(upload)
        df = add_trait_sets(raw.to_csv(index=False))
    st.success(f"Loaded ledger: {len(df):,} rows | {df['row_id'].nunique():,} unique row IDs | {df['winner_core'].nunique():,} cores | {df['winner_member'].nunique():,} members")

    if group_col not in df.columns:
        st.error(f"Selected group column not found: {group_col}")
        st.stop()

    # Group universe
    group_counts = df.groupby(group_col, dropna=False).size().reset_index(name="rows")
    group_counts["distinct_cores"] = df.groupby(group_col, dropna=False)["winner_core"].nunique().values
    if sort_groups_by_size:
        group_counts = group_counts.sort_values(["rows", "distinct_cores"], ascending=[False, False]).reset_index(drop=True)
    else:
        group_counts = group_counts.sort_values(group_col).reset_index(drop=True)
    group_counts["group_ordinal"] = range(1, len(group_counts)+1)

    frames: Dict[str, pd.DataFrame] = {}
    texts = {"README_v40.txt": f"{BUILD_MARKER}\n{DEPLOY_FILENAME_NOTE}\nStage: {stage}\nGroup column: {group_col}\n"}

    if stage.startswith("1"):
        audit = pd.DataFrame([
            {"metric": "app_version", "value": APP_VERSION},
            {"metric": "rows", "value": len(df)},
            {"metric": "unique_row_ids", "value": df["row_id"].nunique()},
            {"metric": "distinct_cores", "value": df["winner_core"].nunique()},
            {"metric": "distinct_members", "value": df["winner_member"].nunique()},
            {"metric": "group_column", "value": group_col},
            {"metric": "distinct_groups", "value": len(group_counts)},
            {"metric": "largest_group", "value": int(group_counts["rows"].max()) if not group_counts.empty else 0},
            {"metric": "accounting_balanced_pre", "value": str(len(df)==df['row_id'].nunique()).upper()},
        ])
        frames = {
            "summary_basic.csv": audit,
            "group_universe_audit.csv": group_counts,
            "seed_count_by_core.csv": df.groupby("winner_core").size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False),
            "seed_count_by_core_member.csv": df.groupby(["winner_core","winner_member"]).size().reset_index(name="seed_rows").sort_values("seed_rows", ascending=False),
        }
        st.subheader("Stage 1 audit")
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.subheader("Group universe sample")
        st.dataframe(group_counts.head(50), use_container_width=True, hide_index=True)

    elif stage.startswith("2"):
        cfg = {
            "min_group_size": min_group_size,
            "accept_dominance_pct": accept_dominance_pct,
            "max_rounds_per_group": max_rounds_per_group,
            "min_pair_support": min_pair_support,
            "min_child_rows": min_child_rows,
            "min_pair_win_rate": min_pair_win_rate,
            "min_pair_gap": min_pair_gap,
            "candidate_pool": candidate_pool,
            "audit_top_n_per_group": audit_top_n_per_group,
        }
        selected = group_counts.iloc[int(start_group)-1:int(start_group)-1+int(groups_to_process)].copy()
        selected_groups = selected[group_col].astype(str).tolist()
        work = df[df[group_col].astype(str).isin(selected_groups)].copy()
        progress = st.progress(0.0)
        status = st.empty()
        all_profiles: List[dict] = []
        all_rules: List[dict] = []
        all_lineage: List[dict] = []
        all_spare: List[dict] = []
        processed_group_rows: List[dict] = []
        t0 = time.time()
        total = max(1, len(selected_groups))
        for i, sig in enumerate(selected_groups, start=1):
            progress.progress((i-1)/total)
            status.info(f"Processing group {i:,}/{total:,}: {str(sig)[:120]}")
            gdf = work[work[group_col].astype(str)==sig].copy()
            if len(gdf) < int(min_group_size):
                for _, r in gdf.iterrows():
                    all_spare.append({
                        "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                        "original_signature": canonical_signature(sig), "current_signature": canonical_signature(sig),
                        "dominant_core_in_spare": "", "dominance_pct_in_spare": 0, "distinct_cores_in_spare": gdf["winner_core"].nunique(),
                        "status": "LOW_SAMPLE_GROUP",
                    })
                    all_lineage.append({
                        "row_id": r["row_id"], "winner_core": r["winner_core"], "winner_member": r["winner_member"],
                        "original_signature": canonical_signature(sig), "child_signature": canonical_signature(sig),
                        "separator_trait": "", "assigned_core": "", "status": "LOW_SAMPLE_GROUP",
                    })
                processed_group_rows.append({"group_ordinal": int(selected.iloc[i-1]["group_ordinal"]), "signature": sig, "rows": len(gdf), "status":"LOW_SAMPLE_GROUP"})
                continue
            profiles, rules, lineage, spare = mine_group_pairwise(gdf, sig, cfg, int(selected.iloc[i-1]["group_ordinal"]))
            all_profiles.extend(profiles); all_rules.extend(rules); all_lineage.extend(lineage); all_spare.extend(spare)
            processed_group_rows.append({
                "group_ordinal": int(selected.iloc[i-1]["group_ordinal"]),
                "signature": sig,
                "rows": len(gdf),
                "distinct_cores": gdf["winner_core"].nunique(),
                "profiles_created": len(profiles),
                "rule_audit_rows": len(rules),
                "spare_rows": len(spare),
                "status": "PROCESSED",
            })
            if i % 10 == 0:
                py_gc.collect()
        progress.progress(1.0)
        status.success(f"Finished {len(selected_groups):,} groups in {time.time()-t0:.2f}s")

        prof_df = pd.DataFrame(all_profiles)
        rules_df = pd.DataFrame(all_rules)
        lineage_df = pd.DataFrame(all_lineage)
        spare_df = pd.DataFrame(all_spare)
        processed_df = pd.DataFrame(processed_group_rows)
        registry_df = build_signature_registry(lineage_df)
        end_df = end_accounting(work, lineage_df, spare_df, prof_df)
        status_counts = lineage_df.groupby("status").size().reset_index(name="rows") if not lineage_df.empty else pd.DataFrame(columns=["status","rows"])

        frames = {
            "summary_basic.csv": pd.DataFrame([
                {"metric":"app_version", "value":APP_VERSION},
                {"metric":"stage", "value":stage},
                {"metric":"group_column", "value":group_col},
                {"metric":"start_group", "value":start_group},
                {"metric":"groups_requested", "value":groups_to_process},
                {"metric":"groups_processed", "value":len(selected_groups)},
                {"metric":"seed_rows_processed", "value":len(work)},
                {"metric":"profiles_created", "value":len(prof_df)},
                {"metric":"rule_audit_rows", "value":len(rules_df)},
                {"metric":"spare_rows", "value":len(spare_df)},
                {"metric":"elapsed_seconds", "value":round(time.time()-t0, 3)},
            ]),
            "core_pairwise_profiles.csv": prof_df,
            "core_pairwise_separator_rule_audit.csv": rules_df,
            "core_seed_lineage_audit.csv": lineage_df,
            "core_spare_revisit_rows.csv": spare_df,
            "core_processed_group_audit.csv": processed_df,
            "global_canonical_signature_registry.csv": registry_df,
            "core_end_accounting_audit.csv": end_df,
            "core_status_counts.csv": status_counts,
        }
        st.subheader("Stage 2 manifest")
        manifest = frames["summary_basic.csv"]
        st.dataframe(manifest, use_container_width=True, hide_index=True)
        st.subheader("Accounting")
        st.dataframe(end_df, use_container_width=True, hide_index=True)
        st.subheader("Status counts")
        st.dataframe(status_counts, use_container_width=True, hide_index=True)

    package = zip_frames(frames, texts)
    st.download_button(
        "Download v40 outputs ZIP",
        package,
        file_name=f"core_affinity_lab_v40_{stage.split()[0]}_{group_col}_start_{int(start_group)}.zip",
        mime="application/zip",
    )
    py_gc.collect()

except Exception as e:
    st.error("v40 stage failed. Full traceback below.")
    st.exception(e)
