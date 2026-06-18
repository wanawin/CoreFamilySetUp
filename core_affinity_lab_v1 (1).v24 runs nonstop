#!/usr/bin/env python3
"""
Core Affinity Lab v24 — Profile Decomposition Debug + Coverage Fix
Coverage-first profile mining for all 120 Pick-4 AABC cores and all 360 AABC members.

Purpose:
- Build multiple profiles per core/member by repeatedly covering the largest remaining winner buckets.
- Then identify profile collisions and mine separator traits only where profiles collide.
- Lab only. No daily-play, budget, RTE, B1Z0, ZLT, or production playlist logic.
"""
from __future__ import annotations

import io
import itertools
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v24"
BUILD_MARKER = "BUILD: core_affinity_lab_v24_PROFILE_DECOMPOSITION_DEBUG_COVERAGE_FIX__2026-06-18"
DEPLOY_FILENAME_NOTE = "For Streamlit Cloud deployment, this file may be renamed to: core_affinity_lab_v1 (1).py"

st.set_page_config(page_title="Core Affinity Lab v24", layout="wide")
st.title("Core Affinity Lab v24 — Profile Decomposition Debug + Coverage Fix")
st.caption(BUILD_MARKER)
st.info(
    "Coverage-first lab: build multiple profiles per core/member, then mine separators only for collisions. "
    "This does not touch daily playlist logic."
)

# -------------------------
# IO
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


def read_table(upload) -> pd.DataFrame:
    raw = _bytes(upload)
    if not raw:
        return pd.DataFrame()
    name = str(getattr(upload, "name", "uploaded")).lower()
    text = raw.decode("utf-8", errors="replace")
    if name.endswith(".csv"):
        try:
            return pd.read_csv(io.StringIO(text), dtype=str)
        except Exception:
            pass
    for sep in ["\t", ",", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
            if df.shape[1] >= 3:
                return df
        except Exception:
            pass
    return pd.read_csv(io.StringIO(text), sep=None, dtype=str, engine="python")


def norm4(x) -> str:
    # only base Pick-4 digits before comma/add-on
    base = str(x).split(",", 1)[0]
    digs = re.findall(r"\d", base)
    if len(digs) < 4:
        return ""
    return "".join(digs[:4])


def classify_aabc(result4: str) -> Tuple[str, str, bool]:
    s = norm4(result4)
    if len(s) != 4:
        return "", "", False
    ss = "".join(sorted(s))
    counts = sorted(Counter(ss).values(), reverse=True)
    if counts != [2, 1, 1]:
        return "", "", False
    core = "".join(sorted(set(ss)))
    member = ss
    return core, member, True


def all_cores() -> List[str]:
    return ["".join(c) for c in itertools.combinations("0123456789", 3)]


def core_members(core: str) -> List[str]:
    return sorted("".join(sorted(core + d)) for d in core)


def parse_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = lower.get("date") or lower.get("draw date")
    state_col = lower.get("state")
    game_col = lower.get("game")
    stream_col = lower.get("streamkey") or lower.get("stream")
    result_col = lower.get("result4") or lower.get("result") or lower.get("draw") or lower.get("winning number")
    if result_col is None:
        # fallback for raw four-column tab file
        if df.shape[1] >= 4:
            cols = list(df.columns)
            date_col, state_col, game_col, result_col = cols[0], cols[1], cols[2], cols[3]
        else:
            raise ValueError("Could not find Result/Result4 column.")
    out = pd.DataFrame()
    out["Date"] = df[date_col].astype(str) if date_col else ""
    out["DateParsed"] = pd.to_datetime(out["Date"], errors="coerce")
    out["State"] = df[state_col].astype(str) if state_col else ""
    out["Game"] = df[game_col].astype(str) if game_col else ""
    if stream_col:
        out["StreamKey"] = df[stream_col].astype(str)
    else:
        out["StreamKey"] = out["State"].str.strip() + " | " + out["Game"].str.strip()
    out["Result4"] = df[result_col].map(norm4)
    out = out[(out["Result4"].str.len() == 4) & out["DateParsed"].notna()].copy()
    out["DateParsed"] = out["DateParsed"].dt.normalize()
    out = out.drop_duplicates(["DateParsed", "StreamKey", "Result4"]).sort_values(["StreamKey", "DateParsed"]).reset_index(drop=True)
    return out


# -------------------------
# Trait generator
# -------------------------
MIRRORS = [(0,5), (1,6), (2,7), (3,8), (4,9)]

def seed_traits(seed: str) -> List[str]:
    s = norm4(seed)
    if len(s) != 4:
        return []
    ds = [int(c) for c in s]
    traits = []
    counts = Counter(ds)
    unique = len(counts)
    max_rep = max(counts.values())
    sorted_s = "".join(sorted(s))
    total = sum(ds)
    root = total
    while root >= 10:
        root = sum(int(c) for c in str(root))
    spread = max(ds) - min(ds)

    traits += [f"seed={s}", f"seed_sorted={sorted_s}"]
    traits += [f"sum={total}", f"sum_last={total%10}", f"root={root}", f"spread={spread}"]
    for lo, hi, label in [(0,9,"0_9"),(10,13,"10_13"),(14,17,"14_17"),(18,21,"18_21"),(22,36,"22_plus")]:
        if lo <= total <= hi:
            traits.append(f"sum_bucket={label}")
            break
    for lo, hi, label in [(0,2,"0_2"),(3,4,"3_4"),(5,6,"5_6"),(7,9,"7_plus")]:
        if lo <= spread <= hi:
            traits.append(f"spread_bucket={label}")
            break

    traits += [f"unique={unique}", f"maxrep={max_rep}", f"shape={'-'.join(map(str, sorted(counts.values(), reverse=True)))}"]
    traits += ["structure=" + ("AAAA" if max_rep == 4 else "AAAB" if max_rep == 3 else "AABB" if sorted(counts.values(), reverse=True)==[2,2] else "AABC" if max_rep==2 else "ABCD")]

    parity_pattern = "".join("E" if d % 2 == 0 else "O" for d in ds)
    highlow_pattern = "".join("H" if d >= 5 else "L" for d in ds)
    traits += [f"parity_pattern={parity_pattern}", f"highlow_pattern={highlow_pattern}"]
    traits += [f"even_count={sum(d%2==0 for d in ds)}", f"odd_count={sum(d%2==1 for d in ds)}"]
    traits += [f"high_count={sum(d>=5 for d in ds)}", f"low_count={sum(d<5 for d in ds)}"]

    for i, d in enumerate(ds, 1):
        traits += [f"pos{i}={d}", f"pos{i}_parity={'E' if d%2==0 else 'O'}", f"pos{i}_hl={'H' if d>=5 else 'L'}"]
    for i, j in itertools.combinations(range(4), 2):
        a, b = ds[i], ds[j]
        label = f"pos{i+1}{j+1}"
        traits += [f"{label}={a}{b}", f"{label}_sorted={''.join(sorted(str(a)+str(b)))}", f"{label}_sum={a+b}", f"{label}_diff={abs(a-b)}"]
        traits += [f"{label}_cmp={'EQ' if a==b else 'LT' if a<b else 'GT'}"]
    traits += [f"first2={s[:2]}", f"mid2={s[1:3]}", f"last2={s[2:]}", f"first_last={s[0]}{s[-1]}"]
    traits += [f"first2_sorted={''.join(sorted(s[:2]))}", f"mid2_sorted={''.join(sorted(s[1:3]))}", f"last2_sorted={''.join(sorted(s[2:]))}", f"first_last_sorted={''.join(sorted(s[0]+s[-1]))}"]
    traits += [f"first2_sum={ds[0]+ds[1]}", f"mid2_sum={ds[1]+ds[2]}", f"last2_sum={ds[2]+ds[3]}", f"firstlast_sum={ds[0]+ds[3]}"]

    for d in range(10):
        c = counts.get(d, 0)
        traits += [f"has{d}={1 if c else 0}", f"no{d}={1 if not c else 0}", f"cnt{d}={c}"]
    present = set(ds)
    for a, b in itertools.combinations(range(10), 2):
        has_pair = a in present and b in present
        traits.append(f"has_pair_{a}{b}={1 if has_pair else 0}")
        traits.append(f"no_pair_{a}{b}={1 if not has_pair else 0}")
    mcnt = 0
    for a, b in MIRRORS:
        flag = a in present and b in present
        if flag:
            mcnt += 1
        traits.append(f"mirror_{a}{b}={1 if flag else 0}")
    traits += [f"mirror_count={mcnt}", f"mirror_ge1={1 if mcnt>=1 else 0}", f"mirror_ge2={1 if mcnt>=2 else 0}"]
    consec = sum(1 for a,b in zip(ds, ds[1:]) if abs(a-b)==1)
    plusminus_any = sum(1 for a,b in itertools.combinations(ds, 2) if abs(a-b)==1)
    traits += [f"adj_consec_links={consec}", f"pm1_pairs={plusminus_any}", f"pm1_ge1={1 if plusminus_any>=1 else 0}"]
    return sorted(set(traits))


@st.cache_data(show_spinner=False)
def prepare_rows_from_csv(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text), dtype=str)
    hist = parse_history(df)
    if hist.empty:
        return pd.DataFrame()
    # seed = previous result within stream; target = current row
    hist["SeedResult"] = hist.groupby("StreamKey")["Result4"].shift(1)
    hist = hist.dropna(subset=["SeedResult"]).copy()
    labels = hist["Result4"].map(classify_aabc)
    hist["core_id"] = labels.map(lambda x: x[0])
    hist["member"] = labels.map(lambda x: x[1])
    hist["is_aabc"] = labels.map(lambda x: x[2])
    aabc = hist[hist["is_aabc"]].copy().reset_index(drop=True)
    aabc["trait_list"] = aabc["SeedResult"].map(seed_traits)
    return aabc


def zip_frames(frames: Dict[str, pd.DataFrame], texts: Dict[str, str] | None = None) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(name, df.to_csv(index=False))
        for name, txt in (texts or {}).items():
            z.writestr(name, str(txt))
    bio.seek(0)
    return bio.getvalue()


# -------------------------
# Profile decomposition
# -------------------------
@dataclass
class ProfileCfg:
    level: str
    max_profiles_per_target: int
    max_stack_depth: int
    min_target_hits: int
    min_precision: float
    min_gain_hits: int
    top_trait_pool: int


def _target_mask(rows: pd.DataFrame, level: str, target: str) -> pd.Series:
    return rows["core_id"].eq(target) if level == "core" else rows["member"].eq(target)


def _trait_counts(rows: pd.DataFrame, idxs: Iterable[int]) -> Counter:
    c = Counter()
    tl = rows["trait_list"].tolist()
    for i in idxs:
        c.update(tl[i])
    return c


def _rows_matching_stack(rows: pd.DataFrame, stack: Tuple[str, ...], candidate_idxs: Iterable[int] | None = None) -> List[int]:
    st = set(stack)
    tl = rows["trait_list"].tolist()
    idxs = candidate_idxs if candidate_idxs is not None else range(len(rows))
    out = []
    for i in idxs:
        if st.issubset(set(tl[i])):
            out.append(i)
    return out


def _score_stack(rows: pd.DataFrame, level: str, target: str, stack: Tuple[str,...], all_idxs: List[int]) -> Dict[str, float]:
    matched = _rows_matching_stack(rows, stack, all_idxs)
    if not matched:
        return {"sample":0,"target_hits":0,"sibling_hits":0,"precision":0.0,"target_coverage":0.0}
    mask = _target_mask(rows.iloc[matched], level, target)
    target_hits = int(mask.sum())
    sample = len(matched)
    sibling_hits = sample - target_hits
    total_target = int(_target_mask(rows, level, target).sum())
    return {
        "sample": sample,
        "target_hits": target_hits,
        "sibling_hits": sibling_hits,
        "precision": target_hits / sample if sample else 0.0,
        "target_coverage": target_hits / total_target if total_target else 0.0,
    }


def _score_stack_for_remaining(rows: pd.DataFrame, level: str, target: str, stack: Tuple[str,...], all_idxs: List[int], remaining_target_idxs: set) -> Dict[str, float]:
    """Coverage-first profile score.

    target_hits = NEW remaining target wins covered by this stack.
    sample/precision are still measured against the full universe so collisions can be audited,
    but precision is NOT allowed to block initial coverage buckets. Separators come later.
    """
    matched_all = _rows_matching_stack(rows, stack, all_idxs)
    if not matched_all:
        return {"sample":0,"target_hits":0,"target_hits_all":0,"sibling_hits":0,"precision":0.0,"target_coverage":0.0,"remaining_coverage":0.0}
    target_hits_all = sum(1 for i in matched_all if (rows.iloc[i]["core_id"] if level=="core" else rows.iloc[i]["member"]) == target)
    target_hits_remaining = len(set(matched_all) & remaining_target_idxs)
    sample = len(matched_all)
    sibling_hits = sample - target_hits_all
    total_target = int(_target_mask(rows, level, target).sum())
    return {
        "sample": sample,
        "target_hits": target_hits_remaining,
        "target_hits_all": target_hits_all,
        "sibling_hits": sibling_hits,
        "precision": target_hits_all / sample if sample else 0.0,
        "target_coverage": target_hits_remaining / total_target if total_target else 0.0,
        "remaining_coverage": target_hits_remaining / len(remaining_target_idxs) if remaining_target_idxs else 0.0,
    }


def debug_candidate_profiles(rows: pd.DataFrame, level: str, target: str, cfg: ProfileCfg, max_rows: int = 250) -> pd.DataFrame:
    """Export top candidate single traits before hard filters for one target."""
    if rows.empty or not target:
        return pd.DataFrame()
    all_idxs = list(range(len(rows)))
    target_idxs = {i for i in all_idxs if (rows.iloc[i]["core_id"] if level=="core" else rows.iloc[i]["member"]) == target}
    if not target_idxs:
        return pd.DataFrame()
    counts = _trait_counts(rows, target_idxs)
    out=[]
    for trait, n in counts.most_common(max(cfg.top_trait_pool, max_rows)):
        stats = _score_stack_for_remaining(rows, level, target, (trait,), all_idxs, target_idxs)
        out.append({
            "level": level,
            "target": target,
            "trait_stack": trait,
            "target_trait_hits": n,
            "sample_all": stats["sample"],
            "target_hits_all": stats["target_hits_all"],
            "sibling_hits": stats["sibling_hits"],
            "precision_all": round(float(stats["precision"]),4),
            "target_coverage_all": round(float(stats["target_coverage"]),4),
            "remaining_coverage": round(float(stats["remaining_coverage"]),4),
            "passes_min_hits": bool(stats["target_hits"] >= cfg.min_target_hits),
            "would_fail_old_precision_gate": bool(stats["precision"] < cfg.min_precision),
            "note": "v24 debug: initial profile discovery is coverage-first; low precision means separator needed later, not rejection now.",
        })
    df=pd.DataFrame(out)
    if not df.empty:
        df=df.sort_values(["target_trait_hits","remaining_coverage","precision_all"], ascending=[False,False,False]).head(max_rows).reset_index(drop=True)
    return df


def build_profiles(rows: pd.DataFrame, level: str, cfg: ProfileCfg, target_limit: int | None = None, target_start: int = 1, target_chunk_size: int | None = None, progress=None, status=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    targets = sorted(rows[level + "_id"].dropna().unique()) if level == "core" else sorted(rows["member"].dropna().unique())
    if target_start < 1:
        target_start = 1
    if target_chunk_size and target_chunk_size > 0:
        targets = targets[target_start-1:target_start-1+target_chunk_size]
    elif target_limit:
        targets = targets[:target_limit]
    all_idxs = list(range(len(rows)))
    profile_rows = []
    residual_rows = []
    total_targets = max(1, len(targets))
    for target_i, target in enumerate(targets, start=1):
        if progress is not None:
            progress.progress(min(1.0, (target_i - 1) / total_targets))
        if status is not None:
            status.info(f"Building {level} profiles: {target_i:,}/{total_targets:,} targets — current {level}: {target}")
        target_idxs_all = {i for i in all_idxs if (rows.iloc[i]["core_id"] if level=="core" else rows.iloc[i]["member"]) == target}
        remaining = set(target_idxs_all)
        total_target = len(target_idxs_all)
        if total_target == 0:
            continue
        for pnum in range(1, cfg.max_profiles_per_target + 1):
            if len(remaining) < cfg.min_target_hits:
                break
            counts = _trait_counts(rows, remaining)
            # IMPORTANT v24 fix:
            # profiles are discovered by target coverage first. Precision vs other targets is reported,
            # not used as a hard gate here; later collision/separator stages handle ambiguity.
            pool = [t for t, n in counts.most_common(cfg.top_trait_pool) if n >= cfg.min_target_hits]
            best_stack = None
            best_stats = None
            best_score = -1.0
            for t in pool:
                stats = _score_stack_for_remaining(rows, level, target, (t,), all_idxs, remaining)
                if stats["target_hits"] < cfg.min_target_hits:
                    continue
                # coverage-first: remaining target hits dominate; precision and margin are tie-breakers only
                margin = stats["target_hits_all"] - stats["sibling_hits"]
                score = (stats["target_hits"] * 10.0) + (stats["remaining_coverage"] * 100.0) + (stats["precision"] * 2.0) + max(0, margin) * 0.05
                if score > best_score:
                    best_score = score; best_stack=(t,); best_stats=stats
            if best_stack is None:
                break
            current = best_stack
            current_stats = best_stats
            for depth in range(2, cfg.max_stack_depth + 1):
                matched_remaining = _rows_matching_stack(rows, current, remaining)
                addon_counts = _trait_counts(rows, matched_remaining)
                add_best = None; add_stats = None; add_score = -1.0
                # Refinement may add traits, but only if it preserves meaningful NEW target coverage.
                min_keep = max(cfg.min_target_hits, math.ceil(current_stats["target_hits"] * 0.50))
                for t, _n in addon_counts.most_common(cfg.top_trait_pool):
                    if t in current:
                        continue
                    cand = tuple(sorted(current + (t,)))
                    stats = _score_stack_for_remaining(rows, level, target, cand, all_idxs, remaining)
                    if stats["target_hits"] < min_keep:
                        continue
                    # accept refinement if it improves precision materially or keeps same precision with better margin,
                    # without turning the coverage bucket into a tiny sliver.
                    margin = stats["target_hits_all"] - stats["sibling_hits"]
                    cur_margin = current_stats["target_hits_all"] - current_stats["sibling_hits"]
                    if stats["precision"] + 0.001 < current_stats["precision"] and margin <= cur_margin:
                        continue
                    score = (stats["target_hits"] * 8.0) + (stats["precision"] * 5.0) + max(0, margin) * 0.10
                    if score > add_score:
                        add_score = score; add_best = cand; add_stats = stats
                if add_best is None:
                    break
                current, current_stats = add_best, add_stats
            covered_remaining = set(_rows_matching_stack(rows, current, remaining))
            new_target_hits = len(covered_remaining)
            if new_target_hits < cfg.min_target_hits:
                break
            profile_rows.append({
                "level": level,
                "target": target,
                "profile_id": f"{level.upper()}_{target}_P{pnum:03d}",
                "profile_num": pnum,
                "trait_stack": " && ".join(current),
                "stack_depth": len(current),
                "total_target_hits": total_target,
                "new_target_hits": int(new_target_hits),
                "target_hits": int(current_stats["target_hits_all"]),
                "sibling_hits": int(current_stats["sibling_hits"]),
                "sample": int(current_stats["sample"]),
                "precision": round(float(current_stats["precision"]), 4),
                "target_coverage": round(float(current_stats["target_coverage"]), 4),
                "remaining_before": len(remaining),
                "remaining_after": len(remaining - covered_remaining),
                "coverage_of_remaining": round(len(covered_remaining) / len(remaining), 4) if remaining else 0,
                "needs_separator": bool(current_stats["precision"] < cfg.min_precision),
                "production_status": "DISCOVERY_PROFILE_NOT_PRODUCTION_RULE",
            })
            remaining -= covered_remaining
        residual_rows.append({
            "level": level,
            "target": target,
            "total_target_hits": total_target,
            "covered_hits": total_target - len(remaining),
            "uncovered_hits": len(remaining),
            "covered_rate": round((total_target-len(remaining))/total_target,4) if total_target else 0,
            "profiles_found": sum(1 for r in profile_rows if r["level"] == level and r["target"] == target),
        })
    if progress is not None:
        progress.progress(1.0)
    if status is not None:
        status.success(f"Finished {level} profile decomposition for {len(targets):,} targets.")
    return pd.DataFrame(profile_rows), pd.DataFrame(residual_rows)

def collision_scan(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty or "trait_stack" not in profiles.columns:
        return pd.DataFrame()
    rows=[]
    for stack, g in profiles.groupby("trait_stack"):
        targets = sorted(g["target"].astype(str).unique())
        if len(targets) <= 1:
            continue
        rows.append({
            "trait_stack": stack,
            "stack_depth": int(g["stack_depth"].max()),
            "collision_target_count": len(targets),
            "targets": ",".join(targets),
            "profile_ids": ";".join(g["profile_id"].astype(str).tolist()),
            "total_sample": int(g["sample"].sum()),
            "total_target_hits": int(g["target_hits"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["collision_target_count","total_sample"], ascending=[False,False]) if rows else pd.DataFrame()


def separator_refinement(rows: pd.DataFrame, profiles: pd.DataFrame, level: str, max_collisions: int, min_hits: int) -> pd.DataFrame:
    col = collision_scan(profiles)
    if col.empty:
        return pd.DataFrame()
    out=[]
    all_idxs = list(range(len(rows)))
    for _, c in col.head(max_collisions).iterrows():
        base_stack = tuple([x.strip() for x in str(c["trait_stack"]).split("&&") if x.strip()])
        targets = str(c["targets"]).split(",")
        matched = _rows_matching_stack(rows, base_stack, all_idxs)
        if len(targets) < 2 or not matched:
            continue
        for target in targets:
            target_idxs = [i for i in matched if (rows.iloc[i]["core_id"] if level=="core" else rows.iloc[i]["member"]) == target]
            if len(target_idxs) < min_hits:
                continue
            counts = _trait_counts(rows, target_idxs)
            for sep_trait, thits in counts.most_common(60):
                if sep_trait in base_stack: continue
                both = _rows_matching_stack(rows, base_stack + (sep_trait,), matched)
                if not both: continue
                target_hits = sum(1 for i in both if (rows.iloc[i]["core_id"] if level=="core" else rows.iloc[i]["member"]) == target)
                sibling_hits = len(both) - target_hits
                if target_hits < min_hits:
                    continue
                precision = target_hits / len(both)
                out.append({
                    "level": level,
                    "collision_stack": " && ".join(base_stack),
                    "favored_target": target,
                    "separator_trait": sep_trait,
                    "refined_stack": " && ".join(base_stack + (sep_trait,)),
                    "sample": len(both),
                    "target_hits": target_hits,
                    "sibling_hits": sibling_hits,
                    "precision": round(precision,4),
                    "hit_margin": target_hits - sibling_hits,
                    "targets_in_collision": ",".join(targets),
                    "production_status": "DISCOVERY_SEPARATOR_NOT_PRODUCTION_RULE",
                })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["precision","hit_margin","target_hits"], ascending=[False,False,False]).reset_index(drop=True)
    return df


# -------------------------
# UI
# -------------------------
upload = st.file_uploader("Upload clean Pick-4 history CSV/TXT", type=["csv","txt","tsv"])

with st.sidebar:
    st.header("v24 settings")
    stage = st.selectbox("Stage", [
        "1 - Data audit + base summaries",
        "2 - Core coverage-first profile decomposition",
        "3 - Member coverage-first profile decomposition",
        "4 - Core profile collision scan + separator refinement",
        "5 - Member profile collision scan + separator refinement",
    ])
    max_profiles = st.slider("Max profiles per target", 1, 20, 8)
    max_depth = st.slider("Max traits per profile stack", 1, 5, 3)
    min_hits = st.slider("Min target hits per profile", 3, 30, 10)
    min_precision = st.slider("Min profile precision", 0.40, 1.00, 0.60, 0.05)
    top_trait_pool = st.slider("Candidate trait pool per round", 25, 300, 120, 5)
    export_top_n = st.slider("Preview/export top N rows", 100, 5000, 1000, 100)
    st.divider()
    target_start = st.number_input("Chunk start target #", min_value=1, max_value=360, value=1, step=1)
    core_chunk_size = st.slider("Core chunk size", 5, 120, 20, 5)
    member_chunk_size = st.slider("Member chunk size", 5, 360, 30, 5)
    st.caption("v24 runs decomposition in chunks. Profile discovery is coverage-first; precision is reported for separator refinement, not used to block initial buckets.")

if not upload:
    st.stop()

try:
    raw_df = read_table(upload)
    # normalize once to CSV text for cache
    csv_text = raw_df.to_csv(index=False)
    with st.spinner("Preparing AABC seed→winner rows..."):
        rows = prepare_rows_from_csv(csv_text)
    if rows.empty:
        st.error("No AABC seed→winner rows could be prepared.")
        st.stop()
    st.success(f"Prepared {len(rows):,} AABC transitions across {rows['StreamKey'].nunique():,} streams, {rows['core_id'].nunique():,} cores, {rows['member'].nunique():,} members.")

    cfg_core = ProfileCfg("core", max_profiles, max_depth, min_hits, min_precision, 1, top_trait_pool)
    cfg_member = ProfileCfg("member", max_profiles, max_depth, min_hits, min_precision, 1, top_trait_pool)
    summary = pd.DataFrame([
        {"metric":"app_version", "value":APP_VERSION},
        {"metric":"chunk_start_target_num", "value":target_start},
        {"metric":"core_chunk_size", "value":core_chunk_size},
        {"metric":"member_chunk_size", "value":member_chunk_size},
        {"metric":"build_marker", "value":BUILD_MARKER},
        {"metric":"stage", "value":stage},
        {"metric":"aabc_transitions", "value":len(rows)},
        {"metric":"streams", "value":rows['StreamKey'].nunique()},
        {"metric":"cores", "value":rows['core_id'].nunique()},
        {"metric":"members", "value":rows['member'].nunique()},
        {"metric":"note", "value":"Profile decomposition is discovery-only; not production play logic."},
    ])

    frames = {"summary.csv": summary}
    texts = {"README_v24.txt": f"{BUILD_MARKER}\n{DEPLOY_FILENAME_NOTE}\nStage run: {stage}\nDiscovery lab only. v24 fixes profile discovery to be coverage-first and exports prefilter diagnostics.\n"}

    if stage.startswith("1"):
        core_counts = rows.groupby("core_id").size().reset_index(name="hits").sort_values("hits", ascending=False)
        member_counts = rows.groupby(["core_id","member"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
        stream_core = rows.groupby(["StreamKey","core_id"]).size().reset_index(name="hits").sort_values("hits", ascending=False)
        frames.update({
            "core_base_counts.csv": core_counts,
            "member_base_counts.csv": member_counts,
            "stream_core_base_counts.csv": stream_core.head(export_top_n),
            "prepared_aabc_rows_sample.csv": rows.drop(columns=["trait_list"]).head(export_top_n),
        })
        st.dataframe(core_counts, use_container_width=True, hide_index=True)

    elif stage.startswith("2"):
        progress = st.progress(0.0)
        status = st.empty()
        with st.spinner("Building core coverage-first profiles for selected chunk..."):
            prof, residual = build_profiles(rows, "core", cfg_core, target_start=int(target_start), target_chunk_size=int(core_chunk_size), progress=progress, status=status)
        first_target = sorted(rows["core_id"].dropna().unique())[int(target_start)-1] if len(sorted(rows["core_id"].dropna().unique())) >= int(target_start) else ""
        debug = debug_candidate_profiles(rows, "core", first_target, cfg_core, max_rows=min(500, export_top_n))
        frames.update({"core_decomposed_profiles.csv": prof.head(export_top_n), "core_profile_residuals.csv": residual, "core_prefilter_candidate_debug.csv": debug})
        st.subheader("Core prefilter candidate debug for first chunk target")
        st.dataframe(debug.head(200), use_container_width=True, hide_index=True)
        st.subheader("Core decomposed profiles")
        st.dataframe(prof.head(export_top_n), use_container_width=True, hide_index=True)
        st.subheader("Core residuals")
        st.dataframe(residual, use_container_width=True, hide_index=True)

    elif stage.startswith("3"):
        progress = st.progress(0.0)
        status = st.empty()
        with st.spinner("Building member coverage-first profiles for selected chunk..."):
            prof, residual = build_profiles(rows, "member", cfg_member, target_start=int(target_start), target_chunk_size=int(member_chunk_size), progress=progress, status=status)
        member_targets = sorted(rows["member"].dropna().unique())
        first_target = member_targets[int(target_start)-1] if len(member_targets) >= int(target_start) else ""
        debug = debug_candidate_profiles(rows, "member", first_target, cfg_member, max_rows=min(500, export_top_n))
        frames.update({"member_decomposed_profiles.csv": prof.head(export_top_n), "member_profile_residuals.csv": residual, "member_prefilter_candidate_debug.csv": debug})
        st.subheader("Member prefilter candidate debug for first chunk target")
        st.dataframe(debug.head(200), use_container_width=True, hide_index=True)
        st.subheader("Member decomposed profiles")
        st.dataframe(prof.head(export_top_n), use_container_width=True, hide_index=True)
        st.subheader("Member residuals")
        st.dataframe(residual.head(export_top_n), use_container_width=True, hide_index=True)

    elif stage.startswith("4"):
        progress = st.progress(0.0)
        status = st.empty()
        with st.spinner("Building core profiles for selected chunk, scanning collisions, refining separators..."):
            prof, residual = build_profiles(rows, "core", cfg_core, target_start=int(target_start), target_chunk_size=int(core_chunk_size), progress=progress, status=status)
            status.info("Scanning core profile collisions...")
            collisions = collision_scan(prof)
            status.info("Refining separators for core profile collisions...")
            sep = separator_refinement(rows, prof, "core", max_collisions=200, min_hits=max(3, min_hits//2))
            status.success("Core collision scan and separator refinement complete.")
        frames.update({
            "core_decomposed_profiles.csv": prof.head(export_top_n),
            "core_profile_residuals.csv": residual,
            "core_prefilter_candidate_debug.csv": debug_candidate_profiles(rows, "core", sorted(rows["core_id"].dropna().unique())[int(target_start)-1] if len(sorted(rows["core_id"].dropna().unique())) >= int(target_start) else "", cfg_core, max_rows=min(500, export_top_n)),
            "core_profile_collisions.csv": collisions.head(export_top_n),
            "core_collision_separator_refinements.csv": sep.head(export_top_n),
        })
        st.subheader("Core profile collisions")
        st.dataframe(collisions.head(export_top_n), use_container_width=True, hide_index=True)
        st.subheader("Core collision separator refinements")
        st.dataframe(sep.head(export_top_n), use_container_width=True, hide_index=True)

    elif stage.startswith("5"):
        progress = st.progress(0.0)
        status = st.empty()
        with st.spinner("Building member profiles for selected chunk, scanning collisions, refining separators..."):
            prof, residual = build_profiles(rows, "member", cfg_member, target_start=int(target_start), target_chunk_size=int(member_chunk_size), progress=progress, status=status)
            status.info("Scanning member profile collisions...")
            collisions = collision_scan(prof)
            status.info("Refining separators for member profile collisions...")
            sep = separator_refinement(rows, prof, "member", max_collisions=300, min_hits=max(3, min_hits//2))
            status.success("Member collision scan and separator refinement complete.")
        frames.update({
            "member_decomposed_profiles.csv": prof.head(export_top_n),
            "member_profile_residuals.csv": residual.head(export_top_n),
            "member_prefilter_candidate_debug.csv": debug_candidate_profiles(rows, "member", sorted(rows["member"].dropna().unique())[int(target_start)-1] if len(sorted(rows["member"].dropna().unique())) >= int(target_start) else "", cfg_member, max_rows=min(500, export_top_n)),
            "member_profile_collisions.csv": collisions.head(export_top_n),
            "member_collision_separator_refinements.csv": sep.head(export_top_n),
        })
        st.subheader("Member profile collisions")
        st.dataframe(collisions.head(export_top_n), use_container_width=True, hide_index=True)
        st.subheader("Member collision separator refinements")
        st.dataframe(sep.head(export_top_n), use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    package = zip_frames(frames, texts)
    st.download_button(
        "Download selected v24 stage outputs ZIP",
        package,
        file_name=f"core_affinity_lab_v24_stage_{stage.split()[0]}_chunk_{int(target_start)}.zip",
        mime="application/zip",
    )

except Exception as e:
    st.error("v24 stage failed. Full traceback below.")
    st.exception(e)
