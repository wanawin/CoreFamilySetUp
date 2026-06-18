#!/usr/bin/env python3
"""
Core Affinity Lab v33 — Discriminative Collision Separator Miner

Purpose:
- Consume ALL v30 core coverage profile ZIPs.
- Merge and deduplicate global core profile buckets.
- Mine separator traits inside each broad coverage bucket:
    target core rows inside bucket vs sibling/non-target rows inside the same bucket.
- Lab only. No daily-play, budget, RTE, B1Z0, ZLT, or production playlist logic.

Deployment note:
- For Streamlit Cloud, this file may be renamed to: core_affinity_lab_v1 (1).py
"""
from __future__ import annotations

import gc
import io
import itertools
import re
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set

import pandas as pd
import streamlit as st

APP_VERSION = "v33"
BUILD_MARKER = "BUILD: core_affinity_lab_v33_DISCRIMINATIVE_COLLISION_MINER__2026-06-18"
DEPLOY_FILENAME_NOTE = "For Streamlit Cloud deployment, this file may be renamed to: core_affinity_lab_v1 (1).py"

st.set_page_config(page_title="Core Affinity Lab v33", layout="wide")
st.title("Core Affinity Lab v33 — Discriminative Collision Separator Miner")
st.caption(BUILD_MARKER)
st.info(
    "Upload clean history plus all v30 Stage-2 core-profile ZIP chunks. "
    "v33 merges v30 coverage buckets globally, deduplicates overlaps, then mines DISCRIMINATIVE separators: target rows vs sibling rows inside each broad bucket. "
    "Lab only — this does not touch daily playlist logic."
)

# -------------------------
# Basic IO / parsing
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
# Universal seed traits — keep aligned with v30
# -------------------------
MIRRORS = [(0,5), (1,6), (2,7), (3,8), (4,9)]

def seed_traits(seed: str) -> List[str]:
    s = norm4(seed)
    if len(s) != 4:
        return []
    ds = [int(c) for c in s]
    traits: List[str] = []
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
    hist["SeedResult"] = hist.groupby("StreamKey")["Result4"].shift(1)
    hist = hist.dropna(subset=["SeedResult"]).copy()
    labels = hist["Result4"].map(classify_aabc)
    hist["core_id"] = labels.map(lambda x: x[0])
    hist["member"] = labels.map(lambda x: x[1])
    hist["is_aabc"] = labels.map(lambda x: x[2])
    aabc = hist[hist["is_aabc"]].copy().reset_index(drop=True)
    aabc["trait_list"] = aabc["SeedResult"].map(seed_traits)
    return aabc

# -------------------------
# v30 profile ZIP handling
# -------------------------
def read_csv_from_zip(upload, desired_name: str) -> pd.DataFrame:
    raw = _bytes(upload)
    if not raw:
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as z:
            names = z.namelist()
            hit = None
            for n in names:
                if n.endswith(desired_name):
                    hit = n
                    break
            if hit is None:
                return pd.DataFrame()
            with z.open(hit) as f:
                return pd.read_csv(f, dtype=str)
    except Exception:
        return pd.DataFrame()


def merge_v30_core_profiles(zips) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profiles = []
    residuals = []
    manifest = []
    for up in zips or []:
        name = getattr(up, "name", "uploaded.zip")
        p = read_csv_from_zip(up, "core_decomposed_profiles.csv")
        r = read_csv_from_zip(up, "core_profile_residuals.csv")
        profiles.append(p)
        residuals.append(r)
        manifest.append({
            "source_zip": name,
            "profile_rows": len(p),
            "residual_rows": len(r),
            "has_profiles": not p.empty,
            "has_residuals": not r.empty,
        })
    prof = pd.concat([p for p in profiles if not p.empty], ignore_index=True) if any(not p.empty for p in profiles) else pd.DataFrame()
    res = pd.concat([r for r in residuals if not r.empty], ignore_index=True) if any(not r.empty for r in residuals) else pd.DataFrame()
    man = safe_display_df(pd.DataFrame(manifest))
    if not prof.empty:
        # normalize numeric where present
        for c in ["profile_num", "total_target_hits", "new_target_hits", "target_hits", "sibling_hits", "sample", "remaining_before", "remaining_after"]:
            if c in prof.columns:
                prof[c] = pd.to_numeric(prof[c], errors="coerce")
        if "precision" in prof.columns:
            prof["precision"] = pd.to_numeric(prof["precision"], errors="coerce")
        if "target_coverage" in prof.columns:
            prof["target_coverage"] = pd.to_numeric(prof["target_coverage"], errors="coerce")
        # dedupe overlap: prefer higher new_target_hits / precision when duplicate target/profile_num appears
        sort_cols = [c for c in ["target", "profile_num", "new_target_hits", "precision"] if c in prof.columns]
        if "new_target_hits" in prof.columns:
            prof = prof.sort_values(["target", "profile_num", "new_target_hits", "precision"], ascending=[True, True, False, False])
        subset = [c for c in ["level", "target", "profile_num"] if c in prof.columns]
        if subset:
            prof = prof.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        else:
            prof = prof.drop_duplicates().reset_index(drop=True)
    if not res.empty:
        for c in ["total_target_hits", "covered_hits", "uncovered_hits", "profiles_found"]:
            if c in res.columns:
                res[c] = pd.to_numeric(res[c], errors="coerce")
        if "covered_rate" in res.columns:
            res["covered_rate"] = pd.to_numeric(res["covered_rate"], errors="coerce")
        subset = [c for c in ["level", "target"] if c in res.columns]
        if subset:
            res = res.sort_values(["target", "covered_rate"], ascending=[True, False]).drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    return prof, res, man

# -------------------------
# Separator mining
# -------------------------
def build_trait_index(rows: pd.DataFrame) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = defaultdict(set)
    for i, traits in enumerate(rows["trait_list"].tolist()):
        for t in traits:
            idx[t].add(i)
    return idx


def stack_to_tuple(stack: str) -> Tuple[str, ...]:
    parts = [x.strip() for x in str(stack).split("&&") if x.strip()]
    return tuple(parts)


def match_stack(trait_index: Dict[str, Set[int]], stack: Tuple[str, ...], all_idxs: Set[int]) -> Set[int]:
    if not stack:
        return set()
    out = None
    for t in stack:
        s = trait_index.get(t, set())
        out = set(s) if out is None else out & s
        if not out:
            return set()
    return out if out is not None else set()


def count_traits(rows: pd.DataFrame, idxs: Iterable[int]) -> Counter:
    c = Counter()
    tl = rows["trait_list"].tolist()
    for i in idxs:
        c.update(tl[i])
    return c


def mine_profile_separators(
    rows: pd.DataFrame,
    profiles: pd.DataFrame,
    profile_start: int,
    profile_chunk_size: int,
    min_target_hits: int,
    min_precision: float,
    min_profile_coverage: float,
    candidate_pool: int,
    top_per_profile: int,
    refine_depth: int,
    progress=None,
    status=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """v33 discriminative collision separator miner.

    v33 mostly selected traits that were common inside the broad coverage bucket,
    even when they were equally common in sibling rows. v33 explicitly compares:

        target rows inside the bucket  VS  sibling/non-target rows inside the bucket

    and ranks traits by precision gain, target-vs-sibling lift, odds ratio,
    and coverage. This is still discovery only, not production play logic.
    """
    if rows.empty or profiles.empty:
        return pd.DataFrame(), pd.DataFrame()
    t0 = time.time()
    eps = 1e-9
    all_idxs = set(range(len(rows)))
    trait_index = build_trait_index(rows)
    core_values = rows["core_id"].tolist()
    core_index: Dict[str, Set[int]] = defaultdict(set)
    for i, c in enumerate(core_values):
        core_index[str(c)].add(i)

    profiles = profiles.copy().reset_index(drop=True)
    profiles["profile_row_num"] = profiles.index + 1
    selected = profiles.iloc[profile_start-1:profile_start-1+profile_chunk_size].copy()
    sep_rows = []
    audit_rows = []
    total = max(1, len(selected))

    def is_redundant_with_base(trait: str, base_stack: Tuple[str, ...]) -> bool:
        """Avoid fake separators that simply restate the base stack.

        Example: base has has_pair_05=0 and candidate no_pair_05=1.
        Those are equivalent and produce no discrimination.
        """
        base = set(base_stack)
        if trait in base:
            return True
        m = re.match(r"no_pair_(\d\d)=1", trait)
        if m and f"has_pair_{m.group(1)}=0" in base:
            return True
        m = re.match(r"has_pair_(\d\d)=0", trait)
        if m and f"no_pair_{m.group(1)}=1" in base:
            return True
        m = re.match(r"no(\d)=1", trait)
        if m and f"has{m.group(1)}=0" in base:
            return True
        m = re.match(r"has(\d)=0", trait)
        if m and f"no{m.group(1)}=1" in base:
            return True
        return False

    def score_candidate(target_set: Set[int], sibling_set: Set[int], matched: Set[int], total_target_hits: int, base_precision: float, profile_target_total: int) -> dict:
        target_hits = len(matched & target_set)
        sibling_hits = len(matched & sibling_set)
        sample = target_hits + sibling_hits
        if sample <= 0:
            return {}
        target_miss = max(0, len(target_set) - target_hits)
        sibling_miss = max(0, len(sibling_set) - sibling_hits)
        precision = target_hits / sample if sample else 0.0
        profile_coverage = target_hits / profile_target_total if profile_target_total else 0.0
        global_coverage = target_hits / total_target_hits if total_target_hits else 0.0
        target_rate = target_hits / len(target_set) if target_set else 0.0
        sibling_rate = sibling_hits / len(sibling_set) if sibling_set else 0.0
        lift = (target_rate + eps) / (sibling_rate + eps)
        odds_ratio = ((target_hits + 0.5) / (target_miss + 0.5)) / ((sibling_hits + 0.5) / (sibling_miss + 0.5))
        precision_gain = precision - base_precision
        hit_margin = target_hits - sibling_hits
        # Discriminative score: precision gain and lift matter most, but coverage keeps us from tiny tricks.
        score = (
            max(0.0, precision_gain) * 500.0
            + min(lift, 50.0) * 8.0
            + min(odds_ratio, 100.0) * 2.0
            + profile_coverage * 80.0
            + global_coverage * 30.0
            + target_hits * 0.75
            + max(0, hit_margin) * 0.10
        )
        return {
            "sample": sample,
            "target_hits": target_hits,
            "sibling_hits": sibling_hits,
            "precision": precision,
            "precision_gain": precision_gain,
            "profile_coverage": profile_coverage,
            "global_target_coverage": global_coverage,
            "target_rate_in_bucket": target_rate,
            "sibling_rate_in_bucket": sibling_rate,
            "target_vs_sibling_lift": lift,
            "odds_ratio": odds_ratio,
            "hit_margin": hit_margin,
            "score": score,
        }

    for n, (_, pr) in enumerate(selected.iterrows(), start=1):
        target = str(pr.get("target", ""))
        base_stack = stack_to_tuple(str(pr.get("trait_stack", "")))
        if progress is not None:
            progress.progress(min(1.0, (n-1)/total))
        if status is not None:
            status.info(f"Mining DISCRIMINATIVE separators: profile {n}/{total} — core {target} profile {pr.get('profile_num','')}")

        target_idxs = set(core_index.get(target, set()))
        bucket = match_stack(trait_index, base_stack, all_idxs)
        bucket_target = bucket & target_idxs
        bucket_sibling = bucket - target_idxs
        total_target_hits = len(target_idxs)
        base_precision = len(bucket_target) / len(bucket) if bucket else 0.0
        base_target_coverage = len(bucket_target) / total_target_hits if total_target_hits else 0.0

        audit_rows.append({
            "profile_row_num": int(pr.get("profile_row_num", n)),
            "target": target,
            "profile_id": pr.get("profile_id", ""),
            "profile_num": pr.get("profile_num", ""),
            "base_stack": " && ".join(base_stack),
            "base_sample": len(bucket),
            "base_target_hits": len(bucket_target),
            "base_sibling_hits": len(bucket_sibling),
            "base_precision": round(base_precision, 6),
            "base_target_coverage": round(base_target_coverage, 6),
            "mined": bool(len(bucket_target) >= min_target_hits and len(bucket) > 0),
            "note": "v33 compares target frequency against sibling frequency inside this bucket.",
        })
        if len(bucket_target) < min_target_hits or not bucket:
            continue

        # Candidate traits come from target rows, but are scored against sibling rows.
        candidate_counts = count_traits(rows, bucket_target)
        primary_candidates = []
        examined = 0
        for trait, _cnt in candidate_counts.most_common(candidate_pool):
            if is_redundant_with_base(trait, base_stack):
                continue
            examined += 1
            matched = bucket & trait_index.get(trait, set())
            stats = score_candidate(bucket_target, bucket_sibling, matched, total_target_hits, base_precision, len(bucket_target))
            if not stats:
                continue
            if stats["target_hits"] < min_target_hits:
                continue
            # Must actually discriminate. Either precision improves, or target rate materially exceeds sibling rate.
            if stats["precision_gain"] <= 0 and stats["target_vs_sibling_lift"] <= 1.05:
                continue
            if stats["precision"] < min_precision and stats["profile_coverage"] < min_profile_coverage:
                continue
            row = {
                "profile_row_num": int(pr.get("profile_row_num", n)),
                "target": target,
                "profile_id": pr.get("profile_id", ""),
                "profile_num": pr.get("profile_num", ""),
                "base_stack": " && ".join(base_stack),
                "separator_stack": trait,
                "refined_stack": " && ".join(base_stack + (trait,)),
                "separator_depth": 1,
                "base_sample": len(bucket),
                "base_target_hits": len(bucket_target),
                "base_sibling_hits": len(bucket_sibling),
                "base_precision": round(base_precision, 6),
                "base_target_coverage": round(base_target_coverage, 6),
                "sample": stats["sample"],
                "target_hits": stats["target_hits"],
                "sibling_hits": stats["sibling_hits"],
                "precision": round(stats["precision"], 6),
                "precision_gain": round(stats["precision_gain"], 6),
                "profile_coverage": round(stats["profile_coverage"], 6),
                "global_target_coverage": round(stats["global_target_coverage"], 6),
                "target_rate_in_bucket": round(stats["target_rate_in_bucket"], 6),
                "sibling_rate_in_bucket": round(stats["sibling_rate_in_bucket"], 6),
                "target_vs_sibling_lift": round(stats["target_vs_sibling_lift"], 6),
                "odds_ratio": round(stats["odds_ratio"], 6),
                "hit_margin": stats["hit_margin"],
                "score": round(stats["score"], 6),
                "candidate_rank_basis": "precision_gain + target_vs_sibling_lift + odds_ratio + coverage",
                "needs_validation": True,
                "production_status": "DISCOVERY_SEPARATOR_NOT_PRODUCTION_RULE",
            }
            sep_rows.append(row)
            primary_candidates.append((stats["score"], trait, matched, row))

        if refine_depth >= 2 and primary_candidates:
            primary_candidates = sorted(primary_candidates, key=lambda x: x[0], reverse=True)[:min(20, top_per_profile)]
            for _score, trait1, match1, _row1 in primary_candidates:
                target_after_1 = match1 & bucket_target
                sibling_after_1 = match1 & bucket_sibling
                if len(target_after_1) < min_target_hits:
                    continue
                counts2 = count_traits(rows, target_after_1)
                base_precision_1 = len(target_after_1) / (len(target_after_1) + len(sibling_after_1)) if (len(target_after_1) + len(sibling_after_1)) else base_precision
                for trait2, _c2 in counts2.most_common(min(candidate_pool, 80)):
                    if trait2 == trait1 or is_redundant_with_base(trait2, base_stack):
                        continue
                    matched2 = match1 & trait_index.get(trait2, set())
                    stats = score_candidate(bucket_target, bucket_sibling, matched2, total_target_hits, base_precision, len(bucket_target))
                    if not stats:
                        continue
                    if stats["target_hits"] < min_target_hits:
                        continue
                    if stats["precision"] <= base_precision_1 and stats["target_vs_sibling_lift"] <= 1.05:
                        continue
                    if stats["precision"] < min_precision and stats["profile_coverage"] < min_profile_coverage:
                        continue
                    sep_rows.append({
                        "profile_row_num": int(pr.get("profile_row_num", n)),
                        "target": target,
                        "profile_id": pr.get("profile_id", ""),
                        "profile_num": pr.get("profile_num", ""),
                        "base_stack": " && ".join(base_stack),
                        "separator_stack": f"{trait1} && {trait2}",
                        "refined_stack": " && ".join(base_stack + tuple(sorted((trait1, trait2)))),
                        "separator_depth": 2,
                        "base_sample": len(bucket),
                        "base_target_hits": len(bucket_target),
                        "base_sibling_hits": len(bucket_sibling),
                        "base_precision": round(base_precision, 6),
                        "base_target_coverage": round(base_target_coverage, 6),
                        "sample": stats["sample"],
                        "target_hits": stats["target_hits"],
                        "sibling_hits": stats["sibling_hits"],
                        "precision": round(stats["precision"], 6),
                        "precision_gain": round(stats["precision_gain"], 6),
                        "profile_coverage": round(stats["profile_coverage"], 6),
                        "global_target_coverage": round(stats["global_target_coverage"], 6),
                        "target_rate_in_bucket": round(stats["target_rate_in_bucket"], 6),
                        "sibling_rate_in_bucket": round(stats["sibling_rate_in_bucket"], 6),
                        "target_vs_sibling_lift": round(stats["target_vs_sibling_lift"], 6),
                        "odds_ratio": round(stats["odds_ratio"], 6),
                        "hit_margin": stats["hit_margin"],
                        "score": round(stats["score"], 6),
                        "candidate_rank_basis": "precision_gain + target_vs_sibling_lift + odds_ratio + coverage",
                        "needs_validation": True,
                        "production_status": "DISCOVERY_SEPARATOR_NOT_PRODUCTION_RULE",
                    })

    if progress is not None:
        progress.progress(1.0)
    if status is not None:
        status.success(f"Finished discriminative separator mining for {len(selected)} profiles in {time.time()-t0:.2f} seconds.")
    seps = pd.DataFrame(sep_rows)
    if not seps.empty:
        seps = seps.sort_values(["score", "precision_gain", "target_vs_sibling_lift", "precision", "profile_coverage", "target_hits"], ascending=[False, False, False, False, False, False]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return seps, audit

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
# v33 Arrow / Streamlit display safety
# -------------------------
def safe_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return an Arrow-safe copy for st.table/st.dataframe.

    Streamlit converts displayed dataframes through PyArrow. Mixed object columns
    such as a manifest value column containing ints, floats, bools, and strings
    can crash with: Expected bytes, got int. Display copies are coerced to str.
    CSV exports keep their original values.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].fillna("").map(lambda x: "" if pd.isna(x) else str(x))
    return out

def manifest_df(metrics: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{"metric": str(k), "value": "" if v is None else str(v)} for k, v in metrics.items()])

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("v33 settings")
    stage = st.selectbox("Stage", [
        "1 - Merge/audit v30 core profile ZIPs",
        "2 - Global core profile separator mining",
    ])
    st.divider()
    profile_start = st.number_input("Profile start row #", min_value=1, max_value=10000, value=1, step=1)
    profile_chunk_size = st.slider("Profile chunk size", 5, 500, 80, 5)
    min_target_hits = st.slider("Min target hits for separator", 3, 50, 8)
    min_precision = st.slider("Min separator precision", 0.01, 1.00, 0.10, 0.01)
    min_profile_coverage = st.slider("Min profile coverage if precision is low", 0.01, 0.75, 0.05, 0.01)
    candidate_pool = st.slider("Separator candidate trait pool", 25, 400, 160, 5)
    refine_depth = st.slider("Separator refinement depth", 1, 2, 2)
    export_top_n = st.slider("CSV export top N rows", 100, 10000, 2000, 100)
    st.caption("v33 is manifest-only by default. It ranks separator candidates by true target-vs-sibling discrimination; outputs are discovery-only.")

history_upload = st.file_uploader("Upload clean Pick-4 history CSV/TXT", type=["csv", "txt", "tsv"], key="history")
zip_uploads = st.file_uploader("Upload ALL v30 Stage-2 core-profile ZIP chunks", type=["zip"], accept_multiple_files=True, key="v30zips")

if "v33_package" not in st.session_state:
    st.session_state["v33_package"] = None
    st.session_state["v33_manifest"] = None
    st.session_state["v33_filename"] = None

run = st.button("Run selected v33 stage and prepare frozen ZIP", type="primary")

if run:
    st.session_state["v33_package"] = None
    st.session_state["v33_manifest"] = None
    st.session_state["v33_filename"] = None
    t_run = time.time()
    try:
        if not zip_uploads:
            st.error("Upload at least one v30 Stage-2 ZIP.")
            st.stop()

        profiles, residuals, zip_manifest = merge_v30_core_profiles(zip_uploads)
        if profiles.empty:
            st.error("No core_decomposed_profiles.csv files found in uploaded v30 ZIPs.")
            st.stop()

        frames: Dict[str, pd.DataFrame] = {}
        texts: Dict[str, str] = {}
        metrics = {
            "app_version": APP_VERSION,
            "build_marker": BUILD_MARKER,
            "stage": stage,
            "v30_zip_count": len(zip_uploads),
            "merged_profile_rows": len(profiles),
            "merged_residual_rows": len(residuals),
            "unique_core_targets": profiles["target"].nunique() if "target" in profiles.columns else 0,
            "profile_start": int(profile_start),
            "profile_chunk_size": int(profile_chunk_size),
        }

        if stage.startswith("1"):
            target_summary = profiles.groupby("target").agg(
                profiles_found=("profile_id", "count"),
                mean_precision=("precision", "mean"),
                mean_target_coverage=("target_coverage", "mean"),
                total_new_target_hits=("new_target_hits", "sum"),
            ).reset_index() if not profiles.empty and "target" in profiles.columns else pd.DataFrame()
            frames["v33_zip_manifest.csv"] = zip_manifest
            frames["merged_core_decomposed_profiles.csv"] = profiles.head(export_top_n)
            frames["merged_core_profile_residuals.csv"] = residuals.head(export_top_n) if not residuals.empty else pd.DataFrame()
            frames["merged_core_target_summary.csv"] = target_summary
            metrics["stage_result"] = "merge_audit_only"

        else:
            if not history_upload:
                st.error("Stage 2 needs the clean history file too.")
                st.stop()
            raw_df = read_table(history_upload)
            csv_text = raw_df.to_csv(index=False)
            prep_t0 = time.time()
            with st.spinner("Preparing AABC seed→winner rows from history..."):
                rows = prepare_rows_from_csv(csv_text)
            prep_seconds = time.time() - prep_t0
            if rows.empty:
                st.error("No AABC rows prepared from history.")
                st.stop()
            progress = st.progress(0.0)
            status = st.empty()
            mining_t0 = time.time()
            seps, audit = mine_profile_separators(
                rows=rows,
                profiles=profiles,
                profile_start=int(profile_start),
                profile_chunk_size=int(profile_chunk_size),
                min_target_hits=int(min_target_hits),
                min_precision=float(min_precision),
                min_profile_coverage=float(min_profile_coverage),
                candidate_pool=int(candidate_pool),
                top_per_profile=25,
                refine_depth=int(refine_depth),
                progress=progress,
                status=status,
            )
            mining_seconds = time.time() - mining_t0
            frames["v33_zip_manifest.csv"] = zip_manifest
            frames["merged_core_decomposed_profiles.csv"] = profiles.head(export_top_n)
            frames["merged_core_profile_residuals.csv"] = residuals.head(export_top_n) if not residuals.empty else pd.DataFrame()
            frames["core_profile_separator_audit.csv"] = audit.head(export_top_n)
            frames["global_core_profile_separator_candidates.csv"] = seps.head(export_top_n)
            metrics.update({
                "stage_result": "global_core_separator_mining",
                "aabc_transitions": len(rows),
                "streams": rows["StreamKey"].nunique(),
                "core_targets_in_history": rows["core_id"].nunique(),
                "history_prepare_seconds": round(prep_seconds, 3),
                "separator_mining_seconds": round(mining_seconds, 3),
                "separator_rows_exported": len(seps),
                "profile_audit_rows": len(audit),
            })
            del rows, seps, audit
            gc.collect()

        metrics["total_run_seconds"] = round(time.time() - t_run, 3)
        manifest = manifest_df(metrics)
        frames["summary.csv"] = manifest
        texts["README_v33.txt"] = (
            f"{BUILD_MARKER}\n{DEPLOY_FILENAME_NOTE}\n"
            f"Stage run: {stage}\n"
            "This is a discovery lab. Separator candidates are NOT production play rules until walk-forward validated.\n"
            "v33 consumes v30 coverage profiles and mines discriminative separator traits using target-vs-sibling frequency, lift, odds ratio, and precision gain.\n"
        )
        package = zip_frames(frames, texts)
        filename = f"core_affinity_lab_v33_stage_{stage.split()[0]}_profiles_{int(profile_start)}.zip"
        st.session_state["v33_package"] = package
        st.session_state["v33_manifest"] = manifest
        st.session_state["v33_filename"] = filename
        # drop big frames before rendering UI
        del frames, profiles, residuals, zip_manifest, package
        gc.collect()
        st.success("v33 frozen ZIP prepared. Download below. No rerun/rebuild is needed.")

    except Exception as e:
        st.error("v33 stage failed. Full traceback below.")
        st.exception(e)

if st.session_state.get("v33_manifest") is not None:
    st.subheader("Frozen package manifest")
    st.table(safe_display_df(st.session_state["v33_manifest"]))

if st.session_state.get("v33_package") is not None:
    st.download_button(
        "Download frozen v33 stage outputs ZIP",
        st.session_state["v33_package"],
        file_name=st.session_state.get("v33_filename", "core_affinity_lab_v33_outputs.zip"),
        mime="application/zip",
        key="v33_download_button",
    )

st.caption("Tip: Upload all v30 Stage-2 chunks together. Use Stage 1 to audit merge/dedupe before Stage 2 separator mining.")
