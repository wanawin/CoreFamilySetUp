#!/usr/bin/env python3
"""
Core Affinity Lab v15 — UNIVERSAL DEEP TRAIT EXPANSION

Purpose:
- Lab only. No daily playlist, no cuts, no B1Z0, no RTE, no ZLT, no rescues.
- Mine universal seed traits for ALL 120 AABC cores and ALL 360 AABC members.
- Save both core-level and member-level trait profiles so member separation is not redone later.
- Add deep trait families inspired by prior 025/389 miners, but generalized to every digit/core/member.
- Run one bounded stage at a time so Streamlit Cloud does not crash.

Deploy note:
- Streamlit Cloud for this existing app expects the physical filename:
  core_affinity_lab_v1 (1).py
- The package includes that exact alias file.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v15"
BUILD_MARKER = "BUILD: core_affinity_lab_v15_UNIVERSAL_DEEP_TRAIT_EXPANSION__2026-06-17"
DIGITS = "0123456789"
MIRROR_PAIRS = [("0","5"),("1","6"),("2","7"),("3","8"),("4","9")]
ALL_CORES = ["".join(c) for c in combinations(DIGITS, 3)]
ALL_MEMBERS = []
for core in ALL_CORES:
    for d in core:
        ALL_MEMBERS.append("".join(sorted(core + d)))
ALL_MEMBERS = sorted(set(ALL_MEMBERS))
MEMBER_TO_CORE = {m: "".join(sorted(set(m))) for m in ALL_MEMBERS}

st.set_page_config(page_title="Core Affinity Lab v15", layout="wide")
st.title("Core Affinity Lab v15 — Universal Deep Trait Expansion")
st.caption(BUILD_MARKER)
st.info("v15 expands the trait universe for ALL 120 cores and ALL 360 members: digit inclusion/exclusion, counts, pairs, positions, mirrors, sums, spread, parity/high-low, and efficient all-core separator validation.")
st.warning("Lab only. No daily playlist, no cuts, no RTE, no B1Z0, no ZLT, no rescue logic.")

# -----------------------------
# Download/session freeze
# -----------------------------
def freeze_bytes(key: str, data: bytes):
    st.session_state[key] = data
    return data

def make_zip(frames: Dict[str, pd.DataFrame], texts: Dict[str, str] | None = None) -> bytes:
    texts = texts or {}
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name, df in frames.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(name if name.endswith(".csv") else name + ".csv", df.to_csv(index=False))
        for name, txt in texts.items():
            z.writestr(name if name.endswith(".txt") else name + ".txt", str(txt))
    return bio.getvalue()

# -----------------------------
# IO / parsing
# -----------------------------
def read_upload(file) -> pd.DataFrame:
    raw = file.getvalue()
    name = str(getattr(file, "name", "")).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    text = raw.decode("utf-8", errors="replace")
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    rows = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            rows.append({"Date": parts[0], "State": parts[1], "Game": parts[2], "Result": parts[3]})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()

def norm4(x) -> str:
    base = str(x).split(",", 1)[0]
    digs = re.findall(r"\d", base)
    return "".join(digs[:4]) if len(digs) >= 4 else ""

def classify_aabc(result4: str) -> Tuple[str, str, bool]:
    s = norm4(result4)
    if len(s) != 4:
        return "", "", False
    sorted_s = "".join(sorted(s))
    counts = sorted(Counter(sorted_s).values(), reverse=True)
    if counts != [2, 1, 1]:
        return "", "", False
    core = "".join(sorted(set(sorted_s)))
    member = sorted_s
    return core, member, True

def prepare_transitions(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame()
    df = hist.copy()
    lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = lower.get("date") or lower.get("drawdate") or lower.get("draw_date")
    result_col = lower.get("result4") or lower.get("result") or lower.get("winning number") or lower.get("number")
    state_col = lower.get("state")
    game_col = lower.get("game")
    stream_col = lower.get("streamkey") or lower.get("stream")
    if date_col is None or result_col is None:
        raise ValueError("History must contain Date and Result/Result4 columns, or raw Date-State-Game-Result lines.")
    df["DateParsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Result4"] = df[result_col].map(norm4)
    if stream_col is None:
        if state_col is None or game_col is None:
            raise ValueError("History must contain StreamKey or State + Game.")
        df["StreamKey"] = df[state_col].fillna("").astype(str) + " | " + df[game_col].fillna("").astype(str)
    else:
        df["StreamKey"] = df[stream_col].astype(str)
    df = df[df["DateParsed"].notna() & df["Result4"].str.len().eq(4) & df["StreamKey"].ne("")].copy()
    df = df.drop_duplicates(["DateParsed", "StreamKey", "Result4"]).sort_values(["StreamKey", "DateParsed"]).reset_index(drop=True)
    df["SeedResult"] = df.groupby("StreamKey")["Result4"].shift(1)
    df["SeedDate"] = df.groupby("StreamKey")["DateParsed"].shift(1)
    trans = df[df["SeedResult"].notna()].copy()
    cls = trans["Result4"].map(classify_aabc)
    trans["ActualCore"] = cls.map(lambda x: x[0])
    trans["ActualMember"] = cls.map(lambda x: x[1])
    trans["IsAABC"] = cls.map(lambda x: x[2])
    trans["DOW"] = trans["DateParsed"].dt.day_name()
    trans["Month"] = trans["DateParsed"].dt.month.astype(str).str.zfill(2)
    trans["SeedAgeDays"] = (trans["DateParsed"] - trans["SeedDate"]).dt.days
    return trans.reset_index(drop=True)

# -----------------------------
# Universal deep trait generator
# -----------------------------
def bucket_sum(v: int) -> str:
    return "00_09" if v <= 9 else "10_13" if v <= 13 else "14_17" if v <= 17 else "18_21" if v <= 21 else "22_plus"

def bucket_pair_sum(v: int) -> str:
    return "00_04" if v <= 4 else "05_08" if v <= 8 else "09_12" if v <= 12 else "13_16" if v <= 16 else "17_18"

def seed_traits(seed: str, stream: str = "", dow: str = "", month: str = "") -> List[str]:
    """Universal deep trait pack. Intentionally broad; filtering/scoring decides what survives."""
    s = norm4(seed)
    if len(s) != 4:
        return []
    ds = [int(c) for c in s]
    counts = Counter(s)
    present = set(s)
    traits: List[str] = []

    # 1) all single-digit inclusion/exclusion/counts
    for d in DIGITS:
        c = counts.get(d, 0)
        traits.append(f"has{d}={1 if c else 0}")
        traits.append(f"no{d}={1 if not c else 0}")
        traits.append(f"cnt{d}={c}")
        traits.append(f"cnt{d}_ge2={1 if c >= 2 else 0}")

    # 2) all unordered digit pairs inclusion/exclusion
    for a, b in combinations(DIGITS, 2):
        hp = int(a in present and b in present)
        traits.append(f"pair_{a}{b}={hp}")
        traits.append(f"nopair_{a}{b}={1-hp}")

    # 3) all adjacent ordered pair inclusion/exclusion and adjacent unordered pair presence
    adjacent = [s[0:2], s[1:3], s[2:4]]
    adjacent_set = set(adjacent)
    adjacent_sorted_set = {"".join(sorted(p)) for p in adjacent}
    for a in DIGITS:
        for b in DIGITS:
            ab = a + b
            hp = int(ab in adjacent_set)
            traits.append(f"adj_order_{ab}={hp}")
            if hp == 0:
                traits.append(f"no_adj_order_{ab}=1")
    for a, b in combinations(DIGITS, 2):
        ab = a + b
        hp = int(ab in adjacent_sorted_set)
        traits.append(f"adj_unordered_{ab}={hp}")

    # 4) positional exact digit and positional exclusion/highlow/evenodd
    for i, ch in enumerate(s, start=1):
        v = int(ch)
        traits.append(f"pos{i}={ch}")
        for d in DIGITS:
            traits.append(f"pos{i}_is{d}={1 if ch == d else 0}")
            if ch != d:
                traits.append(f"pos{i}_no{d}=1")
        traits.append(f"pos{i}_hl={'H' if v >= 5 else 'L'}")
        traits.append(f"pos{i}_eo={'E' if v % 2 == 0 else 'O'}")

    # 5) ordered position pairs and pair sums/spreads/sorted forms
    pair_defs = {
        "first2": s[:2], "mid2": s[1:3], "last2": s[2:], "first_last": s[0] + s[-1],
        "pos13": s[0] + s[2], "pos24": s[1] + s[3], "pos14": s[0] + s[3], "pos23": s[1] + s[2]
    }
    for name, val in pair_defs.items():
        nums = [int(x) for x in val]
        ps = sum(nums)
        pdiff = abs(nums[0] - nums[1])
        sorted_val = "".join(sorted(val))
        traits.append(f"{name}={val}")
        traits.append(f"{name}_sorted={sorted_val}")
        traits.append(f"{name}_sum={ps}")
        traits.append(f"{name}_sum_bucket={bucket_pair_sum(ps)}")
        traits.append(f"{name}_diff={pdiff}")
        traits.append(f"{name}_diff_bucket={'0_1' if pdiff<=1 else '2_4' if pdiff<=4 else '5_plus'}")
        traits.append(f"{name}_mirror={1 if sorted_val in {'05','16','27','38','49'} else 0}")

    # 6) first3/last3 triples and sums
    first3 = s[:3]
    last3 = s[1:]
    for name, val in {"first3": first3, "last3": last3}.items():
        nums = [int(x) for x in val]
        total3 = sum(nums)
        traits.append(f"{name}={val}")
        traits.append(f"{name}_sorted={''.join(sorted(val))}")
        traits.append(f"{name}_sum={total3}")
        traits.append(f"{name}_sum_bucket={bucket_sum(total3)}")
        traits.append(f"{name}_unique={len(set(val))}")

    # 7) mirrors and plus/minus links
    mirror_count = 0
    for a, b in MIRROR_PAIRS:
        hp = int(a in present and b in present)
        mirror_count += hp
        traits.append(f"mirror_{a}{b}={hp}")
    traits.append(f"mirror_count={mirror_count}")
    traits.append(f"has_any_mirror={int(mirror_count > 0)}")
    traits.append(f"mirror_ge2={int(mirror_count >= 2)}")
    plusminus1_pairs = sum(1 for a,b in combinations(ds, 2) if abs(a-b) == 1)
    traits.append(f"plusminus1_pairs={plusminus1_pairs}")
    traits.append(f"plusminus1_bucket={'pm1_0' if plusminus1_pairs==0 else 'pm1_1' if plusminus1_pairs==1 else 'pm1_2plus'}")

    # 8) sums/spread/root
    total = sum(ds)
    spread = max(ds) - min(ds)
    root = total
    while root >= 10:
        root = sum(int(c) for c in str(root))
    traits += [
        f"sum={total}", f"sum_last={total % 10}", f"root_sum={root}", f"sum_bucket={bucket_sum(total)}",
        f"spread={spread}", f"spread_bucket={'0_2' if spread<=2 else '3_4' if spread<=4 else '5_6' if spread<=6 else '7_plus'}",
    ]

    # 9) structure/repeats/patterns
    cnts = sorted(counts.values(), reverse=True)
    if cnts == [4]: structure = "AAAA"
    elif cnts == [3,1]: structure = "AAAB"
    elif cnts == [2,2]: structure = "AABB"
    elif cnts == [2,1,1]: structure = "AABC"
    else: structure = "ABCD"
    traits += [
        f"structure={structure}", f"unique_count={len(counts)}", f"max_repeat={max(cnts)}",
        "hl_pattern=" + "".join("H" if x >= 5 else "L" for x in ds),
        "eo_pattern=" + "".join("E" if x % 2 == 0 else "O" for x in ds),
        f"high_count={sum(x>=5 for x in ds)}", f"low_count={sum(x<5 for x in ds)}",
        f"even_count={sum(x%2==0 for x in ds)}", f"odd_count={sum(x%2==1 for x in ds)}",
        f"consec_links={sum(1 for a,b in zip(ds,ds[1:]) if abs(a-b)==1)}",
        f"repeat_pos12={int(s[0]==s[1])}", f"repeat_pos23={int(s[1]==s[2])}", f"repeat_pos34={int(s[2]==s[3])}",
        f"repeat_any_adjacent={int(s[0]==s[1] or s[1]==s[2] or s[2]==s[3])}",
    ]

    # 10) stream/cadence context, used only in stream+trait stages
    if dow: traits.append(f"dow={dow}")
    if month: traits.append(f"month={month}")
    if stream: traits.append("stream=" + stream)

    # Unique preserves memory and prevents duplicate counting.
    return list(dict.fromkeys(traits))

# -----------------------------
# Scoring helpers
# -----------------------------
def sample_tier(sample: int) -> str:
    return "STRONG_250_PLUS" if sample >= 250 else "STABLE_100_PLUS" if sample >= 100 else "CANDIDATE_50_99" if sample >= 50 else "EXPLORATORY_25_49" if sample >= 25 else "MICRO"

def metric_row(trait: str, target: str, sample: int, hits: int, baseline: float, min_lift: float, stable_den: int):
    hit_rate = hits / sample if sample else 0.0
    lift = hit_rate / baseline if baseline > 0 else 0.0
    if lift < min_lift:
        return None
    stability_weight = min(1.0, sample / max(1, stable_den))
    weighted_lift = lift * stability_weight
    confidence_score = weighted_lift * math.log1p(hits) * hit_rate
    return {
        "trait": trait, "target": target, "sample": sample, "hits": hits,
        "hit_rate": round(hit_rate, 6), "baseline_rate": round(baseline, 6),
        "relative_lift": round(lift, 4), "stability_weight": round(stability_weight, 4),
        "weighted_lift": round(weighted_lift, 4), "confidence_score": round(confidence_score, 6),
        "sample_tier": sample_tier(sample),
    }

def build_rows(trans: pd.DataFrame, include_stream_trait: bool) -> List[dict]:
    aabc = trans[trans["IsAABC"]].copy()
    rows = []
    prog = st.progress(0.0)
    n = len(aabc)
    for i, r in enumerate(aabc.itertuples(index=False), start=1):
        stream = getattr(r, "StreamKey") if include_stream_trait else ""
        traits = seed_traits(getattr(r, "SeedResult"), stream=stream, dow=getattr(r, "DOW", ""), month=getattr(r, "Month", ""))
        rows.append({"ActualCore": getattr(r, "ActualCore"), "ActualMember": getattr(r, "ActualMember"), "Traits": traits})
        if i % 1500 == 0 or i == n:
            prog.progress(i / max(1, n))
    prog.empty()
    return rows

def score_rows(rows: List[dict], target_col: str, min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    total = len(rows)
    target_totals = Counter(r[target_col] for r in rows if r.get(target_col))
    trait_samples = Counter()
    trait_target_hits = Counter()
    prog = st.progress(0.0)
    for i, r in enumerate(rows, start=1):
        target = r.get(target_col, "")
        for tr in r.get("Traits", []):
            trait_samples[tr] += 1
            if target:
                trait_target_hits[(tr, target)] += 1
        if i % 1500 == 0 or i == len(rows):
            prog.progress(i / max(1, len(rows)))
    prog.empty()
    out = []
    for (tr, target), hits in trait_target_hits.items():
        sample = trait_samples[tr]
        if sample < min_sample or hits < min_hits:
            continue
        baseline = target_totals[target] / total if total else 0
        row = metric_row(tr, target, sample, hits, baseline, min_lift, stable_den)
        if row:
            out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score", "weighted_lift", "hits", "sample"], ascending=False).head(top_n).reset_index(drop=True)

def simple_profiles(trans: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aabc = trans[trans["IsAABC"]].copy()
    core = aabc.groupby("ActualCore").size().reset_index(name="hits").sort_values("hits", ascending=False)
    core["share_of_aabc"] = (core["hits"] / max(1, len(aabc))).round(6)
    member = aabc.groupby(["ActualCore", "ActualMember"]).size().reset_index(name="hits").sort_values(["ActualCore", "hits"], ascending=[True, False])
    stream_core = aabc.groupby(["StreamKey", "ActualCore"]).size().reset_index(name="hits")
    stream_total = aabc.groupby("StreamKey").size().rename("stream_sample").reset_index()
    stream_core = stream_core.merge(stream_total, on="StreamKey", how="left")
    stream_core["stream_core_rate"] = (stream_core["hits"] / stream_core["stream_sample"]).round(6)
    return core, member, stream_core.sort_values(["stream_core_rate", "hits"], ascending=False)

def parse_target_cores(text: str) -> List[str]:
    s = str(text).strip().upper()
    if not s or s in {"ALL", "*", "120", "ALL 120"}:
        return list(ALL_CORES)
    vals = []
    for tok in re.split(r"[^0-9]+", s):
        if len(tok) == 3 and len(set(tok)) == 3:
            core = "".join(sorted(tok))
            if core in ALL_CORES and core not in vals:
                vals.append(core)
    return vals or list(ALL_CORES)

def members_for_core(core: str) -> List[str]:
    core = "".join(sorted(str(core)))
    return sorted("".join(sorted(core + d)) for d in core)

# -----------------------------
# Efficient separator miners
# -----------------------------
def build_trait_target_counts(trans: pd.DataFrame, target_col: str, selected_targets: set[str] | None = None, include_stream_trait: bool = False):
    aabc = trans[trans["IsAABC"]].copy()
    if selected_targets is not None:
        aabc = aabc[aabc[target_col].isin(selected_targets)].copy()
    target_totals = Counter(aabc[target_col])
    trait_target = defaultdict(Counter)  # trait -> Counter(target -> hits)
    prog = st.progress(0.0)
    rows = list(aabc.itertuples(index=False))
    for i, r in enumerate(rows, start=1):
        target = getattr(r, target_col)
        stream = getattr(r, "StreamKey") if include_stream_trait else ""
        traits = seed_traits(getattr(r, "SeedResult"), stream=stream, dow=getattr(r, "DOW", ""), month=getattr(r, "Month", ""))
        for tr in traits:
            trait_target[tr][target] += 1
        if i % 1500 == 0 or i == len(rows):
            prog.progress(i / max(1, len(rows)))
    prog.empty()
    return target_totals, trait_target

def validate_all_core_pair_separators(trans: pd.DataFrame, target_cores: List[str], min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int, include_stream_trait: bool = False) -> pd.DataFrame:
    selected = set(target_cores)
    target_totals, trait_target = build_trait_target_counts(trans, "ActualCore", selected, include_stream_trait=include_stream_trait)
    pairs = list(combinations(sorted(selected), 2))
    out = []
    prog = st.progress(0.0)
    trait_items = list(trait_target.items())
    for idx, (tr, counts) in enumerate(trait_items, start=1):
        for a, b in pairs:
            total_pair = target_totals.get(a, 0) + target_totals.get(b, 0)
            if total_pair <= 0:
                continue
            ha = counts.get(a, 0)
            hb = counts.get(b, 0)
            sample = ha + hb
            if sample < min_sample:
                continue
            for favored, hits in [(a, ha), (b, hb)]:
                if hits < min_hits:
                    continue
                baseline = target_totals[favored] / total_pair if total_pair else 0
                row = metric_row(tr, favored, sample, hits, baseline, min_lift, stable_den)
                if row:
                    row.update({"separator_type": "core_vs_core", "pair": f"{a}_vs_{b}", "favored_core": favored})
                    out.append(row)
        if idx % 25 == 0 or idx == len(trait_items):
            prog.progress(idx / max(1, len(trait_items)))
    prog.empty()
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score", "weighted_lift", "hits", "sample"], ascending=False).head(top_n).reset_index(drop=True)

def validate_within_core_member_separators(trans: pd.DataFrame, target_cores: List[str], min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int, include_stream_trait: bool = False) -> pd.DataFrame:
    selected_members = set()
    for c in target_cores:
        selected_members.update(members_for_core(c))
    target_totals, trait_target = build_trait_target_counts(trans, "ActualMember", selected_members, include_stream_trait=include_stream_trait)
    out = []
    # Only compare within each core's 3 members. That is the daily member-selection problem.
    member_pairs = []
    for c in target_cores:
        for a, b in combinations(members_for_core(c), 2):
            member_pairs.append((c, a, b))
    prog = st.progress(0.0)
    trait_items = list(trait_target.items())
    for idx, (tr, counts) in enumerate(trait_items, start=1):
        for core, a, b in member_pairs:
            total_pair = target_totals.get(a, 0) + target_totals.get(b, 0)
            if total_pair <= 0:
                continue
            ha = counts.get(a, 0)
            hb = counts.get(b, 0)
            sample = ha + hb
            if sample < min_sample:
                continue
            for favored, hits in [(a, ha), (b, hb)]:
                if hits < min_hits:
                    continue
                baseline = target_totals[favored] / total_pair if total_pair else 0
                row = metric_row(tr, favored, sample, hits, baseline, min_lift, stable_den)
                if row:
                    row.update({"separator_type": "member_vs_member", "core": core, "pair": f"{a}_vs_{b}", "favored_member": favored})
                    out.append(row)
        if idx % 25 == 0 or idx == len(trait_items):
            prog.progress(idx / max(1, len(trait_items)))
    prog.empty()
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score", "weighted_lift", "hits", "sample"], ascending=False).head(top_n).reset_index(drop=True)

def trait_dictionary() -> pd.DataFrame:
    rows = [
        {"trait_family": "digit_presence", "examples": "has0=1, no9=1, cnt5=2, cnt8_ge2=1"},
        {"trait_family": "unordered_digit_pairs", "examples": "pair_38=1, nopair_05=1"},
        {"trait_family": "adjacent_ordered_pairs", "examples": "adj_order_38=1, no_adj_order_94=1"},
        {"trait_family": "adjacent_unordered_pairs", "examples": "adj_unordered_38=1"},
        {"trait_family": "positional_digits", "examples": "pos1=8, pos2_is3=1, pos4_no9=1"},
        {"trait_family": "position_pair_shapes", "examples": "first2=83, first2_sorted=38, first_last_sum_bucket=13_16"},
        {"trait_family": "mirrors", "examples": "mirror_38=1, mirror_count=2, mirror_ge2=1"},
        {"trait_family": "sum_spread_root", "examples": "sum_bucket=18_21, spread_bucket=7_plus, root_sum=4"},
        {"trait_family": "structure_patterns", "examples": "structure=AABC, hl_pattern=HHLH, eo_pattern=EOOE"},
        {"trait_family": "cadence_context", "examples": "dow=Tuesday, month=06"},
        {"trait_family": "stream_context", "examples": "stream=New York | Win 4 Midday, used only in stream+trait stages"},
    ]
    return pd.DataFrame(rows)

# -----------------------------
# UI controls
# -----------------------------
with st.sidebar:
    st.header("v15 Controls")
    st.caption("Run one validation stage at a time. Default target cores = ALL 120.")
    stage = st.selectbox("Mining stage", [
        "1 - Data audit + base profiles",
        "2 - Core single-trait lift",
        "3 - Member single-trait lift",
        "4 - Core-vs-core separator validation ALL cores",
        "5 - Member-vs-member separator validation ALL members",
        "6 - Core stream+trait lift",
        "7 - Member stream+trait lift",
        "8 - Stream-aware core-vs-core separators",
        "9 - Stream-aware member-vs-member separators",
    ])
    target_core_text = st.text_input("Target cores for separator stages", "ALL")
    min_sample = st.selectbox("Minimum sample to display", [10, 15, 25, 50, 100], index=2)
    min_hits = st.selectbox("Minimum hits for signal", [3, 5, 10, 15, 20], index=2)
    min_lift = st.slider("Minimum relative lift", 1.0, 10.0, 1.5, 0.1)
    stable_den = st.selectbox("Stable sample denominator", [50, 100, 250, 500], index=1)
    top_n = st.slider("Export/display top N", 100, 5000, 500, 100)

uploaded = st.file_uploader("Upload full clean Pick-4 history", type=["csv", "txt", "tsv", "xlsx", "xls"])
if not uploaded:
    st.stop()

if st.button("Run selected v15 stage", type="primary", use_container_width=True):
    try:
        with st.status("Loading and preparing transitions...", expanded=True):
            hist = read_upload(uploaded)
            st.write(f"Loaded rows: {len(hist):,}")
            trans = prepare_transitions(hist)
            st.write(f"Seed→next transitions: {len(trans):,}")
            st.write(f"AABC transitions: {int(trans['IsAABC'].sum()):,}")
            st.write(f"Streams: {trans['StreamKey'].nunique():,}")

        frames: Dict[str, pd.DataFrame] = {}
        summary = pd.DataFrame([
            {"metric": "app_version", "value": APP_VERSION},
            {"metric": "build_marker", "value": BUILD_MARKER},
            {"metric": "loaded_rows", "value": len(hist)},
            {"metric": "seed_next_transitions", "value": len(trans)},
            {"metric": "aabc_transitions", "value": int(trans['IsAABC'].sum())},
            {"metric": "streams", "value": trans['StreamKey'].nunique()},
            {"metric": "stage", "value": stage},
            {"metric": "target_core_text", "value": target_core_text},
            {"metric": "target_core_count", "value": len(parse_target_cores(target_core_text))},
            {"metric": "min_sample", "value": min_sample},
            {"metric": "min_hits", "value": min_hits},
            {"metric": "min_lift", "value": min_lift},
        ])
        frames["summary.csv"] = summary
        frames["trait_dictionary.csv"] = trait_dictionary()

        if stage.startswith("1"):
            st.subheader("Data audit + base profiles")
            core, member, stream_core = simple_profiles(trans)
            frames["core_profiles.csv"] = core
            frames["member_profiles.csv"] = member
            frames["stream_core_profiles.csv"] = stream_core.head(top_n)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.dataframe(core.head(120), use_container_width=True, hide_index=True)

        elif stage.startswith("2"):
            st.subheader("Core single-trait lift — universal deep trait pack")
            rows = build_rows(trans, include_stream_trait=False)
            df = score_rows(rows, "ActualCore", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["core_single_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("3"):
            st.subheader("Member single-trait lift — universal deep trait pack")
            rows = build_rows(trans, include_stream_trait=False)
            df = score_rows(rows, "ActualMember", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["member_single_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("4"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Core-vs-core separator validation — efficient ALL-core capable")
            st.caption(f"Selected cores: {len(target_cores)}")
            df = validate_all_core_pair_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n, include_stream_trait=False)
            frames["core_vs_core_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("5"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Member-vs-member separator validation — within-core all-member capable")
            st.caption(f"Selected cores: {len(target_cores)}; members compared within each core.")
            df = validate_within_core_member_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n, include_stream_trait=False)
            frames["member_vs_member_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("6"):
            st.subheader("Core stream+trait lift — universal deep trait pack")
            rows = build_rows(trans, include_stream_trait=True)
            df = score_rows(rows, "ActualCore", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["core_stream_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("7"):
            st.subheader("Member stream+trait lift — universal deep trait pack")
            rows = build_rows(trans, include_stream_trait=True)
            df = score_rows(rows, "ActualMember", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["member_stream_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("8"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Stream-aware core-vs-core separators")
            st.caption("Heavier than stage 4. Keep top N modest.")
            df = validate_all_core_pair_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n, include_stream_trait=True)
            frames["stream_aware_core_vs_core_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("9"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Stream-aware member-vs-member separators")
            st.caption("Heavier than stage 5. Keep top N modest.")
            df = validate_within_core_member_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n, include_stream_trait=True)
            frames["stream_aware_member_vs_member_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        zip_bytes = make_zip(frames, {"README_v15.txt": "Core Affinity Lab v15 universal-deep-trait output. Run one stage at a time; combine exported CSVs later. Lab only; no playlist logic."})
        freeze_bytes("v15_last_zip", zip_bytes)
        st.success("Stage complete. Download ZIP below.")
        st.download_button("Download v15 selected-stage outputs ZIP", data=zip_bytes, file_name="core_affinity_lab_v15_selected_stage_outputs.zip", mime="application/zip", use_container_width=True)

    except Exception as e:
        st.error("v15 caught the error instead of blank-crashing.")
        st.exception(e)
        st.stop()

if "v15_last_zip" in st.session_state:
    st.download_button("Download previous v15 ZIP again", data=st.session_state["v15_last_zip"], file_name="core_affinity_lab_v15_selected_stage_outputs.zip", mime="application/zip", use_container_width=True)
