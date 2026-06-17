#!/usr/bin/env python3
"""
Core Affinity Lab v14 — SIGNAL VALIDATION

Purpose:
- Lab only. No daily playlist, no cuts, no B1Z0, no RTE, no ZLT, no rescues.
- Mine universal seed traits for all 120 AABC cores and all 360 AABC members.
- Run one bounded stage at a time so Streamlit Cloud does not crash.
- Export only the selected stage outputs.

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
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v14"
BUILD_MARKER = "BUILD: core_affinity_lab_v13_SIGNAL_VALIDATION__2026-06-17"
DIGITS = "0123456789"
MIRROR_PAIRS = [("0","5"),("1","6"),("2","7"),("3","8"),("4","9")]
ALL_CORES = ["".join(c) for c in combinations(DIGITS, 3)]
ALL_MEMBERS = []
for core in ALL_CORES:
    for d in core:
        ALL_MEMBERS.append("".join(sorted(core + d)))
ALL_MEMBERS = sorted(set(ALL_MEMBERS))

st.set_page_config(page_title="Core Affinity Lab v14", layout="wide")
st.title("Core Affinity Lab v14 — Signal Validation + Core/Member Separators")
st.caption(BUILD_MARKER)
st.info("v14 clean build: use this single app file only. Dropdown stages 4 and 5 are the separator validation stages.")
st.info("Stage-safe lab only. Validates which traits separate cores and members. No daily playlist, no cuts, no RTE, no B1Z0, no ZLT, no rescue logic.")

# -----------------------------
# Download/session freeze
# -----------------------------
def freeze_bytes(key: str, data: bytes):
    st.session_state[key] = data
    return data

def df_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")

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
    # clean app-ready csv/tsv first
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    # raw tab history fallback: Date State Game Result
    rows = []
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            rows.append({"Date": parts[0], "State": parts[1], "Game": parts[2], "Result": parts[3]})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()

def norm4(x) -> str:
    # base draw only, ignore fireball/wild/sum after comma
    base = str(x).split(",", 1)[0]
    digs = re.findall(r"\d", base)
    return "".join(digs[:4]) if len(digs) >= 4 else ""

def classify_aabc(result4: str) -> Tuple[str, str, bool]:
    s = norm4(result4)
    if len(s) != 4:
        return "", "", False
    sorted_s = "".join(sorted(s))
    counts = sorted(Counter(sorted_s).values(), reverse=True)
    if counts != [2,1,1]:
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
    # seed -> next result within same stream
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
# Universal traits
# -----------------------------
def seed_traits(seed: str, stream: str = "", dow: str = "", month: str = "") -> List[str]:
    s = norm4(seed)
    if len(s) != 4:
        return []
    ds = [int(c) for c in s]
    traits: List[str] = []
    counts = Counter(s)
    sorted_s = "".join(sorted(s))

    # digit presence/exclusion/count
    for d in DIGITS:
        c = counts.get(d, 0)
        traits.append(f"has{d}={1 if c else 0}")
        traits.append(f"no{d}={1 if not c else 0}")
        traits.append(f"cnt{d}={c}")

    # unordered digit pair inclusion/exclusion
    present = set(s)
    for a,b in combinations(DIGITS, 2):
        has_pair = int(a in present and b in present)
        traits.append(f"pair_{a}{b}={has_pair}")
        traits.append(f"nopair_{a}{b}={1-has_pair}")

    # mirror pairs
    mirror_count = 0
    for a,b in MIRROR_PAIRS:
        hp = int(a in present and b in present)
        mirror_count += hp
        traits.append(f"mirror_{a}{b}={hp}")
    traits.append(f"mirror_count={mirror_count}")
    traits.append(f"has_any_mirror={int(mirror_count > 0)}")

    # positional digits + pos attributes
    for i,ch in enumerate(s, start=1):
        v = int(ch)
        traits.append(f"pos{i}={ch}")
        traits.append(f"pos{i}_hl={'H' if v >= 5 else 'L'}")
        traits.append(f"pos{i}_eo={'E' if v % 2 == 0 else 'O'}")

    # ordered pairs / adjacent / edge
    pair_defs = {
        "first2": s[:2], "mid2": s[1:3], "last2": s[2:], "first_last": s[0]+s[-1],
        "pos13": s[0]+s[2], "pos24": s[1]+s[3]
    }
    for name,val in pair_defs.items():
        traits.append(f"{name}={val}")
        traits.append(f"{name}_sum={sum(int(x) for x in val)}")
        traits.append(f"{name}_sorted={''.join(sorted(val))}")

    # sum/spread/root
    total = sum(ds)
    spread = max(ds) - min(ds)
    root = total
    while root >= 10:
        root = sum(int(c) for c in str(root))
    traits += [
        f"sum={total}", f"sum_last={total%10}", f"root_sum={root}",
        f"sum_bucket={'00_09' if total<=9 else '10_13' if total<=13 else '14_17' if total<=17 else '18_21' if total<=21 else '22_plus'}",
        f"spread={spread}", f"spread_bucket={'0_2' if spread<=2 else '3_4' if spread<=4 else '5_6' if spread<=6 else '7_plus'}",
    ]

    # structure/high-low/parity
    cnts = sorted(counts.values(), reverse=True)
    if cnts == [4]: structure = "AAAA"
    elif cnts == [3,1]: structure = "AAAB"
    elif cnts == [2,2]: structure = "AABB"
    elif cnts == [2,1,1]: structure = "AABC"
    else: structure = "ABCD"
    traits += [
        f"structure={structure}", f"unique_count={len(counts)}", f"max_repeat={max(cnts)}",
        "hl_pattern=" + "".join("H" if x>=5 else "L" for x in ds),
        "eo_pattern=" + "".join("E" if x%2==0 else "O" for x in ds),
        f"high_count={sum(x>=5 for x in ds)}", f"low_count={sum(x<5 for x in ds)}",
        f"even_count={sum(x%2==0 for x in ds)}", f"odd_count={sum(x%2==1 for x in ds)}",
        f"consec_links={sum(1 for a,b in zip(ds,ds[1:]) if abs(a-b)==1)}",
    ]

    # cadence/context as optional traits
    if dow: traits.append(f"dow={dow}")
    if month: traits.append(f"month={month}")
    if stream: traits.append("stream=" + stream)
    return traits

# -----------------------------
# Counter miners
# -----------------------------
def score_rows(rows: List[dict], target_col: str, min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    total = len(rows)
    target_totals = Counter(r[target_col] for r in rows if r.get(target_col))
    trait_samples = Counter()
    trait_target_hits = Counter()
    for r in rows:
        target = r.get(target_col, "")
        trset = r.get("Traits", [])
        for tr in trset:
            trait_samples[tr] += 1
            if target:
                trait_target_hits[(tr,target)] += 1
    out = []
    for (tr,target), hits in trait_target_hits.items():
        sample = trait_samples[tr]
        if sample < min_sample or hits < min_hits:
            continue
        baseline = target_totals[target] / total if total else 0
        hit_rate = hits / sample if sample else 0
        lift = hit_rate / baseline if baseline > 0 else 0
        if lift < min_lift:
            continue
        stability_weight = min(1.0, sample / max(1, stable_den))
        weighted_lift = lift * stability_weight
        confidence_score = weighted_lift * math.log1p(hits) * hit_rate
        out.append({
            "trait": tr, "target": target, "sample": sample, "hits": hits,
            "hit_rate": round(hit_rate, 6), "baseline_rate": round(baseline, 6),
            "relative_lift": round(lift, 4), "stability_weight": round(stability_weight, 4),
            "weighted_lift": round(weighted_lift, 4), "confidence_score": round(confidence_score, 6),
            "sample_tier": "STRONG_250_PLUS" if sample>=250 else "STABLE_100_PLUS" if sample>=100 else "CANDIDATE_50_99" if sample>=50 else "EXPLORATORY_25_49" if sample>=25 else "MICRO",
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score","weighted_lift","hits","sample"], ascending=False).head(top_n).reset_index(drop=True)

def build_rows(trans: pd.DataFrame, include_stream_trait: bool) -> List[dict]:
    aabc = trans[trans["IsAABC"]].copy()
    rows = []
    prog = st.progress(0.0)
    n = len(aabc)
    for i, r in enumerate(aabc.itertuples(index=False), start=1):
        stream = getattr(r, "StreamKey") if include_stream_trait else ""
        traits = seed_traits(getattr(r, "SeedResult"), stream=stream, dow=getattr(r,"DOW",""), month=getattr(r,"Month",""))
        rows.append({"ActualCore": getattr(r,"ActualCore"), "ActualMember": getattr(r,"ActualMember"), "Traits": traits})
        if i % 2500 == 0 or i == n:
            prog.progress(i / max(1,n))
    prog.empty()
    return rows

def simple_profiles(trans: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aabc = trans[trans["IsAABC"]].copy()
    core = aabc.groupby("ActualCore").size().reset_index(name="hits").sort_values("hits", ascending=False)
    core["share_of_aabc"] = (core["hits"] / max(1, len(aabc))).round(6)
    member = aabc.groupby(["ActualCore","ActualMember"]).size().reset_index(name="hits").sort_values(["ActualCore","hits"], ascending=[True,False])
    stream_core = aabc.groupby(["StreamKey","ActualCore"]).size().reset_index(name="hits")
    stream_total = aabc.groupby("StreamKey").size().rename("stream_sample").reset_index()
    stream_core = stream_core.merge(stream_total, on="StreamKey", how="left")
    stream_core["stream_core_rate"] = (stream_core["hits"] / stream_core["stream_sample"]).round(6)
    return core, member, stream_core.sort_values(["stream_core_rate","hits"], ascending=False)



def parse_target_cores(text: str) -> List[str]:
    vals = []
    for tok in re.split(r"[^0-9]+", str(text)):
        if len(tok) == 3 and len(set(tok)) == 3:
            core = "".join(sorted(tok))
            if core in ALL_CORES and core not in vals:
                vals.append(core)
    return vals or ["025","389","168","246","589","019","468","236"]

def members_for_core(core: str) -> List[str]:
    core = "".join(sorted(str(core)))
    return sorted("".join(sorted(core + d)) for d in core)

def validate_pairwise_core_separators(trans: pd.DataFrame, target_cores: List[str], min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int) -> pd.DataFrame:
    """Within selected cores, mine traits that separate one core from another.
    For pair A/B, baseline is A share among all A/B outcomes; trait rate is A share among rows with that trait.
    """
    aabc = trans[trans["IsAABC"] & trans["ActualCore"].isin(target_cores)].copy()
    if aabc.empty or len(target_cores) < 2:
        return pd.DataFrame()
    core_totals = Counter(aabc["ActualCore"])
    pair_totals = Counter()
    pair_trait_samples = Counter()
    pair_trait_hits = Counter()
    rows = list(aabc.itertuples(index=False))
    prog = st.progress(0.0)
    for i, r in enumerate(rows, start=1):
        actual = getattr(r, "ActualCore")
        traits = seed_traits(getattr(r, "SeedResult"), stream="", dow=getattr(r,"DOW",""), month=getattr(r,"Month",""))
        for other in target_cores:
            if other == actual:
                continue
            pair = tuple(sorted((actual, other)))
            pair_totals[pair] += 1
            for tr in traits:
                pair_trait_samples[(pair, tr)] += 1
                pair_trait_hits[(pair, tr, actual)] += 1
        if i % 2500 == 0 or i == len(rows):
            prog.progress(i / max(1, len(rows)))
    prog.empty()
    out = []
    for (pair, tr), sample in pair_trait_samples.items():
        if sample < min_sample:
            continue
        a,b = pair
        total = pair_totals[pair]
        for winner in pair:
            hits = pair_trait_hits.get((pair,tr,winner), 0)
            if hits < min_hits:
                continue
            baseline = core_totals[winner] / max(1, total)
            hit_rate = hits / sample
            lift = hit_rate / baseline if baseline > 0 else 0
            if lift < min_lift:
                continue
            stability_weight = min(1.0, sample / max(1, stable_den))
            weighted_lift = lift * stability_weight
            confidence_score = weighted_lift * math.log1p(hits) * hit_rate
            out.append({
                "separator_type":"core_vs_core", "pair":"_vs_".join(pair), "trait":tr,
                "favored_core":winner, "sample":sample, "hits":hits,
                "hit_rate":round(hit_rate,6), "pair_baseline_rate":round(baseline,6),
                "relative_lift":round(lift,4), "stability_weight":round(stability_weight,4),
                "weighted_lift":round(weighted_lift,4), "confidence_score":round(confidence_score,6),
                "sample_tier": "STRONG_250_PLUS" if sample>=250 else "STABLE_100_PLUS" if sample>=100 else "CANDIDATE_50_99" if sample>=50 else "EXPLORATORY_25_49" if sample>=25 else "MICRO",
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score","weighted_lift","hits","sample"], ascending=False).head(top_n).reset_index(drop=True)

def validate_member_separators(trans: pd.DataFrame, target_cores: List[str], min_sample: int, min_hits: int, min_lift: float, stable_den: int, top_n: int) -> pd.DataFrame:
    """Within each selected core, mine traits that separate its 3 members."""
    selected_members = set()
    member_to_core = {}
    for c in target_cores:
        for m in members_for_core(c):
            selected_members.add(m)
            member_to_core[m] = c
    aabc = trans[trans["IsAABC"] & trans["ActualMember"].isin(selected_members)].copy()
    if aabc.empty:
        return pd.DataFrame()
    member_totals = Counter(aabc["ActualMember"])
    pair_totals = Counter()
    pair_trait_samples = Counter()
    pair_trait_hits = Counter()
    rows = list(aabc.itertuples(index=False))
    prog = st.progress(0.0)
    for i, r in enumerate(rows, start=1):
        actual = getattr(r, "ActualMember")
        core = member_to_core.get(actual, "")
        if not core:
            continue
        traits = seed_traits(getattr(r, "SeedResult"), stream="", dow=getattr(r,"DOW",""), month=getattr(r,"Month",""))
        for other in members_for_core(core):
            if other == actual:
                continue
            pair = tuple(sorted((actual, other)))
            pair_totals[pair] += 1
            for tr in traits:
                pair_trait_samples[(pair, tr)] += 1
                pair_trait_hits[(pair, tr, actual)] += 1
        if i % 2500 == 0 or i == len(rows):
            prog.progress(i / max(1, len(rows)))
    prog.empty()
    out = []
    for (pair, tr), sample in pair_trait_samples.items():
        if sample < min_sample:
            continue
        total = pair_totals[pair]
        for winner in pair:
            hits = pair_trait_hits.get((pair,tr,winner), 0)
            if hits < min_hits:
                continue
            baseline = member_totals[winner] / max(1, total)
            hit_rate = hits / sample
            lift = hit_rate / baseline if baseline > 0 else 0
            if lift < min_lift:
                continue
            stability_weight = min(1.0, sample / max(1, stable_den))
            weighted_lift = lift * stability_weight
            confidence_score = weighted_lift * math.log1p(hits) * hit_rate
            out.append({
                "separator_type":"member_vs_member", "core":member_to_core.get(winner,""), "pair":"_vs_".join(pair), "trait":tr,
                "favored_member":winner, "sample":sample, "hits":hits,
                "hit_rate":round(hit_rate,6), "pair_baseline_rate":round(baseline,6),
                "relative_lift":round(lift,4), "stability_weight":round(stability_weight,4),
                "weighted_lift":round(weighted_lift,4), "confidence_score":round(confidence_score,6),
                "sample_tier": "STRONG_250_PLUS" if sample>=250 else "STABLE_100_PLUS" if sample>=100 else "CANDIDATE_50_99" if sample>=50 else "EXPLORATORY_25_49" if sample>=25 else "MICRO",
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["confidence_score","weighted_lift","hits","sample"], ascending=False).head(top_n).reset_index(drop=True)

# -----------------------------
# UI controls
# -----------------------------
with st.sidebar:
    st.header("v14 Controls")
    st.caption("Run one validation stage at a time. This prevents Streamlit Cloud crashes.")
    stage = st.selectbox("Mining stage", [
        "1 - Data audit + base profiles",
        "2 - Core single-trait lift",
        "3 - Member single-trait lift",
        "4 - Core-vs-core separator validation",
        "5 - Member-vs-member separator validation",
        "6 - Core stream+trait lift",
        "7 - Member stream+trait lift",
    ])
    target_core_text = st.text_input("Target cores for separator stages", "025,389,168,246,589,019,468,236")
    min_sample = st.selectbox("Minimum sample to display", [10,15,25,50,100], index=2)
    min_hits = st.selectbox("Minimum hits for signal", [3,5,10,15,20], index=2)
    min_lift = st.slider("Minimum relative lift", 1.0, 10.0, 1.5, 0.1)
    stable_den = st.selectbox("Stable sample denominator", [50,100,250,500], index=1)
    top_n = st.slider("Export/display top N", 100, 3000, 500, 100)

uploaded = st.file_uploader("Upload full clean Pick-4 history", type=["csv","txt","tsv","xlsx","xls"])
if not uploaded:
    st.stop()

if st.button("Run selected v14 stage", type="primary", use_container_width=True):
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
            {"metric":"app_version", "value":APP_VERSION},
            {"metric":"build_marker", "value":BUILD_MARKER},
            {"metric":"loaded_rows", "value":len(hist)},
            {"metric":"seed_next_transitions", "value":len(trans)},
            {"metric":"aabc_transitions", "value":int(trans['IsAABC'].sum())},
            {"metric":"streams", "value":trans['StreamKey'].nunique()},
            {"metric":"stage", "value":stage},
            {"metric":"min_sample", "value":min_sample},
            {"metric":"min_hits", "value":min_hits},
            {"metric":"min_lift", "value":min_lift},
        ])
        frames["summary.csv"] = summary

        if stage.startswith("1"):
            st.subheader("Data audit + base profiles")
            core, member, stream_core = simple_profiles(trans)
            frames["core_profiles.csv"] = core
            frames["member_profiles.csv"] = member
            frames["stream_core_profiles.csv"] = stream_core.head(top_n)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.dataframe(core.head(120), use_container_width=True, hide_index=True)

        elif stage.startswith("2"):
            st.subheader("Core single-trait lift")
            rows = build_rows(trans, include_stream_trait=False)
            df = score_rows(rows, "ActualCore", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["core_single_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("3"):
            st.subheader("Member single-trait lift")
            rows = build_rows(trans, include_stream_trait=False)
            df = score_rows(rows, "ActualMember", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["member_single_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("4"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Core-vs-core separator validation")
            st.caption("Selected cores: " + ", ".join(target_cores))
            df = validate_pairwise_core_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n)
            frames["core_vs_core_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("5"):
            target_cores = parse_target_cores(target_core_text)
            st.subheader("Member-vs-member separator validation")
            st.caption("Within-core member separators for: " + ", ".join(target_cores))
            df = validate_member_separators(trans, target_cores, min_sample, min_hits, min_lift, stable_den, top_n)
            frames["member_vs_member_separators.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("6"):
            st.subheader("Core stream+trait lift")
            rows = build_rows(trans, include_stream_trait=True)
            df = score_rows(rows, "ActualCore", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["core_stream_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif stage.startswith("7"):
            st.subheader("Member stream+trait lift")
            rows = build_rows(trans, include_stream_trait=True)
            df = score_rows(rows, "ActualMember", min_sample, min_hits, min_lift, stable_den, top_n)
            frames["member_stream_trait_lift.csv"] = df
            st.dataframe(df, use_container_width=True, hide_index=True)

        zip_bytes = make_zip(frames, {"README_v14.txt": "Core Affinity Lab v14 signal-validation output. Run one stage at a time; combine exported CSVs later. Lab only; no playlist logic."})
        freeze_bytes("v14_last_zip", zip_bytes)
        st.success("Stage complete. Download ZIP below.")
        st.download_button("Download v14 selected-stage outputs ZIP", data=zip_bytes, file_name="core_affinity_lab_v14_selected_stage_outputs.zip", mime="application/zip", use_container_width=True)

    except Exception as e:
        st.error("v14 caught the error instead of blank-crashing.")
        st.exception(e)
        st.stop()

if "v14_last_zip" in st.session_state:
    st.download_button("Download previous v14 ZIP again", data=st.session_state["v14_last_zip"], file_name="core_affinity_lab_v14_selected_stage_outputs.zip", mime="application/zip", use_container_width=True)
