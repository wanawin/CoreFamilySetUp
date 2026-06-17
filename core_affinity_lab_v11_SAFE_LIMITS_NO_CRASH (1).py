#!/usr/bin/env python3
"""
Core Affinity Lab v11 — Universal Trait + Member Profiles SAFE LIMITS

Purpose:
- Build a universal trait universe from Pick-4 seed transitions.
- Mine traits for BOTH core-level separation and member-level separation.
- Export reusable core/member profile tables for future multi-core daily platform.

Locked behavior:
- Lab only. No daily playlist logic.
- No B1Z0/RTE/ZLT/rescue/budget/cut logic.
- No simulations presented as real play results.
"""
from __future__ import annotations

import io
import itertools
import math
import re
import zipfile
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "v11"
BUILD_MARKER = "BUILD: core_affinity_lab_v11_SAFE_LIMITS_NO_CRASH__2026-06-17"

st.set_page_config(page_title="Core Affinity Lab v11", layout="wide")
st.title("Core Affinity Lab v11 — Universal Trait + Member Profiles SAFE LIMITS")
st.caption(BUILD_MARKER)
st.info(
    "Lab only. Mines universal seed traits across all 120 AABC cores and all 360 AABC members. "
    "No daily playlist, no cuts, no RTE, no B1Z0, no ZLT, no rescue logic. v11 adds safe limits/checkpoints to avoid Streamlit memory crashes."
)

# -----------------------------
# Constants
# -----------------------------
DIGITS = "0123456789"
MIRROR_PAIRS = {"05", "16", "27", "38", "49"}
ALL_CORES = ["".join(c) for c in itertools.combinations(DIGITS, 3)]


def all_members_for_core(core_id: str) -> List[str]:
    digs = list(str(core_id))
    return sorted("".join(sorted(digs + [d])) for d in digs)


ALL_MEMBERS = sorted([m for c in ALL_CORES for m in all_members_for_core(c)])

# -----------------------------
# Download/session freeze helpers
# -----------------------------
def _freeze_outputs(outputs: Dict[str, pd.DataFrame], texts: Dict[str, str] | None = None) -> bytes:
    texts = texts or {}
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name, df in outputs.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(name, df.to_csv(index=False).encode("utf-8"))
        for name, txt in texts.items():
            zf.writestr(name, str(txt).encode("utf-8"))
    data = bio.getvalue()
    st.session_state["v11_outputs"] = outputs
    st.session_state["v11_texts"] = texts
    st.session_state["v11_zip_bytes"] = data
    return data


def _download_df(label: str, key: str, filename: str):
    outputs = st.session_state.get("v11_outputs", {})
    df = outputs.get(key)
    if isinstance(df, pd.DataFrame):
        st.download_button(
            label,
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
            key=f"download_{APP_VERSION}_{key}",
            use_container_width=True,
        )

# -----------------------------
# I/O and normalization
# -----------------------------
def read_upload(file) -> pd.DataFrame:
    raw = file.getvalue()
    name = str(file.name).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str)
    text = raw.decode("utf-8", errors="replace")
    if name.endswith(".csv"):
        return pd.read_csv(io.StringIO(text), dtype=str)
    if name.endswith(".tsv"):
        return pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    # Try common delimiters.
    for sep in ["\t", ",", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")
            if df.shape[1] >= 4:
                return df
        except Exception:
            pass
    return pd.read_csv(io.StringIO(text), sep=None, dtype=str, engine="python")


def norm4(x) -> str:
    # Use only the base Pick-4 digits before any add-on comma fields.
    base = str(x).split(",", 1)[0]
    digs = re.findall(r"\d", base)
    return "".join(digs[:4]) if len(digs) >= 4 else ""


def classify_aabc(result4: str) -> Tuple[str, str, str, bool]:
    s = norm4(result4)
    if len(s) != 4:
        return "", "", "", False
    sorted_s = "".join(sorted(s))
    c = Counter(sorted_s)
    counts = sorted(c.values(), reverse=True)
    if counts != [2, 1, 1]:
        return "", "", "", False
    repeat_digit = next(d for d, n in c.items() if n == 2)
    core_id = "".join(sorted(c.keys()))
    member = sorted_s
    return core_id, member, repeat_digit, True


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = lower.get("date") or lower.get("drawdate") or lower.get("draw_date")
    result_col = None
    for k in ["result4", "result", "winning number", "winningnumber", "number", "draw"]:
        if k in lower:
            result_col = lower[k]
            break
    stream_col = lower.get("streamkey")
    state_col = lower.get("state")
    game_col = lower.get("game")
    if stream_col is None and state_col and game_col:
        stream_col = "__generated_streamkey__"
    if date_col is None or result_col is None:
        raise ValueError("Expected a Date column and a Result/Result4 column.")
    return {"date": date_col, "result": result_col, "stream": stream_col, "state": state_col, "game": game_col}


def prepare_seed_transition_rows(hist: pd.DataFrame) -> pd.DataFrame:
    cols = resolve_columns(hist)
    df = hist.copy()
    if cols["stream"] == "__generated_streamkey__":
        df["__generated_streamkey__"] = df[cols["state"]].astype(str) + " | " + df[cols["game"]].astype(str)
    df["DateParsed"] = pd.to_datetime(df[cols["date"]], errors="coerce")
    df["StreamKey"] = df[cols["stream"]].astype(str)
    df["Result4"] = df[cols["result"]].map(norm4)
    df = df.dropna(subset=["DateParsed"]).copy()
    df = df[df["Result4"].str.len() == 4].copy()
    df = df.drop_duplicates(["DateParsed", "StreamKey", "Result4"]).sort_values(["StreamKey", "DateParsed"]).reset_index(drop=True)

    df["SeedResult"] = df.groupby("StreamKey")["Result4"].shift(1)
    df["SeedDate"] = df.groupby("StreamKey")["DateParsed"].shift(1)
    trans = df.dropna(subset=["SeedResult", "SeedDate"]).copy()
    trans["SeedAgeDays"] = (trans["DateParsed"] - trans["SeedDate"]).dt.days

    fam = trans["Result4"].map(classify_aabc)
    trans["ActualCore"] = fam.map(lambda x: x[0])
    trans["ActualMember"] = fam.map(lambda x: x[1])
    trans["RepeatDigit"] = fam.map(lambda x: x[2])
    trans["IsAABC"] = fam.map(lambda x: x[3])
    return trans.reset_index(drop=True)

# -----------------------------
# Universal trait generator
# -----------------------------
def _sum_bucket(n: int) -> str:
    if n <= 9:
        return "00_09"
    if n <= 13:
        return "10_13"
    if n <= 17:
        return "14_17"
    if n <= 21:
        return "18_21"
    return "22_plus"


def _spread_bucket(n: int) -> str:
    if n <= 2:
        return "0_2"
    if n <= 4:
        return "3_4"
    if n <= 6:
        return "5_6"
    return "7_9"


def seed_traits(seed: str) -> List[str]:
    s = norm4(seed).zfill(4)[-4:]
    ds = [int(x) for x in s]
    traits: List[str] = []

    # Digits: inclusion/exclusion/counts for all 10 digits.
    for d in DIGITS:
        cnt = s.count(d)
        traits.append(f"has_digit:{d}={1 if cnt else 0}")
        traits.append(f"no_digit:{d}={1 if cnt == 0 else 0}")
        traits.append(f"cnt_digit:{d}={cnt}")

    # Unordered digit-pair inclusion/exclusion for all 45 pairs.
    unique = set(s)
    for a, b in itertools.combinations(DIGITS, 2):
        pair = f"{a}{b}"
        has_pair = int(a in unique and b in unique)
        traits.append(f"has_unordered_pair:{pair}={has_pair}")
        traits.append(f"no_unordered_pair:{pair}={1 - has_pair}")

    # Ordered and positional pairs.
    pair_positions = {
        "first2": s[0:2],
        "mid2": s[1:3],
        "last2": s[2:4],
        "firstlast": s[0] + s[3],
        "pos1pos3": s[0] + s[2],
        "pos2pos4": s[1] + s[3],
    }
    for name, val in pair_positions.items():
        traits.append(f"ordered_pair:{name}={val}")
        traits.append(f"unordered_pair:{name}={''.join(sorted(val))}")
        traits.append(f"pair_sum:{name}={int(val[0])+int(val[1])}")
        traits.append(f"pair_sum_bucket:{name}={_sum_bucket(int(val[0])+int(val[1]))}")
        traits.append(f"pair_parity:{name}={''.join('E' if int(x)%2==0 else 'O' for x in val)}")
        traits.append(f"pair_highlow:{name}={''.join('H' if int(x)>=5 else 'L' for x in val)}")

    # Positional traits.
    for i, ch in enumerate(s, start=1):
        v = int(ch)
        traits.append(f"pos_digit:p{i}={ch}")
        traits.append(f"pos_highlow:p{i}={'H' if v >= 5 else 'L'}")
        traits.append(f"pos_parity:p{i}={'E' if v % 2 == 0 else 'O'}")
        traits.append(f"pos_mod3:p{i}={v % 3}")
        traits.append(f"pos_mod5:p{i}={v % 5}")

    # Mirror traits.
    mirror_count = 0
    for mp in sorted(MIRROR_PAIRS):
        a, b = mp[0], mp[1]
        present = int(a in unique and b in unique)
        mirror_count += present
        traits.append(f"mirror_pair:{mp}={present}")
    traits.append(f"mirror_count={mirror_count}")
    traits.append(f"has_any_mirror={1 if mirror_count else 0}")

    # Sum, spread, root, parity, high/low, structure.
    seed_sum = sum(ds)
    root = seed_sum
    while root >= 10:
        root = sum(int(c) for c in str(root))
    spread = max(ds) - min(ds)
    counts = sorted(Counter(s).values(), reverse=True)
    if counts == [4]:
        structure = "AAAA"
    elif counts == [3, 1]:
        structure = "AAAB"
    elif counts == [2, 2]:
        structure = "AABB"
    elif counts == [2, 1, 1]:
        structure = "AABC"
    else:
        structure = "ABCD"
    traits.extend([
        f"seed_sum={seed_sum}",
        f"seed_sum_bucket={_sum_bucket(seed_sum)}",
        f"seed_sum_lastdigit={seed_sum % 10}",
        f"seed_root_sum={root}",
        f"spread={spread}",
        f"spread_bucket={_spread_bucket(spread)}",
        f"parity_pattern={''.join('E' if x%2==0 else 'O' for x in ds)}",
        f"highlow_pattern={''.join('H' if x>=5 else 'L' for x in ds)}",
        f"even_count={sum(1 for x in ds if x%2==0)}",
        f"odd_count={sum(1 for x in ds if x%2==1)}",
        f"high_count={sum(1 for x in ds if x>=5)}",
        f"low_count={sum(1 for x in ds if x<5)}",
        f"unique_count={len(set(s))}",
        f"max_repeat={max(counts)}",
        f"structure={structure}",
        f"consec_links={sum(1 for a,b in zip(ds, ds[1:]) if abs(a-b)==1)}",
        f"plusminus1_pairs={sum(1 for a,b in itertools.combinations(ds,2) if abs(a-b)==1)}",
    ])
    return traits


def trait_category(trait: str) -> str:
    return str(trait).split(":", 1)[0].split("=", 1)[0]

# -----------------------------
# Profile / lift utilities
# -----------------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def add_signal_metrics(df: pd.DataFrame, sample_col: str, hits_col: str, base_rate_col: str, min_stable_sample: int) -> pd.DataFrame:
    out = df.copy()
    out["hit_rate"] = out.apply(lambda r: _safe_div(r[hits_col], r[sample_col]), axis=1)
    out["relative_lift"] = out.apply(lambda r: _safe_div(r["hit_rate"], r[base_rate_col]), axis=1)
    out["absolute_lift"] = out["hit_rate"] - out[base_rate_col]
    out["stability_weight"] = (out[sample_col].astype(float) / float(max(1, min_stable_sample))).clip(upper=1.0)
    out["weighted_lift"] = out["relative_lift"] * out["stability_weight"]
    out["confidence_score"] = out["weighted_lift"] * np.log1p(out[hits_col].astype(float)) * out["hit_rate"]
    out["sample_tier"] = pd.cut(
        out[sample_col].astype(float),
        bins=[-1, 9, 24, 49, 99, 249, float("inf")],
        labels=["tiny_0_9", "exploratory_10_24", "weak_25_49", "candidate_50_99", "stable_100_249", "strong_250_plus"],
    ).astype(str)
    return out


def lift_by_group(rows: pd.DataFrame, group_cols: Sequence[str], target_col: str, total_target_counts: pd.Series,
                  total_rows: int, min_stable_sample: int) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    sample = rows.groupby(list(group_cols)).size().rename("sample_size").reset_index()
    hits = rows.groupby(list(group_cols) + [target_col]).size().rename("hits").reset_index()
    merged = hits.merge(sample, on=list(group_cols), how="left")
    merged["baseline_hits"] = merged[target_col].map(total_target_counts).fillna(0).astype(int)
    merged["baseline_rate"] = merged["baseline_hits"] / max(1, total_rows)
    merged = add_signal_metrics(merged, "sample_size", "hits", "baseline_rate", min_stable_sample)
    return merged.sort_values(["confidence_score", "hits", "sample_size"], ascending=[False, False, False]).reset_index(drop=True)



def _counter_lift_table(sample_counter: Counter, hit_counter: Counter, group_names: Sequence[str], target_col: str,
                        total_target_counts: pd.Series, total_rows: int, min_stable_sample: int) -> pd.DataFrame:
    """Memory-safe lift builder from counters. Avoids exploding millions of rows into a dataframe."""
    records = []
    for key_tuple, sample_size in sample_counter.items():
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        for target, hits in hit_counter.get(key_tuple, {}).items():
            baseline_hits = int(total_target_counts.get(target, 0))
            baseline_rate = baseline_hits / max(1, total_rows)
            hit_rate = hits / max(1, sample_size)
            rel_lift = hit_rate / baseline_rate if baseline_rate else 0.0
            rec = {name: val for name, val in zip(group_names, key_tuple)}
            rec.update({
                target_col: target,
                "sample_size": int(sample_size),
                "hits": int(hits),
                "baseline_hits": baseline_hits,
                "baseline_rate": baseline_rate,
                "hit_rate": hit_rate,
                "relative_lift": rel_lift,
                "absolute_lift": hit_rate - baseline_rate,
                "stability_weight": min(1.0, sample_size / float(max(1, min_stable_sample))),
            })
            rec["weighted_lift"] = rec["relative_lift"] * rec["stability_weight"]
            rec["confidence_score"] = rec["weighted_lift"] * math.log1p(hits) * rec["hit_rate"]
            records.append(rec)
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["sample_tier"] = pd.cut(
        out["sample_size"].astype(float),
        bins=[-1, 9, 24, 49, 99, 249, float("inf")],
        labels=["tiny_0_9", "exploratory_10_24", "weak_25_49", "candidate_50_99", "stable_100_249", "strong_250_plus"],
    ).astype(str)
    return out.sort_values(["confidence_score", "hits", "sample_size"], ascending=[False, False, False]).reset_index(drop=True)



def build_profiles(aabc: pd.DataFrame, min_stable_sample: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build core/member and stream x core/member profiles.
    v9 surgical crash fix: v8 accidentally called this function without defining it after the
    memory-safe refactor. This restores the profile builder only; mining/scoring logic unchanged.
    """
    total = len(aabc)
    core_counts = aabc["ActualCore"].value_counts().reindex(ALL_CORES, fill_value=0)
    member_counts = aabc["ActualMember"].value_counts().reindex(ALL_MEMBERS, fill_value=0)

    core_profiles = pd.DataFrame({"core_id": core_counts.index, "hits": core_counts.values})
    core_profiles["baseline_rate"] = core_profiles["hits"] / max(1, total)
    core_profiles["members"] = core_profiles["core_id"].map(lambda c: ",".join(all_members_for_core(c)))
    core_profiles["profile_rank_by_hits"] = core_profiles["hits"].rank(method="first", ascending=False).astype(int)

    member_profiles = pd.DataFrame({"member": member_counts.index, "hits": member_counts.values})
    member_profiles["core_id"] = member_profiles["member"].map(lambda m: "".join(sorted(set(str(m)))))
    member_profiles["baseline_rate"] = member_profiles["hits"] / max(1, total)
    member_profiles["profile_rank_by_hits"] = member_profiles["hits"].rank(method="first", ascending=False).astype(int)
    member_profiles["member_role_by_core_freq"] = (
        member_profiles.groupby("core_id")["hits"]
        .rank(method="first", ascending=False)
        .map({1.0: "strongest_candidate", 2.0: "middle_candidate", 3.0: "suppressed_candidate"})
        .fillna("unknown")
    )

    stream_sample = aabc.groupby("StreamKey").size().rename("stream_sample").reset_index()

    stream_core = aabc.groupby(["StreamKey", "ActualCore"]).size().rename("hits").reset_index()
    stream_core = stream_core.merge(stream_sample, on="StreamKey", how="left")
    stream_core["stream_core_hit_rate"] = stream_core["hits"] / stream_core["stream_sample"].clip(lower=1)
    stream_core = stream_core.merge(
        core_profiles[["core_id", "baseline_rate"]],
        left_on="ActualCore", right_on="core_id", how="left"
    ).drop(columns=["core_id"])
    stream_core["relative_lift"] = stream_core["stream_core_hit_rate"] / stream_core["baseline_rate"].replace(0, np.nan)
    stream_core["stability_weight"] = (stream_core["stream_sample"] / max(1, min_stable_sample)).clip(upper=1.0)
    stream_core["tie_breaker_score"] = stream_core["relative_lift"].fillna(0) * stream_core["stability_weight"] * np.log1p(stream_core["hits"])
    stream_core = stream_core.sort_values(["tie_breaker_score", "hits"], ascending=[False, False]).reset_index(drop=True)

    stream_member = aabc.groupby(["StreamKey", "ActualMember"]).size().rename("hits").reset_index()
    stream_member = stream_member.merge(stream_sample, on="StreamKey", how="left")
    stream_member["stream_member_hit_rate"] = stream_member["hits"] / stream_member["stream_sample"].clip(lower=1)
    stream_member = stream_member.merge(
        member_profiles[["member", "baseline_rate"]],
        left_on="ActualMember", right_on="member", how="left"
    ).drop(columns=["member"])
    stream_member["relative_lift"] = stream_member["stream_member_hit_rate"] / stream_member["baseline_rate"].replace(0, np.nan)
    stream_member["stability_weight"] = (stream_member["stream_sample"] / max(1, min_stable_sample)).clip(upper=1.0)
    stream_member["tie_breaker_score"] = stream_member["relative_lift"].fillna(0) * stream_member["stability_weight"] * np.log1p(stream_member["hits"])
    stream_member = stream_member.sort_values(["tie_breaker_score", "hits"], ascending=[False, False]).reset_index(drop=True)

    return core_profiles, member_profiles, stream_core, stream_member

def build_single_trait_lifts(aabc: pd.DataFrame, min_stable_sample: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_core_counts = aabc["ActualCore"].value_counts()
    base_member_counts = aabc["ActualMember"].value_counts()
    total = len(aabc)
    sample_counter = Counter()
    core_hit_counter: Dict[Tuple[str, str], Counter] = {}
    member_hit_counter: Dict[Tuple[str, str], Counter] = {}
    for _, r in aabc[["ActualCore", "ActualMember", "TraitList"]].iterrows():
        for trait in r["TraitList"]:
            key = (trait, trait_category(trait))
            sample_counter[key] += 1
            core_hit_counter.setdefault(key, Counter())[r["ActualCore"]] += 1
            member_hit_counter.setdefault(key, Counter())[r["ActualMember"]] += 1
    trait_dictionary = pd.DataFrame([
        {"trait": k[0], "trait_category": k[1], "transition_count": v}
        for k, v in sample_counter.items()
    ]).sort_values("transition_count", ascending=False).reset_index(drop=True)
    core_lift = _counter_lift_table(sample_counter, core_hit_counter, ["trait", "trait_category"], "ActualCore", base_core_counts, total, min_stable_sample)
    member_lift = _counter_lift_table(sample_counter, member_hit_counter, ["trait", "trait_category"], "ActualMember", base_member_counts, total, min_stable_sample)
    return trait_dictionary, core_lift, member_lift


def build_stream_trait_lifts(aabc: pd.DataFrame, min_stable_sample: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_core_counts = aabc["ActualCore"].value_counts()
    base_member_counts = aabc["ActualMember"].value_counts()
    total = len(aabc)
    sample_counter = Counter()
    core_hit_counter: Dict[Tuple[str, str, str], Counter] = {}
    member_hit_counter: Dict[Tuple[str, str, str], Counter] = {}
    for _, r in aabc[["StreamKey", "ActualCore", "ActualMember", "TraitList"]].iterrows():
        stream = r["StreamKey"]
        for trait in r["TraitList"]:
            key = (stream, trait, trait_category(trait))
            sample_counter[key] += 1
            core_hit_counter.setdefault(key, Counter())[r["ActualCore"]] += 1
            member_hit_counter.setdefault(key, Counter())[r["ActualMember"]] += 1
    core = _counter_lift_table(sample_counter, core_hit_counter, ["StreamKey", "trait", "trait_category"], "ActualCore", base_core_counts, total, min_stable_sample)
    member = _counter_lift_table(sample_counter, member_hit_counter, ["StreamKey", "trait", "trait_category"], "ActualMember", base_member_counts, total, min_stable_sample)
    return core, member


def build_stacked_lifts(aabc: pd.DataFrame, trait_core_lift: pd.DataFrame, min_stable_sample: int,
                        include_pairs: bool, include_triples: bool, max_base_traits: int, max_traits_per_row: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not include_pairs and not include_triples:
        return pd.DataFrame(), pd.DataFrame()
    if trait_core_lift is None or trait_core_lift.empty:
        return pd.DataFrame(), pd.DataFrame()
    top_traits = set(trait_core_lift.head(max_base_traits)["trait"].astype(str).tolist())
    base_core_counts = aabc["ActualCore"].value_counts()
    base_member_counts = aabc["ActualMember"].value_counts()
    total = len(aabc)
    sample_counter = Counter()
    core_hit_counter: Dict[Tuple[str, int], Counter] = {}
    member_hit_counter: Dict[Tuple[str, int], Counter] = {}
    stack_sizes = []
    if include_pairs:
        stack_sizes.append(2)
    if include_triples:
        stack_sizes.append(3)
    for _, r in aabc[["ActualCore", "ActualMember", "TraitList"]].iterrows():
        traits = [t for t in r["TraitList"] if t in top_traits][:max_traits_per_row]
        for stack_size in stack_sizes:
            if len(traits) < stack_size:
                continue
            for combo in itertools.combinations(traits, stack_size):
                key = (" && ".join(combo), stack_size)
                sample_counter[key] += 1
                core_hit_counter.setdefault(key, Counter())[r["ActualCore"]] += 1
                member_hit_counter.setdefault(key, Counter())[r["ActualMember"]] += 1
    if not sample_counter:
        return pd.DataFrame(), pd.DataFrame()
    core = _counter_lift_table(sample_counter, core_hit_counter, ["stack_trait", "stack_size"], "ActualCore", base_core_counts, total, min_stable_sample)
    member = _counter_lift_table(sample_counter, member_hit_counter, ["stack_trait", "stack_size"], "ActualMember", base_member_counts, total, min_stable_sample)
    return core, member


def filter_signals(df: pd.DataFrame, min_sample: int, min_hits: int, min_lift: float, sort_col: str = "confidence_score") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    sample_col = "sample_size" if "sample_size" in out.columns else "stream_sample"
    lift_col = "relative_lift"
    if sample_col in out.columns:
        out = out[pd.to_numeric(out[sample_col], errors="coerce").fillna(0) >= min_sample]
    if "hits" in out.columns:
        out = out[pd.to_numeric(out["hits"], errors="coerce").fillna(0) >= min_hits]
    if lift_col in out.columns:
        out = out[pd.to_numeric(out[lift_col], errors="coerce").fillna(0) >= min_lift]
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out.reset_index(drop=True)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("v11 Controls")
st.sidebar.caption("v11 safe limits: stacked mining OFF by default, capped when enabled, staged error reporting.")
min_display_sample = st.sidebar.selectbox("Minimum sample to display", [10, 15, 25, 50, 100], index=2)
min_hits_for_signal = st.sidebar.selectbox("Minimum hits for signal", [3, 5, 10, 15, 20], index=2)
min_lift_for_signal = st.sidebar.slider("Minimum relative lift", 1.0, 10.0, 1.5, 0.1)
min_stable_sample = st.sidebar.selectbox("Stable sample denominator", [50, 75, 100, 150, 250], index=2)
export_top_n = st.sidebar.slider("Export top N per heavy table", 100, 3000, 500, 100)
include_pair_stacks = st.sidebar.checkbox("Mine 2-trait stacked signals (heavy — keep OFF unless needed)", value=False)
include_triple_stacks = st.sidebar.checkbox("Mine 3-trait stacked signals", value=False)
max_base_traits = st.sidebar.slider("Stack miner: max base single traits", 10, 100, 25, 5)
max_traits_per_row = st.sidebar.slider("Stack miner: max traits per row", 3, 12, 6, 1)

history_file = st.file_uploader("Upload full clean Pick-4 history", type=["csv", "txt", "tsv", "xlsx", "xls"])

if not history_file:
    st.stop()

run_btn = st.button("Run v11 universal trait/member profiler", type="primary", use_container_width=True)

if run_btn:
    try:
        progress = st.progress(0, text="Loading history...")
        hist = read_upload(history_file)
        progress.progress(8, text="Preparing seed transitions...")
        trans = prepare_seed_transition_rows(hist)
        trans = trans.reset_index(drop=True)
        trans["TransitionID"] = np.arange(1, len(trans) + 1)

        progress.progress(18, text="Generating universal trait universe...")
        trans["TraitList"] = trans["SeedResult"].map(seed_traits)
        aabc = trans[trans["IsAABC"]].copy().reset_index(drop=True)

        progress.progress(30, text="Building core/member profiles...")
        core_profiles, member_profiles, stream_core_profiles, stream_member_profiles = build_profiles(aabc, min_stable_sample)

        progress.progress(43, text="Mining single-trait core/member lift...")
        trait_dictionary, seed_trait_core_lift, seed_trait_member_lift = build_single_trait_lifts(aabc, min_stable_sample)

        progress.progress(58, text="Mining stream + trait core/member lift...")
        stream_seed_trait_core_lift, stream_seed_trait_member_lift = build_stream_trait_lifts(aabc, min_stable_sample)

        progress.progress(72, text="Mining optional stacked traits...")
        stacked_trait_core_lift, stacked_trait_member_lift = build_stacked_lifts(
        aabc,
        filter_signals(seed_trait_core_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal),
        min_stable_sample,
        include_pair_stacks,
        include_triple_stacks,
        max_base_traits,
        max_traits_per_row,
        )

        progress.progress(84, text="Filtering signal candidates...")
        core_signal_candidates = filter_signals(seed_trait_core_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)
        member_signal_candidates = filter_signals(seed_trait_member_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)
        stream_core_signal_candidates = filter_signals(stream_seed_trait_core_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)
        stream_member_signal_candidates = filter_signals(stream_seed_trait_member_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)
        stacked_core_signal_candidates = filter_signals(stacked_trait_core_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)
        stacked_member_signal_candidates = filter_signals(stacked_trait_member_lift, min_display_sample, min_hits_for_signal, min_lift_for_signal).head(export_top_n)

        progress.progress(92, text="Preparing exports...")
        prepared_export = trans[[
        "TransitionID", "DateParsed", "StreamKey", "SeedDate", "SeedResult", "SeedAgeDays", "Result4", "ActualCore", "ActualMember", "RepeatDigit", "IsAABC"
        ]].copy()
        prepared_export["DateParsed"] = pd.to_datetime(prepared_export["DateParsed"]).dt.strftime("%Y-%m-%d")
        prepared_export["SeedDate"] = pd.to_datetime(prepared_export["SeedDate"]).dt.strftime("%Y-%m-%d")

        summary = pd.DataFrame([
        {"metric": "app_version", "value": APP_VERSION},
        {"metric": "build_marker", "value": BUILD_MARKER},
        {"metric": "history_rows_loaded", "value": len(hist)},
        {"metric": "seed_transitions", "value": len(trans)},
        {"metric": "AABC_transitions", "value": len(aabc)},
        {"metric": "streams", "value": trans["StreamKey"].nunique()},
        {"metric": "cores", "value": len(ALL_CORES)},
        {"metric": "members", "value": len(ALL_MEMBERS)},
        {"metric": "trait_dictionary_rows", "value": len(trait_dictionary)},
        {"metric": "core_signal_candidates", "value": len(core_signal_candidates)},
        {"metric": "member_signal_candidates", "value": len(member_signal_candidates)},
        {"metric": "stream_core_signal_candidates", "value": len(stream_core_signal_candidates)},
        {"metric": "stream_member_signal_candidates", "value": len(stream_member_signal_candidates)},
        {"metric": "stacked_core_signal_candidates", "value": len(stacked_core_signal_candidates)},
        {"metric": "stacked_member_signal_candidates", "value": len(stacked_member_signal_candidates)},
        {"metric": "min_display_sample", "value": min_display_sample},
        {"metric": "min_hits_for_signal", "value": min_hits_for_signal},
        {"metric": "min_lift_for_signal", "value": min_lift_for_signal},
        {"metric": "min_stable_sample", "value": min_stable_sample},
        ])

        outputs = {
        "summary.csv": summary,
        "prepared_seed_transition_rows.csv": prepared_export,
        "trait_pack_dictionary.csv": trait_dictionary.head(export_top_n),
        "core_profiles.csv": core_profiles,
        "member_profiles.csv": member_profiles,
        "stream_core_profiles.csv": stream_core_profiles.head(export_top_n),
        "stream_member_profiles.csv": stream_member_profiles.head(export_top_n),
        "seed_trait_core_lift.csv": seed_trait_core_lift.head(export_top_n),
        "seed_trait_member_lift.csv": seed_trait_member_lift.head(export_top_n),
        "stream_seed_trait_core_lift.csv": stream_seed_trait_core_lift.head(export_top_n),
        "stream_seed_trait_member_lift.csv": stream_seed_trait_member_lift.head(export_top_n),
        "core_signal_candidates.csv": core_signal_candidates,
        "member_signal_candidates.csv": member_signal_candidates,
        "stream_core_signal_candidates.csv": stream_core_signal_candidates,
        "stream_member_signal_candidates.csv": stream_member_signal_candidates,
        "stacked_trait_core_lift.csv": stacked_trait_core_lift.head(export_top_n),
        "stacked_trait_member_lift.csv": stacked_trait_member_lift.head(export_top_n),
        "stacked_core_signal_candidates.csv": stacked_core_signal_candidates,
        "stacked_member_signal_candidates.csv": stacked_member_signal_candidates,
        }
        readme = (
        f"{BUILD_MARKER}\n"
        "Universal trait expansion for all 120 AABC cores and all 360 AABC members.\n"
        "Core-level and member-level outputs are both exported so member separation does not have to be redone later.\n"
        "Lab only: no daily playlist, no cuts, no RTE, no B1Z0, no ZLT, no rescue logic.\n"
        )
        zip_bytes = _freeze_outputs(outputs, {"README_v11.txt": readme})
        progress.progress(100, text="v11 complete.")

    except MemoryError as e:
        st.error("The run exceeded Streamlit memory before exports could be created. Use v11 defaults: stacked mining OFF and export top N 500.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("v11 stopped at the current checkpoint instead of crashing the whole app. Copy this traceback if you need a patch.")
        st.exception(e)
        st.stop()

if "v11_outputs" not in st.session_state:
    st.stop()

outputs = st.session_state["v11_outputs"]
st.success("v11 outputs ready. Download buttons use frozen session bytes.")
st.subheader("Summary")
st.dataframe(outputs["summary.csv"], use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top core signal candidates")
    st.dataframe(outputs["core_signal_candidates.csv"].head(50), use_container_width=True, hide_index=True)
with col2:
    st.subheader("Top member signal candidates")
    st.dataframe(outputs["member_signal_candidates.csv"].head(50), use_container_width=True, hide_index=True)

st.subheader("Downloads")
st.download_button(
    "Download all v11 outputs ZIP",
    data=st.session_state.get("v11_zip_bytes", b""),
    file_name="core_affinity_lab_v11_all_outputs.zip",
    mime="application/zip",
    key="download_v11_all_zip",
    use_container_width=True,
)
for key in [
    "summary.csv", "core_profiles.csv", "member_profiles.csv", "trait_pack_dictionary.csv",
    "core_signal_candidates.csv", "member_signal_candidates.csv",
    "stream_core_signal_candidates.csv", "stream_member_signal_candidates.csv",
    "stacked_core_signal_candidates.csv", "stacked_member_signal_candidates.csv",
]:
    _download_df(f"Download {key}", key, key)
