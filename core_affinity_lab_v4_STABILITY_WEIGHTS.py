#!/usr/bin/env python3
# Core Affinity Lab v4 — Profile Builder + Stability Weights
# BUILD: core_affinity_lab_v4__2026-06-16_PROFILE_BUILDER_STABILITY_WEIGHTS
# Purpose: Build stable stream/core/seed/cadence profiles for all 120 AABC Pick-4 core families.
# This is a LAB ONLY. It does not use or change daily playlist logic, reductions, B1Z0, RTE, ZLT, rescues, or budget logic.

from __future__ import annotations

import itertools
import math
import re
import zipfile
from collections import Counter
from io import BytesIO, StringIO
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core_affinity_lab_v4__2026-06-16_PROFILE_BUILDER_STABILITY_WEIGHTS"
ALL_DIGITS = "0123456789"

st.set_page_config(page_title="Core Affinity Lab v4", layout="wide")
st.title("Core Affinity Lab v4 — Profile Builder + Stability Weights")
st.caption(BUILD_MARKER)
st.info(
    "Lab-only profile builder. Mines stream, core, stream×core, seed-trait, and cadence profiles "
    "from history, then weights signals by sample size/hits/lift stability. No daily-play, cuts, RTE, B1Z0, ZLT, rescues, or budget logic is used."
)


def _read_upload(file) -> pd.DataFrame:
    raw = file.getvalue().decode("utf-8", errors="replace")
    name = str(getattr(file, "name", "")).lower()
    if name.endswith(".csv"):
        return pd.read_csv(StringIO(raw), dtype=str)
    if name.endswith(".tsv"):
        return pd.read_csv(StringIO(raw), sep="\t", dtype=str)
    try:
        return pd.read_csv(StringIO(raw), sep="\t", dtype=str)
    except Exception:
        return pd.read_csv(StringIO(raw), sep=None, engine="python", dtype=str)


def _norm4(x) -> str:
    digits = re.findall(r"\d", str(x))
    if len(digits) < 4:
        return ""
    return "".join(digits[:4])


def _sorted4(x) -> str:
    s = _norm4(x)
    return "".join(sorted(s)) if len(s) == 4 else ""


def classify_aabc(result4: str) -> Tuple[str, str, str, bool]:
    s = _sorted4(result4)
    if len(s) != 4:
        return "", "", "", False
    c = Counter(s)
    counts = sorted(c.values(), reverse=True)
    if counts != [2, 1, 1]:
        return "", s, "", False
    repeat_digit = next(d for d, n in c.items() if n == 2)
    core_id = "".join(sorted(c.keys()))
    member = s
    return core_id, member, repeat_digit, True


def all_120_cores() -> List[str]:
    return ["".join(c) for c in itertools.combinations(ALL_DIGITS, 3)]


def core_members(core_id: str) -> List[str]:
    digs = list(str(core_id))
    return sorted("".join(sorted(digs + [d])) for d in digs)


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _seed_features(seed: str) -> Dict[str, str]:
    s = _norm4(seed)
    if len(s) != 4:
        s = "0000"
    vals = [int(ch) for ch in s]
    cnt = Counter(s)
    counts = sorted(cnt.values(), reverse=True)
    if counts == [4]:
        shape = "quad"
    elif counts == [3, 1]:
        shape = "triple"
    elif counts == [2, 2]:
        shape = "double_double"
    elif counts == [2, 1, 1]:
        shape = "one_pair"
    else:
        shape = "all_unique"
    sm = sum(vals)
    high = sum(v >= 5 for v in vals)
    low = 4 - high
    even = sum(v % 2 == 0 for v in vals)
    odd = 4 - even
    spread = max(vals) - min(vals)
    consec_links = sum(1 for a, b in zip(vals, vals[1:]) if abs(a - b) == 1)
    mirror_pairs = 0
    mirrors = {"0":"5","1":"6","2":"7","3":"8","4":"9","5":"0","6":"1","7":"2","8":"3","9":"4"}
    for ch in set(s):
        if mirrors[ch] in s:
            mirror_pairs += 1
    mirror_pairs //= 2
    digit_family = "".join(sorted(set(s)))
    sorted_seed = "".join(sorted(s))
    return {
        "seed": s,
        "seed_sorted": sorted_seed,
        "seed_digit_family": digit_family,
        "seed_shape": shape,
        "seed_sum": str(sm),
        "seed_sum_bucket": "sum_00_09" if sm <= 9 else "sum_10_13" if sm <= 13 else "sum_14_17" if sm <= 17 else "sum_18_21" if sm <= 21 else "sum_22_plus",
        "seed_sum_mod5": str(sm % 5),
        "seed_sum_end": str(sm % 10),
        "seed_high_count": str(high),
        "seed_low_count": str(low),
        "seed_even_count": str(even),
        "seed_odd_count": str(odd),
        "seed_parity": "".join("E" if v % 2 == 0 else "O" for v in vals),
        "seed_highlow": "".join("H" if v >= 5 else "L" for v in vals),
        "seed_spread": str(spread),
        "seed_spread_bucket": "spread_0_2" if spread <= 2 else "spread_3_4" if spread <= 4 else "spread_5_6" if spread <= 6 else "spread_7_plus",
        "seed_consec_links": str(consec_links),
        "seed_mirror_pairs": str(mirror_pairs),
        "seed_has0": str("0" in s),
        "seed_has1": str("1" in s),
        "seed_has2": str("2" in s),
        "seed_has3": str("3" in s),
        "seed_has4": str("4" in s),
        "seed_has5": str("5" in s),
        "seed_has6": str("6" in s),
        "seed_has7": str("7" in s),
        "seed_has8": str("8" in s),
        "seed_has9": str("9" in s),
    }


def _prepare_history(hist: pd.DataFrame) -> pd.DataFrame:
    result_col = _find_col(hist, ["Result4", "Result", "Winning Number", "WinningNumber", "Number"])
    date_col = _find_col(hist, ["Date", "DrawDate", "draw_date"])
    state_col = _find_col(hist, ["State"])
    game_col = _find_col(hist, ["Game"])
    stream_col = _find_col(hist, ["StreamKey", "Stream", "stream_key"])
    if result_col is None:
        raise ValueError("Could not find Result4/Result column.")
    if date_col is None:
        raise ValueError("Could not find Date column.")
    out = hist.copy()
    out["Date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["Date"]).copy()
    out["Result4"] = out[result_col].map(_norm4)
    out = out[out["Result4"].str.len() == 4].copy()
    if stream_col and stream_col in out.columns:
        out["StreamKey"] = out[stream_col].astype(str)
    elif state_col and game_col:
        out["StreamKey"] = out[state_col].astype(str) + " | " + out[game_col].astype(str)
    else:
        out["StreamKey"] = "ALL_STREAMS"
    if state_col and state_col in out.columns:
        out["State"] = out[state_col].astype(str)
    else:
        out["State"] = out["StreamKey"].astype(str).str.split(" | ", regex=False).str[0]
    if game_col and game_col in out.columns:
        out["Game"] = out[game_col].astype(str)
    else:
        out["Game"] = out["StreamKey"].astype(str).str.split(" | ", regex=False).str[-1]
    fam = out["Result4"].map(classify_aabc)
    out["ActualCore"] = fam.map(lambda x: x[0])
    out["ActualMember"] = fam.map(lambda x: x[1])
    out["RepeatDigit"] = fam.map(lambda x: x[2])
    out["IsAABC"] = fam.map(lambda x: bool(x[3]))
    out = out.sort_values(["StreamKey", "Date"]).reset_index(drop=True)
    out["Seed"] = out.groupby("StreamKey")["Result4"].shift(1)
    out["SeedDate"] = out.groupby("StreamKey")["Date"].shift(1)
    out["GapDaysFromSeed"] = (out["Date"] - out["SeedDate"]).dt.days
    out = out.dropna(subset=["Seed"]).copy()
    feat = pd.DataFrame([_seed_features(x) for x in out["Seed"]], index=out.index)
    out = pd.concat([out, feat], axis=1)
    out["DayOfWeek"] = out["Date"].dt.day_name()
    out["Month"] = out["Date"].dt.month.astype(str).str.zfill(2)
    out["IsWeekend"] = out["Date"].dt.dayofweek.isin([5, 6]).astype(str)
    return out.reset_index(drop=True)


def _add_lift_cols(df: pd.DataFrame, hit_col="hit_count", sample_col="sample_size", baseline_col="baseline_rate") -> pd.DataFrame:
    out = df.copy()
    out["hit_rate"] = np.where(out[sample_col] > 0, out[hit_col] / out[sample_col], np.nan)
    out["lift"] = out["hit_rate"] - out[baseline_col]
    out["lift_pct_points"] = (out["lift"] * 100).round(3)
    out["hit_rate_pct"] = (out["hit_rate"] * 100).round(3)
    out["baseline_rate_pct"] = (out[baseline_col] * 100).round(3)
    out["relative_lift_x"] = np.where(out[baseline_col] > 0, out["hit_rate"] / out[baseline_col], np.nan)
    out["relative_lift_x"] = out["relative_lift_x"].round(3)
    return out


def _add_stability_cols(
    df: pd.DataFrame,
    min_display_sample: int = 25,
    candidate_sample: int = 50,
    stable_sample: int = 100,
    strong_sample: int = 250,
    min_hits_for_signal: int = 5,
    min_relative_lift_for_signal: float = 1.5,
) -> pd.DataFrame:
    """
    v4 stability layer. This does NOT change mined counts. It only labels/ranks signals so
    tiny high-lift rows do not outrank larger, more reliable patterns.
    """
    out = df.copy()
    sample = pd.to_numeric(out.get("sample_size", 0), errors="coerce").fillna(0).astype(float)
    hits = pd.to_numeric(out.get("hit_count", 0), errors="coerce").fillna(0).astype(float)
    hit_rate = pd.to_numeric(out.get("hit_rate", 0), errors="coerce").fillna(0).astype(float)
    rel_lift = pd.to_numeric(out.get("relative_lift_x", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    lift_pp = pd.to_numeric(out.get("lift_pct_points", 0), errors="coerce").fillna(0).astype(float)

    out["stability_weight"] = np.minimum(1.0, sample / max(float(stable_sample), 1.0)).round(4)
    out["hit_weight"] = np.minimum(1.0, hits / max(float(min_hits_for_signal), 1.0)).round(4)
    out["weighted_lift_pct_points"] = (lift_pp * out["stability_weight"]).round(4)

    # Conservative confidence: rewards lift, hit rate, hits, and sample stability.
    out["confidence_score"] = (
        out["weighted_lift_pct_points"].clip(lower=0)
        * np.log1p(hits)
        * hit_rate.clip(lower=0)
        * out["hit_weight"]
    ).round(6)

    def tier(n):
        try:
            n = float(n)
        except Exception:
            n = 0
        if n < min_display_sample:
            return "HIDE_BELOW_DISPLAY_MIN"
        if n < candidate_sample:
            return "EXPLORATORY_25_49" if min_display_sample >= 25 else "EXPLORATORY"
        if n < stable_sample:
            return "CANDIDATE_50_99"
        if n < strong_sample:
            return "STABLE_100_249"
        return "STRONG_STABLE_250_PLUS"

    out["sample_tier"] = sample.map(tier)
    out["meets_signal_floor"] = (
        (sample >= float(candidate_sample))
        & (hits >= float(min_hits_for_signal))
        & (rel_lift >= float(min_relative_lift_for_signal))
        & (lift_pp > 0)
    )
    return out


def _core_baseline(trans: pd.DataFrame) -> pd.DataFrame:
    allcores = pd.DataFrame({"ActualCore": all_120_cores()})
    total = len(trans)
    hits = trans[trans["IsAABC"]].groupby("ActualCore").size().rename("core_hits").reset_index()
    base = allcores.merge(hits, on="ActualCore", how="left").fillna({"core_hits": 0})
    base["core_hits"] = base["core_hits"].astype(int)
    base["total_transitions"] = total
    base["baseline_rate"] = np.where(total > 0, base["core_hits"] / total, 0.0)
    return base.rename(columns={"ActualCore": "core_id"})


def build_profiles(trans: pd.DataFrame, min_sample: int = 25, top_n_rule_candidates: int = 500, candidate_sample: int = 50, stable_sample: int = 100, strong_sample: int = 250, min_hits_for_signal: int = 5, min_relative_lift_for_signal: float = 1.5):
    allcores = all_120_cores()
    baseline = _core_baseline(trans)
    base_lookup = dict(zip(baseline["core_id"], baseline["baseline_rate"]))

    # Core profiles.
    core_profiles = baseline.copy()
    core_profiles["members"] = core_profiles["core_id"].map(lambda x: ",".join(core_members(x)))
    core_profiles = core_profiles.sort_values(["core_hits", "core_id"], ascending=[False, True]).reset_index(drop=True)
    core_profiles.insert(0, "core_rank_by_frequency", range(1, len(core_profiles) + 1))
    core_profiles["baseline_rate_pct"] = (core_profiles["baseline_rate"] * 100).round(3)

    # Stream profiles: total rows + AABC rate.
    stream_total = trans.groupby("StreamKey").size().rename("stream_total_transitions").reset_index()
    stream_aabc = trans[trans["IsAABC"]].groupby("StreamKey").size().rename("stream_aabc_hits").reset_index()
    stream_profiles = stream_total.merge(stream_aabc, on="StreamKey", how="left").fillna({"stream_aabc_hits": 0})
    stream_profiles["stream_aabc_hits"] = stream_profiles["stream_aabc_hits"].astype(int)
    stream_profiles["stream_aabc_rate_pct"] = (stream_profiles["stream_aabc_hits"] / stream_profiles["stream_total_transitions"] * 100).round(3)
    stream_profiles = stream_profiles.sort_values(["stream_aabc_hits", "stream_aabc_rate_pct", "StreamKey"], ascending=[False, False, True])

    # Stream×Core profiles.
    stc_total = trans.groupby("StreamKey").size().rename("sample_size").reset_index()
    stc_hits = trans[trans["IsAABC"]].groupby(["StreamKey", "ActualCore"]).size().rename("hit_count").reset_index().rename(columns={"ActualCore": "core_id"})
    streams = stc_total["StreamKey"].tolist()
    grid = pd.MultiIndex.from_product([streams, allcores], names=["StreamKey", "core_id"]).to_frame(index=False)
    stream_core = grid.merge(stc_total, on="StreamKey", how="left").merge(stc_hits, on=["StreamKey", "core_id"], how="left").fillna({"hit_count": 0})
    stream_core["hit_count"] = stream_core["hit_count"].astype(int)
    stream_core["baseline_rate"] = stream_core["core_id"].map(base_lookup).fillna(0.0)
    stream_core = _add_lift_cols(stream_core)
    stream_core = _add_stability_cols(stream_core, int(min_sample), int(candidate_sample), int(stable_sample), int(strong_sample), int(min_hits_for_signal), float(min_relative_lift_for_signal))
    stream_core = stream_core[stream_core["sample_size"] >= int(min_sample)].copy()
    stream_core = stream_core.sort_values(["confidence_score", "weighted_lift_pct_points", "hit_count", "sample_size"], ascending=[False, False, False, False])

    # Seed trait profiles.
    seed_trait_cols = [
        "seed_sorted", "seed_digit_family", "seed_shape", "seed_sum_bucket", "seed_sum_mod5", "seed_sum_end",
        "seed_high_count", "seed_even_count", "seed_parity", "seed_highlow", "seed_spread_bucket",
        "seed_consec_links", "seed_mirror_pairs", "DayOfWeek", "Month", "IsWeekend",
        "seed_has0", "seed_has1", "seed_has2", "seed_has3", "seed_has4", "seed_has5", "seed_has6", "seed_has7", "seed_has8", "seed_has9"
    ]
    seed_trait_frames = []
    for col in seed_trait_cols:
        total = trans.groupby(col).size().rename("sample_size").reset_index().rename(columns={col: "trait_value"})
        total["trait_name"] = col
        hits = trans[trans["IsAABC"]].groupby([col, "ActualCore"]).size().rename("hit_count").reset_index().rename(columns={col: "trait_value", "ActualCore": "core_id"})
        grid2 = total[["trait_name", "trait_value", "sample_size"]].merge(pd.DataFrame({"core_id": allcores}), how="cross")
        prof = grid2.merge(hits, on=["trait_value", "core_id"], how="left").fillna({"hit_count": 0})
        prof["hit_count"] = prof["hit_count"].astype(int)
        prof["baseline_rate"] = prof["core_id"].map(base_lookup).fillna(0.0)
        seed_trait_frames.append(prof)
    seed_trait_core_lift = pd.concat(seed_trait_frames, ignore_index=True)
    seed_trait_core_lift = seed_trait_core_lift[seed_trait_core_lift["sample_size"] >= int(min_sample)].copy()
    seed_trait_core_lift = _add_lift_cols(seed_trait_core_lift)
    seed_trait_core_lift = _add_stability_cols(seed_trait_core_lift, int(min_sample), int(candidate_sample), int(stable_sample), int(strong_sample), int(min_hits_for_signal), float(min_relative_lift_for_signal))
    seed_trait_core_lift = seed_trait_core_lift.sort_values(["confidence_score", "weighted_lift_pct_points", "hit_count", "sample_size"], ascending=[False, False, False, False])

    # Stream×SeedTrait×Core profiles for high-value context-specific signals.
    stream_trait_cols = ["seed_shape", "seed_sum_bucket", "seed_parity", "seed_highlow", "seed_spread_bucket", "seed_consec_links", "DayOfWeek"]
    st_trait_frames = []
    for col in stream_trait_cols:
        total = trans.groupby(["StreamKey", col]).size().rename("sample_size").reset_index().rename(columns={col: "trait_value"})
        total["trait_name"] = col
        hits = trans[trans["IsAABC"]].groupby(["StreamKey", col, "ActualCore"]).size().rename("hit_count").reset_index().rename(columns={col: "trait_value", "ActualCore": "core_id"})
        grid3 = total[["StreamKey", "trait_name", "trait_value", "sample_size"]].merge(pd.DataFrame({"core_id": allcores}), how="cross")
        prof = grid3.merge(hits, on=["StreamKey", "trait_value", "core_id"], how="left").fillna({"hit_count": 0})
        prof["hit_count"] = prof["hit_count"].astype(int)
        prof["baseline_rate"] = prof["core_id"].map(base_lookup).fillna(0.0)
        st_trait_frames.append(prof)
    stream_seed_trait_core_lift = pd.concat(st_trait_frames, ignore_index=True)
    stream_seed_trait_core_lift = stream_seed_trait_core_lift[stream_seed_trait_core_lift["sample_size"] >= int(min_sample)].copy()
    stream_seed_trait_core_lift = _add_lift_cols(stream_seed_trait_core_lift)
    stream_seed_trait_core_lift = _add_stability_cols(stream_seed_trait_core_lift, int(min_sample), int(candidate_sample), int(stable_sample), int(strong_sample), int(min_hits_for_signal), float(min_relative_lift_for_signal))
    stream_seed_trait_core_lift = stream_seed_trait_core_lift.sort_values(["confidence_score", "weighted_lift_pct_points", "hit_count", "sample_size"], ascending=[False, False, False, False])

    # Cadence profiles: previous same-core gap per stream/core event history.
    aabc = trans[trans["IsAABC"]].copy().sort_values(["StreamKey", "ActualCore", "Date"])
    aabc["PrevSameCoreDate"] = aabc.groupby(["StreamKey", "ActualCore"])["Date"].shift(1)
    aabc["SameCoreGapDays"] = (aabc["Date"] - aabc["PrevSameCoreDate"]).dt.days
    def gap_bucket(x):
        if pd.isna(x): return "first_seen"
        x = int(x)
        if x <= 7: return "gap_001_007"
        if x <= 14: return "gap_008_014"
        if x <= 30: return "gap_015_030"
        if x <= 60: return "gap_031_060"
        if x <= 120: return "gap_061_120"
        return "gap_121_plus"
    aabc["SameCoreGapBucket"] = aabc["SameCoreGapDays"].map(gap_bucket)
    cadence_core = aabc.groupby(["ActualCore", "SameCoreGapBucket"]).size().rename("hit_count").reset_index().rename(columns={"ActualCore": "core_id"})
    cadence_total = aabc.groupby("SameCoreGapBucket").size().rename("sample_size").reset_index()
    cadence_core = cadence_core.merge(cadence_total, on="SameCoreGapBucket", how="left")
    cadence_core["baseline_rate"] = cadence_core["core_id"].map(base_lookup).fillna(0.0)
    cadence_core = _add_lift_cols(cadence_core)
    cadence_core = _add_stability_cols(cadence_core, int(min_sample), int(candidate_sample), int(stable_sample), int(strong_sample), int(min_hits_for_signal), float(min_relative_lift_for_signal))
    cadence_core = cadence_core.sort_values(["confidence_score", "weighted_lift_pct_points", "hit_count"], ascending=[False, False, False])

    # Member role profile by core.
    member_hits = trans[trans["IsAABC"]].groupby(["ActualCore", "ActualMember"]).size().rename("hit_count").reset_index().rename(columns={"ActualCore": "core_id", "ActualMember": "member"})
    matrix_rows = []
    for c in allcores:
        sub = member_hits[member_hits["core_id"] == c]
        lookup = dict(zip(sub["member"], sub["hit_count"]))
        total_hits = sum(lookup.values())
        rows = []
        for m in core_members(c):
            rows.append({"core_id": c, "member": m, "member_hits": int(lookup.get(m, 0)), "core_hits": int(total_hits)})
        tmp = pd.DataFrame(rows).sort_values(["member_hits", "member"], ascending=[False, True]).reset_index(drop=True)
        labels = ["strongest_candidate", "middle_candidate", "suppressed_candidate"]
        tmp["matrix_slot"] = [labels[i] for i in range(len(tmp))]
        tmp["member_share_pct"] = np.where(tmp["core_hits"] > 0, tmp["member_hits"] / tmp["core_hits"] * 100, 0).round(3)
        matrix_rows.append(tmp)
    member_role_profile = pd.concat(matrix_rows, ignore_index=True)

    # Rule candidates: strongest global seed traits + stream-core + stream-seed traits.
    cand1 = seed_trait_core_lift.copy()
    cand1["rule_type"] = "seed_trait_core"
    cand1["rule"] = cand1["trait_name"].astype(str) + "==" + cand1["trait_value"].astype(str) + " -> core " + cand1["core_id"].astype(str)
    cand2 = stream_core.copy()
    cand2["rule_type"] = "stream_core"
    cand2["rule"] = "StreamKey==" + cand2["StreamKey"].astype(str) + " -> core " + cand2["core_id"].astype(str)
    cand3 = stream_seed_trait_core_lift.copy()
    cand3["rule_type"] = "stream_seed_trait_core"
    cand3["rule"] = "StreamKey==" + cand3["StreamKey"].astype(str) + " AND " + cand3["trait_name"].astype(str) + "==" + cand3["trait_value"].astype(str) + " -> core " + cand3["core_id"].astype(str)
    common_cols = ["rule_type", "rule", "core_id", "sample_size", "sample_tier", "hit_count", "hit_rate_pct", "baseline_rate_pct", "lift_pct_points", "relative_lift_x", "stability_weight", "weighted_lift_pct_points", "confidence_score", "meets_signal_floor"]
    affinity_rule_candidates = pd.concat([cand1[common_cols], cand2[common_cols], cand3[common_cols]], ignore_index=True)
    affinity_rule_candidates = affinity_rule_candidates[(affinity_rule_candidates["sample_size"] >= int(min_sample)) & (affinity_rule_candidates["hit_count"] > 0)].copy()
    affinity_rule_candidates = affinity_rule_candidates.sort_values(["meets_signal_floor", "confidence_score", "weighted_lift_pct_points", "hit_count", "sample_size"], ascending=[False, False, False, False, False]).head(int(top_n_rule_candidates)).reset_index(drop=True)
    affinity_rule_candidates.insert(0, "candidate_rank", range(1, len(affinity_rule_candidates) + 1))

    # Latest seed preview: current latest seed per stream against profile candidates. Not a playlist.
    latest = trans.sort_values("Date").groupby("StreamKey", as_index=False).tail(1).copy()
    preview_rows = []
    top_trait_lookup = seed_trait_core_lift[seed_trait_core_lift["sample_size"] >= int(min_sample)].copy()
    for _, r in latest.iterrows():
        stream = r["StreamKey"]
        # combine stream-core lift + matching seed trait lifts
        score = pd.DataFrame({"core_id": allcores})
        score["affinity_score"] = 0.0
        score["support_notes"] = ""
        sc = stream_core[stream_core["StreamKey"] == stream][["core_id", "lift_pct_points", "weighted_lift_pct_points", "confidence_score", "hit_rate_pct", "sample_size"]].copy()
        if not sc.empty:
            sc = sc.rename(columns={"lift_pct_points": "stream_lift_pp", "weighted_lift_pct_points": "stream_weighted_lift_pp", "confidence_score": "stream_confidence_score", "hit_rate_pct": "stream_hit_rate_pct", "sample_size": "stream_sample"})
            score = score.merge(sc, on="core_id", how="left")
            score["affinity_score"] += score["stream_weighted_lift_pp"].fillna(0)
        else:
            score["stream_lift_pp"] = 0.0
            score["stream_weighted_lift_pp"] = 0.0
            score["stream_confidence_score"] = 0.0
            score["stream_hit_rate_pct"] = np.nan
            score["stream_sample"] = np.nan
        matching_traits = []
        for col in seed_trait_cols:
            if col in r.index:
                val = str(r[col])
                tmp = top_trait_lookup[(top_trait_lookup["trait_name"] == col) & (top_trait_lookup["trait_value"].astype(str) == val)][["core_id", "weighted_lift_pct_points"]]
                if not tmp.empty:
                    tmp = tmp.rename(columns={"weighted_lift_pct_points": f"{col}_weighted_lift_pp"})
                    score = score.merge(tmp, on="core_id", how="left")
                    score["affinity_score"] += score[f"{col}_weighted_lift_pp"].fillna(0) * 0.25
                    matching_traits.append(col)
        score["StreamKey"] = stream
        score["Seed"] = r["Seed"]
        score["SeedDate"] = r["SeedDate"]
        score["PlayDateCandidate"] = r["Date"]
        score["matching_trait_count"] = len(matching_traits)
        top = score.sort_values(["affinity_score", "stream_confidence_score", "stream_lift_pp"], ascending=[False, False, False]).head(10)
        preview_rows.append(top[["StreamKey", "Seed", "SeedDate", "PlayDateCandidate", "core_id", "affinity_score", "stream_lift_pp", "stream_weighted_lift_pp", "stream_confidence_score", "stream_hit_rate_pct", "stream_sample", "matching_trait_count"]])
    latest_seed_affinity_preview = pd.concat(preview_rows, ignore_index=True) if preview_rows else pd.DataFrame()

    summary = pd.DataFrame([
        {"metric": "valid_seed_to_next_result_transitions", "value": len(trans)},
        {"metric": "aabc_winning_transitions", "value": int(trans["IsAABC"].sum())},
        {"metric": "streams_profiled", "value": int(trans["StreamKey"].nunique())},
        {"metric": "cores_profiled", "value": 120},
        {"metric": "min_display_sample_threshold", "value": int(min_sample)},
        {"metric": "candidate_sample_threshold", "value": int(candidate_sample)},
        {"metric": "stable_sample_threshold", "value": int(stable_sample)},
        {"metric": "strong_sample_threshold", "value": int(strong_sample)},
        {"metric": "min_hits_for_signal", "value": int(min_hits_for_signal)},
        {"metric": "min_relative_lift_for_signal", "value": float(min_relative_lift_for_signal)},
        {"metric": "rule_candidates_exported", "value": int(len(affinity_rule_candidates))},
        {"metric": "rule_candidates_meeting_signal_floor", "value": int(affinity_rule_candidates.get("meets_signal_floor", pd.Series(dtype=bool)).sum())},
    ])

    return {
        "summary": summary,
        "core_profiles": core_profiles,
        "stream_profiles": stream_profiles,
        "stream_core_profiles": stream_core,
        "seed_trait_core_lift": seed_trait_core_lift,
        "stream_seed_trait_core_lift": stream_seed_trait_core_lift,
        "cadence_core_lift": cadence_core,
        "member_role_profiles": member_role_profile,
        "affinity_rule_candidates": affinity_rule_candidates,
        "latest_seed_affinity_preview": latest_seed_affinity_preview,
        "prepared_seed_transition_rows": trans,
    }


def freeze_exports(outputs: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    frozen = {}
    for name, df in outputs.items():
        if isinstance(df, pd.DataFrame):
            frozen[f"{name}.csv"] = df.to_csv(index=False).encode("utf-8")
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for fname, data in frozen.items():
            zf.writestr(fname, data)
        zf.writestr("README.txt", (BUILD_MARKER + "\nLab-only profile builder. No daily-play logic included.\n").encode("utf-8"))
    frozen["core_affinity_lab_v4_all_outputs.zip"] = bio.getvalue()
    return frozen


with st.sidebar:
    st.header("Inputs")
    history_file = st.file_uploader("Upload full clean Pick-4 history", type=["csv", "txt", "tsv"])
    min_sample = st.select_slider("Minimum sample size to display", options=[10, 15, 25, 50, 100], value=25)
    candidate_sample = st.number_input("Candidate signal sample threshold", min_value=10, max_value=1000, value=50, step=5)
    stable_sample = st.number_input("Stable signal sample threshold", min_value=25, max_value=2000, value=100, step=25)
    strong_sample = st.number_input("Strong stable sample threshold", min_value=50, max_value=5000, value=250, step=50)
    min_hits_for_signal = st.number_input("Minimum hits for signal floor", min_value=1, max_value=100, value=5, step=1)
    min_relative_lift_for_signal = st.number_input("Minimum relative lift × for signal floor", min_value=1.0, max_value=25.0, value=1.5, step=0.1)
    top_n_rules = st.number_input("Max rule candidates to export", min_value=50, max_value=5000, value=1000, step=50)
    st.caption("Tiers: below display min hidden; 25–49 exploratory; 50–99 candidate; 100–249 stable; 250+ strong stable.")
    run_btn = st.button("Build weighted profiles", type="primary", use_container_width=True)

if "v4_outputs" not in st.session_state:
    st.session_state["v4_outputs"] = None
if "v4_export_bytes" not in st.session_state:
    st.session_state["v4_export_bytes"] = None

if run_btn:
    if history_file is None:
        st.error("Upload full history first.")
        st.stop()
    try:
        prog = st.progress(0, text="Loading history")
        hist = _read_upload(history_file)
        prog.progress(15, text=f"Preparing seed transitions from {len(hist):,} rows")
        trans = _prepare_history(hist)
        prog.progress(35, text="Building stream/core/seed/cadence profiles")
        outputs = build_profiles(trans, min_sample=int(min_sample), top_n_rule_candidates=int(top_n_rules), candidate_sample=int(candidate_sample), stable_sample=int(stable_sample), strong_sample=int(strong_sample), min_hits_for_signal=int(min_hits_for_signal), min_relative_lift_for_signal=float(min_relative_lift_for_signal))
        prog.progress(85, text="Freezing downloads")
        exports = freeze_exports(outputs)
        st.session_state["v4_outputs"] = outputs
        st.session_state["v4_export_bytes"] = exports
        prog.progress(100, text="Done")
        st.success("Profiles built and downloads frozen. Download buttons will not rebuild the app.")
    except Exception as e:
        st.error(f"Profile build failed: {e}")
        st.stop()

outputs = st.session_state.get("v4_outputs")
exports = st.session_state.get("v4_export_bytes")

if not isinstance(outputs, dict):
    st.warning("Upload history and click Build profiles.")
    st.stop()

st.subheader("Summary")
st.dataframe(outputs["summary"], use_container_width=True, hide_index=True)

st.subheader("Core Profiles")
st.caption("Historical baseline by core. This is profile context, not a daily prediction.")
st.dataframe(outputs["core_profiles"].head(120), use_container_width=True, hide_index=True)

st.subheader("Stream Profiles")
st.dataframe(outputs["stream_profiles"], use_container_width=True, hide_index=True)

st.subheader("Stream × Core Profiles — strongest lift")
st.dataframe(outputs["stream_core_profiles"].head(500), use_container_width=True, hide_index=True)

st.subheader("Seed Trait × Core Lift — strongest global seed traits")
st.dataframe(outputs["seed_trait_core_lift"].head(500), use_container_width=True, hide_index=True)

st.subheader("Stream × Seed Trait × Core Lift — strongest context traits")
st.dataframe(outputs["stream_seed_trait_core_lift"].head(500), use_container_width=True, hide_index=True)

st.subheader("Cadence × Core Lift")
st.dataframe(outputs["cadence_core_lift"], use_container_width=True, hide_index=True)

st.subheader("Member Role Profiles")
st.dataframe(outputs["member_role_profiles"].head(360), use_container_width=True, hide_index=True)

st.subheader("Affinity Rule Candidates")
st.caption("Ranked by signal floor + confidence_score + weighted lift. Candidate rules still require later validation before production use.")
st.dataframe(outputs["affinity_rule_candidates"].head(1000), use_container_width=True, hide_index=True)

st.subheader("Latest Seed Affinity Preview")
st.caption("Preview only. Not a playlist and not a betting recommendation.")
st.dataframe(outputs["latest_seed_affinity_preview"], use_container_width=True, hide_index=True)

st.subheader("Downloads")
if isinstance(exports, dict):
    st.download_button(
        "Download ALL v4 outputs ZIP",
        data=exports.get("core_affinity_lab_v4_all_outputs.zip", b""),
        file_name="core_affinity_lab_v4_all_outputs.zip",
        mime="application/zip",
        use_container_width=True,
        key="dl_v4_all_zip",
    )
    for fname in [
        "summary.csv",
        "core_profiles.csv",
        "stream_profiles.csv",
        "stream_core_profiles.csv",
        "seed_trait_core_lift.csv",
        "stream_seed_trait_core_lift.csv",
        "cadence_core_lift.csv",
        "member_role_profiles.csv",
        "affinity_rule_candidates.csv",
        "latest_seed_affinity_preview.csv",
    ]:
        if fname in exports:
            st.download_button(
                f"Download {fname}",
                data=exports[fname],
                file_name=fname,
                mime="text/csv",
                use_container_width=True,
                key=f"dl_v4_{fname}",
            )
