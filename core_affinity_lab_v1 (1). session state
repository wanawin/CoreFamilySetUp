#!/usr/bin/env python3
# Core Affinity Lab v1
# Mines seed/stream/core affinity for all Pick-4 AABC core families.
# Separate lab app only. No daily playlist logic, no cuts, no RTE, no B1Z0, no rescues.

from __future__ import annotations

import io
import itertools
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core_affinity_lab_v1__2026-06-16_ALL_CORES_STREAM_SEED_PROFILE"

# -----------------------------------------------------------------------------
# LOCKED PURPOSE
# -----------------------------------------------------------------------------
# This app is a LAB / RESEARCH app only.
# It does NOT produce a betting playlist.
# It does NOT apply budget cuts.
# It does NOT apply RTE, B1Z0, B1Z1, B2Z2, ZLT, rescues, or straight logic.
# It studies which seed traits and streams historically favor each core family.
# All reported accuracy values are historical backtest / holdout measurements,
# not guarantees and not simulated future results.
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Core Affinity Lab v1", layout="wide")
st.title("Core Affinity Lab v1")
st.caption(BUILD_MARKER)
st.info(
    "Research-only lab: mines seed-trait, stream, member, and cadence affinity for all 120 AABC core families. "
    "No playlist, no cuts, no RTE, no B1Z0, no rescues, and no betting logic."
)


# =============================================================================
# Basic parsing / family utilities
# =============================================================================

def read_upload(file) -> pd.DataFrame:
    raw = file.getvalue().decode("utf-8", errors="replace")
    name = str(file.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.StringIO(raw), dtype=str)
    if name.endswith(".tsv"):
        return pd.read_csv(io.StringIO(raw), sep="\t", dtype=str)
    # Try comma first, then tab, then python sniff.
    for sep in [",", "\t", None]:
        try:
            return pd.read_csv(io.StringIO(raw), sep=sep, engine="python", dtype=str)
        except Exception:
            pass
    raise ValueError("Could not parse uploaded history file.")


def norm4(x) -> str:
    digits = re.findall(r"\d", str(x))
    if len(digits) < 4:
        return ""
    return "".join(digits[:4])


def sorted_digits4(x) -> str:
    s = norm4(x)
    if len(s) != 4:
        return ""
    return "".join(sorted(s))


def all_120_cores() -> List[str]:
    return ["".join(c) for c in itertools.combinations("0123456789", 3)]


def core_members(core_id: str) -> List[str]:
    digs = list(str(core_id))
    return sorted("".join(sorted(digs + [d])) for d in digs)


def classify_aabc(result4: str) -> Tuple[str, str, str, bool]:
    s = sorted_digits4(result4)
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


def infer_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    date_col = None
    for c in ["date", "drawdate", "draw_date", "play_date"]:
        if c in cols_lower:
            date_col = cols_lower[c]
            break
    result_col = None
    for c in ["result4", "result", "winning number", "winningnumber", "number"]:
        if c in cols_lower:
            result_col = cols_lower[c]
            break
    stream_col = None
    for c in ["streamkey", "stream", "stream_key"]:
        if c in cols_lower:
            stream_col = cols_lower[c]
            break
    if stream_col is None:
        state = cols_lower.get("state")
        game = cols_lower.get("game")
        if state and game:
            stream_col = "__StreamKey_factory"
            df[stream_col] = df[state].fillna("").astype(str).str.strip() + " | " + df[game].fillna("").astype(str).str.strip()
    if date_col is None or result_col is None or stream_col is None:
        raise ValueError("History must include Date, Result/Result4, and StreamKey or State+Game columns.")
    return date_col, result_col, stream_col


# =============================================================================
# Seed traits / cadence traits
# =============================================================================

def digit_list4(x: str) -> List[int]:
    s = norm4(x)
    if len(s) != 4:
        return [0, 0, 0, 0]
    return [int(d) for d in s]


def seed_traits(seed: str) -> Dict[str, str]:
    digs = digit_list4(seed)
    s = "".join(str(d) for d in digs)
    cnt = Counter(s)
    vals = sorted(digs)
    total = sum(digs)
    spread = max(digs) - min(digs)
    evens = sum(1 for d in digs if d % 2 == 0)
    highs = sum(1 for d in digs if d >= 5)
    unique = len(cnt)
    repeats = sorted(cnt.values(), reverse=True)

    if repeats == [4]:
        shape = "quad"
    elif repeats == [3, 1]:
        shape = "triple"
    elif repeats == [2, 2]:
        shape = "double_double"
    elif repeats == [2, 1, 1]:
        shape = "one_pair"
    else:
        shape = "all_unique"

    consec_links = 0
    for a, b in zip(vals, vals[1:]):
        if b - a == 1:
            consec_links += 1

    mirror_pairs = 0
    mirror_set = {("0", "5"), ("1", "6"), ("2", "7"), ("3", "8"), ("4", "9")}
    chars = set(s)
    for a, b in mirror_set:
        if a in chars and b in chars:
            mirror_pairs += 1

    traits = {
        "seed_sum_bucket": bucket_num(total, [9, 13, 17, 21], ["sum_00_09", "sum_10_13", "sum_14_17", "sum_18_21", "sum_22_plus"]),
        "seed_sum_mod5": str(total % 5),
        "seed_sum_end": str(total % 10),
        "seed_parity": "".join("E" if d % 2 == 0 else "O" for d in digs),
        "seed_even_count": str(evens),
        "seed_highlow": "".join("H" if d >= 5 else "L" for d in digs),
        "seed_high_count": str(highs),
        "seed_shape": shape,
        "seed_unique_count": str(unique),
        "seed_spread_bucket": bucket_num(spread, [2, 4, 6], ["spread_0_2", "spread_3_4", "spread_5_6", "spread_7_9"]),
        "seed_consec_links": str(consec_links),
        "seed_mirror_pairs": str(mirror_pairs),
        "seed_first_digit": str(digs[0]),
        "seed_last_digit": str(digs[-1]),
        "seed_min_digit": str(min(digs)),
        "seed_max_digit": str(max(digs)),
    }
    for d in range(10):
        traits[f"seed_has_{d}"] = "1" if str(d) in s else "0"
    return traits


def bucket_num(x: int, cuts: List[int], labels: List[str]) -> str:
    for cut, lab in zip(cuts, labels):
        if x <= cut:
            return lab
    return labels[-1]


def add_seed_transition_rows(hist: pd.DataFrame, date_col: str, result_col: str, stream_col: str) -> pd.DataFrame:
    df = hist.copy()
    df["Date_affinity"] = pd.to_datetime(df[date_col], errors="coerce")
    df["Result4_affinity"] = df[result_col].map(norm4)
    df["StreamKey_affinity"] = df[stream_col].astype(str).str.strip()
    df = df.dropna(subset=["Date_affinity"]).copy()
    df = df[df["Result4_affinity"].str.len() == 4].copy()
    df = df.sort_values(["StreamKey_affinity", "Date_affinity"]).reset_index(drop=True)

    df["SeedResult4"] = df.groupby("StreamKey_affinity")["Result4_affinity"].shift(1)
    df["SeedDate"] = df.groupby("StreamKey_affinity")["Date_affinity"].shift(1)
    df["DaysSinceSeed"] = (df["Date_affinity"] - df["SeedDate"]).dt.days
    df = df[df["SeedResult4"].fillna("").astype(str).str.len() == 4].copy()

    fam = df["Result4_affinity"].map(classify_aabc)
    df["ActualCore"] = fam.map(lambda x: x[0])
    df["ActualMember"] = fam.map(lambda x: x[1])
    df["ActualRepeatDigit"] = fam.map(lambda x: x[2])
    df["IsAABC"] = fam.map(lambda x: x[3])

    # Calendar/cadence features available at prediction time.
    df["day_of_week"] = df["Date_affinity"].dt.day_name()
    df["month"] = df["Date_affinity"].dt.month.astype(str).str.zfill(2)
    df["days_since_seed_bucket"] = pd.cut(
        pd.to_numeric(df["DaysSinceSeed"], errors="coerce").fillna(0),
        bins=[-1, 1, 2, 3, 7, 9999],
        labels=["gap_1", "gap_2", "gap_3", "gap_4_7", "gap_8_plus"],
    ).astype(str)

    trait_rows = [seed_traits(x) for x in df["SeedResult4"]]
    traits_df = pd.DataFrame(trait_rows, index=df.index)
    out = pd.concat([df.reset_index(drop=True), traits_df.reset_index(drop=True)], axis=1)
    out["event_id"] = np.arange(len(out))
    return out


# =============================================================================
# Mining profiles
# =============================================================================

@dataclass
class LabConfig:
    min_sample: int = 25
    smoothing: float = 5.0
    test_fraction: float = 0.30
    use_stream_score: bool = True
    use_trait_score: bool = True
    use_cadence_score: bool = True
    top_k_event_export: int = 10


def train_test_split_by_date(events: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = sorted(events["Date_affinity"].dropna().unique())
    if not dates:
        return events.copy(), events.iloc[0:0].copy(), pd.NaT
    idx = max(1, int(len(dates) * (1.0 - float(test_fraction))))
    idx = min(idx, len(dates) - 1) if len(dates) > 1 else 0
    cutoff = pd.Timestamp(dates[idx])
    train = events[events["Date_affinity"] < cutoff].copy()
    test = events[events["Date_affinity"] >= cutoff].copy()
    if train.empty or test.empty:
        # fallback: 70/30 row split
        cut = int(len(events) * 0.70)
        train = events.iloc[:cut].copy()
        test = events.iloc[cut:].copy()
        cutoff = test["Date_affinity"].min() if not test.empty else pd.NaT
    return train, test, cutoff


def build_core_member_profile(events: pd.DataFrame, cores: List[str]) -> pd.DataFrame:
    aabc = events[events["IsAABC"]].copy()
    rows = []
    for core in cores:
        sub = aabc[aabc["ActualCore"] == core]
        members = core_members(core)
        total = int(len(sub))
        for m in members:
            rows.append({
                "core_id": core,
                "member": m,
                "member_hit_count": int((sub["ActualMember"] == m).sum()),
                "core_hit_count": total,
                "member_share_within_core_pct": round((sub["ActualMember"] == m).mean() * 100, 2) if total else 0.0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["member_role_by_count"] = df.groupby("core_id")["member_hit_count"].rank(method="first", ascending=False).map({1.0: "strongest_candidate", 2.0: "middle_candidate", 3.0: "suppressed_candidate"})
    return df.sort_values(["core_id", "member_hit_count", "member"], ascending=[True, False, True])


def build_core_rank(events: pd.DataFrame, cores: List[str]) -> pd.DataFrame:
    total_events = len(events)
    aabc = events[events["IsAABC"]].copy()
    total_aabc = len(aabc)
    counts = aabc.groupby("ActualCore").size().to_dict()
    rows = []
    for core in cores:
        hit = int(counts.get(core, 0))
        rows.append({
            "core_id": core,
            "core_hit_count": hit,
            "pct_of_all_seed_transitions": round(hit / total_events * 100, 4) if total_events else 0.0,
            "pct_of_AABC_winners": round(hit / total_aabc * 100, 4) if total_aabc else 0.0,
            "members": ",".join(core_members(core)),
        })
    out = pd.DataFrame(rows).sort_values(["core_hit_count", "core_id"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "core_affinity_build_rank", range(1, len(out) + 1))
    return out


def build_stream_core_profile(train: pd.DataFrame, cores: List[str], smoothing: float) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Tuple[float, int]]]:
    aabc = train[train["IsAABC"]].copy()
    total_aabc = max(1, len(aabc))
    base_counts = aabc.groupby("ActualCore").size().to_dict()
    base_rate = {c: (base_counts.get(c, 0) + smoothing) / (total_aabc + smoothing * len(cores)) for c in cores}

    stream_totals = aabc.groupby("StreamKey_affinity").size().to_dict()
    stream_core_counts = aabc.groupby(["StreamKey_affinity", "ActualCore"]).size().to_dict()
    lookup = {}
    rows = []
    for stream, stotal in stream_totals.items():
        for core in cores:
            hits = int(stream_core_counts.get((stream, core), 0))
            rate = (hits + smoothing * base_rate[core]) / (stotal + smoothing)
            lift = rate / base_rate[core] if base_rate[core] > 0 else 1.0
            lookup[(stream, core)] = (float(lift), int(stotal))
            rows.append({
                "StreamKey": stream,
                "core_id": core,
                "stream_AABC_count": int(stotal),
                "stream_core_hits": hits,
                "stream_core_rate_pct": round(rate * 100, 4),
                "core_base_rate_pct": round(base_rate[core] * 100, 4),
                "stream_core_lift": round(lift, 4),
            })
    df = pd.DataFrame(rows).sort_values(["StreamKey", "stream_core_lift", "stream_core_hits"], ascending=[True, False, False]) if rows else pd.DataFrame()
    return df, lookup


def trait_columns() -> List[str]:
    base = [
        "seed_sum_bucket", "seed_sum_mod5", "seed_sum_end", "seed_parity", "seed_even_count",
        "seed_highlow", "seed_high_count", "seed_shape", "seed_unique_count", "seed_spread_bucket",
        "seed_consec_links", "seed_mirror_pairs", "seed_first_digit", "seed_last_digit",
        "seed_min_digit", "seed_max_digit", "day_of_week", "month", "days_since_seed_bucket",
    ]
    base.extend([f"seed_has_{d}" for d in range(10)])
    return base


def build_trait_core_profile(train: pd.DataFrame, cores: List[str], smoothing: float, min_sample: int) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], Tuple[float, int]]]:
    aabc = train[train["IsAABC"]].copy()
    total_aabc = max(1, len(aabc))
    base_counts = aabc.groupby("ActualCore").size().to_dict()
    base_rate = {c: (base_counts.get(c, 0) + smoothing) / (total_aabc + smoothing * len(cores)) for c in cores}

    rows = []
    lookup: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
    for tcol in trait_columns():
        if tcol not in aabc.columns:
            continue
        totals = aabc.groupby(tcol).size().to_dict()
        core_counts = aabc.groupby([tcol, "ActualCore"]).size().to_dict()
        for val, n in totals.items():
            val_s = str(val)
            for core in cores:
                hits = int(core_counts.get((val, core), 0))
                rate = (hits + smoothing * base_rate[core]) / (int(n) + smoothing)
                lift = rate / base_rate[core] if base_rate[core] > 0 else 1.0
                lookup[(tcol, val_s, core)] = (float(lift), int(n))
                if int(n) >= int(min_sample):
                    rows.append({
                        "trait": tcol,
                        "trait_value": val_s,
                        "core_id": core,
                        "trait_sample_size": int(n),
                        "trait_core_hits": hits,
                        "trait_core_rate_pct": round(rate * 100, 4),
                        "core_base_rate_pct": round(base_rate[core] * 100, 4),
                        "trait_core_lift": round(lift, 4),
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["trait_core_lift", "trait_sample_size", "trait_core_hits"], ascending=[False, False, False])
    return df, lookup


def build_cadence_core_profile(train: pd.DataFrame, cores: List[str], smoothing: float, min_sample: int) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], Tuple[float, int]]]:
    # Cadence v1: stream + day-of-week and stream + gap bucket interaction.
    aabc = train[train["IsAABC"]].copy()
    total_aabc = max(1, len(aabc))
    base_counts = aabc.groupby("ActualCore").size().to_dict()
    base_rate = {c: (base_counts.get(c, 0) + smoothing) / (total_aabc + smoothing * len(cores)) for c in cores}

    interaction_specs = [
        ("stream_x_dow", ["StreamKey_affinity", "day_of_week"]),
        ("stream_x_gap", ["StreamKey_affinity", "days_since_seed_bucket"]),
        ("stream_x_seed_shape", ["StreamKey_affinity", "seed_shape"]),
    ]

    rows = []
    lookup = {}
    for label, cols in interaction_specs:
        temp = aabc.copy()
        temp[label] = temp[cols].astype(str).agg(" || ".join, axis=1)
        totals = temp.groupby(label).size().to_dict()
        core_counts = temp.groupby([label, "ActualCore"]).size().to_dict()
        for val, n in totals.items():
            for core in cores:
                hits = int(core_counts.get((val, core), 0))
                rate = (hits + smoothing * base_rate[core]) / (int(n) + smoothing)
                lift = rate / base_rate[core] if base_rate[core] > 0 else 1.0
                lookup[(label, str(val), core)] = (float(lift), int(n))
                if int(n) >= int(min_sample):
                    rows.append({
                        "cadence_trait": label,
                        "cadence_value": str(val),
                        "core_id": core,
                        "cadence_sample_size": int(n),
                        "cadence_core_hits": hits,
                        "cadence_core_rate_pct": round(rate * 100, 4),
                        "core_base_rate_pct": round(base_rate[core] * 100, 4),
                        "cadence_core_lift": round(lift, 4),
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cadence_core_lift", "cadence_sample_size", "cadence_core_hits"], ascending=[False, False, False])
    return df, lookup


# =============================================================================
# Cross-core backtest scoring
# =============================================================================

def _safe_log_lift(lift: float, cap: float = 4.0) -> float:
    try:
        lift = float(lift)
        if not math.isfinite(lift) or lift <= 0:
            return 0.0
        lift = max(1.0 / cap, min(cap, lift))
        return math.log(lift)
    except Exception:
        return 0.0


def score_event_core(row: pd.Series, core: str, stream_lookup, trait_lookup, cadence_lookup, cfg: LabConfig) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []
    if cfg.use_stream_score:
        lift, n = stream_lookup.get((row["StreamKey_affinity"], core), (1.0, 0))
        score += 1.25 * _safe_log_lift(lift)
        if lift >= 1.25:
            reasons.append(f"stream_lift={lift:.2f}/n={n}")
    if cfg.use_trait_score:
        trait_scores = []
        for tcol in trait_columns():
            if tcol not in row.index:
                continue
            val = str(row[tcol])
            lift, n = trait_lookup.get((tcol, val, core), (1.0, 0))
            # downweight weak samples even if lookup exists
            w = min(1.0, n / max(1, cfg.min_sample))
            trait_scores.append(w * _safe_log_lift(lift))
        if trait_scores:
            # average avoids excessive reward from many correlated traits
            score += 2.0 * float(np.mean(trait_scores))
    if cfg.use_cadence_score:
        cadence_values = {
            "stream_x_dow": f"{row['StreamKey_affinity']} || {row['day_of_week']}",
            "stream_x_gap": f"{row['StreamKey_affinity']} || {row['days_since_seed_bucket']}",
            "stream_x_seed_shape": f"{row['StreamKey_affinity']} || {row['seed_shape']}",
        }
        cad_scores = []
        for label, val in cadence_values.items():
            lift, n = cadence_lookup.get((label, val, core), (1.0, 0))
            w = min(1.0, n / max(1, cfg.min_sample))
            cad_scores.append(w * _safe_log_lift(lift))
            if lift >= 1.50 and n >= cfg.min_sample:
                reasons.append(f"{label}_lift={lift:.2f}/n={n}")
        if cad_scores:
            score += 1.5 * float(np.mean(cad_scores))
    return score, reasons[:5]


def run_cross_core_backtest(train: pd.DataFrame, test: pd.DataFrame, cores: List[str], stream_lookup, trait_lookup, cadence_lookup, cfg: LabConfig, progress=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_aabc = test[test["IsAABC"]].copy()
    rows = []
    detail_rows = []
    total = max(1, len(test_aabc))
    for i, (_, row) in enumerate(test_aabc.iterrows(), start=1):
        scored = []
        for core in cores:
            sc, reasons = score_event_core(row, core, stream_lookup, trait_lookup, cadence_lookup, cfg)
            scored.append((core, sc, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        actual = row["ActualCore"]
        ranks = {core: idx + 1 for idx, (core, _, _) in enumerate(scored)}
        actual_rank = int(ranks.get(actual, 9999))
        top1_core, top1_score, top1_reasons = scored[0]
        rows.append({
            "Date": row["Date_affinity"].strftime("%Y-%m-%d"),
            "StreamKey": row["StreamKey_affinity"],
            "SeedResult4": row["SeedResult4"],
            "ActualResult4": row["Result4_affinity"],
            "ActualCore": actual,
            "ActualMember": row["ActualMember"],
            "Top1Core": top1_core,
            "Top1Score": round(top1_score, 6),
            "Top1CorrectCore": bool(top1_core == actual),
            "ActualCoreRank": actual_rank,
            "Top3CorrectCore": bool(actual_rank <= 3),
            "Top5CorrectCore": bool(actual_rank <= 5),
            "Top10CorrectCore": bool(actual_rank <= 10),
            "Top1Reasons": "; ".join(top1_reasons),
        })
        for rank_idx, (core, sc, reasons) in enumerate(scored[: int(cfg.top_k_event_export)], start=1):
            detail_rows.append({
                "Date": row["Date_affinity"].strftime("%Y-%m-%d"),
                "StreamKey": row["StreamKey_affinity"],
                "SeedResult4": row["SeedResult4"],
                "ActualCore": actual,
                "ActualMember": row["ActualMember"],
                "CandidateRank": rank_idx,
                "CandidateCore": core,
                "CandidateScore": round(sc, 6),
                "IsActualCore": bool(core == actual),
                "Reasons": "; ".join(reasons),
            })
        if progress is not None and (i % 100 == 0 or i == total):
            progress.progress(min(1.0, i / total), text=f"Backtesting cross-core affinity: {i:,}/{total:,} AABC winner events")
    event_df = pd.DataFrame(rows)
    detail_df = pd.DataFrame(detail_rows)

    if event_df.empty:
        summary = pd.DataFrame()
        stream_summary = pd.DataFrame()
    else:
        summary_rows = [{
            "metric": "test_AABC_events",
            "value": int(len(event_df)),
        }, {
            "metric": "top1_core_accuracy_pct",
            "value": round(event_df["Top1CorrectCore"].mean() * 100, 2),
        }, {
            "metric": "top3_core_accuracy_pct",
            "value": round(event_df["Top3CorrectCore"].mean() * 100, 2),
        }, {
            "metric": "top5_core_accuracy_pct",
            "value": round(event_df["Top5CorrectCore"].mean() * 100, 2),
        }, {
            "metric": "top10_core_accuracy_pct",
            "value": round(event_df["Top10CorrectCore"].mean() * 100, 2),
        }, {
            "metric": "median_actual_core_rank",
            "value": round(float(event_df["ActualCoreRank"].median()), 2),
        }]
        summary = pd.DataFrame(summary_rows)
        stream_summary = event_df.groupby("StreamKey").agg(
            test_AABC_events=("ActualCore", "size"),
            top1_core_accuracy_pct=("Top1CorrectCore", lambda x: round(float(np.mean(x)) * 100, 2)),
            top3_core_accuracy_pct=("Top3CorrectCore", lambda x: round(float(np.mean(x)) * 100, 2)),
            median_actual_core_rank=("ActualCoreRank", "median"),
        ).reset_index().sort_values(["test_AABC_events", "top1_core_accuracy_pct"], ascending=[False, False])
    return event_df, detail_df, summary, stream_summary


def build_latest_seed_daily_affinity(events: pd.DataFrame, cores: List[str], stream_lookup, trait_lookup, cadence_lookup, cfg: LabConfig) -> pd.DataFrame:
    latest = events.sort_values("Date_affinity").groupby("StreamKey_affinity", as_index=False).tail(1).copy()
    rows = []
    for _, row in latest.iterrows():
        scored = []
        for core in cores:
            sc, reasons = score_event_core(row, core, stream_lookup, trait_lookup, cadence_lookup, cfg)
            scored.append((core, sc, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        for rank_idx, (core, sc, reasons) in enumerate(scored[:10], start=1):
            rows.append({
                "StreamKey": row["StreamKey_affinity"],
                "LatestHistoryDate": row["Date_affinity"].strftime("%Y-%m-%d"),
                "CurrentSeedResult4": row["Result4_affinity"],
                "CandidateRank": rank_idx,
                "CandidateCore": core,
                "CoreAffinityScore": round(sc, 6),
                "Members": ",".join(core_members(core)),
                "Reasons": "; ".join(reasons),
            })
    return pd.DataFrame(rows).sort_values(["StreamKey", "CandidateRank"])


def bytes_csv(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False).encode("utf-8")


def make_zip(named_dfs: Dict[str, pd.DataFrame], readme: str) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.writestr("README_core_affinity_lab_v1.txt", readme.encode("utf-8"))
        for name, df in named_dfs.items():
            zf.writestr(name, bytes_csv(df))
    return bio.getvalue()


# =============================================================================
# Streamlit UI
# =============================================================================

with st.sidebar:
    st.header("Inputs")
    history_file = st.file_uploader("Upload full clean Pick-4 history", type=["csv", "txt", "tsv"])
    st.header("Mining controls")
    default_targets = "ALL"
    target_text = st.text_area(
        "Target cores",
        value=default_targets,
        help="Use ALL for all 120 cores, or comma-separated cores like 025,389,168,589.",
        height=80,
    )
    min_sample = st.number_input("Minimum sample size for displayed trait/cadence profiles", min_value=5, max_value=500, value=25, step=5)
    smoothing = st.number_input("Smoothing strength", min_value=0.1, max_value=50.0, value=5.0, step=0.5)
    test_fraction = st.slider("Holdout test fraction", min_value=0.10, max_value=0.50, value=0.30, step=0.05)
    top_k_event_export = st.slider("Candidate cores exported per test event", min_value=3, max_value=120, value=10, step=1)
    use_stream_score = st.checkbox("Use stream-core score", value=True)
    use_trait_score = st.checkbox("Use seed-trait score", value=True)
    use_cadence_score = st.checkbox("Use cadence interaction score", value=True)
    run_button = st.button("Run Core Affinity Lab", type="primary", use_container_width=True)

if not history_file:
    st.warning("Upload history to begin.")
    st.stop()

if target_text.strip().upper() == "ALL":
    selected_cores = all_120_cores()
else:
    selected_cores = []
    for tok in re.split(r"[,\s]+", target_text.strip()):
        tok = tok.strip()
        if not tok:
            continue
        tok = "".join(sorted(re.findall(r"\d", tok)))
        if len(tok) == 3 and len(set(tok)) == 3:
            selected_cores.append(tok)
    selected_cores = sorted(set(selected_cores))
    if not selected_cores:
        st.error("No valid target cores found. Use ALL or cores like 025,389,168.")
        st.stop()

cfg = LabConfig(
    min_sample=int(min_sample),
    smoothing=float(smoothing),
    test_fraction=float(test_fraction),
    use_stream_score=bool(use_stream_score),
    use_trait_score=bool(use_trait_score),
    use_cadence_score=bool(use_cadence_score),
    top_k_event_export=int(top_k_event_export),
)

if not run_button:
    st.info("Choose settings, then click Run Core Affinity Lab.")
    st.stop()

progress = st.progress(0, text="Loading history")
try:
    hist = read_upload(history_file)
    date_col, result_col, stream_col = infer_cols(hist)
except Exception as e:
    st.error(f"Could not load history: {e}")
    st.stop()

progress.progress(0.10, text="Building seed→next-winner transition rows")
events = add_seed_transition_rows(hist, date_col, result_col, stream_col)
if events.empty:
    st.error("No usable seed→next-winner transition rows could be built.")
    st.stop()

progress.progress(0.18, text="Splitting train/test by date")
train, test, cutoff = train_test_split_by_date(events, cfg.test_fraction)

progress.progress(0.25, text="Building core rank and member profiles")
core_rank = build_core_rank(events, selected_cores)
member_profile = build_core_member_profile(events, selected_cores)

progress.progress(0.35, text="Mining stream×core profiles")
stream_core_profile, stream_lookup = build_stream_core_profile(train, selected_cores, cfg.smoothing)

progress.progress(0.50, text="Mining seed-trait×core profiles")
trait_core_profile, trait_lookup = build_trait_core_profile(train, selected_cores, cfg.smoothing, cfg.min_sample)

progress.progress(0.65, text="Mining cadence×core profiles")
cadence_core_profile, cadence_lookup = build_cadence_core_profile(train, selected_cores, cfg.smoothing, cfg.min_sample)

progress.progress(0.75, text="Running historical holdout accuracy test")
event_predictions, event_candidate_detail, accuracy_summary, stream_accuracy_summary = run_cross_core_backtest(
    train, test, selected_cores, stream_lookup, trait_lookup, cadence_lookup, cfg, progress=progress
)

progress.progress(0.95, text="Scoring latest-seed stream profiles")
latest_seed_affinity = build_latest_seed_daily_affinity(events, selected_cores, stream_lookup, trait_lookup, cadence_lookup, cfg)

manifest = pd.DataFrame([
    {"item": "build_marker", "value": BUILD_MARKER},
    {"item": "history_rows_loaded", "value": len(hist)},
    {"item": "seed_transition_rows", "value": len(events)},
    {"item": "train_rows", "value": len(train)},
    {"item": "test_rows", "value": len(test)},
    {"item": "holdout_cutoff_date", "value": "" if pd.isna(cutoff) else pd.Timestamp(cutoff).strftime("%Y-%m-%d")},
    {"item": "target_cores", "value": ",".join(selected_cores)},
    {"item": "min_sample", "value": cfg.min_sample},
    {"item": "smoothing", "value": cfg.smoothing},
    {"item": "test_fraction", "value": cfg.test_fraction},
    {"item": "use_stream_score", "value": cfg.use_stream_score},
    {"item": "use_trait_score", "value": cfg.use_trait_score},
    {"item": "use_cadence_score", "value": cfg.use_cadence_score},
])

progress.progress(1.0, text="Done")

st.success(f"Built Core Affinity profiles for {len(selected_cores)} cores using {len(events):,} seed→next-winner transitions.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Transition rows", f"{len(events):,}")
with c2:
    st.metric("Train rows", f"{len(train):,}")
with c3:
    st.metric("Test rows", f"{len(test):,}")
with c4:
    top1_value = ""
    if not accuracy_summary.empty:
        found = accuracy_summary[accuracy_summary["metric"] == "top1_core_accuracy_pct"]
        if not found.empty:
            top1_value = str(found.iloc[0]["value"]) + "%"
    st.metric("Top1 core accuracy", top1_value or "n/a")

st.subheader("Historical Holdout Accuracy")
st.caption("Accuracy is measured only on historical AABC winner events in the holdout period. This is not a guarantee of future results.")
st.dataframe(accuracy_summary, use_container_width=True, hide_index=True)

st.subheader("Core Rank by Actual Frequency")
st.dataframe(core_rank.head(40), use_container_width=True, hide_index=True)

st.subheader("Member Role Profile")
st.dataframe(member_profile.head(120), use_container_width=True, hide_index=True)

st.subheader("Stream × Core Profile")
st.caption("Shows which streams historically over/under-favor each core in the training period.")
st.dataframe(stream_core_profile.head(500), use_container_width=True, hide_index=True)

st.subheader("Seed Trait × Core Profile")
st.caption("Shows seed traits that over/under-favor cores. Filtered by minimum sample size for display/export table.")
st.dataframe(trait_core_profile.head(500), use_container_width=True, hide_index=True)

st.subheader("Cadence × Core Profile")
st.caption("V1 cadence interactions: stream×day-of-week, stream×gap bucket, stream×seed shape.")
st.dataframe(cadence_core_profile.head(500), use_container_width=True, hide_index=True)

st.subheader("Stream Accuracy Summary")
st.dataframe(stream_accuracy_summary.head(500), use_container_width=True, hide_index=True)

st.subheader("Latest Seed Core Affinity Preview")
st.caption("Research preview only. This is not a playlist. It ranks core affinity for each stream's latest known seed.")
st.dataframe(latest_seed_affinity.head(500), use_container_width=True, hide_index=True)

st.subheader("Downloads")
outputs = {
    "core_affinity_manifest_v1.csv": manifest,
    "core_affinity_core_rank_v1.csv": core_rank,
    "core_affinity_member_profile_v1.csv": member_profile,
    "core_affinity_stream_core_profile_v1.csv": stream_core_profile,
    "core_affinity_seed_trait_core_profile_v1.csv": trait_core_profile,
    "core_affinity_cadence_core_profile_v1.csv": cadence_core_profile,
    "core_affinity_holdout_accuracy_summary_v1.csv": accuracy_summary,
    "core_affinity_holdout_event_predictions_v1.csv": event_predictions,
    "core_affinity_holdout_event_candidate_detail_v1.csv": event_candidate_detail,
    "core_affinity_stream_accuracy_summary_v1.csv": stream_accuracy_summary,
    "core_affinity_latest_seed_preview_v1.csv": latest_seed_affinity,
}
readme = f"""Core Affinity Lab v1
{BUILD_MARKER}

Purpose:
Mine stream/core, seed-trait/core, and cadence/core relationships from Pick-4 history.
This is a research-only lab. It does not produce a betting playlist and does not use cuts,
RTE, B1Z0, ZLT, rescues, budget logic, or straight logic.

Key files:
- core_affinity_core_rank_v1.csv: actual frequency ranking of target cores.
- core_affinity_member_profile_v1.csv: member hit distribution and starter roles.
- core_affinity_stream_core_profile_v1.csv: stream-specific core affinity/lift.
- core_affinity_seed_trait_core_profile_v1.csv: seed trait core affinity/lift.
- core_affinity_cadence_core_profile_v1.csv: stream/cadence interactions.
- core_affinity_holdout_accuracy_summary_v1.csv: historical holdout accuracy percentages.
- core_affinity_latest_seed_preview_v1.csv: latest-seed core affinity preview, not a playlist.

Interpretation rule:
Use this lab to discover which cores should compete on each stream. Do not treat its output as
member selection or final play selection. Member engines and daily platform come later.
"""
zip_bytes = make_zip(outputs, readme)
st.download_button(
    "Download ALL Core Affinity Lab outputs ZIP",
    data=zip_bytes,
    file_name="core_affinity_lab_v1_outputs.zip",
    mime="application/zip",
    use_container_width=True,
)
for name, df in outputs.items():
    st.download_button(
        f"Download {name}",
        data=bytes_csv(df),
        file_name=name,
        mime="text/csv",
        use_container_width=True,
    )
