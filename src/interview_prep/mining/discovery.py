"""Process discovery, variant analysis, and bottleneck detection."""

from __future__ import annotations

import warnings
from typing import Literal

import pandas as pd
import pm4py
from pydantic import BaseModel, field_validator

from interview_prep.mining.parser import (
    COL_ACTIVITY,
    COL_CASE,
    COL_RESOURCE,
    COL_TIMESTAMP,
)

Algorithm = Literal["inductive", "alpha", "heuristics"]

_REQUIRED = {COL_CASE, COL_ACTIVITY, COL_TIMESTAMP}


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class ProcessModel(BaseModel):
    """Discovered process model summary."""

    algorithm: str
    num_activities: int   # labeled (non-silent) transitions in the Petri net
    num_transitions: int  # total arcs (place→transition + transition→place)
    fitness_score: float  # log fitness via token-based replay, 0.0–1.0

    @field_validator("fitness_score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class Variant(BaseModel):
    """A unique trace variant with frequency and average duration."""

    activities: tuple[str, ...]
    frequency: int                  # number of cases following this variant
    avg_duration_seconds: float     # mean case span (first → last event)


class Bottleneck(BaseModel):
    """Activity-level bottleneck metrics."""

    activity: str
    avg_wait_time_seconds: float  # mean wait since the previous event in the case
    frequency: int                # total occurrences across all cases
    rework_rate: float            # fraction of cases where this activity repeats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_required(df: pd.DataFrame) -> None:
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty.")


def _to_pm4py_log(df: pd.DataFrame) -> pm4py.objects.log.obj.EventLog:
    rename = {
        COL_CASE: "case:concept:name",
        COL_ACTIVITY: "concept:name",
        COL_TIMESTAMP: "time:timestamp",
    }
    if COL_RESOURCE in df.columns:
        rename[COL_RESOURCE] = "org:resource"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm4py.convert_to_event_log(df.rename(columns=rename))


def _discover_net(
    log: pm4py.objects.log.obj.EventLog,
    algorithm: Algorithm,
) -> tuple:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if algorithm == "inductive":
            return pm4py.discover_petri_net_inductive(log)
        if algorithm == "alpha":
            return pm4py.discover_petri_net_alpha(log)
        if algorithm == "heuristics":
            return pm4py.discover_petri_net_heuristics(log)
    raise ValueError(  # pragma: no cover
        f"Unknown algorithm '{algorithm}'. Choose from: inductive, alpha, heuristics."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_process(
    df: pd.DataFrame,
    algorithm: str = "inductive",
) -> ProcessModel:
    """Discover a Petri-net process model from an event log DataFrame.

    Args:
        df: Normalised event log (output of parse_xes / parse_csv).
        algorithm: Discovery algorithm — one of ``inductive``, ``alpha``,
            ``heuristics``. Defaults to ``inductive``.

    Returns:
        ProcessModel with algorithm used, activity/transition counts, and
        token-replay fitness score.

    Raises:
        ValueError: If df is missing required columns, is empty, or the
            algorithm name is not recognised.
    """
    _assert_required(df)
    if algorithm not in ("inductive", "alpha", "heuristics"):
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            "Choose from: inductive, alpha, heuristics."
        )

    log = _to_pm4py_log(df)
    net, im, fm = _discover_net(log, algorithm)  # type: ignore[arg-type]

    num_activities = sum(1 for t in net.transitions if t.label is not None)
    num_transitions = len(net.arcs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitness_result = pm4py.fitness_token_based_replay(log, net, im, fm)

    fitness_score = float(fitness_result.get("log_fitness", 0.0))

    return ProcessModel(
        algorithm=algorithm,
        num_activities=num_activities,
        num_transitions=num_transitions,
        fitness_score=fitness_score,
    )


def get_variants(df: pd.DataFrame) -> list[Variant]:
    """Extract all trace variants with frequency and average duration.

    A variant is a unique ordered sequence of activities within a case.
    Duration is measured from the timestamp of the first event to the last.

    Args:
        df: Normalised event log DataFrame.

    Returns:
        List of Variant objects sorted by frequency descending.

    Raises:
        ValueError: If df is missing required columns or is empty.
    """
    _assert_required(df)

    df_sorted = df.sort_values([COL_CASE, COL_TIMESTAMP])

    # build (variant_tuple, duration_seconds) per case
    records: list[tuple[tuple[str, ...], float]] = []
    for _, case_df in df_sorted.groupby(COL_CASE, sort=False):
        activities = tuple(case_df[COL_ACTIVITY].tolist())
        ts = case_df[COL_TIMESTAMP]
        duration = (ts.max() - ts.min()).total_seconds()
        records.append((activities, duration))

    # aggregate by variant
    from collections import defaultdict
    freq: dict[tuple[str, ...], int] = defaultdict(int)
    total_dur: dict[tuple[str, ...], float] = defaultdict(float)
    for variant, dur in records:
        freq[variant] += 1
        total_dur[variant] += dur

    variants = [
        Variant(
            activities=variant,
            frequency=freq[variant],
            avg_duration_seconds=total_dur[variant] / freq[variant],
        )
        for variant in freq
    ]
    return sorted(variants, key=lambda v: v.frequency, reverse=True)


def get_bottlenecks(df: pd.DataFrame, top_n: int = 5) -> list[Bottleneck]:
    """Identify the top-N activity bottlenecks by average wait time.

    Wait time for an event is the elapsed seconds since the previous event in
    the same case (NaN for the first event per case, excluded from the mean).
    Rework rate is the fraction of cases where the activity appears more than
    once.

    Args:
        df: Normalised event log DataFrame.
        top_n: Number of bottlenecks to return, ordered by avg_wait_time desc.

    Returns:
        List of at most ``top_n`` Bottleneck objects.

    Raises:
        ValueError: If df is missing required columns or is empty.
    """
    _assert_required(df)
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    df_s = df.sort_values([COL_CASE, COL_TIMESTAMP]).copy()
    df_s["_prev_ts"] = df_s.groupby(COL_CASE)[COL_TIMESTAMP].shift(1)
    df_s["_wait"] = (df_s[COL_TIMESTAMP] - df_s["_prev_ts"]).dt.total_seconds()

    # per-activity aggregate wait and frequency
    agg = (
        df_s.groupby(COL_ACTIVITY)
        .agg(
            avg_wait=("_wait", "mean"),       # NaN first-events excluded by mean
            frequency=(COL_ACTIVITY, "count"),
        )
        .reset_index()
    )
    agg["avg_wait"] = agg["avg_wait"].fillna(0.0)

    # rework rate: fraction of cases where activity count > 1
    case_act_counts = (
        df.groupby([COL_CASE, COL_ACTIVITY])
        .size()
        .reset_index(name="_cnt")
    )
    total_cases = df[COL_CASE].nunique()
    rework = (
        case_act_counts[case_act_counts["_cnt"] > 1]
        .groupby(COL_ACTIVITY)
        .size()
        .reset_index(name="_rework_cases")
    )
    agg = agg.merge(rework, on=COL_ACTIVITY, how="left")
    agg["_rework_cases"] = agg["_rework_cases"].fillna(0)
    agg["rework_rate"] = agg["_rework_cases"] / total_cases

    top = agg.nlargest(top_n, "avg_wait")

    return [
        Bottleneck(
            activity=row[COL_ACTIVITY],
            avg_wait_time_seconds=float(row["avg_wait"]),
            frequency=int(row["frequency"]),
            rework_rate=float(row["rework_rate"]),
        )
        for _, row in top.iterrows()
    ]
