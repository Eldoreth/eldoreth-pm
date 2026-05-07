"""Event log parsing and validation for XES and CSV formats."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import pm4py
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class EventLog(BaseModel):
    """Single event record within a process event log."""

    case_id: str
    activity: str
    timestamp: datetime
    resource: str | None = None


class EventLogStats(BaseModel):
    """Summary statistics produced by validate_eventlog."""

    total_cases: int
    total_events: int
    unique_activities: int
    date_range: tuple[datetime, datetime]
    quality_score: float  # 0.0 – 1.0

    @field_validator("quality_score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Column name constants (pm4py XES defaults)
# ---------------------------------------------------------------------------

_XES_CASE = "case:concept:name"
_XES_ACTIVITY = "concept:name"
_XES_TIMESTAMP = "time:timestamp"
_XES_RESOURCE = "org:resource"

# Canonical column names used internally after parsing
COL_CASE = "case_id"
COL_ACTIVITY = "activity"
COL_TIMESTAMP = "timestamp"
COL_RESOURCE = "resource"

_REQUIRED_COLS = {COL_CASE, COL_ACTIVITY, COL_TIMESTAMP}

# Common alternative column names accepted when parsing CSV files
_CSV_ALIASES: dict[str, list[str]] = {
    COL_CASE: ["case_id", "case", "CaseID", "case:concept:name", "traceid"],
    COL_ACTIVITY: ["activity", "Activity", "concept:name", "task", "Task", "event"],
    COL_TIMESTAMP: [
        "timestamp",
        "Timestamp",
        "time:timestamp",
        "time",
        "Time",
        "date",
        "Date",
        "start_time",
    ],
    COL_RESOURCE: ["resource", "Resource", "org:resource", "user", "User", "agent"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw pm4py / CSV column names to canonical names."""
    rename: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}

    for canonical, aliases in _CSV_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
            if alias.lower() in lower_cols and lower_cols[alias.lower()] not in rename:
                rename[lower_cols[alias.lower()]] = canonical
                break

    df = df.rename(columns=rename)
    if COL_CASE in df.columns:
        df[COL_CASE] = df[COL_CASE].astype(str)
    if COL_TIMESTAMP in df.columns:
        df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], utc=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_xes(filepath: str) -> pd.DataFrame:
    """Parse an XES event log file into a normalised DataFrame.

    Args:
        filepath: Absolute or relative path to a .xes or .xes.gz file.

    Returns:
        DataFrame with columns: case_id, activity, timestamp, resource (when present).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing after parsing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"XES file not found: {filepath}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw: pd.DataFrame = pm4py.read_xes(str(path))

    df = _normalise(raw)
    _assert_required(df, filepath)

    keep = [COL_CASE, COL_ACTIVITY, COL_TIMESTAMP]
    if COL_RESOURCE in df.columns:
        keep.append(COL_RESOURCE)

    return df[keep].reset_index(drop=True)


def parse_csv(filepath: str) -> pd.DataFrame:
    """Parse a CSV event log file into a normalised DataFrame.

    The CSV must contain columns (or recognised aliases) for case_id, activity,
    and timestamp. A resource column is optional.

    Args:
        filepath: Absolute or relative path to a .csv file.

    Returns:
        DataFrame with columns: case_id, activity, timestamp, resource (when present).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing after alias resolution.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    raw = pd.read_csv(str(path))
    df = _normalise(raw)
    _assert_required(df, filepath)

    keep = [COL_CASE, COL_ACTIVITY, COL_TIMESTAMP]
    if COL_RESOURCE in df.columns:
        keep.append(COL_RESOURCE)

    return df[keep].reset_index(drop=True)


def validate_eventlog(df: pd.DataFrame) -> EventLogStats:
    """Compute summary statistics and a quality score for an event log DataFrame.

    Quality score is 1.0 when: no null activities, no null timestamps, no
    duplicate events, and at least 2 distinct activities.  Each failing check
    deducts 0.25.

    Args:
        df: DataFrame previously produced by parse_xes or parse_csv.

    Returns:
        EventLogStats with counts, date range, and quality score.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    _assert_required(df, "input DataFrame")

    if df.empty:
        raise ValueError("Event log DataFrame is empty.")

    total_cases = df[COL_CASE].nunique()
    total_events = len(df)
    unique_activities = df[COL_ACTIVITY].nunique()
    date_range = (df[COL_TIMESTAMP].min().to_pydatetime(),
                  df[COL_TIMESTAMP].max().to_pydatetime())

    deductions = 0.0
    if df[COL_ACTIVITY].isna().any():
        deductions += 0.25
    if df[COL_TIMESTAMP].isna().any():
        deductions += 0.25
    if df.duplicated().any():
        deductions += 0.25
    if unique_activities < 2:
        deductions += 0.25

    return EventLogStats(
        total_cases=total_cases,
        total_events=total_events,
        unique_activities=unique_activities,
        date_range=date_range,
        quality_score=1.0 - deductions,
    )


def _assert_required(df: pd.DataFrame, source: str) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in '{source}'. "
            f"Found: {list(df.columns)}"
        )
