"""Tests for the event log parser module."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from interview_prep.mining.parser import (
    COL_ACTIVITY,
    COL_CASE,
    COL_RESOURCE,
    COL_TIMESTAMP,
    EventLog,
    EventLogStats,
    parse_csv,
    parse_xes,
    validate_eventlog,
)

# Path to the bundled XES fixture generated at project setup
SAMPLE_XES = Path(__file__).parent.parent / "data" / "sample_orders.xes"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def orders_df() -> pd.DataFrame:
    """Parse the bundled sample XES once and reuse across tests."""
    return parse_xes(str(SAMPLE_XES))


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    """Minimal valid DataFrame with required columns only."""
    return pd.DataFrame(
        {
            COL_CASE: ["1", "1", "2"],
            COL_ACTIVITY: ["A", "B", "A"],
            COL_TIMESTAMP: pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01"], utc=True
            ),
        }
    )


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """Write a small CSV event log using canonical column names."""
    content = textwrap.dedent("""\
        case_id,activity,timestamp,resource
        C1,Create Order,2024-03-01 09:00:00,Alice
        C1,Approve Order,2024-03-01 11:00:00,Bob
        C2,Create Order,2024-03-02 10:00:00,Carol
        C2,Approve Order,2024-03-02 14:00:00,Alice
        C2,Ship Order,2024-03-03 08:00:00,Bob
    """)
    p = tmp_path / "orders.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def csv_alias_file(tmp_path: Path) -> Path:
    """CSV using non-canonical but recognised column aliases."""
    content = textwrap.dedent("""\
        traceid,task,date,user
        T1,Receive,2024-05-01,Alice
        T1,Process,2024-05-02,Bob
        T2,Receive,2024-05-01,Carol
    """)
    p = tmp_path / "alias_log.csv"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# 1. parse_xes — happy path
# ---------------------------------------------------------------------------

class TestParseXes:
    def test_returns_dataframe(self, orders_df: pd.DataFrame) -> None:
        assert isinstance(orders_df, pd.DataFrame)

    def test_required_columns_present(self, orders_df: pd.DataFrame) -> None:
        for col in (COL_CASE, COL_ACTIVITY, COL_TIMESTAMP):
            assert col in orders_df.columns

    def test_resource_column_present(self, orders_df: pd.DataFrame) -> None:
        assert COL_RESOURCE in orders_df.columns

    def test_expected_case_count(self, orders_df: pd.DataFrame) -> None:
        assert orders_df[COL_CASE].nunique() == 20

    def test_expected_activity_count(self, orders_df: pd.DataFrame) -> None:
        assert orders_df[COL_ACTIVITY].nunique() == 3

    def test_expected_event_count(self, orders_df: pd.DataFrame) -> None:
        # 20 cases × 3 activities each
        assert len(orders_df) == 60

    def test_timestamp_is_datetime(self, orders_df: pd.DataFrame) -> None:
        assert pd.api.types.is_datetime64_any_dtype(orders_df[COL_TIMESTAMP])

    def test_case_id_is_string(self, orders_df: pd.DataFrame) -> None:
        assert pd.api.types.is_string_dtype(orders_df[COL_CASE])

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_xes("/nonexistent/path/log.xes")


# ---------------------------------------------------------------------------
# 2. parse_csv — happy path and aliases
# ---------------------------------------------------------------------------

class TestParseCsv:
    def test_returns_dataframe(self, csv_file: Path) -> None:
        df = parse_csv(str(csv_file))
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, csv_file: Path) -> None:
        df = parse_csv(str(csv_file))
        for col in (COL_CASE, COL_ACTIVITY, COL_TIMESTAMP):
            assert col in df.columns

    def test_resource_parsed(self, csv_file: Path) -> None:
        df = parse_csv(str(csv_file))
        assert COL_RESOURCE in df.columns
        assert set(df[COL_RESOURCE]) == {"Alice", "Bob", "Carol"}

    def test_correct_row_count(self, csv_file: Path) -> None:
        df = parse_csv(str(csv_file))
        assert len(df) == 5

    def test_alias_columns_resolved(self, csv_alias_file: Path) -> None:
        df = parse_csv(str(csv_alias_file))
        for col in (COL_CASE, COL_ACTIVITY, COL_TIMESTAMP):
            assert col in df.columns

    def test_timestamp_is_utc(self, csv_file: Path) -> None:
        df = parse_csv(str(csv_file))
        assert df[COL_TIMESTAMP].dt.tz is not None

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_csv("/nonexistent/log.csv")

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("col_a,col_b\n1,foo\n2,bar\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            parse_csv(str(bad))


# ---------------------------------------------------------------------------
# 3. validate_eventlog
# ---------------------------------------------------------------------------

class TestValidateEventlog:
    def test_returns_stats_model(self, minimal_df: pd.DataFrame) -> None:
        stats = validate_eventlog(minimal_df)
        assert isinstance(stats, EventLogStats)

    def test_counts_are_correct(self, minimal_df: pd.DataFrame) -> None:
        stats = validate_eventlog(minimal_df)
        assert stats.total_cases == 2
        assert stats.total_events == 3
        assert stats.unique_activities == 2

    def test_date_range_tuple(self, minimal_df: pd.DataFrame) -> None:
        stats = validate_eventlog(minimal_df)
        assert isinstance(stats.date_range, tuple)
        assert len(stats.date_range) == 2
        assert stats.date_range[0] <= stats.date_range[1]

    def test_perfect_quality_score(self, minimal_df: pd.DataFrame) -> None:
        stats = validate_eventlog(minimal_df)
        assert stats.quality_score == 1.0

    def test_quality_penalised_for_null_activity(self) -> None:
        df = pd.DataFrame(
            {
                COL_CASE: ["1", "2"],
                COL_ACTIVITY: [None, "A"],
                COL_TIMESTAMP: pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            }
        )
        stats = validate_eventlog(df)
        assert stats.quality_score < 1.0

    def test_quality_penalised_for_duplicates(self) -> None:
        df = pd.DataFrame(
            {
                COL_CASE: ["1", "1"],
                COL_ACTIVITY: ["A", "A"],
                COL_TIMESTAMP: pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
            }
        )
        stats = validate_eventlog(df)
        assert stats.quality_score < 1.0

    def test_quality_penalised_for_single_activity(self) -> None:
        df = pd.DataFrame(
            {
                COL_CASE: ["1", "2"],
                COL_ACTIVITY: ["A", "A"],
                COL_TIMESTAMP: pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            }
        )
        stats = validate_eventlog(df)
        assert stats.quality_score < 1.0

    def test_empty_dataframe_raises(self) -> None:
        empty = pd.DataFrame(
            {COL_CASE: pd.Series([], dtype=str),
             COL_ACTIVITY: pd.Series([], dtype=str),
             COL_TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]")}
        )
        with pytest.raises(ValueError, match="empty"):
            validate_eventlog(empty)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"case_id": ["1"], "activity": ["A"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_eventlog(df)

    def test_quality_score_clamped_to_one(self, minimal_df: pd.DataFrame) -> None:
        stats = validate_eventlog(minimal_df)
        assert 0.0 <= stats.quality_score <= 1.0

    def test_full_pipeline_xes(self, orders_df: pd.DataFrame) -> None:
        """End-to-end: parse XES then validate."""
        stats = validate_eventlog(orders_df)
        assert stats.total_cases == 20
        assert stats.total_events == 60
        assert stats.quality_score == 1.0


# ---------------------------------------------------------------------------
# 4. EventLog pydantic model
# ---------------------------------------------------------------------------

class TestEventLogModel:
    def test_valid_model(self) -> None:
        e = EventLog(
            case_id="case_1",
            activity="Create Order",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert e.resource is None

    def test_with_resource(self) -> None:
        e = EventLog(
            case_id="c1",
            activity="A",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            resource="Alice",
        )
        assert e.resource == "Alice"

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(Exception):
            EventLog(case_id="1", timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))  # type: ignore[call-arg]
