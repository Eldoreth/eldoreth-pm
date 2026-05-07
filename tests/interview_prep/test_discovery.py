"""Tests for the process discovery module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from interview_prep.mining.discovery import (
    Bottleneck,
    ProcessModel,
    Variant,
    discover_process,
    get_bottlenecks,
    get_variants,
)
from interview_prep.mining.parser import (
    COL_ACTIVITY,
    COL_CASE,
    COL_TIMESTAMP,
    parse_xes,
)
from pathlib import Path

SAMPLE_XES = Path(__file__).parent.parent / "data" / "sample_orders.xes"

_UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def orders_df() -> pd.DataFrame:
    """20-case, 3-activity linear log from the bundled XES fixture."""
    return parse_xes(str(SAMPLE_XES))


def _ts(day: int, hour: int = 9) -> datetime:
    return datetime(2024, 1, day, hour, 0, 0, tzinfo=_UTC)


@pytest.fixture()
def multi_variant_df() -> pd.DataFrame:
    """Log with two variants and a rework activity to exercise all metrics."""
    rows = [
        # Variant A: Create → Approve → Ship  (cases 1, 2, 3)
        ("1", "Create",  _ts(1, 9)),
        ("1", "Approve", _ts(1, 11)),  # wait 2 h
        ("1", "Ship",    _ts(1, 15)),  # wait 4 h
        ("2", "Create",  _ts(2, 9)),
        ("2", "Approve", _ts(2, 12)),  # wait 3 h
        ("2", "Ship",    _ts(2, 18)),  # wait 6 h
        ("3", "Create",  _ts(3, 9)),
        ("3", "Approve", _ts(3, 10)),  # wait 1 h
        ("3", "Ship",    _ts(3, 14)),  # wait 4 h
        # Variant B: Create → Reject  (cases 4, 5)
        ("4", "Create",  _ts(4, 9)),
        ("4", "Reject",  _ts(4, 17)),  # wait 8 h
        ("5", "Create",  _ts(5, 9)),
        ("5", "Reject",  _ts(5, 13)),  # wait 4 h
        # Rework in case 6: Approve appears twice
        ("6", "Create",  _ts(6, 9)),
        ("6", "Approve", _ts(6, 10)),
        ("6", "Approve", _ts(6, 12)),  # rework
        ("6", "Ship",    _ts(6, 16)),
    ]
    return pd.DataFrame(rows, columns=[COL_CASE, COL_ACTIVITY, COL_TIMESTAMP])


# ---------------------------------------------------------------------------
# discover_process tests
# ---------------------------------------------------------------------------

class TestDiscoverProcess:
    def test_returns_process_model(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df)
        assert isinstance(model, ProcessModel)

    def test_inductive_algorithm_default(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df)
        assert model.algorithm == "inductive"

    def test_alpha_algorithm(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df, algorithm="alpha")
        assert model.algorithm == "alpha"

    def test_heuristics_algorithm(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df, algorithm="heuristics")
        assert model.algorithm == "heuristics"

    def test_num_activities_matches_unique(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df)
        assert model.num_activities == orders_df[COL_ACTIVITY].nunique()

    def test_num_transitions_positive(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df)
        assert model.num_transitions > 0

    def test_fitness_score_range(self, orders_df: pd.DataFrame) -> None:
        model = discover_process(orders_df)
        assert 0.0 <= model.fitness_score <= 1.0

    def test_perfect_fitness_on_clean_log(self, orders_df: pd.DataFrame) -> None:
        """Linear 3-step process must replay with fitness 1.0."""
        model = discover_process(orders_df)
        assert model.fitness_score == 1.0

    def test_all_algorithms_same_activity_count(self, orders_df: pd.DataFrame) -> None:
        counts = {
            alg: discover_process(orders_df, algorithm=alg).num_activities
            for alg in ("inductive", "alpha", "heuristics")
        }
        assert len(set(counts.values())) == 1  # all identical

    def test_unknown_algorithm_raises(self, orders_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown algorithm"):
            discover_process(orders_df, algorithm="magic")

    def test_empty_df_raises(self) -> None:
        empty = pd.DataFrame({
            COL_CASE: pd.Series([], dtype=str),
            COL_ACTIVITY: pd.Series([], dtype=str),
            COL_TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]"),
        })
        with pytest.raises(ValueError):
            discover_process(empty)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"case_id": ["1"], "activity": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            discover_process(df)


# ---------------------------------------------------------------------------
# get_variants tests
# ---------------------------------------------------------------------------

class TestGetVariants:
    def test_returns_list_of_variant(self, orders_df: pd.DataFrame) -> None:
        variants = get_variants(orders_df)
        assert isinstance(variants, list)
        assert all(isinstance(v, Variant) for v in variants)

    def test_single_variant_on_linear_log(self, orders_df: pd.DataFrame) -> None:
        """All 20 cases follow the same 3-step sequence."""
        variants = get_variants(orders_df)
        assert len(variants) == 1

    def test_variant_frequency_sums_to_total_cases(
        self, orders_df: pd.DataFrame
    ) -> None:
        total = orders_df[COL_CASE].nunique()
        assert sum(v.frequency for v in get_variants(orders_df)) == total

    def test_multiple_variants_detected(
        self, multi_variant_df: pd.DataFrame
    ) -> None:
        variants = get_variants(multi_variant_df)
        assert len(variants) >= 2

    def test_variants_sorted_by_frequency_desc(
        self, multi_variant_df: pd.DataFrame
    ) -> None:
        variants = get_variants(multi_variant_df)
        freqs = [v.frequency for v in variants]
        assert freqs == sorted(freqs, reverse=True)

    def test_variant_activities_is_tuple(self, orders_df: pd.DataFrame) -> None:
        v = get_variants(orders_df)[0]
        assert isinstance(v.activities, tuple)

    def test_avg_duration_positive(self, orders_df: pd.DataFrame) -> None:
        for v in get_variants(orders_df):
            assert v.avg_duration_seconds >= 0.0

    def test_rework_case_counted_separately(
        self, multi_variant_df: pd.DataFrame
    ) -> None:
        """Case 6 has Approve twice → its variant must differ from the clean path."""
        variants = get_variants(multi_variant_df)
        activity_seqs = [v.activities for v in variants]
        rework_variant = ("Create", "Approve", "Approve", "Ship")
        assert rework_variant in activity_seqs

    def test_empty_df_raises(self) -> None:
        empty = pd.DataFrame({
            COL_CASE: pd.Series([], dtype=str),
            COL_ACTIVITY: pd.Series([], dtype=str),
            COL_TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]"),
        })
        with pytest.raises(ValueError):
            get_variants(empty)


# ---------------------------------------------------------------------------
# get_bottlenecks tests
# ---------------------------------------------------------------------------

class TestGetBottlenecks:
    def test_returns_list_of_bottleneck(self, orders_df: pd.DataFrame) -> None:
        bns = get_bottlenecks(orders_df)
        assert isinstance(bns, list)
        assert all(isinstance(b, Bottleneck) for b in bns)

    def test_default_top_n_at_most_5(self, orders_df: pd.DataFrame) -> None:
        bns = get_bottlenecks(orders_df)
        assert len(bns) <= 5

    def test_custom_top_n(self, orders_df: pd.DataFrame) -> None:
        bns = get_bottlenecks(orders_df, top_n=2)
        assert len(bns) <= 2

    def test_sorted_by_avg_wait_desc(self, multi_variant_df: pd.DataFrame) -> None:
        bns = get_bottlenecks(multi_variant_df)
        waits = [b.avg_wait_time_seconds for b in bns]
        assert waits == sorted(waits, reverse=True)

    def test_first_activity_wait_is_zero(self, orders_df: pd.DataFrame) -> None:
        """'Create Order' is always first → avg wait should be 0."""
        bns = get_bottlenecks(orders_df, top_n=10)
        create = next((b for b in bns if b.activity == "Create Order"), None)
        assert create is not None
        assert create.avg_wait_time_seconds == 0.0

    def test_frequency_matches_occurrence_count(
        self, orders_df: pd.DataFrame
    ) -> None:
        bns = get_bottlenecks(orders_df, top_n=10)
        for b in bns:
            expected = int((orders_df[COL_ACTIVITY] == b.activity).sum())
            assert b.frequency == expected

    def test_rework_rate_nonzero_when_rework_present(
        self, multi_variant_df: pd.DataFrame
    ) -> None:
        """Approve appears twice in case 6 out of 6 total cases."""
        bns = get_bottlenecks(multi_variant_df, top_n=10)
        approve = next(b for b in bns if b.activity == "Approve")
        assert approve.rework_rate > 0.0

    def test_rework_rate_zero_on_clean_log(self, orders_df: pd.DataFrame) -> None:
        bns = get_bottlenecks(orders_df, top_n=10)
        assert all(b.rework_rate == 0.0 for b in bns)

    def test_rework_rate_between_0_and_1(
        self, multi_variant_df: pd.DataFrame
    ) -> None:
        bns = get_bottlenecks(multi_variant_df, top_n=10)
        assert all(0.0 <= b.rework_rate <= 1.0 for b in bns)

    def test_invalid_top_n_raises(self, orders_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="top_n"):
            get_bottlenecks(orders_df, top_n=0)

    def test_empty_df_raises(self) -> None:
        empty = pd.DataFrame({
            COL_CASE: pd.Series([], dtype=str),
            COL_ACTIVITY: pd.Series([], dtype=str),
            COL_TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]"),
        })
        with pytest.raises(ValueError):
            get_bottlenecks(empty)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"case_id": ["1"], "activity": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            get_bottlenecks(df)
