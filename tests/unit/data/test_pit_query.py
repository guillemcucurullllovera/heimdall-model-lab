"""P1-4: PIT as-of query contract — adversarial tests proving a SNAPSHOT_ONLY
series cannot be misused as if it had genuine historical PIT depth, and that
AS_RELEASED vintage selection is correct."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from model_lab.contracts.dataset_snapshot import PitClass, PitRow
from model_lab.data.pit_query import AsOfStatus, query_as_of

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _row(period: date, value: str, knowledge_time: datetime, realtime_start: date | None = None) -> PitRow:
    return PitRow(
        series_id="X", observation_period=period, value=value, knowledge_time_utc=knowledge_time,
        realtime_start=realtime_start, source_snapshot_sha="a" * 64,
    )


# --- PitRow structural temporal ordering (adversarial review's own repro) --


def test_EXPLOIT_backdated_knowledge_time_is_rejected() -> None:
    """The review's exact repro: knowledge_time_utc six years before
    realtime_start, previously accepted silently."""
    with pytest.raises(ValidationError, match="before realtime_start"):
        PitRow(
            series_id="X", observation_period=date(2020, 1, 1), value="1",
            realtime_start=date(2026, 1, 1),
            knowledge_time_utc=datetime(2020, 1, 1, tzinfo=UTC),  # 6 years before realtime_start
            source_snapshot_sha="a" * 64,
        )


def test_knowledge_time_equal_to_realtime_start_is_allowed() -> None:
    # This is the REAL, honest pattern for SNAPSHOT_ONLY series (FRED
    # returns "today" as realtime_start for every historical row) -- not a
    # backdate, so must be accepted.
    row = PitRow(
        series_id="X", observation_period=date(2003, 1, 1), value="1",
        realtime_start=date(2026, 9, 30),
        knowledge_time_utc=datetime(2026, 9, 30, tzinfo=UTC),
        source_snapshot_sha="a" * 64,
    )
    assert row.knowledge_time_utc.date() == row.realtime_start


def test_realtime_end_before_realtime_start_is_rejected() -> None:
    with pytest.raises(ValidationError, match="before realtime_start"):
        PitRow(
            series_id="X", observation_period=date(2020, 1, 1), value="1",
            realtime_start=date(2020, 6, 1), realtime_end=date(2020, 1, 1),
            knowledge_time_utc=datetime(2020, 6, 1, tzinfo=UTC), source_snapshot_sha="a" * 64,
        )


# --- query_as_of: SNAPSHOT_ONLY misuse prevention ---------------------------


def test_EXPLOIT_snapshot_only_deep_historical_query_is_refused() -> None:
    """The exact misuse the review warned about: treating a SNAPSHOT_ONLY
    observation (single current vintage) as if it were knowable decades
    ago."""
    retrieved_at = NOW
    rows = [_row(date(1962, 1, 2), "2.5", knowledge_time=retrieved_at, realtime_start=retrieved_at.date())]
    result = query_as_of(
        rows, series_id="X", observation_period=date(1962, 1, 2),
        as_of=datetime(1962, 1, 2, tzinfo=UTC),  # asking "what was known in 1962"
        pit_class=PitClass.SNAPSHOT_ONLY, retrieved_at_utc=retrieved_at,
    )
    assert result.status == AsOfStatus.UNSAFE_SNAPSHOT_ONLY_HISTORICAL_QUERY
    assert result.value is None  # never fabricates an answer


def test_snapshot_only_query_within_safe_horizon_of_fetch_returns_the_value() -> None:
    retrieved_at = NOW
    rows = [_row(date(2026, 9, 29), "4.5", knowledge_time=retrieved_at, realtime_start=retrieved_at.date())]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2026, 9, 29),
        as_of=retrieved_at - timedelta(days=1),  # within the safe horizon
        pit_class=PitClass.SNAPSHOT_ONLY, retrieved_at_utc=retrieved_at,
    )
    assert result.status == AsOfStatus.KNOWN
    assert result.value == "4.5"


def test_snapshot_only_query_just_outside_safe_horizon_is_refused() -> None:
    retrieved_at = NOW
    rows = [_row(date(2026, 1, 1), "4.5", knowledge_time=retrieved_at, realtime_start=retrieved_at.date())]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2026, 1, 1),
        as_of=retrieved_at - timedelta(days=6),  # just past the 5-day safe horizon
        pit_class=PitClass.SNAPSHOT_ONLY, retrieved_at_utc=retrieved_at,
    )
    assert result.status == AsOfStatus.UNSAFE_SNAPSHOT_ONLY_HISTORICAL_QUERY


# --- query_as_of: AS_RELEASED genuine vintage selection ---------------------


def test_as_released_picks_the_latest_vintage_known_by_as_of() -> None:
    rows = [
        _row(date(2020, 1, 1), "first-release-100", knowledge_time=datetime(2020, 2, 1, tzinfo=UTC), realtime_start=date(2020, 2, 1)),
        _row(date(2020, 1, 1), "revised-105", knowledge_time=datetime(2020, 3, 1, tzinfo=UTC), realtime_start=date(2020, 3, 1)),
        _row(date(2020, 1, 1), "final-103", knowledge_time=datetime(2020, 6, 1, tzinfo=UTC), realtime_start=date(2020, 6, 1)),
    ]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2020, 1, 1),
        as_of=datetime(2020, 4, 1, tzinfo=UTC),  # between the 2nd and 3rd vintage
        pit_class=PitClass.AS_RELEASED, retrieved_at_utc=datetime(2020, 6, 1, tzinfo=UTC),
    )
    assert result.status == AsOfStatus.KNOWN
    assert result.value == "revised-105"  # the vintage current as of 2020-04-01, NOT the final one


def test_as_released_query_before_first_release_is_unknown() -> None:
    rows = [_row(date(2020, 1, 1), "first-release-100", knowledge_time=datetime(2020, 2, 1, tzinfo=UTC), realtime_start=date(2020, 2, 1))]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2020, 1, 1),
        as_of=datetime(2020, 1, 15, tzinfo=UTC),  # before the observation was ever released
        pit_class=PitClass.AS_RELEASED, retrieved_at_utc=datetime(2020, 2, 1, tzinfo=UTC),
    )
    assert result.status == AsOfStatus.UNKNOWN_BEFORE_KNOWLEDGE_TIME
    assert result.value is None


def test_as_released_no_hindsight_never_returns_a_future_vintage() -> None:
    """A query at t must never return a vintage whose knowledge_time is
    AFTER t, even if that vintage exists in the row set (the classic
    no-hindsight/leakage failure mode)."""
    rows = [
        _row(date(2020, 1, 1), "early", knowledge_time=datetime(2020, 2, 1, tzinfo=UTC), realtime_start=date(2020, 2, 1)),
        _row(date(2020, 1, 1), "future-revision", knowledge_time=datetime(2025, 1, 1, tzinfo=UTC), realtime_start=date(2025, 1, 1)),
    ]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2020, 1, 1),
        as_of=datetime(2020, 6, 1, tzinfo=UTC),
        pit_class=PitClass.AS_RELEASED, retrieved_at_utc=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert result.value == "early"
    assert result.value != "future-revision"


# --- AS_REVISED and missing series -------------------------------------------


def test_as_revised_is_refused_outright() -> None:
    rows = [_row(date(2020, 1, 1), "x", knowledge_time=NOW)]
    result = query_as_of(
        rows, series_id="X", observation_period=date(2020, 1, 1), as_of=NOW,
        pit_class=PitClass.AS_REVISED, retrieved_at_utc=NOW,
    )
    assert result.status == AsOfStatus.AS_REVISED_NOT_PIT_SAFE
    assert result.value is None


def test_no_matching_series_returns_explicit_status_not_none_silently() -> None:
    result = query_as_of(
        [], series_id="NOPE", observation_period=date(2020, 1, 1), as_of=NOW,
        pit_class=PitClass.AS_RELEASED, retrieved_at_utc=NOW,
    )
    assert result.status == AsOfStatus.NO_MATCHING_SERIES
