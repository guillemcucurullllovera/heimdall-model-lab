"""AT-11: PIT/vintage tests (as-released vs as-revised); first-bar
assertion; DQ suites; licence propagation (an UNRESOLVED source cannot feed
a promotion-evidence task)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.contracts.task import (
    DataRequirement,
    DataRequirementStatus,
    PromotionClass,
    TaskStatus,
    derive_promotion_class,
    derive_task_status,
)
from model_lab.data import snapshot as snap_mod
from model_lab.data.registry import default_source_registry, get_entry

pytestmark = pytest.mark.acceptance


# --- PIT / vintage: as-released vs as-revised ------------------------------


def test_as_released_snapshot_carries_pit_class() -> None:
    row = PitRow(
        series_id="UNRATE",
        observation_period=date(2024, 1, 1),
        value="3.7",
        realtime_start=date(2024, 2, 2),
        realtime_end=date(9999, 12, 31),
        knowledge_time_utc=datetime(2024, 2, 2, tzinfo=UTC),
        source_snapshot_sha="a" * 64,
    )
    snap = DatasetSnapshot(
        snapshot_id="s1",
        source_id="fred_alfred",
        endpoint_or_path="/fred/series/observations",
        retrieved_at_utc=datetime(2024, 2, 3, tzinfo=UTC),
        licence_status=LicenceStatus.CLEAR,
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        logical_content_hash="b" * 64,
        schema_id="macro_pit_vintage.v1",
        rows=[row],
        calendar=CalendarInfo(exchange=None, timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="none"),
        pit_class=PitClass.AS_RELEASED,
    )
    assert snap.pit_class == PitClass.AS_RELEASED
    # knowledge_time_utc (our conservative available_at) must not precede
    # the vintage's own realtime_start -- a PIT snapshot can never claim to
    # have known a value before FRED itself published that vintage.
    assert snap.rows[0].knowledge_time_utc.date() >= snap.rows[0].realtime_start


def test_as_revised_series_flagged_snapshot_only_when_vintage_query_degrades() -> None:
    # Mirrors pit_macro.py's real behaviour: a series whose vintage query
    # degrades (e.g. DGS10 exceeding FRED's 2000-vintage-date cap) is
    # recorded as SNAPSHOT_ONLY, not silently treated as full AS_RELEASED
    # PIT history -- §9 rule 5: AS_REVISED-style/degraded series must be
    # distinguishable and excluded from any PIT-required task.
    row = PitRow(
        series_id="DGS10",
        observation_period=date(2024, 1, 1),
        value="4.1",
        realtime_start=date(2026, 9, 30),
        realtime_end=date(9999, 12, 31),
        knowledge_time_utc=datetime(2026, 9, 30, tzinfo=UTC),
        source_snapshot_sha="a" * 64,
    )
    snap = DatasetSnapshot(
        snapshot_id="s2",
        source_id="fred_alfred",
        endpoint_or_path="/fred/series/observations",
        retrieved_at_utc=datetime(2026, 9, 30, tzinfo=UTC),
        licence_status=LicenceStatus.CLEAR,
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        logical_content_hash="c" * 64,
        schema_id="macro_pit_vintage.v1",
        rows=[row],
        calendar=CalendarInfo(exchange=None, timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="none"),
        pit_class=PitClass.SNAPSHOT_ONLY,
    )
    assert snap.pit_class == PitClass.SNAPSHOT_ONLY
    assert snap.pit_class != PitClass.AS_RELEASED


# --- first-bar assertion ---------------------------------------------------


def test_first_bar_assertion_rejects_undeclared_lag() -> None:
    with pytest.raises(ValidationError, match="first_bar_lag_days"):
        DatasetSnapshot(
            snapshot_id="s3",
            source_id="alpaca_bars",
            endpoint_or_path="/v2/stocks/SPY/bars",
            retrieved_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
            licence_status=LicenceStatus.UNRESOLVED,
            research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
            logical_content_hash="d" * 64,
            schema_id="ohlcv_daily.v1",
            rows_path="snapshots/s3/rows.ndjson",
            calendar=CalendarInfo(exchange="XNYS", timezone="America/New_York"),
            adjustments=AdjustmentInfo(kind="raw"),
            pit_class=PitClass.AS_RELEASED,
            requested_start=date(1990, 1, 1),
            first_bar_date=date(2016, 1, 4),
        )


def test_real_acquired_snapshots_declare_first_bar_lag_honestly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduces the real bootstrap's finding: Alpaca's actual first bar for
    every ETF (2016-01-04) is far later than the requested_start
    (1990-01-01) -- the contract must carry an explicit, non-zero
    first_bar_lag_days rather than silently accepting the clipped window.

    Uses `monkeypatch` (not a bare module-attribute assignment) so these
    globals are restored after the test -- a prior direct-assignment version
    of this test permanently repointed `model_lab.data.snapshot.
    DATA_ESTATE_ROOT`/`SNAPSHOTS_ROOT` at this test's own `tmp_path` for the
    REST of the test session, silently breaking any later test in the same
    process that reads those globals at call time (found when MODEL-2A's
    `DatasetSnapshotRegistry.get_rows` started doing exactly that)."""
    rows = [
        PitRow(
            series_id="SPY",
            observation_period=date(2016, 1, 4),
            value='{"c": "200.49"}',
            knowledge_time_utc=datetime(2016, 1, 4, tzinfo=UTC),
            source_snapshot_sha="e" * 64,
        )
    ]
    import model_lab.data.snapshot as sm

    monkeypatch.setattr(sm, "DATA_ESTATE_ROOT", tmp_path)
    monkeypatch.setattr(sm, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(sm, "SNAPSHOTS_ROOT", tmp_path / "snapshots")

    snap = sm.build_etf_bar_snapshot(
        snapshot_id="at11-real-shape",
        source_id="alpaca_bars",
        symbol="SPY",
        endpoint_or_path="/v2/stocks/SPY/bars",
        params={"start": "1990-01-01", "end": "2026-09-29"},
        licence_status=LicenceStatus.UNRESOLVED,
        rows=rows,
        raw_payload_hashes=["f" * 64],
        requested_start=date(1990, 1, 1),
        adjustment_kind="raw",
    )
    assert snap.first_bar_lag_days == (date(2016, 1, 4) - date(1990, 1, 1)).days
    assert snap.first_bar_lag_days == 9499


# --- DQ suite ---------------------------------------------------------------


def test_dq_report_written_and_hash_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snap_mod, "DATA_ESTATE_ROOT", tmp_path)
    monkeypatch.setattr(snap_mod, "RAW_ROOT", tmp_path / "raw")
    monkeypatch.setattr(snap_mod, "SNAPSHOTS_ROOT", tmp_path / "snapshots")

    rows = [
        PitRow(
            series_id="SPY",
            observation_period=date(2024, 1, 2),
            value="{}",
            knowledge_time_utc=datetime(2024, 1, 2, tzinfo=UTC),
            source_snapshot_sha="a" * 64,
        )
    ]
    snap = snap_mod.build_etf_bar_snapshot(
        snapshot_id="at11-dq",
        source_id="alpaca_bars",
        symbol="SPY",
        endpoint_or_path="/v2/stocks/SPY/bars",
        params={},
        licence_status=LicenceStatus.UNRESOLVED,
        rows=rows,
        raw_payload_hashes=["a" * 64],
        requested_start=date(2024, 1, 2),
        adjustment_kind="raw",
    )
    assert snap.dq_report_hash is not None
    dq_path = snap_mod.SNAPSHOTS_ROOT / "at11-dq" / "dq_report.json"
    assert dq_path.exists()


# --- licence propagation: UNRESOLVED source cannot feed promotion evidence -


def test_unresolved_source_research_state_blocks_promotion_class() -> None:
    entry = get_entry("alpaca_bars", default_source_registry())
    assert entry.licence_status == LicenceStatus.UNRESOLVED
    assert entry.research_state == ResearchState.RESEARCH_ONLY_UNLICENSED

    req = DataRequirement(
        dataset_role="etf_daily_bars",
        dataset_id="alpaca_SPY_daily_raw",
        min_depth_years=1.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
        research_state=entry.research_state,
    )
    task_status = derive_task_status([req])
    promotion_class = derive_promotion_class(task_status, [req])
    assert task_status == TaskStatus.DATA_READY  # coverage is fine...
    assert promotion_class == PromotionClass.RESEARCH_ONLY  # ...but licence caps it, mechanically
    assert promotion_class != PromotionClass.PROMOTION_READY


def test_clear_source_can_reach_promotion_ready() -> None:
    entry = get_entry("fred_alfred", default_source_registry())
    req = DataRequirement(
        dataset_role="macro_pit",
        dataset_id="macro_pit_INDPRO",
        min_depth_years=1.0,
        frequency="monthly",
        pit_required=True,
        status=DataRequirementStatus.READY,
        research_state=entry.research_state,
    )
    task_status = derive_task_status([req])
    promotion_class = derive_promotion_class(task_status, [req])
    assert promotion_class == PromotionClass.PROMOTION_READY


def test_mixed_requirements_one_unresolved_caps_the_whole_task() -> None:
    clear_req = DataRequirement(
        dataset_role="macro_pit",
        dataset_id="placeholder-macro-pit",  # P0-1: research_state requires a dataset_id
        min_depth_years=1.0,
        frequency="monthly",
        pit_required=True,
        status=DataRequirementStatus.READY,
        research_state=ResearchState.PROMOTION_ELIGIBLE,
    )
    unresolved_req = DataRequirement(
        dataset_role="etf_daily_bars",
        dataset_id="placeholder-etf-bars",  # P0-1: research_state requires a dataset_id
        min_depth_years=1.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
    )
    task_status = derive_task_status([clear_req, unresolved_req])
    promotion_class = derive_promotion_class(task_status, [clear_req, unresolved_req])
    assert promotion_class == PromotionClass.RESEARCH_ONLY
