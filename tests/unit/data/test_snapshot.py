from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from model_lab.contracts.dataset_snapshot import PitRow
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.data import snapshot as snap_mod


@pytest.fixture(autouse=True)
def _isolated_data_estate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "heimdall-model-lab-data"
    monkeypatch.setattr(snap_mod, "DATA_ESTATE_ROOT", root)
    monkeypatch.setattr(snap_mod, "RAW_ROOT", root / "raw")
    monkeypatch.setattr(snap_mod, "SNAPSHOTS_ROOT", root / "snapshots")
    return root


def test_store_raw_is_content_addressed(tmp_path: Path) -> None:
    h1 = snap_mod.store_raw(b"hello world")
    h2 = snap_mod.store_raw(b"hello world")
    assert h1 == h2
    path = snap_mod.RAW_ROOT / h1[:2] / h1
    assert path.exists()
    assert path.read_bytes() == b"hello world"


def test_store_raw_different_content_different_hash() -> None:
    h1 = snap_mod.store_raw(b"a")
    h2 = snap_mod.store_raw(b"b")
    assert h1 != h2


def _row(d: date) -> PitRow:
    return PitRow(
        series_id="SPY",
        observation_period=d,
        value="100.0",
        knowledge_time_utc=datetime(d.year, d.month, d.day, tzinfo=UTC),
        source_snapshot_sha="a" * 64,
    )


def test_compute_calendar_gaps_flags_missing_real_trading_day() -> None:
    # P0-4/E.2 remediation: real XNYS calendar, not a weekday approximation
    # (2024-01-01 is New Year's Day, an exchange holiday -- a weekday
    # approximation would have wrongly treated it as an expected trading
    # day; a real calendar correctly never expects it). 2024-01-02 and
    # 2024-01-04 are real trading days; 2024-01-03 (also a real trading
    # day) is the one deliberately omitted here.
    gaps = snap_mod.compute_calendar_gaps([date(2024, 1, 2), date(2024, 1, 4)])
    assert len(gaps) == 1
    assert gaps[0].start == date(2024, 1, 3)
    assert gaps[0].end == date(2024, 1, 3)


def test_compute_calendar_gaps_no_gap_across_weekend() -> None:
    # Friday 2024-01-05 -> Monday 2024-01-08 is NOT a gap (weekend + no
    # holiday in between, per the real XNYS calendar).
    gaps = snap_mod.compute_calendar_gaps([date(2024, 1, 5), date(2024, 1, 8)])
    assert gaps == []


def test_compute_calendar_gaps_does_not_flag_a_real_holiday() -> None:
    # 2024-01-01 (New Year's Day, XNYS closed) must NOT be reported as a
    # gap -- this is exactly the false-positive the old weekday
    # approximation produced (finding E.2).
    gaps = snap_mod.compute_calendar_gaps([date(2023, 12, 29), date(2024, 1, 2)])
    assert gaps == []


def test_build_dq_report_counts_duplicates_and_gaps() -> None:
    rows = [_row(date(2024, 1, 1)), _row(date(2024, 1, 1)), _row(date(2024, 1, 3))]
    dq = snap_mod.build_dq_report(rows)
    assert dq.n_rows == 3
    assert dq.n_duplicate_dates == 1
    assert dq.n_gap_days == 1


def test_build_etf_bar_snapshot_first_bar_matches_requested_start() -> None:
    rows = [_row(date(2024, 1, 2)), _row(date(2024, 1, 3))]
    snap = snap_mod.build_etf_bar_snapshot(
        snapshot_id="test-snap-1",
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
    assert snap.first_bar_date == date(2024, 1, 2)
    assert snap.first_bar_lag_days is None
    assert snap.research_state == ResearchState.RESEARCH_ONLY_UNLICENSED


def test_build_etf_bar_snapshot_declares_lag_when_first_bar_later_than_requested() -> None:
    rows = [_row(date(2016, 1, 4)), _row(date(2016, 1, 5))]
    snap = snap_mod.build_etf_bar_snapshot(
        snapshot_id="test-snap-2",
        source_id="alpaca_bars",
        symbol="SPY",
        endpoint_or_path="/v2/stocks/SPY/bars",
        params={},
        licence_status=LicenceStatus.UNRESOLVED,
        rows=rows,
        raw_payload_hashes=["a" * 64],
        requested_start=date(1990, 1, 1),
        adjustment_kind="raw",
    )
    assert snap.first_bar_date == date(2016, 1, 4)
    assert snap.first_bar_lag_days == (date(2016, 1, 4) - date(1990, 1, 1)).days
    # The contract's own validator would have rejected this snapshot if
    # first_bar_lag_days had been left None -- constructing it successfully
    # here IS the assertion that the rule fired correctly.


def test_rows_ndjson_written_and_hashed(tmp_path: Path) -> None:
    rows = [_row(date(2024, 1, 2))]
    rel_path, content_hash = snap_mod.write_rows_ndjson("test-snap-3", rows)
    full_path = snap_mod.DATA_ESTATE_ROOT / rel_path
    assert full_path.exists()
    from model_lab.canon.jcs import sha256_bytes

    assert sha256_bytes(full_path.read_bytes()) == content_hash
