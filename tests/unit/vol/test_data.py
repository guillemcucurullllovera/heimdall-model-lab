from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pytest

from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.vol.data import load_return_series

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _bar_snapshot(snapshot_id: str, ticker: str, closes: list[float]) -> DatasetSnapshot:
    rows = [
        PitRow(
            series_id=ticker,
            observation_period=date(2024, 1, 1 + i),
            value=json.dumps({"o": str(c), "h": str(c), "l": str(c), "c": str(c), "v": "1000"}),
            knowledge_time_utc=NOW,
            source_snapshot_sha="a" * 64,
        )
        for i, c in enumerate(closes)
    ]
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        source_id="test",
        endpoint_or_path="/test",
        retrieved_at_utc=NOW,
        licence_status=LicenceStatus.UNRESOLVED,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        logical_content_hash="b" * 64,
        schema_id="ohlcv_daily.v1",
        rows=rows,
        calendar=CalendarInfo(exchange="XNYS", timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="raw"),
        pit_class=PitClass.AS_RELEASED,
    )


def test_load_return_series_derives_correct_log_returns() -> None:
    closes = [100.0, 110.0, 99.0, 108.9]
    snap = _bar_snapshot("test_TICK_daily_raw", "TICK", closes)
    registry = DatasetSnapshotRegistry.from_snapshots([snap])
    series = load_return_series("TICK", registry, dataset_id_map={"TICK": "test_TICK_daily_raw"})
    assert series.ticker == "TICK"
    assert len(series.closes) == 4
    assert len(series.log_returns) == 3
    np.testing.assert_allclose(series.log_returns, np.diff(np.log(np.array(closes))))
    assert series.data_fingerprint == "b" * 64


def test_load_return_series_fails_closed_on_unregistered_dataset() -> None:
    registry = DatasetSnapshotRegistry.empty()
    with pytest.raises(ValueError, match="not registered"):
        load_return_series("TICK", registry, dataset_id_map={"TICK": "does-not-exist"})


def test_load_return_series_filters_to_the_right_series_id() -> None:
    """A registry row list might in principle contain other series_ids
    (not true for real ETF bar snapshots today, but the loader must not
    silently mix them in if it ever happened)."""
    closes = [100.0, 101.0, 102.0]
    snap = _bar_snapshot("mixed_daily_raw", "TICK", closes)
    extra_row = PitRow(
        series_id="OTHER",
        observation_period=date(2024, 1, 5),
        value=json.dumps({"o": "1", "h": "1", "l": "1", "c": "1", "v": "1"}),
        knowledge_time_utc=NOW,
        source_snapshot_sha="a" * 64,
    )
    snap_with_extra = snap.model_copy(update={"rows": [*snap.rows, extra_row]})
    registry = DatasetSnapshotRegistry.from_snapshots([snap_with_extra])
    series = load_return_series("TICK", registry, dataset_id_map={"TICK": "mixed_daily_raw"})
    assert len(series.closes) == 3  # OTHER's row excluded
