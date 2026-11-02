from __future__ import annotations

import json
from datetime import UTC, date, datetime

import numpy as np
import pytest

from model_lab.anomaly.panel import build_return_panel
from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _bar_snapshot(snapshot_id: str, ticker: str, dates: list[date], closes: list[float]) -> DatasetSnapshot:
    rows = [
        PitRow(
            series_id=ticker,
            observation_period=d,
            value=json.dumps({"o": str(c), "h": str(c), "l": str(c), "c": str(c), "v": "1000"}),
            knowledge_time_utc=NOW,
            source_snapshot_sha="a" * 64,
        )
        for d, c in zip(dates, closes, strict=True)
    ]
    return DatasetSnapshot(
        snapshot_id=snapshot_id, source_id="test", endpoint_or_path="/test", retrieved_at_utc=NOW,
        licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        logical_content_hash="b" * 64, schema_id="ohlcv_daily.v1", rows=rows,
        calendar=CalendarInfo(exchange="XNYS", timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="raw"), pit_class=PitClass.AS_RELEASED,
    )


def test_build_return_panel_aligns_on_common_dates_only() -> None:
    dates_a = [date(2024, 1, d) for d in range(1, 6)]
    dates_b = [date(2024, 1, d) for d in (1, 2, 4, 5, 6)]  # missing 1/3, has an extra 1/6
    snap_a = _bar_snapshot("A_raw", "A", dates_a, [100.0, 101.0, 102.0, 103.0, 104.0])
    snap_b = _bar_snapshot("B_raw", "B", dates_b, [50.0, 51.0, 52.0, 53.0, 54.0])
    registry = DatasetSnapshotRegistry.from_snapshots([snap_a, snap_b])

    panel = build_return_panel(["A", "B"], registry, dataset_id_map={"A": "A_raw", "B": "B_raw"})
    # Common return dates: A has returns on 1/2,1/3,1/4,1/5; B has returns on 1/2,1/4,1/5,1/6.
    # Intersection: 1/2, 1/4, 1/5.
    assert panel.dates == ["2024-01-02", "2024-01-04", "2024-01-05"]
    assert panel.returns.shape == (3, 2)


def test_build_return_panel_raises_on_disjoint_dates() -> None:
    dates_a = [date(2024, 1, 1), date(2024, 1, 2)]
    dates_b = [date(2025, 6, 1), date(2025, 6, 2)]
    snap_a = _bar_snapshot("A_raw", "A", dates_a, [100.0, 101.0])
    snap_b = _bar_snapshot("B_raw", "B", dates_b, [50.0, 51.0])
    registry = DatasetSnapshotRegistry.from_snapshots([snap_a, snap_b])
    with pytest.raises(ValueError, match="no common dates"):
        build_return_panel(["A", "B"], registry, dataset_id_map={"A": "A_raw", "B": "B_raw"})


def test_build_return_panel_values_match_manual_log_returns() -> None:
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    snap_a = _bar_snapshot("A_raw", "A", dates, [100.0, 110.0, 99.0])
    snap_b = _bar_snapshot("B_raw", "B", dates, [10.0, 10.5, 10.2])
    registry = DatasetSnapshotRegistry.from_snapshots([snap_a, snap_b])
    panel = build_return_panel(["A", "B"], registry, dataset_id_map={"A": "A_raw", "B": "B_raw"})
    np.testing.assert_allclose(panel.returns[:, 0], np.diff(np.log([100.0, 110.0, 99.0])))
    np.testing.assert_allclose(panel.returns[:, 1], np.diff(np.log([10.0, 10.5, 10.2])))
