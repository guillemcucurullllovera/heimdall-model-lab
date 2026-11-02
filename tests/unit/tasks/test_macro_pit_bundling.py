"""P1-3: macro_pit_panel's bundled status must be mechanically derived from
each field's real `pit_class`, never hand-set."""

from __future__ import annotations

from datetime import UTC, date, datetime

from model_lab.contracts.dataset_snapshot import (
    AdjustmentInfo,
    CalendarInfo,
    DatasetSnapshot,
    PitClass,
    PitRow,
)
from model_lab.contracts.licence import LicenceStatus, ResearchState
from model_lab.contracts.task import DataRequirementStatus
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import MACRO_PIT_DATASET_IDS, build_default_task_registry

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _macro_snapshot(snapshot_id: str, pit_class: PitClass) -> DatasetSnapshot:
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        source_id="fred_alfred",
        endpoint_or_path="/fred/series/observations",
        retrieved_at_utc=NOW,
        licence_status=LicenceStatus.CLEAR,
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        logical_content_hash="a" * 64,
        schema_id="macro_pit_vintage.v1",
        rows=[PitRow(series_id="X", observation_period=date(2024, 1, 2), value="1", knowledge_time_utc=NOW, source_snapshot_sha="b" * 64)],
        calendar=CalendarInfo(exchange=None, timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="none"),
        pit_class=pit_class,
    )


def test_all_fields_as_released_gives_ready_bundle() -> None:
    registry = DatasetSnapshotRegistry.from_snapshots(
        [_macro_snapshot(sid, PitClass.AS_RELEASED) for sid in MACRO_PIT_DATASET_IDS.values()]
    )
    reg = build_default_task_registry(registry)
    macro_req = next(r for r in reg["T-STATE"].data_requirements if r.dataset_role == "macro_pit_panel")
    assert macro_req.status == DataRequirementStatus.READY
    assert macro_req.blockers == []


def test_one_snapshot_only_field_downgrades_bundle_to_partial() -> None:
    fields = list(MACRO_PIT_DATASET_IDS.items())
    snapshots = []
    for i, (_field, sid) in enumerate(fields):
        pit_class = PitClass.SNAPSHOT_ONLY if i == 0 else PitClass.AS_RELEASED
        snapshots.append(_macro_snapshot(sid, pit_class))
    registry = DatasetSnapshotRegistry.from_snapshots(snapshots)
    reg = build_default_task_registry(registry)
    macro_req = next(r for r in reg["T-STATE"].data_requirements if r.dataset_role == "macro_pit_panel")
    assert macro_req.status == DataRequirementStatus.PARTIAL
    assert fields[0][0] in macro_req.blockers[0]


def test_one_missing_field_blocks_the_whole_bundle() -> None:
    fields = list(MACRO_PIT_DATASET_IDS.items())
    snapshots = [_macro_snapshot(sid, PitClass.AS_RELEASED) for _field, sid in fields[1:]]  # skip the first field
    registry = DatasetSnapshotRegistry.from_snapshots(snapshots)
    reg = build_default_task_registry(registry)
    macro_req = next(r for r in reg["T-STATE"].data_requirements if r.dataset_role == "macro_pit_panel")
    assert macro_req.status == DataRequirementStatus.BLOCKED
    assert fields[0][0] in macro_req.blockers[0]


def test_real_data_estate_has_five_snapshot_only_fields() -> None:
    """Confirms against the ACTUAL on-disk data estate: T10YIE/DFF/DGS3MO/
    DGS2/DGS10 are the five degraded (SNAPSHOT_ONLY) fields, matching the
    adversarial review's finding D.4 exactly."""
    reg = build_default_task_registry()
    macro_req = next(r for r in reg["T-STATE"].data_requirements if r.dataset_role == "macro_pit_panel")
    assert macro_req.status == DataRequirementStatus.PARTIAL
    for expected_field in ("inflation_breakeven", "policy_rate", "yield_3m", "yield_2y", "yield_10y"):
        assert expected_field in macro_req.blockers[0]
