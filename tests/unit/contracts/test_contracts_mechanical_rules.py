"""Tests for the mechanical (non-discretionary) rules embedded in the
contracts: research_state derivation, task status derivation, lifecycle
transition validity, and the DatasetSnapshot first-bar assertion.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from model_lab.contracts import (
    AdjustmentInfo,
    CalendarInfo,
    DataRequirement,
    DataRequirementStatus,
    DatasetSnapshot,
    LicenceStatus,
    LifecycleState,
    PitClass,
    PitRow,
    PromotionClass,
    ResearchState,
    SplitterScheme,
    SplitterSpec,
    Task,
    TaskStatus,
    TaskTarget,
    TaskUniverse,
    build_task,
    derive_promotion_class,
    derive_research_state,
    derive_task_status,
    is_influence_increase,
    is_valid_lifecycle_transition,
)
from model_lab.contracts.task import InfluenceTier, MdeSpec, ScoreSpec
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry


def _pit_row() -> PitRow:
    return PitRow(
        series_id="SPY",
        observation_period=date(2024, 1, 2),
        value="470.12",
        knowledge_time_utc=datetime(2024, 1, 2, 21, 5, tzinfo=UTC),
        source_snapshot_sha="a" * 64,
    )


def _snapshot(**overrides: object) -> DatasetSnapshot:
    kwargs = dict(
        snapshot_id="snap-1",
        source_id="alpaca_bars",
        endpoint_or_path="/v2/stocks/SPY/bars",
        retrieved_at_utc=datetime(2024, 1, 3, tzinfo=UTC),
        licence_status=LicenceStatus.UNRESOLVED,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        logical_content_hash="b" * 64,
        schema_id="ohlcv_daily.v1",
        rows=[_pit_row()],
        calendar=CalendarInfo(exchange="XNYS", timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind="none"),
        pit_class=PitClass.AS_RELEASED,
    )
    kwargs.update(overrides)
    return DatasetSnapshot(**kwargs)


# --- licence -> research_state derivation ---------------------------------


def test_derive_research_state_clear() -> None:
    assert derive_research_state(LicenceStatus.CLEAR) == ResearchState.PROMOTION_ELIGIBLE


def test_derive_research_state_unresolved() -> None:
    assert derive_research_state(LicenceStatus.UNRESOLVED) == ResearchState.RESEARCH_ONLY_UNLICENSED


def test_derive_research_state_restricted() -> None:
    assert derive_research_state(LicenceStatus.RESTRICTED) == ResearchState.QUARANTINED


def test_dataset_snapshot_rejects_mismatched_research_state() -> None:
    with pytest.raises(ValidationError, match="research_state"):
        _snapshot(licence_status=LicenceStatus.CLEAR, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED)


def test_dataset_snapshot_accepts_matching_research_state() -> None:
    snap = _snapshot(licence_status=LicenceStatus.CLEAR, research_state=ResearchState.PROMOTION_ELIGIBLE)
    assert snap.research_state == ResearchState.PROMOTION_ELIGIBLE


def test_dataset_snapshot_requires_exactly_one_row_source() -> None:
    with pytest.raises(ValidationError, match="rows"):
        _snapshot(rows=None, rows_path=None)
    with pytest.raises(ValidationError, match="rows"):
        _snapshot(rows=[_pit_row()], rows_path="snapshots/snap-1/rows.ndjson")


def test_dataset_snapshot_first_bar_assertion_requires_declared_lag() -> None:
    with pytest.raises(ValidationError, match="first_bar_lag_days"):
        _snapshot(requested_start=date(2020, 1, 1), first_bar_date=date(2020, 1, 3))


def test_dataset_snapshot_first_bar_assertion_ok_with_declared_lag() -> None:
    snap = _snapshot(
        requested_start=date(2020, 1, 1),
        first_bar_date=date(2020, 1, 3),
        first_bar_lag_days=2,
    )
    assert snap.first_bar_lag_days == 2


# --- task status derivation -------------------------------------------------


def _requirement(status: DataRequirementStatus, research_state: ResearchState | None = None) -> DataRequirement:
    return DataRequirement(
        dataset_role="etf_daily_bars",
        # P0-1: research_state can only be set alongside a dataset_id (the
        # DataRequirement-level validator enforces this even outside of any
        # Task/registry context). The tests in this section that pass a
        # research_state only ever exercise derive_promotion_class directly
        # (a pure function, no registry lookup involved), so a placeholder
        # dataset_id is sufficient here -- it never needs to resolve.
        dataset_id="placeholder-dataset-id" if research_state is not None else None,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=status,
        research_state=research_state,
    )


def test_derive_task_status_blocked_wins() -> None:
    reqs = [_requirement(DataRequirementStatus.READY), _requirement(DataRequirementStatus.BLOCKED)]
    assert derive_task_status(reqs) == TaskStatus.BLOCKED


def test_derive_task_status_partial_when_no_blocked() -> None:
    reqs = [_requirement(DataRequirementStatus.READY), _requirement(DataRequirementStatus.PARTIAL)]
    assert derive_task_status(reqs) == TaskStatus.PARTIAL


def test_derive_task_status_ready_when_all_ready() -> None:
    reqs = [_requirement(DataRequirementStatus.READY), _requirement(DataRequirementStatus.READY)]
    assert derive_task_status(reqs) == TaskStatus.DATA_READY


def test_derive_promotion_class_blocked() -> None:
    assert derive_promotion_class(TaskStatus.BLOCKED, []) == PromotionClass.BLOCKED


def test_derive_promotion_class_research_only_when_any_unlicensed() -> None:
    reqs = [_requirement(DataRequirementStatus.READY, ResearchState.RESEARCH_ONLY_UNLICENSED)]
    assert derive_promotion_class(TaskStatus.DATA_READY, reqs) == PromotionClass.RESEARCH_ONLY


def test_derive_promotion_class_promotion_ready_when_all_clear() -> None:
    reqs = [_requirement(DataRequirementStatus.READY, ResearchState.PROMOTION_ELIGIBLE)]
    assert derive_promotion_class(TaskStatus.DATA_READY, reqs) == PromotionClass.PROMOTION_READY


def _task(**overrides: object) -> Task:
    reqs = overrides.pop("data_requirements", [_requirement(DataRequirementStatus.BLOCKED)])
    status = overrides.pop("status", TaskStatus.BLOCKED)
    promotion_class = overrides.pop("promotion_class", PromotionClass.BLOCKED)
    kwargs = dict(
        task_id="T-VOL",
        version=1,
        frozen_at=datetime(2026, 9, 30, tzinfo=UTC),
        spec_sha256="c" * 64,
        target=TaskTarget(
            variable="realised_variance",
            horizon_days=[1, 5],
            aggregation="sum",
            label_end_time_rule="t+h close",
            knowledge_time_rule="close+latency",
            units="variance",
        ),
        universe=TaskUniverse(instruments=["SPY", "QQQ", "TLT", "IEF", "GLD", "USO"]),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="cpcv_6_2", scheme=SplitterScheme.CPCV, embargo_rule="h", purge_rule="afml_3cond"
        ),
        primary_score=ScoreSpec(name="QLIKE", orientation="lower_is_better", proper=True),
        mde=MdeSpec(table_id="R4_5a", assumptions="see report"),
        status=status,
        promotion_class=promotion_class,
        influence_ceiling=InfluenceTier.T0,
    )
    kwargs.update(overrides)
    # P0-1: Task construction requires an authoritative DatasetSnapshotRegistry
    # in context. Every _requirement() used in this section has dataset_id=None
    # (no research_state passed), so an empty registry is sufficient -- nothing
    # here claims a dataset_id that would need to resolve.
    return build_task(dataset_registry=DatasetSnapshotRegistry.empty(), **kwargs)


def test_task_rejects_mismatched_status() -> None:
    with pytest.raises(ValidationError, match="status"):
        _task(
            data_requirements=[_requirement(DataRequirementStatus.READY)],
            status=TaskStatus.BLOCKED,
            promotion_class=PromotionClass.BLOCKED,
        )


def test_task_accepts_matching_status() -> None:
    t = _task(
        data_requirements=[_requirement(DataRequirementStatus.BLOCKED)],
        status=TaskStatus.BLOCKED,
        promotion_class=PromotionClass.BLOCKED,
    )
    assert t.status == TaskStatus.BLOCKED


# --- lifecycle transitions ---------------------------------------------------


def test_forward_lifecycle_transition_valid() -> None:
    assert is_valid_lifecycle_transition(LifecycleState.RESEARCH, LifecycleState.REPRODUCED)
    assert is_valid_lifecycle_transition(LifecycleState.SHADOW, LifecycleState.PROSPECTIVE_CHALLENGER)


def test_skipping_a_lifecycle_stage_is_invalid() -> None:
    assert not is_valid_lifecycle_transition(LifecycleState.RESEARCH, LifecycleState.SHADOW)


def test_any_non_promoted_state_can_move_to_a_side_state() -> None:
    assert is_valid_lifecycle_transition(LifecycleState.SHADOW, LifecycleState.QUARANTINED)
    assert is_valid_lifecycle_transition(LifecycleState.RESEARCH, LifecycleState.REJECTED)


def test_promoted_can_only_move_to_superseded() -> None:
    assert is_valid_lifecycle_transition(LifecycleState.PROMOTED, LifecycleState.SUPERSEDED)
    assert not is_valid_lifecycle_transition(LifecycleState.PROMOTED, LifecycleState.QUARANTINED)


def test_is_influence_increase() -> None:
    assert is_influence_increase(InfluenceTier.T0, InfluenceTier.T1)
    assert not is_influence_increase(InfluenceTier.T2, InfluenceTier.T1)
    assert not is_influence_increase(InfluenceTier.T1, InfluenceTier.T1)
