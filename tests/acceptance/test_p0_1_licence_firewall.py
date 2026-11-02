"""P0-1 exploit regression — adversarial review finding C.1: "Licence/
promotion firewall is bypassable — source authority is not cryptographically
or structurally bound, only trusted metadata."

Reproduces the review's exact working repro (a fabricated `DataRequirement`
claiming `PROMOTION_ELIGIBLE` with `dataset_id=None`, validating with zero
errors and yielding `promotion_class=PROMOTION_READY`) and proves it now
fails closed, plus the full set of adjacent bypass attempts the remediation
brief calls for.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

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
    InfluenceTier,
    MdeSpec,
    PromotionClass,
    ScoreSpec,
    SplitterScheme,
    SplitterSpec,
    Task,
    TaskStatus,
    TaskTarget,
    TaskUniverse,
    build_task,
)
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry

pytestmark = pytest.mark.acceptance

NOW = datetime(2026, 9, 30, tzinfo=UTC)


def _base_task_kwargs(data_requirements: list[DataRequirement], *, status: TaskStatus, promotion_class: PromotionClass) -> dict:
    return dict(
        task_id="T-EXPLOIT",
        version=1,
        frozen_at=NOW,
        spec_sha256="a" * 64,
        target=TaskTarget(
            variable="x", horizon_days=[1], aggregation="sum", label_end_time_rule="r",
            knowledge_time_rule="r", units="u",
        ),
        universe=TaskUniverse(instruments=["SPY"]),
        data_requirements=data_requirements,
        splitter=SplitterSpec(config_id="c", scheme=SplitterScheme.EXPANDING, embargo_rule="e", purge_rule="p"),
        primary_score=ScoreSpec(name="s", orientation="lower_is_better", proper=True),
        mde=MdeSpec(table_id="t", assumptions="a"),
        status=status,
        promotion_class=promotion_class,
        influence_ceiling=InfluenceTier.T0,
    )


def _real_snapshot(
    snapshot_id: str,
    *,
    licence_status: LicenceStatus,
    research_state: ResearchState,
    adjustment_kind: str = "raw",
    params: dict | None = None,
    rows: list[PitRow] | None = None,
    first_bar_date: date | None = None,
    last_bar_date: date | None = None,
) -> DatasetSnapshot:
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        source_id="test_source",
        endpoint_or_path="/test",
        params=params or {},
        retrieved_at_utc=NOW,
        licence_status=licence_status,
        research_state=research_state,
        logical_content_hash="b" * 64,
        schema_id="test.v1",
        rows=rows
        if rows is not None
        else [
            PitRow(
                series_id="X", observation_period=date(2024, 1, 2), value="1",
                knowledge_time_utc=NOW, source_snapshot_sha="c" * 64,
            )
        ],
        calendar=CalendarInfo(exchange="XNYS", timezone="America/New_York"),
        adjustments=AdjustmentInfo(kind=adjustment_kind),
        pit_class=PitClass.AS_RELEASED,
        first_bar_date=first_bar_date,
        last_bar_date=last_bar_date,
    )


def _corporate_actions_snapshot(
    snapshot_id: str, *, ticker: str, split_events: list[PitRow] | None = None, params_override: dict | None = None
) -> DatasetSnapshot:
    """A real-shaped `corporate_actions_<TICKER>` companion snapshot, mirroring
    `scripts/build_corporate_actions_and_adjusted_uso.py`: `params={"ticker": ...}`,
    one row per real split event (`series_id=ticker`), zero rows means zero events."""
    return _real_snapshot(
        snapshot_id,
        licence_status=LicenceStatus.UNRESOLVED,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="none",
        params=params_override if params_override is not None else {"ticker": ticker},
        rows=split_events if split_events is not None else [],
    )


# --- the review's exact repro ------------------------------------------------


def test_EXPLOIT_fabricated_promotion_eligible_with_no_dataset_id_is_rejected() -> None:
    """The review's literal repro: DataRequirement(dataset_id=None,
    research_state=PROMOTION_ELIGIBLE, ...) — must now be rejected at
    DataRequirement construction time, before it could ever reach a Task."""
    with pytest.raises(ValidationError, match="dataset_id"):
        DataRequirement(
            dataset_role="etf_daily_ohlcv",
            dataset_id=None,
            research_state=ResearchState.PROMOTION_ELIGIBLE,
            min_depth_years=10.0,
            frequency="daily",
            pit_required=False,
            status=DataRequirementStatus.READY,
        )


def test_EXPLOIT_direct_task_construction_without_registry_fails_closed() -> None:
    """Even if a fabricated DataRequirement somehow carried a dataset_id
    (the next layer down), direct `Task(**kwargs)` construction — no
    registry in context — must refuse to build at all."""
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv",
        dataset_id="fabricated-id",
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.PROMOTION_READY)
    with pytest.raises(ValidationError, match="DatasetSnapshotRegistry"):
        Task(**kwargs)


def test_EXPLOIT_fabricated_dataset_id_that_does_not_resolve_fails_closed() -> None:
    """build_task WITH a registry, but the claimed dataset_id names nothing
    real — must fail closed (P0-1 requirement 5), not silently pass."""
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv",
        dataset_id="totally-fabricated-snapshot-id",
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.PROMOTION_READY)
    with pytest.raises(ValidationError, match="does not resolve"):
        build_task(dataset_registry=DatasetSnapshotRegistry.empty(), **kwargs)


def test_EXPLOIT_claiming_stronger_authority_than_the_real_snapshot_fails_closed() -> None:
    """The real snapshot exists and is UNRESOLVED/research_only_unlicensed,
    but the DataRequirement claims PROMOTION_ELIGIBLE for it -- a lie about
    an otherwise-real dataset. Must fail closed."""
    real_snapshot = _real_snapshot(
        "real-but-unlicensed", licence_status=LicenceStatus.UNRESOLVED,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
    )
    registry = DatasetSnapshotRegistry.from_snapshots([real_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv",
        dataset_id="real-but-unlicensed",
        research_state=ResearchState.PROMOTION_ELIGIBLE,  # LIE: registry says research_only_unlicensed
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.PROMOTION_READY)
    with pytest.raises(ValidationError, match="actually has research_state"):
        build_task(dataset_registry=registry, **kwargs)


def test_legitimate_promotion_ready_task_with_a_real_clear_snapshot_succeeds() -> None:
    """Positive control: a genuinely CLEAR/promotion_eligible real snapshot,
    honestly claimed, DOES produce a valid PROMOTION_READY task -- the fix
    closes the fabrication bypass without blocking legitimate evidence."""
    real_snapshot = _real_snapshot(
        "real-and-clear", licence_status=LicenceStatus.CLEAR, research_state=ResearchState.PROMOTION_ELIGIBLE,
    )
    registry = DatasetSnapshotRegistry.from_snapshots([real_snapshot])
    req = DataRequirement(
        dataset_role="macro_series",
        dataset_id="real-and-clear",
        research_state=ResearchState.PROMOTION_ELIGIBLE,
        min_depth_years=1.0,
        frequency="daily",
        pit_required=True,
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.PROMOTION_READY)
    task = build_task(dataset_registry=registry, **kwargs)
    assert task.promotion_class == PromotionClass.PROMOTION_READY


def test_research_only_snapshot_correctly_caps_at_research_only_never_promotion_ready() -> None:
    real_snapshot = _real_snapshot(
        "real-unlicensed", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
    )
    registry = DatasetSnapshotRegistry.from_snapshots([real_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv",
        dataset_id="real-unlicensed",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    task = build_task(dataset_registry=registry, **kwargs)
    assert task.promotion_class == PromotionClass.RESEARCH_ONLY


# --- P0-4: adjustment gating -------------------------------------------------


def test_EXPLOIT_P0_4_unadjusted_raw_snapshot_claiming_adjustment_satisfied_fails_closed() -> None:
    """The USO gap, reproduced directly: a requirement DECLARES it needs a
    corporate-action-safe series (adjustments=[...]) but points at a raw
    (unadjusted) snapshot, with no companion adjustment_dataset_id either.
    Must fail closed."""
    raw_snapshot = _real_snapshot(
        "uso-raw", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="raw",
    )
    registry = DatasetSnapshotRegistry.from_snapshots([raw_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_USO",
        dataset_id="uso-raw",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    with pytest.raises(ValidationError, match="corporate-action-safety"):
        build_task(dataset_registry=registry, **kwargs)


def test_P0_4_adjusted_snapshot_satisfies_the_requirement() -> None:
    adjusted_snapshot = _real_snapshot(
        "uso-adjusted", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="split_adjusted",
    )
    registry = DatasetSnapshotRegistry.from_snapshots([adjusted_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_USO",
        dataset_id="uso-adjusted",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    task = build_task(dataset_registry=registry, **kwargs)
    assert task.status == TaskStatus.DATA_READY


def test_P0_4_companion_corporate_actions_snapshot_satisfies_the_requirement_for_raw_data() -> None:
    """A ticker with ZERO corporate actions in-window is genuinely safe to
    use raw, evidenced by a real corporate-actions snapshot confirming this
    -- not by transforming the price series itself."""
    raw_snapshot = _real_snapshot(
        "spy-raw", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="raw",
    )
    ca_snapshot = _corporate_actions_snapshot("corporate_actions_SPY", ticker="SPY", split_events=[])
    registry = DatasetSnapshotRegistry.from_snapshots([raw_snapshot, ca_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_SPY",
        dataset_id="spy-raw",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id="corporate_actions_SPY",
        instrument="SPY",
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    task = build_task(dataset_registry=registry, **kwargs)
    assert task.status == TaskStatus.DATA_READY


def test_EXPLOIT_fabricated_adjustment_dataset_id_that_does_not_resolve_fails_closed() -> None:
    raw_snapshot = _real_snapshot(
        "spy-raw2", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="raw",
    )
    registry = DatasetSnapshotRegistry.from_snapshots([raw_snapshot])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_SPY",
        dataset_id="spy-raw2",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id="fabricated-corporate-actions-id",
        instrument="SPY",
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    with pytest.raises(ValidationError, match="does not resolve"):
        build_task(dataset_registry=registry, **kwargs)


# --- P1 (MODEL-2A, carried from MODEL1_REMEDIATION_REREVIEW.md sec I): -----
# companion existence is necessary but no longer sufficient -- its CONTENT
# must confirm zero in-window events for the right instrument.


def test_EXPLOIT_P1_companion_with_real_documented_split_no_longer_satisfies_raw_primary() -> None:
    """The rereview's exact new finding, reproduced directly: a requirement
    pairs a RAW (contaminated) primary dataset with a companion that itself
    documents a REAL, unmitigated split for that instrument. Previously
    ACCEPTED (existence alone was treated as "zero events"); must now be
    REJECTED (fail closed) -- the companion's content proves the opposite of
    what the requirement claims."""
    uso_raw = _real_snapshot(
        "uso-raw-contaminated", licence_status=LicenceStatus.UNRESOLVED,
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED, adjustment_kind="raw",
        first_bar_date=date(2016, 1, 4), last_bar_date=date(2026, 9, 29),
    )
    real_split_row = PitRow(
        series_id="USO", observation_period=date(2020, 4, 29),
        value='{"split_from": "8", "split_to": "1"}',
        knowledge_time_utc=datetime(2020, 4, 29, tzinfo=UTC), source_snapshot_sha="d" * 64,
    )
    ca_uso = _corporate_actions_snapshot("corporate_actions_USO", ticker="USO", split_events=[real_split_row])
    registry = DatasetSnapshotRegistry.from_snapshots([uso_raw, ca_uso])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_USO",
        dataset_id="uso-raw-contaminated",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id="corporate_actions_USO",
        instrument="USO",
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    with pytest.raises(ValidationError, match="corporate-action-safety"):
        build_task(dataset_registry=registry, **kwargs)


def test_EXPLOIT_P1_companion_scoped_to_a_different_instrument_fails_closed() -> None:
    """A companion that genuinely has zero rows -- but for the WRONG ticker
    -- must not be treated as evidence for this instrument. Mismatched
    instrument scope fails closed."""
    qqq_raw = _real_snapshot(
        "qqq-raw", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="raw",
    )
    ca_wrong_ticker = _corporate_actions_snapshot("corporate_actions_SPY", ticker="SPY", split_events=[])
    registry = DatasetSnapshotRegistry.from_snapshots([qqq_raw, ca_wrong_ticker])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_QQQ",
        dataset_id="qqq-raw",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id="corporate_actions_SPY",  # wrong companion for QQQ
        instrument="QQQ",
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    with pytest.raises(ValidationError, match="corporate-action-safety"):
        build_task(dataset_registry=registry, **kwargs)


def test_EXPLOIT_P1_adjustment_dataset_id_without_instrument_fails_closed_at_construction() -> None:
    """A DataRequirement cannot set adjustment_dataset_id without naming the
    instrument it's supposed to be scoped to -- fails closed at
    DataRequirement construction, before it could ever reach a Task."""
    with pytest.raises(ValidationError, match="instrument"):
        DataRequirement(
            dataset_role="etf_daily_ohlcv_SPY",
            dataset_id="spy-raw",
            research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
            min_depth_years=10.0,
            frequency="daily",
            pit_required=False,
            adjustments=["corporate_actions_snapshot"],
            adjustment_dataset_id="corporate_actions_SPY",
            status=DataRequirementStatus.READY,
        )


def test_EXPLOIT_P1_companion_bounded_window_not_covering_primary_history_fails_closed() -> None:
    """Defensive check for a future windowed companion fetch (today's real
    fetch_splits is unbounded): a companion that declares its own fetch
    window (params start/end) which does NOT cover the primary's actual bar
    history cannot certify the whole of it -- mismatched date scope fails
    closed even with zero rows in that partial window."""
    spy_raw = _real_snapshot(
        "spy-raw3", licence_status=LicenceStatus.UNRESOLVED, research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        adjustment_kind="raw", first_bar_date=date(2016, 1, 4), last_bar_date=date(2026, 9, 29),
    )
    ca_partial_window = _corporate_actions_snapshot(
        "corporate_actions_SPY_partial", ticker="SPY", split_events=[],
        params_override={"ticker": "SPY", "start": "2020-01-01", "end": "2021-01-01"},
    )
    registry = DatasetSnapshotRegistry.from_snapshots([spy_raw, ca_partial_window])
    req = DataRequirement(
        dataset_role="etf_daily_ohlcv_SPY",
        dataset_id="spy-raw3",
        research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id="corporate_actions_SPY_partial",
        instrument="SPY",
        status=DataRequirementStatus.READY,
    )
    kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
    with pytest.raises(ValidationError, match="corporate-action-safety"):
        build_task(dataset_registry=registry, **kwargs)


def test_P1_companion_zero_events_via_rows_path_on_disk_is_read_correctly() -> None:
    """The content check must actually READ rows_path-backed snapshots (the
    real on-disk shape, per `data/snapshot.py::build_reference_snapshot`),
    not just embedded `rows` -- exercises `DatasetSnapshotRegistry.get_rows`'s
    disk-loading branch end to end against a real temp file."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        import model_lab.data.snapshot as snapshot_mod

        original_root = snapshot_mod.DATA_ESTATE_ROOT
        snapshot_mod.DATA_ESTATE_ROOT = Path(tmpdir)
        try:
            snap_dir = Path(tmpdir) / "snapshots" / "corporate_actions_GLD"
            snap_dir.mkdir(parents=True)
            rows_file = snap_dir / "rows.ndjson"
            rows_file.write_text("", encoding="utf-8")  # zero events, on disk

            ca_gld = DatasetSnapshot(
                snapshot_id="corporate_actions_GLD",
                source_id="massive_reference_splits",
                endpoint_or_path="/v3/reference/splits",
                params={"ticker": "GLD"},
                retrieved_at_utc=NOW,
                licence_status=LicenceStatus.UNRESOLVED,
                research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
                logical_content_hash="e" * 64,
                schema_id="corporate_action_splits.v1",
                rows=None,
                rows_path=str(rows_file.relative_to(Path(tmpdir))),
                calendar=CalendarInfo(exchange=None, timezone="America/New_York"),
                adjustments=AdjustmentInfo(kind="none"),
                pit_class=PitClass.SNAPSHOT_ONLY,
            )
            gld_raw = _real_snapshot(
                "gld-raw", licence_status=LicenceStatus.UNRESOLVED,
                research_state=ResearchState.RESEARCH_ONLY_UNLICENSED, adjustment_kind="raw",
            )
            registry = DatasetSnapshotRegistry.from_snapshots([gld_raw, ca_gld])
            req = DataRequirement(
                dataset_role="etf_daily_ohlcv_GLD",
                dataset_id="gld-raw",
                research_state=ResearchState.RESEARCH_ONLY_UNLICENSED,
                min_depth_years=10.0,
                frequency="daily",
                pit_required=False,
                adjustments=["corporate_actions_snapshot"],
                adjustment_dataset_id="corporate_actions_GLD",
                instrument="GLD",
                status=DataRequirementStatus.READY,
            )
            kwargs = _base_task_kwargs([req], status=TaskStatus.DATA_READY, promotion_class=PromotionClass.RESEARCH_ONLY)
            task = build_task(dataset_registry=registry, **kwargs)
            assert task.status == TaskStatus.DATA_READY
            assert json.loads(rows_file.read_text() or "null") is None  # sanity: file genuinely empty
        finally:
            snapshot_mod.DATA_ESTATE_ROOT = original_root


# --- real production registry, end-to-end -----------------------------------


def test_real_task_registry_builds_cleanly_against_the_real_data_estate() -> None:
    """Confirms the whole chain works against the ACTUAL on-disk data estate
    (not a synthetic fixture) -- build_default_task_registry() must not
    raise, and T-TAIL must be the one unlocked task."""
    from model_lab.tasks.registry import build_default_task_registry

    registry = build_default_task_registry()
    assert registry["T-TAIL"].status == TaskStatus.DATA_READY
    assert registry["T-TAIL"].promotion_class == PromotionClass.RESEARCH_ONLY
    for task_id, task in registry.items():
        if task_id != "T-TAIL":
            assert task.status == TaskStatus.BLOCKED
