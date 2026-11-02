"""Task/1 — MODEL1_foundation_contracts.md §8.

```
Task/1 {
  task_id, version, frozen_at, spec_sha256,
  target: {variable, horizon_days[], aggregation, label_end_time_rule, knowledge_time_rule, units},
  universe: {instruments[], universe_snapshot_id},
  data_requirements: [{dataset_role, dataset_id | UNRESOLVED, min_depth_years, preferred_depth_years,
                       frequency, pit_required: bool, adjustments: [...], status: READY|PARTIAL|BLOCKED,
                       blockers[]}],
  splitter: {config_id, scheme: "expanding"|"CPCV", embargo_rule, purge_rule},
  incumbents: [model_id...], naive: model_id,
  primary_score: {name, orientation, bounds, proper: true|false},
  guards: [{name, margin}], secondary_scores: [...],
  mde: {table_id, assumptions, n_available, n_required_80pct},
  status: DATA_READY | PARTIAL | BLOCKED (derived mechanically from data_requirements),
  amc_seams_allowed: [S1..S9], influence_ceiling: T0|T1|T2|T3
}
```

"A task's `status` is *derived*: it is BLOCKED if any required dataset is
BLOCKED or unlicensed, and a blocked task cannot leave RESEARCH-state
experiments (mechanical gate, not discretion)." — §8. `derive_task_status`
below is the single function that performs this; `Task.status` is validated
to equal its output, exactly like DatasetSnapshot.research_state.

MODEL-1's own extension (needed to satisfy the operator's tri-state request
in item G of the task and AT-12's mechanical-blocking requirement): each
data requirement also carries `research_state` (derived from the dataset
snapshot's licence, if resolved) so a task can be distinguished as
TECHNICALLY_READY (coverage/depth ok) vs RESEARCH_ONLY (coverage ok but
every satisfying dataset is research_only_unlicensed) vs PROMOTION_READY
(coverage ok and every satisfying dataset is promotion_eligible). See
`derive_promotion_class`.

## P0-1 remediation (adversarial review finding C.1)

"Licence/promotion firewall is bypassable — source authority is not
cryptographically or structurally bound, only trusted metadata... A
fabricated DataRequirement can claim PROMOTION_ELIGIBLE and produce
PROMOTION_READY without resolving a real DatasetSnapshot." The review's
working repro: `DataRequirement(dataset_id=None, research_state=
ResearchState.PROMOTION_ELIGIBLE, ...)` validated with zero errors.

Fix: `Task` can no longer be constructed at all without an authoritative
`DatasetSnapshotRegistry` supplied via pydantic validation context (`Task.
model_validate(data, context={"dataset_registry": registry})` — see
`build_task` below, the required construction path). Direct `Task(**kwargs)`
construction (no context) now FAILS CLOSED with a clear error, because
`info.context` is `None` on that path — this is a language-level guarantee,
not a convention someone could forget to follow. Every `DataRequirement`
that claims a `dataset_id` is resolved against the registry and its
self-declared `research_state`/`licence_status`-derived fields must exactly
match what the REAL `DatasetSnapshot` says; any mismatch, or a `dataset_id`
that doesn't resolve to anything, raises (fail closed, P0-1 requirement 5).
A `DataRequirement` may never claim a `research_state` without also naming
the `dataset_id` that backs it (enforced at the `DataRequirement` level,
requirement 2).

## P0-4 remediation (adversarial review finding C.4)

"the very metadata field that names the corporate-actions risk
(`DataRequirement.adjustments`) is never consulted by the status-derivation
functions." Fix: if `DataRequirement.adjustments` is non-empty (the
requirement DECLARES it needs a corporate-action-safe series), the resolved
dataset's `adjustment_kind` must not be `"none"` — a requirement pointing at
an unadjusted raw series while claiming to need adjustment now fails closed
to BLOCKED, it cannot silently read as READY.

## P1 remediation (MODEL-2A, carried forward from `MODEL1_REMEDIATION_REREVIEW.md` §I)

The independent remediation re-review found the P0-4 fix's companion-dataset
path ("does a real `adjustment_dataset_id` exist?") checks only that the
companion snapshot RESOLVES, never that its CONTENT actually confirms zero
in-window corporate-action events for the right instrument. Working repro
from the review: a `DataRequirement` pointing `dataset_id` at the raw,
contaminated `alpaca_USO_daily_raw` snapshot and `adjustment_dataset_id` at
`corporate_actions_USO` (whose rows document the REAL, unmitigated 1-for-8
split) was ACCEPTED — existence of the companion was silently treated as
"zero events," even though its own rows prove the opposite. Not live in
production (the real task registry correctly routes USO through the
adjusted series instead), but reintroducible for any future instrument wired
the same wrong way.

Fix: `DataRequirement` gains `instrument` (the series/ticker this
requirement is about — required whenever `adjustment_dataset_id` is set, so
a companion's claim can be mechanically scoped, not just trusted). The
companion path now:
  1. confirms the companion's own declared scope (`params["ticker"]`, and,
     if the companion declares a bounded fetch window, that window) matches
     `instrument` and the primary dataset's real coverage — a mismatched
     instrument or date scope fails closed, never silently accepted;
  2. reads the companion's REAL rows (`DatasetSnapshotRegistry.get_rows`,
     loading from disk if needed) and requires zero rows whose `series_id`
     equals `instrument` — any row present IS a real corporate-action event,
     so a non-empty result fails the zero-event claim closed, it is never
     treated as "existence is enough."
Existence of a resolvable `adjustment_dataset_id` is necessary but no longer
sufficient; see `_companion_confirms_zero_events` below.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, ValidationInfo, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.licence import ResearchState


class DataRequirementStatus(StrEnum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class TaskStatus(StrEnum):
    DATA_READY = "DATA_READY"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class PromotionClass(StrEnum):
    """MODEL-1's tri-state readiness classification (operator's item G)."""

    TECHNICALLY_READY = "TECHNICALLY_READY"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    PROMOTION_READY = "PROMOTION_READY"
    BLOCKED = "BLOCKED"


class SplitterScheme(StrEnum):
    EXPANDING = "expanding"
    CPCV = "CPCV"


class InfluenceTier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class TaskTarget(ModelLabBaseModel):
    variable: str
    horizon_days: list[int]
    aggregation: str
    label_end_time_rule: str
    knowledge_time_rule: str
    units: str


class TaskUniverse(ModelLabBaseModel):
    instruments: list[str]
    universe_snapshot_id: str | None = None


class DataRequirement(ModelLabBaseModel):
    dataset_role: str
    dataset_id: str | None = Field(default=None, description="None means UNRESOLVED")
    min_depth_years: float
    preferred_depth_years: float | None = None
    frequency: str
    pit_required: bool
    adjustments: list[str] = Field(
        default_factory=list, description="adjustment kinds this requirement NEEDS, e.g. 'corporate_actions_snapshot'"
    )
    status: DataRequirementStatus
    research_state: ResearchState | None = Field(
        default=None,
        description="the research_state of the dataset satisfying this requirement, if any. "
        "MUST be None when dataset_id is None (P0-1 requirement 2) -- a requirement can never "
        "claim authority without naming the dataset that backs it. When dataset_id IS set, this "
        "value is cross-checked against a real DatasetSnapshot by Task's validator, not trusted "
        "here in isolation.",
    )
    blockers: list[str] = Field(default_factory=list)
    adjustment_dataset_id: str | None = Field(
        default=None,
        description="P0-4: an optional companion dataset (e.g. a corporate-actions-metadata "
        "snapshot) that clears this requirement's `adjustments` need WITHOUT the primary "
        "dataset_id itself being an adjusted series -- e.g. a ticker with zero splits in-window "
        "is genuinely safe to use raw, evidenced by a corporate-actions snapshot confirming zero "
        "events, not by the price series being transformed. Resolved against the same registry as "
        "dataset_id; if set, it must ALSO resolve to a real DatasetSnapshot whose CONTENT -- not "
        "merely its existence -- confirms zero in-window events for `instrument` (P1 remediation, "
        "MODEL1_REMEDIATION_REREVIEW.md sec I: existence alone was found insufficient).",
    )
    instrument: str | None = Field(
        default=None,
        description="P1 remediation: the series/ticker this requirement is about, e.g. 'USO'. "
        "Required whenever adjustment_dataset_id is set -- a companion's zero-event claim can only "
        "be mechanically scoped to the right instrument if one is named.",
    )

    @model_validator(mode="after")
    def _research_state_requires_dataset_id(self) -> DataRequirement:
        if self.research_state is not None and self.dataset_id is None:
            raise ValueError(
                "DataRequirement.research_state is set but dataset_id is None -- a requirement can "
                "never self-assert a research_state without naming the real dataset that backs it "
                "(P0-1 requirement 2; adversarial review finding C.1)"
            )
        return self

    @model_validator(mode="after")
    def _adjustment_dataset_requires_instrument(self) -> DataRequirement:
        if self.adjustment_dataset_id is not None and self.instrument is None:
            raise ValueError(
                "DataRequirement.adjustment_dataset_id is set but instrument is None -- a companion "
                "zero-event claim cannot be mechanically scoped to the right series without naming "
                "it (P1 remediation, MODEL1_REMEDIATION_REREVIEW.md sec I: 'existence alone... does "
                "not catch a future ticker wired the same wrong way')"
            )
        return self


class SplitterSpec(ModelLabBaseModel):
    config_id: str
    scheme: SplitterScheme
    embargo_rule: str
    purge_rule: str


class ScoreSpec(ModelLabBaseModel):
    name: str
    orientation: str = Field(description="'lower_is_better' or 'higher_is_better'")
    bounds: tuple[float | None, float | None] | None = None
    proper: bool


class GuardSpec(ModelLabBaseModel):
    name: str
    margin: float


class MdeSpec(ModelLabBaseModel):
    table_id: str
    assumptions: str
    n_available: float | None = None
    n_required_80pct: float | None = None


def derive_task_status(data_requirements: list[DataRequirement]) -> TaskStatus:
    """Mechanical status derivation (§8, no discretion)."""
    if any(r.status == DataRequirementStatus.BLOCKED for r in data_requirements):
        return TaskStatus.BLOCKED
    if any(r.status == DataRequirementStatus.PARTIAL for r in data_requirements):
        return TaskStatus.PARTIAL
    return TaskStatus.DATA_READY


def derive_promotion_class(
    task_status: TaskStatus, data_requirements: list[DataRequirement]
) -> PromotionClass:
    """MODEL-1's tri-state extension. A required-but-BLOCKED dataset always
    wins (BLOCKED). Otherwise: PROMOTION_READY only if every requirement that
    has a research_state is promotion_eligible; RESEARCH_ONLY if at least one
    satisfying dataset is research_only_unlicensed or quarantined;
    TECHNICALLY_READY if coverage is READY/PARTIAL but no research_state has
    been assigned yet (e.g. licence review pending)."""
    if task_status == TaskStatus.BLOCKED:
        return PromotionClass.BLOCKED
    states = [r.research_state for r in data_requirements if r.research_state is not None]
    if not states:
        return PromotionClass.TECHNICALLY_READY
    if all(s == ResearchState.PROMOTION_ELIGIBLE for s in states):
        return PromotionClass.PROMOTION_READY
    return PromotionClass.RESEARCH_ONLY


def accepts_prereg(task_status: TaskStatus) -> bool:
    """AT-12: a task with a BLOCKED dataset cannot accept a prereg."""
    return task_status != TaskStatus.BLOCKED


def _companion_confirms_zero_events(
    *, registry: Any, dataset_id: str, adjustment_dataset_id: str, instrument: str
) -> bool:
    """P1 remediation (`MODEL1_REMEDIATION_REREVIEW.md` sec I): does the
    companion `adjustment_dataset_id` snapshot's REAL CONTENT -- not merely
    its existence -- confirm zero in-window corporate-action events for
    `instrument`? Fails closed (returns False) on every ambiguous or
    adverse case, never silently True:

    - the companion's own declared scope (`params["ticker"]`) doesn't name
      `instrument` -- a companion covering the wrong series proves nothing
      about this one (mismatched instrument scope);
    - the companion declares a bounded fetch window (`params["start"]`/
      `params["end"]`) that does not cover the primary dataset's actual
      first/last bar dates -- a companion that only checked part of the
      primary's history cannot certify the whole of it (mismatched date
      scope). Today's real corporate-actions fetch
      (`providers/massive.py::fetch_splits`) is deliberately unbounded (no
      start/end params sent), so this branch is a defensive check for any
      future windowed companion, not a live constraint on the current data;
    - the companion's rows (read via `registry.get_rows`, which loads from
      disk if needed -- not just a presence check) cannot be loaded at all;
    - ANY row with `series_id == instrument` is present -- a present row IS
      a real corporate-action event (see
      `scripts/build_corporate_actions_and_adjusted_uso.py`: each row
      records one real split; a ticker with zero events in-window has zero
      rows for it, never a placeholder "no event" row).
    """
    companion = registry.get_snapshot(adjustment_dataset_id)
    if companion is None:
        return False
    declared_ticker = companion.params.get("ticker") if isinstance(companion.params, dict) else None
    if declared_ticker != instrument:
        return False
    primary = registry.get_snapshot(dataset_id)
    if primary is not None and primary.first_bar_date is not None and primary.last_bar_date is not None:
        window_start = companion.params.get("start") if isinstance(companion.params, dict) else None
        window_end = companion.params.get("end") if isinstance(companion.params, dict) else None
        if window_start is not None and date.fromisoformat(window_start) > primary.first_bar_date:
            return False
        if window_end is not None and date.fromisoformat(window_end) < primary.last_bar_date:
            return False
    rows = registry.get_rows(adjustment_dataset_id)
    if rows is None:
        return False
    events_for_instrument = [r for r in rows if r.series_id == instrument]
    return len(events_for_instrument) == 0


class Task(ModelLabBaseModel):
    schema_name: str = "Task"
    schema_version: int = 1

    task_id: str
    version: int = Field(ge=1)
    frozen_at: datetime
    spec_sha256: Sha256Hex

    target: TaskTarget
    universe: TaskUniverse
    data_requirements: list[DataRequirement]
    splitter: SplitterSpec
    incumbents: list[str] = Field(default_factory=list)
    naive: str | None = None
    primary_score: ScoreSpec
    guards: list[GuardSpec] = Field(default_factory=list)
    secondary_scores: list[ScoreSpec] = Field(default_factory=list)
    mde: MdeSpec
    status: TaskStatus
    promotion_class: PromotionClass
    amc_seams_allowed: list[str] = Field(default_factory=list)
    influence_ceiling: InfluenceTier

    @model_validator(mode="after")
    def _check_status_is_mechanical(self, info: ValidationInfo) -> Task:
        context: dict[str, Any] | None = info.context
        if not context or "dataset_registry" not in context:
            raise ValueError(
                "Task cannot be constructed without an authoritative DatasetSnapshotRegistry in "
                "validation context (P0-1 remediation, adversarial review finding C.1): "
                "DataRequirement.research_state must be resolved against real DatasetSnapshot "
                "objects, never self-asserted. Use model_lab.contracts.task.build_task(...) or "
                "Task.model_validate(data, context={'dataset_registry': registry}) -- direct "
                "Task(**kwargs) construction is refused by design."
            )
        registry = context["dataset_registry"]

        for req in self.data_requirements:
            if req.dataset_id is None:
                continue
            resolved = registry.resolve(req.dataset_id)
            if resolved is None:
                raise ValueError(
                    f"data_requirements[{req.dataset_role!r}].dataset_id={req.dataset_id!r} does not "
                    "resolve to any registered DatasetSnapshot -- a requirement cannot claim authority "
                    "for a dataset that does not exist (fail closed, P0-1 requirement 5)"
                )
            if req.research_state is not None and ResearchState(req.research_state) != resolved.research_state:
                raise ValueError(
                    f"data_requirements[{req.dataset_role!r}] claims research_state="
                    f"{req.research_state!r} but the registered DatasetSnapshot {req.dataset_id!r} "
                    f"actually has research_state={resolved.research_state!r} -- a DataRequirement "
                    "can never claim a stronger (or different) authority than its real snapshot "
                    "(P0-1 requirement 1/2, adversarial review finding C.1)"
                )
            # P0-4: a requirement that DECLARES an adjustment need must have that need
            # cleared one of two ways -- pointing at a raw/unadjusted snapshot with
            # NEITHER is exactly the USO gap (C.4):
            #   (a) the primary dataset_id itself resolves to an already-adjusted series
            #       (adjustment_kind not in {"raw", "none"} -- real vocabulary from
            #       data/snapshot.py: "raw" = unadjusted price data, "none" = no
            #       adjustment concept applies / non-price reference data); or
            #   (b) a companion adjustment_dataset_id resolves to a real DatasetSnapshot
            #       WHOSE CONTENT confirms zero in-window events for this instrument (P1
            #       remediation below -- existence of the companion alone is NOT enough,
            #       per MODEL1_REMEDIATION_REREVIEW.md sec I).
            if req.adjustments:
                UNADJUSTED_KINDS = {"raw", "none"}
                primary_satisfies = resolved.adjustment_kind not in UNADJUSTED_KINDS
                companion_satisfies = False
                if req.adjustment_dataset_id is not None:
                    companion = registry.resolve(req.adjustment_dataset_id)
                    if companion is None:
                        raise ValueError(
                            f"data_requirements[{req.dataset_role!r}].adjustment_dataset_id="
                            f"{req.adjustment_dataset_id!r} does not resolve to any registered "
                            "DatasetSnapshot (fail closed, P0-1/P0-4)"
                        )
                    # req.instrument is guaranteed non-None here by
                    # DataRequirement._adjustment_dataset_requires_instrument above.
                    assert req.instrument is not None
                    companion_satisfies = _companion_confirms_zero_events(
                        registry=registry,
                        dataset_id=req.dataset_id,
                        adjustment_dataset_id=req.adjustment_dataset_id,
                        instrument=req.instrument,
                    )
                if not (primary_satisfies or companion_satisfies):
                    raise ValueError(
                        f"data_requirements[{req.dataset_role!r}] declares adjustments="
                        f"{req.adjustments!r} but neither the primary dataset {req.dataset_id!r} "
                        f"(adjustment_kind={resolved.adjustment_kind!r}) nor a companion "
                        "adjustment_dataset_id clears the corporate-action-safety requirement "
                        "(fail closed, P0-4 remediation, adversarial review finding C.4)"
                    )

        expected = derive_task_status(self.data_requirements)
        if TaskStatus(self.status) != expected:
            raise ValueError(
                f"status {self.status!r} does not match the mechanical derivation "
                f"from data_requirements (expected {expected!r})"
            )
        expected_class = derive_promotion_class(expected, self.data_requirements)
        if PromotionClass(self.promotion_class) != expected_class:
            raise ValueError(
                f"promotion_class {self.promotion_class!r} does not match the mechanical "
                f"derivation (expected {expected_class!r})"
            )
        return self


def build_task(*, dataset_registry: Any, **kwargs: Any) -> Task:
    """The required construction path for a `Task` (P0-1). Thin wrapper
    around `Task.model_validate` that supplies the validation context --
    exists so call sites read naturally (`build_task(dataset_registry=reg,
    task_id=..., ...)`) instead of every caller hand-building a context
    dict."""
    return Task.model_validate(kwargs, context={"dataset_registry": dataset_registry})
