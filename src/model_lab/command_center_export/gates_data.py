"""Structured (machine-readable) twin of `reports/MODEL2_PROMOTION_GATES.md`'s
own gate table. This module is the canonical SOURCE for gate states -- the
markdown document is written for a human reader, this is written for the
export builder; if the two ever disagree, this module is wrong and should
be fixed to match the (human-reviewable, more detailed) markdown, not the
other way around.

Every gate state here must be independently checkable against real
evidence (listed in each entry's own `evidence` string pointing at a
specific file) -- never asserted from memory of what the state "should"
be.

CLARIFICATION (AMC Promotion Gap Analysis, 2026-10-01, `reports/
MODEL_LAB_AMC_PROMOTION_GAP_ANALYSIS.md` §5): every `GateState` value in
this module is a HAND-AUTHORED evidence assessment -- a human (this
session) read the real underlying evidence and asserted a state. It is
NOT the output of running `gate_v2/branch_s.py`/`branch_f.py`/
`branch_p.py`'s own real, executable, independently-tested gate-
evaluation code against this candidate. That code exists and works (its
own unit tests exercise it against synthetic/pinned data), but has never
been invoked against any real Model Lab candidate -- no `GateResult` has
ever been constructed for MODEL-2 or anything else in this repository.
Do not read a `PASSED` state here as "Gate v2 ran and passed this" --
read it as "a direct evidence review found this specific criterion met."
"""

from __future__ import annotations

from dataclasses import dataclass

from model_lab.contracts.model2_export import GateState


@dataclass(frozen=True)
class GateDefinition:
    gate_id: str
    name: str
    state: GateState
    evidence: str
    maps_to_lifecycle_state: str | None


GATES: tuple[GateDefinition, ...] = (
    GateDefinition(
        gate_id="G1_RESEARCH",
        name="Research / walk-forward evaluation",
        state=GateState.PASSED,
        evidence="reports/vol_ladder/*.json (24 cells, 9 calendar-year expanding folds each)",
        maps_to_lifecycle_state="RESEARCH -> REPRODUCED -> WALK_FORWARD",
    ),
    GateDefinition(
        gate_id="G2_SHADOW_CANDIDATE",
        name="Shadow candidate (real ModelVersion/ModelOutput records)",
        state=GateState.PASSED,
        evidence="shadow/build_combo.py + reports/shadow/*.json; tested in tests/unit/shadow/test_build_combo.py",
        maps_to_lifecycle_state="SHADOW",
    ),
    GateDefinition(
        gate_id="G3_PROSPECTIVE_COLLECTION",
        name="Prospective evidence collection mechanism running",
        state=GateState.PENDING,
        evidence="src/model_lab/prospective/ + reports/prospective/*.json (mechanism exercised; "
        "ModelVersion.lifecycle_state deliberately left at SHADOW, not advanced to "
        "PROSPECTIVE_CHALLENGER, pending an explicit governance decision -- see "
        "MODEL2_PROMOTION_GATES.md)",
        maps_to_lifecycle_state="PROSPECTIVE_CHALLENGER (not yet set on any record)",
    ),
    GateDefinition(
        gate_id="G4_PROSPECTIVE_SUFFICIENT",
        name="Prospective evidence statistically sufficient",
        state=GateState.NOT_ESTABLISHED,
        evidence="6 resolved, non-replay observations (reports/prospective/resolved_*.json) -- far "
        "short of Gate v2 Branch P's own [P]roposed E>=20/E>=100 thresholds (gate_v2/branch_p.py)",
        maps_to_lifecycle_state=None,
    ),
    GateDefinition(
        gate_id="G5_INDEPENDENT_REVIEW",
        name="Independent adversarial review accepted",
        state=GateState.NOT_ESTABLISHED,
        evidence="Not started -- by explicit operator instruction, this session does not perform it",
        maps_to_lifecycle_state=None,
    ),
    GateDefinition(
        gate_id="G6_PROMOTION_ELIGIBLE",
        name="Promotion eligible",
        state=GateState.NOT_ESTABLISHED,
        evidence="Blocked on G4 and G5",
        maps_to_lifecycle_state="PROMOTION_ELIGIBLE",
    ),
    GateDefinition(
        gate_id="G7_AMC_ELIGIBLE",
        name="AMC eligible",
        state=GateState.NOT_ESTABLISHED,
        evidence="Not sought. decision_relevant=False and tier_ceiling_signed=False on every "
        "record this project has ever produced for this candidate, structurally enforced by tests, "
        "not merely by convention",
        maps_to_lifecycle_state="PROMOTED",
    ),
)
