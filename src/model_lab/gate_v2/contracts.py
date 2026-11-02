"""Gate v2 result contracts — the 10 identity fields (§2.1 of the prose
spec / `identity_fields` in gate_v2_spec.json) plus the closed-list
component-table schema (`gate_v2_spec.json`'s own per-component schema:
id/branch/name/statistic/rule/threshold/threshold_status/gating/kind/
missing_input_policy/notes).

"Every gate result carries `gate_version`, `gate_config_sha256`,
`gate_code_commit`, `spec_sha256`, `registry_head_hash`, `lineage_id`,
`family_id`, `input_hashes`, `seeds`, `environment_manifest`. A result
without all of these is invalid (fail-closed)." -- enforced here via
`extra="forbid"` + all-required fields on `ModelLabBaseModel`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex


class Verdict(StrEnum):
    """"verdict tokens: SUFFICIENT | INSUFFICIENT | FAIL; gate never
    increases model influence" (principle 2). No other tokens exist."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    FAIL = "FAIL"


class ComponentKind(StrEnum):
    VETO = "veto"
    POWER = "power"
    ADEQUACY = "adequacy"
    METHOD = "method"
    REPORT_ONLY = "report_only"


class Branch(StrEnum):
    S = "S"
    F = "F"
    P = "P"


class ComponentResult(ModelLabBaseModel):
    """One row of a branch's closed component list. `missing_input_policy`
    is always "FAIL" per the verified research extraction (every S0-S16 /
    F1-F9 / P1-P4 component observed in gate_v2_spec.json sets this to
    "FAIL", never a softer fallback)."""

    id: str
    branch: Branch
    name: str
    statistic: str
    evidence: str = Field(description="provenance tag, e.g. 'V-paper Eq.(2)' or 'PROPOSED'")
    rule: str
    threshold: str | None = None
    threshold_status: str = Field(description="'PROPOSED' | 'structural' | 'n/a'")
    gating: bool
    kind: ComponentKind
    missing_input_policy: str = "FAIL"
    notes: str = ""

    passed: bool | None = Field(
        default=None, description="None means the component could not be computed (missing input -> FAIL at the branch level)"
    )
    value: str | None = None
    missing_input: bool = False


class GateResult(ModelLabBaseModel):
    schema_name: str = "GateResult"
    schema_version: int = 1

    # --- the 10 identity fields (§2.1) ---
    gate_version: str
    gate_config_sha256: Sha256Hex
    gate_code_commit: str
    spec_sha256: Sha256Hex
    registry_head_hash: Sha256Hex
    lineage_id: str
    family_id: str
    input_hashes: list[Sha256Hex]
    seeds: list[int]
    environment_manifest: dict[str, str]

    # --- verdict ---
    branch: Branch
    verdict: Verdict
    components: list[ComponentResult]
    gating_component_ids: list[str] = Field(description="closed list; must equal {c.id for c in components if c.gating}")
    report_only_component_ids: list[str] = Field(
        description="closed list; must equal {c.id for c in components if not c.gating}"
    )
    first_failing_veto_id: str | None = None
