"""TransitionRequest/1 — MODEL1_foundation_contracts.md §17 (D2 promotion
governance).

"the lab produces `transition_request.json` and consumes detached `.sig`
files"; "request bound to `prev_ledger_hash` and short expiry"; "valid-before
<= 12 months"; "namespaces per purpose"; the verifier (a pure function)
"enforces `influence_after > influence_before` => validator AND operator,
both NOT IN contributors".

The companion records this request is eventually paired with
(ValidationAttestation, Approval, and the final Transition itself) live in
governance/contracts.py — they are not among the 11 pydantic types §6 names
for M1-C, but ARE `LedgerRecordType` members (§12), so they need schemas;
keeping them next to the verifier that consumes them (governance/) avoids a
circular import between contracts/ and governance/.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.model_version import LifecycleState
from model_lab.contracts.task import InfluenceTier

_TIER_ORDER = {InfluenceTier.T0: 0, InfluenceTier.T1: 1, InfluenceTier.T2: 2, InfluenceTier.T3: 3}


def influence_rank(tier: InfluenceTier) -> int:
    return _TIER_ORDER[InfluenceTier(tier)]


def is_influence_increase(before: InfluenceTier, after: InfluenceTier) -> bool:
    return influence_rank(after) > influence_rank(before)


class TransitionRequest(ModelLabBaseModel):
    schema_name: str = "TransitionRequest"
    schema_version: int = 1

    request_id: str
    model_id: str
    model_version_hash: Sha256Hex

    lifecycle_before: LifecycleState
    lifecycle_after: LifecycleState
    influence_before: InfluenceTier
    influence_after: InfluenceTier

    reason: str
    evidence_report_hash: Sha256Hex | None = None
    gate_result_hash: Sha256Hex | None = None

    requested_by: str
    contributors: list[str] = Field(
        default_factory=list, description="everyone who authored the model/evidence this request cites"
    )

    namespace: str = Field(description="ssh-keygen -Y sign/verify namespace, scoped per purpose")
    prev_ledger_hash: Sha256Hex
    created_at_utc: datetime
    valid_before_utc: datetime

    request_sha256: Sha256Hex

    @model_validator(mode="after")
    def _valid_before_within_12_months(self) -> TransitionRequest:
        max_delta_days = 366  # <= 12 months, generous leap-year allowance
        delta = (self.valid_before_utc - self.created_at_utc).total_seconds() / 86400.0
        if delta <= 0:
            raise ValueError("valid_before_utc must be after created_at_utc")
        if delta > max_delta_days:
            raise ValueError(f"valid_before_utc exceeds the 12-month cap ({delta:.1f} days)")
        return self
