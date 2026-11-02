"""Governance record bodies referenced by `LedgerRecordType` (§12) but not
among the 11 pydantic types §6 names for M1-C: `ValidationAttestation`,
`Approval`, and the final `Transition` record produced once a
`TransitionRequest` is verified and applied. Kept beside the verifier that
consumes them to avoid a circular import with contracts/.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.model_version import LifecycleState
from model_lab.contracts.task import InfluenceTier


class ValidationAttestation(ModelLabBaseModel):
    """Signed by the model-validator role (a human/account distinct from
    every contributor). Accompanies a TransitionRequest as one of the two
    required signatures for any influence increase."""

    schema_name: str = "ValidationAttestation"
    schema_version: int = 1

    request_id: str
    request_sha256: Sha256Hex
    validator_id: str
    namespace: str
    attested_at_utc: datetime
    notes: str | None = None


class Approval(ModelLabBaseModel):
    """Signed by the operator role. The second required signature for any
    influence increase."""

    schema_name: str = "Approval"
    schema_version: int = 1

    request_id: str
    request_sha256: Sha256Hex
    operator_id: str
    namespace: str
    approved_at_utc: datetime
    notes: str | None = None


class Transition(ModelLabBaseModel):
    """The final applied transition — written to the ledger only after the
    verifier (verifier.py) accepts a TransitionRequest plus its required
    signatures. Ledger reads are pure functions of these records (§11:
    "replaying the ledger reproduces the current state")."""

    schema_name: str = "Transition"
    schema_version: int = 1

    request_id: str
    model_id: str
    model_version_hash: Sha256Hex
    lifecycle_before: LifecycleState
    lifecycle_after: LifecycleState
    influence_before: InfluenceTier
    influence_after: InfluenceTier
    applied_at_utc: datetime
    applied_by_role: str = Field(description="'automaton' | 'validator+operator'")
    request_sha256: Sha256Hex
    validator_signer_id: str | None = Field(
        default=None,
        description="P0-2 auditability: the specific registry-confirmed validator principal, if any",
    )
    operator_signer_id: str | None = Field(
        default=None,
        description="P0-2 auditability: the specific registry-confirmed operator principal, if any. "
        "For any influence increase, validator_signer_id != operator_signer_id is a verifier invariant "
        "(governance/verifier.py), checkable directly from these two fields without re-running the verifier.",
    )
