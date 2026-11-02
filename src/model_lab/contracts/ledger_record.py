"""LedgerRecord/1 — MODEL1_foundation_contracts.md §12.

`LedgerRecord/1 { seq, prev_hash, record_type, body, body_hash, author_id,
ts_utc (audit only), signatures[] }`; canonical bytes by RFC 8785 JCS;
append-only; no update/delete API exists. Record types per §12.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex


class LedgerRecordType(StrEnum):
    HYPOTHESIS = "hypothesis"
    PREREG = "prereg"
    SEARCH_SPACE = "search_space"
    TRIAL_BATCH = "trial_batch"
    DATASET_SNAPSHOT = "dataset_snapshot"
    FEATURE_SET = "feature_set"
    RUN = "run"
    EVIDENCE_REPORT = "evidence_report"
    MODEL_VERSION = "model_version"
    SHADOW_RECORD = "shadow_record"
    DECISION_DIFF = "decision_diff"
    TRANSITION_REQUEST = "transition_request"
    VALIDATION_ATTESTATION = "validation_attestation"
    APPROVAL = "approval"
    TRANSITION = "transition"
    LIMIT_TABLE = "limit_table"
    INCIDENT = "incident"
    NEGATIVE_RESULT = "negative_result"
    DECISION = "decision"
    SOURCE_LICENCE_STATUS = "source_licence_status"
    OPTION_ARCHIVE_MANIFEST = "option_archive_manifest"
    ACCEPTANCE_TEST_RESULT = "acceptance_test_result"  # MODEL-1's own extension: AT-01..17 results


class Signature(ModelLabBaseModel):
    """A detached signature over a record's body_hash.

    Produced out-of-band by `ssh-keygen -Y sign` (or, in tests, throwaway
    keys) and verified with `ssh-keygen -Y verify` / `find-principals`
    against a role-restricted `allowed_signers` file. MODEL-1 never
    generates operator or validator keys itself (D2).
    """

    signer_role: str = Field(description="e.g. 'author', 'validator', 'operator', 'automaton'")
    signer_id: str = Field(description="principal identity as it appears in the allowed_signers file")
    namespace: str = Field(description="ssh-keygen -Y sign/verify namespace")
    signature_format: str = Field(default="ssh-ed25519-Y-sign")
    signature_b64: str
    signed_at_utc: datetime


class LedgerRecord(ModelLabBaseModel):
    schema_name: str = "LedgerRecord"
    schema_version: int = 1

    seq: int = Field(ge=0)
    prev_hash: Sha256Hex | None = Field(
        default=None, description="None only for seq==0 (genesis record)"
    )
    record_type: LedgerRecordType
    body: dict[str, Any] = Field(description="the record_type-specific payload, e.g. a Prereg/1 dump")
    body_hash: Sha256Hex = Field(description="canonical_sha256(body)")
    author_id: str
    ts_utc: datetime = Field(description="audit-only; not used for any PIT/knowledge-time decision")
    signatures: list[Signature] = Field(default_factory=list)

    def hashable_dict(self) -> dict[str, Any]:
        """Fields that feed this record's own record_hash (everything except
        the record_hash field itself, which does not exist on this model —
        the ledger writer computes it externally as
        canonical_sha256({seq, prev_hash, record_type, body_hash, author_id, ts_utc})
        deliberately excluding `body` and `signatures` from the chain hash:
        body is already committed via body_hash, and signatures are appended
        after the record is chained (a signature over the chained record
        would create a circular dependency)."""
        d = self.canonical_dict()
        d.pop("body", None)
        d.pop("signatures", None)
        d.pop("schema_name", None)
        d.pop("schema_version", None)
        return d
