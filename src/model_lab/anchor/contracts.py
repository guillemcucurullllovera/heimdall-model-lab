"""Checkpoint/anchor contract — P0-3 remediation (adversarial review finding
C.3): "Ledger tail-truncation is completely undetected... The intended
defense -- 'an off-host operator-signed checkpoint (head hash committed to
git)' -- is named in a code comment... but does not exist anywhere in the
code."

A hash chain alone cannot prove completeness without an external anchor —
this module IS that anchor, as a generic, reusable primitive shared by the
evidence ledger, the option archive, and (P1-1) publication, rather than
three separate bespoke mechanisms.

A `Checkpoint` binds: which subject it anchors (`subject_kind`/`subject_id`),
the last position seen (`anchored_seq`, None for non-sequential subjects
like a single publication manifest), the hash AT that position
(`anchored_head_hash`), an optional record count, and a timestamp/namespace.
It is signed (via the SAME `SignerRegistry`/`ssh-keygen -Y` machinery as
governance transitions — P0-3 requirement 2) by a principal holding the
OPERATOR role, and its OWN bytes are never stored inside the mutable
evidence file it anchors (`ledger/anchor.py` / `options_archive/anchor.py`
store checkpoints in a SEPARATE sibling file — see those modules' docstrings
for why, and for the honest limitation this still carries today).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel, Sha256Hex
from model_lab.contracts.ledger_record import Signature


class SubjectKind(StrEnum):
    LEDGER = "ledger"
    OPTION_ARCHIVE = "option_archive"
    PUBLICATION_MANIFEST = "publication_manifest"


class Checkpoint(ModelLabBaseModel):
    schema_name: str = "Checkpoint"
    schema_version: int = 1

    subject_kind: SubjectKind
    subject_id: str = Field(description="identifies WHICH ledger/archive/publication this anchors")
    namespace: str = Field(description="ssh-keygen -Y sign/verify namespace, scoped per purpose")

    anchored_seq: int | None = Field(
        default=None, description="last seq/archive_seq at checkpoint time; None for non-sequential subjects"
    )
    anchored_head_hash: Sha256Hex = Field(description="chain_hash/record_hash/manifest_hash at checkpoint time")
    anchored_record_count: int | None = None

    created_at_utc: datetime
    checkpoint_sha256: Sha256Hex = Field(description="self-referential hash, canonical_sha256_excluding this field")


class SignedCheckpoint(ModelLabBaseModel):
    """A Checkpoint plus its signature(s). Persisted as one NDJSON line in
    the sibling `.checkpoints.ndjson` file (never inside the subject's own
    mutable data file)."""

    schema_name: str = "SignedCheckpoint"
    schema_version: int = 1

    checkpoint: Checkpoint
    signatures: list[Signature] = Field(default_factory=list)


class AnchorStatus(StrEnum):
    UNANCHORED = "UNANCHORED"  # no checkpoint exists/provided; internal chain integrity only
    ANCHORED_VALID = "ANCHORED_VALID"  # chain matches or legitimately extends the most recent valid checkpoint
    TAIL_TRUNCATED = "TAIL_TRUNCATED"  # current state is behind or inconsistent with the checkpoint
    CHECKPOINT_INVALID = "CHECKPOINT_INVALID"  # the checkpoint's own signature does not verify
