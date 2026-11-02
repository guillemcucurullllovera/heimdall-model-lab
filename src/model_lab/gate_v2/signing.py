"""Fail-closed signed-threshold loader — §11 of the verified research
extraction / §2.2 of the prose spec.

"Signed thresholds. Thresholds live in one signed file (`signed_by`,
`signed_at`, signature over the file hash). The loader is fail-closed
exactly like v1's `graduation_config.py`: null/absent signature => refuse
to run. Nothing is a code default."

`gate_config_sha256` = SHA-256 of the RFC-8785 canonical JSON of the
*signed* threshold file (§2.1 identity_fields description) -- computed via
`model_lab.canon.jcs`, matching the self-referential-hash pattern
(`canonical_sha256_excluding`) used by `gate_v2_spec.json`'s own
`spec_sha256_of_this_json_excluding_this_field`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.ledger.signing import verify_signature


class UnsignedThresholdFileError(RuntimeError):
    """Raised when the threshold file's signed_by/signature is null/absent
    -- the loader refuses to run rather than falling back to a code
    default (§2.2: "Nothing is a code default")."""


class InvalidThresholdSignatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedThresholds:
    thresholds: dict[str, Any]
    gate_config_sha256: str
    signed_by: str
    signed_at: str


def write_unsigned_threshold_placeholder(path: Path, thresholds: dict[str, Any]) -> None:
    """Write the UNSIGNED placeholder form (signed_by/signed_at/signature
    all null) -- this is what MODEL-1 produces; it must NEVER be signed by
    MODEL-1 itself (operator/validator signing only, off-host)."""
    doc = {
        "thresholds": thresholds,
        "signed_by": None,
        "signed_at": None,
        "signature": None,
        "note": "operator/validator to sign (e.g. ssh-keygen -Y sign over this file's canonical hash) -- MODEL-1 holds no signing key",
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_signed_thresholds(path: Path, *, allowed_signers_path: Path, namespace: str = "model-lab-gate-thresholds") -> LoadedThresholds:
    """Fail-closed: raises `UnsignedThresholdFileError` if signed_by/
    signature is null/absent. Raises `InvalidThresholdSignatureError` if a
    signature is present but does not verify."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("signed_by") or not doc.get("signature"):
        raise UnsignedThresholdFileError(
            f"{path}: signed_by/signature is null or absent -- the gate refuses to load unsigned "
            "thresholds (this is expected for MODEL-1: no real thresholds are calibrated/signed yet)"
        )
    config_hash = canonical_sha256_excluding(doc, "signature")
    ok = verify_signature(
        config_hash.encode("utf-8"),
        doc["signature"],
        allowed_signers_path=allowed_signers_path,
        principal=doc["signed_by"],
        namespace=namespace,
    )
    if not ok:
        raise InvalidThresholdSignatureError(f"{path}: signature does not verify for signer {doc['signed_by']!r}")
    return LoadedThresholds(
        thresholds=doc["thresholds"], gate_config_sha256=config_hash, signed_by=doc["signed_by"], signed_at=doc["signed_at"]
    )
