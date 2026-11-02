from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.gate_v2.signing import (
    InvalidThresholdSignatureError,
    UnsignedThresholdFileError,
    load_signed_thresholds,
    write_unsigned_threshold_placeholder,
)
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    sign_bytes,
    write_allowed_signers,
)


def test_unsigned_placeholder_refuses_to_load(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    write_unsigned_threshold_placeholder(path, {"theta_dsr": 0.95})
    with pytest.raises(UnsignedThresholdFileError):
        load_signed_thresholds(path, allowed_signers_path=tmp_path / "allowed_signers")


def test_properly_signed_threshold_file_loads(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "operator", comment="operator")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available")

    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("operator-x", ["model-lab-gate-thresholds"], key.public_key_text)])

    import json

    # thresholds carry precision-sensitive values as STRINGS per this repo's
    # own JCS convention (see canon/jcs.py) -- a raw JSON float like 0.95
    # would raise NonCanonicalFloatError.
    doc = {"thresholds": {"theta_dsr": "0.95"}, "signed_by": "operator-x", "signed_at": "2026-09-30T00:00:00Z", "signature": None}
    config_hash = canonical_sha256_excluding(doc, "signature")
    sig = sign_bytes(config_hash.encode("utf-8"), key.private_key_path, "model-lab-gate-thresholds")
    doc["signature"] = sig

    path = tmp_path / "thresholds_signed.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    loaded = load_signed_thresholds(path, allowed_signers_path=allowed_signers)
    assert loaded.thresholds == {"theta_dsr": "0.95"}
    assert loaded.gate_config_sha256 == config_hash
    assert loaded.signed_by == "operator-x"


def test_tampered_signed_threshold_file_rejected(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "operator", comment="operator")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available")

    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("operator-x", ["model-lab-gate-thresholds"], key.public_key_text)])

    import json

    doc = {"thresholds": {"theta_dsr": "0.95"}, "signed_by": "operator-x", "signed_at": "2026-09-30T00:00:00Z", "signature": None}
    config_hash = canonical_sha256_excluding(doc, "signature")
    sig = sign_bytes(config_hash.encode("utf-8"), key.private_key_path, "model-lab-gate-thresholds")
    doc["signature"] = sig
    doc["thresholds"]["theta_dsr"] = "0.50"  # tamper AFTER signing

    path = tmp_path / "thresholds_tampered.json"
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(InvalidThresholdSignatureError):
        load_signed_thresholds(path, allowed_signers_path=allowed_signers)
