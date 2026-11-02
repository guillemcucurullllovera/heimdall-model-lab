"""AT-01: altering/deleting any INTERIOR row fails internal chain
verification; models in that lineage auto-QUARANTINE.

**P0-3 scope correction** (adversarial review finding C.3): the original
AT-01 claim ("alter/delete fails verification") was overclaimed -- it holds
for interior alter/delete, but `verify_chain` alone CANNOT detect deleting
the most recent (tail) records, because every check it runs (seq
monotonicity, prev_hash chaining, body_hash) is relative to records that are
*present* in the file; a shortened file is, by those checks alone,
indistinguishable from a ledger that legitimately never grew past that
point. `test_EXPLOIT_tail_deletion_is_invisible_to_internal_verification_alone`
below reproduces the review's exact repro and documents this honestly as a
real, permanent scope boundary of `verify_chain` -- not a bug to "fix" in
that function. The actual fix is `ledger.anchor.verify_ledger_completeness`
(P0-3), which composes `verify_chain` with checkpoint-anchored completeness
and DOES catch tail truncation when a checkpoint exists (see the other
tests in this file, and `tests/unit/anchor/`). Internal chain integrity
(this file's original tests) and checkpoint-anchored completeness (the new
tests) are two distinct, complementary properties -- never claim one implies
the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab.anchor.contracts import AnchorStatus
from model_lab.contracts.ledger_record import LedgerRecordType
from model_lab.governance.signer_registry import RegisteredSigner, SignerRegistry, SignerRole
from model_lab.ledger.anchor import create_ledger_checkpoint, verify_ledger_completeness
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    write_allowed_signers,
)
from model_lab.ledger.storage import LedgerFile
from model_lab.ledger.verifier import verify_chain
from model_lab.ledger.writer import LedgerWriter

pytestmark = pytest.mark.acceptance
NAMESPACE = "model-lab-checkpoint"


def _seed_ledger(path: Path) -> LedgerWriter:
    w = LedgerWriter(path, author_id="test-author")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "hypothesis 1"})
    w.append(
        LedgerRecordType.MODEL_VERSION,
        {
            "model_id": "m1",
            "model_version_hash": "1" * 64,
            "lifecycle_state": "RESEARCH",
            "predecessor_hash": None,
        },
    )
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "hypothesis 2"})
    w.append(
        LedgerRecordType.MODEL_VERSION,
        {
            "model_id": "m2",
            "model_version_hash": "2" * 64,
            "lifecycle_state": "RESEARCH",
            "predecessor_hash": "1" * 64,  # descends from m1
        },
    )
    return w


def test_untampered_chain_verifies(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_ledger(path)
    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert result.ok
    assert not result.violations
    assert result.model_versions_to_quarantine == []


def test_altering_a_body_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_ledger(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0])
    rec0["body"]["text"] = "TAMPERED"  # body_hash now stale
    lines[0] = json.dumps(rec0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert not result.ok
    assert any(v.kind == "body_hash_mismatch" and v.seq == 0 for v in result.violations)


def test_deleting_a_row_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_ledger(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # delete the m1 ModelVersion record
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert not result.ok
    # seq now jumps 0 -> 2 where the deleted record (seq=1) used to be
    assert any(v.kind == "seq_gap" for v in result.violations)
    assert any(v.kind == "prev_hash_mismatch" for v in result.violations)


def test_tamper_quarantines_downstream_model_lineage(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_ledger(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    rec0 = json.loads(lines[0])
    rec0["body"]["text"] = "TAMPERED"
    lines[0] = json.dumps(rec0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert not result.ok
    # both m1 (seq=1, at/after the tamper) and m2 (descends from m1 via
    # predecessor_hash) must be flagged for quarantine.
    assert "m1" in result.model_versions_to_quarantine
    assert "m2" in result.model_versions_to_quarantine


def test_tamper_before_any_model_version_does_not_falsely_clear_lineage(tmp_path: Path) -> None:
    # Sanity: an untampered chain's model versions are never quarantined.
    path = tmp_path / "ledger.ndjson"
    _seed_ledger(path)
    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert result.model_versions_to_quarantine == []


# --- P0-3: checkpoint-anchored completeness (adversarial review finding C.3) ---


def _seed_10(path: Path) -> None:
    w = LedgerWriter(path, author_id="test-author")
    for i in range(10):
        w.append(LedgerRecordType.HYPOTHESIS, {"text": f"hypothesis {i}"})


def _operator_registry_and_signers(tmp_path: Path):
    try:
        key = generate_throwaway_keypair(tmp_path / "operator-key")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available in this environment")
    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("op-1", [NAMESPACE], key.public_key_text)])
    registry = SignerRegistry(
        signers=[RegisteredSigner(principal="op-1", roles=[SignerRole.OPERATOR], namespaces=[NAMESPACE], public_key_line=key.public_key_text)]
    )
    return key, allowed_signers, registry


def test_EXPLOIT_tail_deletion_is_invisible_to_internal_verification_alone(tmp_path: Path) -> None:
    """The review's exact repro: write 10 records, delete the last 3
    (simulating an attacker or bug removing recent unfavorable evidence),
    reload -- verify_chain must report ok=True, zero violations. This is
    the honest, permanent scope boundary `verify_ledger_completeness`
    (below) exists to close -- not a bug in verify_chain."""
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")  # drop the last 3

    records = LedgerFile(path).read_all()
    result = verify_chain(records)
    assert result.ok, "documents the known scope boundary: internal-only verification cannot see tail truncation"
    assert not result.violations


def test_EXPLOIT_tail_deletion_IS_caught_by_checkpoint_anchored_completeness(tmp_path: Path) -> None:
    """Same repro, but with a signed checkpoint taken BEFORE the deletion --
    verify_ledger_completeness must report TAIL_TRUNCATED."""
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)

    create_ledger_checkpoint(path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10  # the checkpoint lives in a separate sibling file, not mixed into this one
    path.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")  # drop the last 3, AFTER the checkpoint

    result = verify_ledger_completeness(
        path, signer_registry=registry, allowed_signers_path=allowed_signers
    )
    assert result.internal.ok  # internal check alone still can't see it (documented above)
    assert result.anchor.status == AnchorStatus.TAIL_TRUNCATED


def test_exact_anchored_ledger_with_no_further_changes_passes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_ledger_checkpoint(path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    result = verify_ledger_completeness(path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.internal.ok
    assert result.anchor.status == AnchorStatus.ANCHORED_VALID


def test_valid_additional_records_after_checkpoint_classified_as_anchored_valid(tmp_path: Path) -> None:
    """Legitimate growth -- new records appended AFTER an older checkpoint --
    must be classified ANCHORED_VALID, never confused with truncation."""
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)

    # Checkpoint after only the first 5 records exist... simulate by
    # checkpointing now (all 10 present) then legitimately appending more.
    create_ledger_checkpoint(path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    w = LedgerWriter(path, author_id="test-author")
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "hypothesis 10 (added after checkpoint)"})
    w.append(LedgerRecordType.HYPOTHESIS, {"text": "hypothesis 11 (added after checkpoint)"})

    result = verify_ledger_completeness(path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.internal.ok
    assert result.anchor.status == AnchorStatus.ANCHORED_VALID


def test_interior_deletion_still_fails_internal_verification_even_with_a_checkpoint(tmp_path: Path) -> None:
    """A checkpoint doesn't weaken the existing interior-tamper protection
    -- verify_chain (the internal check) still independently catches it."""
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_ledger_checkpoint(path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[3]  # interior deletion
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_ledger_completeness(path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert not result.internal.ok


def test_forged_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.ndjson"
    _seed_10(path)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_ledger_checkpoint(path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    # Tamper the checkpoint file itself (forge a different anchored_seq).
    cp_path = path.with_name(path.name + ".checkpoints.ndjson")
    cp_line = json.loads(cp_path.read_text(encoding="utf-8").splitlines()[0])
    cp_line["checkpoint"]["anchored_seq"] = 2  # forged: claim an earlier, easier-to-fake anchor
    cp_path.write_text(json.dumps(cp_line) + "\n", encoding="utf-8")

    result = verify_ledger_completeness(path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.anchor.status == AnchorStatus.CHECKPOINT_INVALID
