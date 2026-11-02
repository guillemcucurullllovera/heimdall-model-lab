"""AT-14: append-only/hash-chain verification; re-derivation from raw; no
new network call is made by the ingester (asserted by a network-denied test
run).

**P0-3 scope note** (adversarial review finding C.3, same root cause as
AT-01): `verify_archive_chain` alone has the identical tail-truncation blind
spot as the ledger's `verify_chain` -- see the "P0-3" tests below, which
apply the shared `options_archive.anchor.verify_archive_completeness`
composition to close it, mirroring AT-01's ledger tests exactly.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from model_lab.anchor.contracts import AnchorStatus
from model_lab.governance.signer_registry import RegisteredSigner, SignerRegistry, SignerRole
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    write_allowed_signers,
)
from model_lab.options_archive.anchor import create_archive_checkpoint, verify_archive_completeness
from model_lab.options_archive.ingester import finalize_and_append, ingest_candidate_file
from model_lab.options_archive.storage import OptionArchiveFile
from model_lab.options_archive.verifier import verify_archive_chain
from model_lab.replay.amc_root import CYCLES_DIR

CYCLE_003 = CYCLES_DIR / "cycle-003-20260908_candidates.json"
CYCLE_004 = CYCLES_DIR / "cycle-004-20260908_candidates.json"

REAL_FILES_AVAILABLE = CYCLE_003.exists() and CYCLE_004.exists()
pytestmark = [pytest.mark.acceptance, pytest.mark.skipif(not REAL_FILES_AVAILABLE, reason="AMC root not present")]
NAMESPACE = "model-lab-checkpoint"


class _NetworkDeniedSocket(socket.socket):
    def connect(self, *a, **kw):
        raise AssertionError("network call attempted during option-archive ingestion (must be zero)")

    def connect_ex(self, *a, **kw):
        raise AssertionError("network call attempted during option-archive ingestion (must be zero)")


def test_ingestion_makes_zero_network_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    archive_path = tmp_path / "archive.ndjson"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)
    monkeypatch.setattr(socket, "socket", _NetworkDeniedSocket)

    drafts = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )
    assert len(drafts) == 736  # 736 contracts confirmed this session

    archive = OptionArchiveFile(archive_path)
    finalized = finalize_and_append(archive, drafts)
    assert len(finalized) == 736


def test_hash_chain_verifies_and_reports_no_violations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    archive_path = tmp_path / "archive.ndjson"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)

    archive = OptionArchiveFile(archive_path)
    drafts_003 = ingest_candidate_file(
        CYCLE_003, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-003-20260908"
    )
    finalize_and_append(archive, drafts_003)
    drafts_004 = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )
    finalize_and_append(archive, drafts_004)

    records = archive.read_all()
    assert len(records) == 736 * 2
    assert [r.archive_seq for r in records] == list(range(len(records)))

    result = verify_archive_chain(records)
    assert result.ok, result.violations


def test_tampering_a_record_fails_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    archive_path = tmp_path / "archive.ndjson"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)

    archive = OptionArchiveFile(archive_path)
    drafts = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )[:5]
    finalize_and_append(archive, drafts)

    import json

    lines = archive_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["quote"]["bid"] = "999999.99"
    lines[0] = json.dumps(rec)
    archive_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = archive.read_all()
    result = verify_archive_chain(records)
    assert not result.ok
    assert any(v.kind == "record_hash_mismatch" for v in result.violations)


def test_raw_payload_is_content_addressed_and_re_derivable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)

    drafts = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )
    raw_path = Path(drafts[0].raw_ref.content_addressed_path)
    assert raw_path.exists()
    assert raw_path.read_bytes() == CYCLE_004.read_bytes()

    import hashlib

    assert raw_path.name == hashlib.sha256(CYCLE_004.read_bytes()).hexdigest()


def test_licence_status_is_research_only_unlicensed_never_promotion_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)

    drafts = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )
    assert all(d.licence.status == "UNRESOLVED" for d in drafts)


# --- P0-3: checkpoint-anchored completeness for the option archive ---------


def _small_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 10) -> Path:
    raw_root = tmp_path / "raw"
    archive_path = tmp_path / "archive.ndjson"
    monkeypatch.setattr("model_lab.options_archive.ingester.RAW_STORE_ROOT", raw_root)
    archive = OptionArchiveFile(archive_path)
    drafts = ingest_candidate_file(
        CYCLE_004, amc_experiment_id="amc4-prospective-v1-20260908", amc_cycle_id="cycle-004-20260908"
    )[:n]
    finalize_and_append(archive, drafts)
    return archive_path


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


def test_EXPLOIT_archive_tail_deletion_invisible_to_internal_verification_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _small_archive(tmp_path, monkeypatch, n=10)
    lines = archive_path.read_text(encoding="utf-8").splitlines()
    archive_path.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")  # drop the last 3

    records = OptionArchiveFile(archive_path).read_all()
    result = verify_archive_chain(records)
    assert result.ok, "documents the known scope boundary, same as the ledger (AT-01)"


def test_EXPLOIT_archive_tail_deletion_IS_caught_by_checkpoint_anchored_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _small_archive(tmp_path, monkeypatch, n=10)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_archive_checkpoint(archive_path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    lines = archive_path.read_text(encoding="utf-8").splitlines()
    archive_path.write_text("\n".join(lines[:7]) + "\n", encoding="utf-8")  # drop the last 3, AFTER the checkpoint

    result = verify_archive_completeness(archive_path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.internal.ok
    assert result.anchor.status == AnchorStatus.TAIL_TRUNCATED


def test_exact_anchored_archive_with_no_further_changes_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = _small_archive(tmp_path, monkeypatch, n=10)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_archive_checkpoint(archive_path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    result = verify_archive_completeness(archive_path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.internal.ok
    assert result.anchor.status == AnchorStatus.ANCHORED_VALID


def test_interior_deletion_still_fails_internal_verification_even_with_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _small_archive(tmp_path, monkeypatch, n=10)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_archive_checkpoint(archive_path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    lines = archive_path.read_text(encoding="utf-8").splitlines()
    del lines[3]
    archive_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_archive_completeness(archive_path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert not result.internal.ok


def test_forged_archive_checkpoint_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    archive_path = _small_archive(tmp_path, monkeypatch, n=10)
    key, allowed_signers, registry = _operator_registry_and_signers(tmp_path)
    create_archive_checkpoint(archive_path, namespace=NAMESPACE, private_key_path=key.private_key_path, signer_id="op-1")

    cp_path = archive_path.with_name(archive_path.name + ".checkpoints.ndjson")
    cp_line = json.loads(cp_path.read_text(encoding="utf-8").splitlines()[0])
    cp_line["checkpoint"]["anchored_seq"] = 2
    cp_path.write_text(json.dumps(cp_line) + "\n", encoding="utf-8")

    result = verify_archive_completeness(archive_path, signer_registry=registry, allowed_signers_path=allowed_signers)
    assert result.anchor.status == AnchorStatus.CHECKPOINT_INVALID
