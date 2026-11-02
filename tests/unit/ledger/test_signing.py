from __future__ import annotations

from pathlib import Path

import pytest

from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    find_principals,
    generate_throwaway_keypair,
    sign_bytes,
    verify_signature,
    write_allowed_signers,
)


def test_write_allowed_signers_format(tmp_path: Path) -> None:
    path = tmp_path / "allowed_signers"
    write_allowed_signers(path, [("alice", ["ns1", "ns2"], "ssh-ed25519 AAAA fake")])
    text = path.read_text(encoding="utf-8")
    assert text.strip() == 'alice namespaces="ns1,ns2" ssh-ed25519 AAAA fake'


def test_sign_and_verify_round_trip(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "k")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed = tmp_path / "allowed_signers"
    write_allowed_signers(allowed, [("alice", ["ns"], key.public_key_text)])

    sig = sign_bytes(b"hello world", key.private_key_path, "ns")
    assert verify_signature(b"hello world", sig, allowed_signers_path=allowed, principal="alice", namespace="ns")


def test_verify_fails_for_wrong_message(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "k")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed = tmp_path / "allowed_signers"
    write_allowed_signers(allowed, [("alice", ["ns"], key.public_key_text)])

    sig = sign_bytes(b"original message", key.private_key_path, "ns")
    assert not verify_signature(b"tampered message", sig, allowed_signers_path=allowed, principal="alice", namespace="ns")


def test_verify_fails_for_wrong_namespace(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "k")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed = tmp_path / "allowed_signers"
    write_allowed_signers(allowed, [("alice", ["ns-a"], key.public_key_text)])

    sig = sign_bytes(b"hello", key.private_key_path, "ns-a")
    assert not verify_signature(b"hello", sig, allowed_signers_path=allowed, principal="alice", namespace="ns-b")


def test_verify_fails_for_unlisted_principal(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "k")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed = tmp_path / "allowed_signers"
    write_allowed_signers(allowed, [("alice", ["ns"], key.public_key_text)])

    sig = sign_bytes(b"hello", key.private_key_path, "ns")
    assert not verify_signature(b"hello", sig, allowed_signers_path=allowed, principal="bob", namespace="ns")


def test_find_principals_returns_the_signer(tmp_path: Path) -> None:
    try:
        key = generate_throwaway_keypair(tmp_path / "k")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen unavailable")
    allowed = tmp_path / "allowed_signers"
    write_allowed_signers(allowed, [("alice", ["ns"], key.public_key_text)])

    sig = sign_bytes(b"hello", key.private_key_path, "ns")
    principals = find_principals(sig, allowed)
    assert "alice" in principals
