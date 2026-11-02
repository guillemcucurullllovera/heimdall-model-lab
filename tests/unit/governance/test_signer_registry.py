from __future__ import annotations

from pathlib import Path

from model_lab.governance.signer_registry import (
    RegisteredSigner,
    SignerRegistry,
    SignerRole,
    load_registry,
    save_registry,
)


def _signer(principal: str, roles: list[SignerRole], namespaces: list[str] | None = None, revoked: bool = False) -> RegisteredSigner:
    return RegisteredSigner(
        principal=principal, roles=roles, namespaces=namespaces or ["ns1"], public_key_line="ssh-ed25519 AAAA fake", revoked=revoked
    )


def test_has_role_true_for_registered_role() -> None:
    reg = SignerRegistry(signers=[_signer("alice", [SignerRole.VALIDATOR])])
    assert reg.has_role("alice", SignerRole.VALIDATOR)


def test_has_role_false_for_unregistered_role() -> None:
    reg = SignerRegistry(signers=[_signer("alice", [SignerRole.VALIDATOR])])
    assert not reg.has_role("alice", SignerRole.OPERATOR)


def test_has_role_false_for_unknown_principal() -> None:
    reg = SignerRegistry(signers=[_signer("alice", [SignerRole.VALIDATOR])])
    assert not reg.has_role("bob", SignerRole.VALIDATOR)


def test_has_role_false_for_revoked_signer() -> None:
    reg = SignerRegistry(signers=[_signer("alice", [SignerRole.VALIDATOR], revoked=True)])
    assert not reg.has_role("alice", SignerRole.VALIDATOR)


def test_has_role_respects_namespace_when_given() -> None:
    reg = SignerRegistry(signers=[_signer("alice", [SignerRole.VALIDATOR], namespaces=["ns-a"])])
    assert reg.has_role("alice", SignerRole.VALIDATOR, namespace="ns-a")
    assert not reg.has_role("alice", SignerRole.VALIDATOR, namespace="ns-b")


def test_role_holders() -> None:
    reg = SignerRegistry(
        signers=[
            _signer("alice", [SignerRole.VALIDATOR]),
            _signer("bob", [SignerRole.VALIDATOR, SignerRole.OPERATOR]),
            _signer("carol", [SignerRole.OPERATOR], revoked=True),
        ]
    )
    assert reg.role_holders(SignerRole.VALIDATOR) == {"alice", "bob"}
    assert reg.role_holders(SignerRole.OPERATOR) == {"bob"}  # carol revoked, excluded


def test_to_allowed_signers_entries_excludes_revoked() -> None:
    reg = SignerRegistry(
        signers=[
            _signer("alice", [SignerRole.VALIDATOR], namespaces=["ns1", "ns2"]),
            _signer("bob", [SignerRole.OPERATOR], revoked=True),
        ]
    )
    entries = reg.to_allowed_signers_entries()
    assert entries == [("alice", ["ns1", "ns2"], "ssh-ed25519 AAAA fake")]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    reg = SignerRegistry(
        signers=[
            _signer("alice", [SignerRole.VALIDATOR]),
            _signer("bob", [SignerRole.OPERATOR, SignerRole.AUTOMATON]),
        ]
    )
    save_registry(reg, path)
    loaded = load_registry(path)
    assert {s.principal for s in loaded.signers} == {"alice", "bob"}
    assert loaded.has_role("bob", SignerRole.AUTOMATON)


def test_load_missing_file_returns_empty_registry(tmp_path: Path) -> None:
    reg = load_registry(tmp_path / "does_not_exist.json")
    assert reg.signers == []
    assert not reg.has_role("anyone", SignerRole.VALIDATOR)
