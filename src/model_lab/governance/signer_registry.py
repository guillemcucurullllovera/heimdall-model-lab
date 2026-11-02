"""Signer registry — remediates P0-2 from the independent adversarial review
(`research-corpus/model-lab/reviews/MODEL1_ADVERSARIAL_REVIEW.md` finding C.2):

"Ledger governance: a single real signer can satisfy the 'validator AND
operator' two-person-integrity requirement... nothing anywhere... binds the
self-declared `Signature.signer_role` field to identity." The review's
working repro: one real keypair, registered under principal `lone-wolf`,
signs once; the identical `signature_b64` is wrapped in two `Signature`
objects — one tagged `signer_role="validator"`, one `signer_role="operator"`
— both `signer_id="lone-wolf"`. `verify_transition` accepted it.

The fix: role is never taken from the caller-supplied, self-declared
`Signature.signer_role` string. It is looked up from this registry — a
structured, independently-maintained record of which principal actually
holds which role — and a transition requiring two distinct roles additionally
checks that the two role-satisfying signatures come from two DISTINCT
principals, even if a bug or attacker supplies more than two `Signature`
objects or duplicates one under different claimed labels.

This registry is itself just data (JSON, persisted under
`heimdall-model-lab-data/governance/signer_registry.json`) — MODEL-1 does not
generate real operator/validator keys (D2); production entries are added by
the operator out-of-band, exactly like the existing `allowed_signers`
mechanism. Tests use throwaway keys.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from model_lab.contracts.base import ModelLabBaseModel

DEFAULT_REGISTRY_PATH = Path("./local-data/governance/signer_registry.json")


class SignerRole(StrEnum):
    VALIDATOR = "validator"
    OPERATOR = "operator"
    AUTOMATON = "automaton"


class RegisteredSigner(ModelLabBaseModel):
    """One principal's registered identity. A principal MAY legitimately hold
    multiple roles across the registry (e.g. in a small test fixture or a
    future automaton-operated-by-operator scenario) — what the verifier
    enforces is not "one role per principal" in the abstract, but "the two
    principals satisfying a single transition's two REQUIRED distinct roles
    must not be the same principal" (checked in governance/verifier.py, not
    here)."""

    principal: str
    roles: list[SignerRole]
    namespaces: list[str] = Field(default_factory=list)
    public_key_line: str = Field(description="the raw authorized_keys-style public key line")
    revoked: bool = False


class SignerRegistry(ModelLabBaseModel):
    schema_name: str = "SignerRegistry"
    schema_version: int = 1
    signers: list[RegisteredSigner] = Field(default_factory=list)

    def _active(self, principal: str) -> RegisteredSigner | None:
        for s in self.signers:
            if s.principal == principal and not s.revoked:
                return s
        return None

    def has_role(self, principal: str, role: SignerRole, *, namespace: str | None = None) -> bool:
        """The ONLY authoritative role check. Never trust a caller-supplied
        role label — always ask the registry."""
        signer = self._active(principal)
        if signer is None or role not in signer.roles:
            return False
        return namespace is None or namespace in signer.namespaces

    def role_holders(self, role: SignerRole) -> set[str]:
        return {s.principal for s in self.signers if not s.revoked and role in s.roles}

    def public_key_line_of(self, principal: str) -> str | None:
        signer = self._active(principal)
        return signer.public_key_line if signer else None

    def to_allowed_signers_entries(self) -> list[tuple[str, list[str], str]]:
        """(principal, namespaces, public_key_line) triples, the exact shape
        `ledger.signing.write_allowed_signers` expects — the registry is the
        single source of truth; the OpenSSH `allowed_signers` file used for
        the actual `ssh-keygen -Y verify` call is DERIVED from it, never
        maintained separately (a second, hand-edited allowed_signers file
        would reintroduce exactly the kind of disconnected-metadata gap this
        fix exists to close)."""
        return [(s.principal, s.namespaces, s.public_key_line) for s in self.signers if not s.revoked]


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> SignerRegistry:
    if not path.exists():
        return SignerRegistry(signers=[])
    return SignerRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def save_registry(registry: SignerRegistry, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json.loads(registry.model_dump_json()), indent=2, sort_keys=True), encoding="utf-8")
