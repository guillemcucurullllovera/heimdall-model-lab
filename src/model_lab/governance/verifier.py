"""D2 transition verifier — MODEL1_foundation_contracts.md §17.

"the verifier (pure function) enforces `influence_after > influence_before`
=> validator AND operator, both NOT IN contributors; ... property tests for
the invariants (no path to PROMOTED without both signatures; every
automaton-signed transition has influence_after <= influence_before; ledger
replay reproduces state)."

**P0-2 remediation** (adversarial review finding C.2): role is NEVER taken
from the caller-supplied, self-declared `Signature.signer_role` field. It is
looked up from a `SignerRegistry` (governance/signer_registry.py) — the only
authoritative source of "which principal holds which role." A transition
requiring two distinct roles (validator, operator) additionally requires the
two satisfying signatures to come from two DIFFERENT principals, even when
the registry (mis)configures one principal with both roles, and even when
more than two `Signature` objects are supplied under arbitrary self-declared
labels. This closes the review's working repro: one keypair, registered
under principal `lone-wolf`, signing once and being wrapped in two
`Signature` objects labelled `signer_role="validator"`/`"operator"` no
longer satisfies anything — `lone-wolf` would need the registry to actually
list them under both roles AND a second, distinct principal to co-sign for
the transition to proceed.

This module is a pure function: `verify_transition` takes a request, its
attached signatures, a `SignerRegistry`, and the caller-supplied ledger head
hash / current time, and returns either an accepted `Transition` body or
raises `TransitionRejected` with a specific, machine-checkable reason. It has
no side effects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.contracts.ledger_record import Signature
from model_lab.contracts.model_version import is_valid_lifecycle_transition
from model_lab.contracts.transition_request import TransitionRequest, is_influence_increase
from model_lab.governance.contracts import Transition
from model_lab.governance.signer_registry import SignerRegistry, SignerRole
from model_lab.ledger.signing import verify_signature

# Kept as plain strings (not SignerRole) for backwards-compatible construction
# of Transition.applied_by_role and for any external string comparison; the
# authoritative role SOURCE is always SignerRegistry/SignerRole now.
AUTOMATON_ROLE = "automaton"
VALIDATOR_ROLE = "validator"
OPERATOR_ROLE = "operator"


class RejectionReason(StrEnum):
    EXPIRED = "expired"
    WRONG_LEDGER_HEAD = "wrong_ledger_head"  # anti-replay: request not bound to the current head
    WRONG_NAMESPACE = "wrong_namespace"
    MISSING_VALIDATOR_SIGNATURE = "missing_validator_signature"
    MISSING_OPERATOR_SIGNATURE = "missing_operator_signature"
    SIGNER_IS_CONTRIBUTOR_OR_AUTHOR = "signer_is_contributor_or_author"
    AUTOMATON_CANNOT_SIGN_INCREASE = "automaton_cannot_sign_increase"
    NO_VALID_SIGNATURE = "no_valid_signature"
    INVALID_LIFECYCLE_TRANSITION = "invalid_lifecycle_transition"
    REQUEST_INTEGRITY_MISMATCH = "request_integrity_mismatch"
    SAME_SIGNER_FOR_BOTH_ROLES = "same_signer_for_both_roles"


class TransitionRejected(Exception):
    def __init__(self, reason: RejectionReason, detail: str) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


def _request_signed_payload(request: TransitionRequest) -> bytes:
    """The bytes a signature is over: the request's own self-referential
    hash (excluding the request_sha256 field itself), matching the
    gate-v2/v1-manifest signing convention already established elsewhere in
    this codebase (sign over the file's own hash, not raw bytes)."""
    expected = canonical_sha256_excluding(request.canonical_dict(), "request_sha256")
    if expected != request.request_sha256:
        raise TransitionRejected(
            RejectionReason.REQUEST_INTEGRITY_MISMATCH,
            f"request_sha256 {request.request_sha256!r} does not match the recomputed hash {expected!r}",
        )
    return expected.encode("utf-8")


def verify_transition(
    request: TransitionRequest,
    signatures: list[Signature],
    *,
    signer_registry: SignerRegistry,
    current_ledger_head_hash: str,
    now: datetime | None = None,
    allowed_signers_path: Path,
    skip_crypto_verification: bool = False,
) -> Transition:
    """Pure function. Raises `TransitionRejected` on any failure; returns the
    `Transition` record body to append on success. `signer_registry` is
    mandatory and is the ONLY source of role authority (P0-2) — a caller
    cannot skip it. `skip_crypto_verification` exists only for unit tests
    that want to exercise the invariant logic without invoking `ssh-keygen`
    (crypto correctness itself is covered by ledger/signing tests) —
    production callers must never set it. `allowed_signers_path` must be an
    OpenSSH allowed_signers file derived from the SAME registry (see
    `SignerRegistry.to_allowed_signers_entries` / `ledger.signing.
    write_allowed_signers`) — the caller is responsible for keeping them in
    sync; a hand-maintained, disconnected `allowed_signers` file would
    reintroduce a version of the exact gap this fix closes.
    """
    now = now or datetime.now(UTC)
    payload = _request_signed_payload(request)

    if request.prev_ledger_hash != current_ledger_head_hash:
        raise TransitionRejected(
            RejectionReason.WRONG_LEDGER_HEAD,
            f"request bound to prev_ledger_hash={request.prev_ledger_hash!r} but the current "
            f"ledger head is {current_ledger_head_hash!r} (stale or replayed request)",
        )
    if now >= request.valid_before_utc:
        raise TransitionRejected(
            RejectionReason.EXPIRED, f"request expired at {request.valid_before_utc!r}, now is {now!r}"
        )
    if not is_valid_lifecycle_transition(request.lifecycle_before, request.lifecycle_after):
        raise TransitionRejected(
            RejectionReason.INVALID_LIFECYCLE_TRANSITION,
            f"{request.lifecycle_before!r} -> {request.lifecycle_after!r} is not a structurally valid step",
        )

    excluded_signers = set(request.contributors) | {request.requested_by}

    def _crypto_valid(sig: Signature) -> bool:
        if sig.namespace != request.namespace:
            return False
        if skip_crypto_verification:
            return True
        return verify_signature(
            payload,
            sig.signature_b64,
            allowed_signers_path=allowed_signers_path,
            principal=sig.signer_id,
            namespace=sig.namespace,
        )

    # Role is determined EXCLUSIVELY by the registry (P0-2) -- sig.signer_role
    # (the self-declared field) is never consulted for authorization. A
    # single crypto-valid signature can populate multiple role buckets if
    # (and only if) the registry itself lists that principal under multiple
    # roles; the distinct-identity check below is what actually prevents one
    # principal from satisfying two REQUIRED roles on one transition.
    valid_by_role: dict[SignerRole, list[Signature]] = {}
    any_crypto_valid = False
    for sig in signatures:
        if not _crypto_valid(sig):
            continue
        any_crypto_valid = True
        for role in SignerRole:
            if signer_registry.has_role(sig.signer_id, role, namespace=sig.namespace):
                valid_by_role.setdefault(role, []).append(sig)

    if not any_crypto_valid:
        raise TransitionRejected(RejectionReason.NO_VALID_SIGNATURE, "no cryptographically valid signature found")

    increase = is_influence_increase(request.influence_before, request.influence_after)

    if increase:
        validators = [s for s in valid_by_role.get(SignerRole.VALIDATOR, []) if s.signer_id not in excluded_signers]
        operators = [s for s in valid_by_role.get(SignerRole.OPERATOR, []) if s.signer_id not in excluded_signers]

        if not validators:
            conflicted = [
                s for s in valid_by_role.get(SignerRole.VALIDATOR, []) if s.signer_id in excluded_signers
            ]
            if conflicted:
                raise TransitionRejected(
                    RejectionReason.SIGNER_IS_CONTRIBUTOR_OR_AUTHOR,
                    f"validator signature from {conflicted[0].signer_id!r} is a contributor/author of this request",
                )
            raise TransitionRejected(
                RejectionReason.MISSING_VALIDATOR_SIGNATURE,
                "an influence increase requires a signature from a registry-confirmed validator; none present",
            )
        if not operators:
            conflicted = [s for s in valid_by_role.get(SignerRole.OPERATOR, []) if s.signer_id in excluded_signers]
            if conflicted:
                raise TransitionRejected(
                    RejectionReason.SIGNER_IS_CONTRIBUTOR_OR_AUTHOR,
                    f"operator signature from {conflicted[0].signer_id!r} is a contributor/author of this request",
                )
            raise TransitionRejected(
                RejectionReason.MISSING_OPERATOR_SIGNATURE,
                "an influence increase requires a signature from a registry-confirmed operator; none present",
            )

        # P0-2 core fix: require at least one (validator, operator) pair with
        # DIFFERENT signer_id -- rejects a single principal registered under
        # both roles (or two Signature objects relabeling the same signer)
        # unless a genuinely distinct second principal also signed.
        distinct_pair = next(
            ((v, o) for v in validators for o in operators if v.signer_id != o.signer_id), None
        )
        if distinct_pair is None:
            raise TransitionRejected(
                RejectionReason.SAME_SIGNER_FOR_BOTH_ROLES,
                f"the only registry-confirmed validator/operator signature(s) come from the same "
                f"principal ({validators[0].signer_id!r}) -- an influence increase requires two "
                f"DIFFERENT people, not one person's signature relabelled twice",
            )
        validator_sig, operator_sig = distinct_pair
        applied_by_role = "validator+operator"
        validator_signer_id: str | None = validator_sig.signer_id
        operator_signer_id: str | None = operator_sig.signer_id
    else:
        # Influence unchanged or reduced: an automaton signature suffices.
        # An automaton signature is NEVER sufficient for an increase, even
        # if cryptographically valid and registry-confirmed -- checked
        # explicitly (not merely by omission) so the invariant is visible in
        # code, not implied.
        has_automaton = bool(valid_by_role.get(SignerRole.AUTOMATON))
        has_human = bool(valid_by_role.get(SignerRole.VALIDATOR)) or bool(valid_by_role.get(SignerRole.OPERATOR))
        if not (has_automaton or has_human):
            raise TransitionRejected(
                RejectionReason.NO_VALID_SIGNATURE,
                "no registry-confirmed automaton/validator/operator signature found for a non-increasing transition",
            )
        applied_by_role = AUTOMATON_ROLE if has_automaton and not has_human else "validator+operator"
        validator_sigs = valid_by_role.get(SignerRole.VALIDATOR, [])
        operator_sigs = valid_by_role.get(SignerRole.OPERATOR, [])
        validator_signer_id = validator_sigs[0].signer_id if validator_sigs else None
        operator_signer_id = operator_sigs[0].signer_id if operator_sigs else None

    if increase and SignerRole.AUTOMATON in valid_by_role and applied_by_role != "validator+operator":
        # Defensive: unreachable given the logic above, asserted explicitly
        # per §17's own emphasis that this must hold even in the presence of
        # a technically-valid, registry-confirmed automaton signature.
        raise TransitionRejected(
            RejectionReason.AUTOMATON_CANNOT_SIGN_INCREASE,
            "an automaton signature can never authorise an influence increase",
        )

    return Transition(
        request_id=request.request_id,
        model_id=request.model_id,
        model_version_hash=request.model_version_hash,
        lifecycle_before=request.lifecycle_before,
        lifecycle_after=request.lifecycle_after,
        influence_before=request.influence_before,
        influence_after=request.influence_after,
        applied_at_utc=now,
        applied_by_role=applied_by_role,
        request_sha256=request.request_sha256,
        validator_signer_id=validator_signer_id,
        operator_signer_id=operator_signer_id,
    )
