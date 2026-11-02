"""AT-03: property tests — no influence increase without validator AND
operator (both NOT IN contributors); automaton cannot sign an increase even
with a valid signature; expired/wrong-namespace/replayed request rejected.

**P0-2 remediation** (adversarial review finding C.2): role is authoritative
only via a `SignerRegistry`, never via the self-declared `Signature.
signer_role` field. This file's default fixture registry registers
"validator-x" as VALIDATOR, "operator-y" as OPERATOR, and "automaton-1" as
AUTOMATON -- three distinct principals -- so the "happy path" tests below
exercise the registry-backed path already. The exploit-regression tests in
the "P0-2 exploit regressions" section below reproduce the review's exact
repro (one principal registered under, or claiming, both roles) and prove it
now fails closed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_lab.canon.jcs import canonical_sha256_excluding
from model_lab.contracts.ledger_record import Signature
from model_lab.contracts.model_version import LifecycleState
from model_lab.contracts.task import InfluenceTier
from model_lab.contracts.transition_request import TransitionRequest
from model_lab.governance.signer_registry import RegisteredSigner, SignerRegistry, SignerRole
from model_lab.governance.verifier import (
    AUTOMATON_ROLE,
    OPERATOR_ROLE,
    VALIDATOR_ROLE,
    RejectionReason,
    TransitionRejected,
    verify_transition,
)
from model_lab.ledger.signing import (
    SigningToolUnavailableError,
    generate_throwaway_keypair,
    sign_bytes,
    write_allowed_signers,
)

pytestmark = pytest.mark.acceptance

HEAD = "a" * 64
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
NAMESPACE = "model-lab-transition"


def _request(**overrides: object) -> TransitionRequest:
    base = dict(
        request_id="req-1",
        model_id="m1",
        model_version_hash="b" * 64,
        lifecycle_before=LifecycleState.SHADOW,
        lifecycle_after=LifecycleState.PROSPECTIVE_CHALLENGER,
        influence_before=InfluenceTier.T0,
        influence_after=InfluenceTier.T1,
        reason="promoting after clean shadow window",
        requested_by="author-1",
        contributors=["author-1", "contributor-2"],
        namespace=NAMESPACE,
        prev_ledger_hash=HEAD,
        created_at_utc=NOW,
        valid_before_utc=NOW + timedelta(days=30),
        request_sha256="0" * 64,  # placeholder, fixed below
    )
    base.update(overrides)
    draft = TransitionRequest(**base)
    real_hash = canonical_sha256_excluding(draft.canonical_dict(), "request_sha256")
    base["request_sha256"] = real_hash
    return TransitionRequest(**base)


def _sig(role: str, signer_id: str, namespace: str = NAMESPACE) -> Signature:
    return Signature(
        signer_role=role,
        signer_id=signer_id,
        namespace=namespace,
        signature_b64="unused-in-skip-crypto-mode",
        signed_at_utc=NOW,
    )


def _registry(*signers: RegisteredSigner) -> SignerRegistry:
    return SignerRegistry(signers=list(signers))


def _registered(principal: str, roles: list[SignerRole], namespaces: list[str] | None = None) -> RegisteredSigner:
    return RegisteredSigner(
        principal=principal, roles=roles, namespaces=namespaces or [NAMESPACE], public_key_line="ssh-ed25519 AAAA...fake"
    )


DEFAULT_REGISTRY = _registry(
    _registered("validator-x", [SignerRole.VALIDATOR]),
    _registered("operator-y", [SignerRole.OPERATOR]),
    _registered("automaton-1", [SignerRole.AUTOMATON]),
)


def _verify(req: TransitionRequest, sigs: list[Signature], *, registry: SignerRegistry = DEFAULT_REGISTRY, **kw):
    kw.setdefault("current_ledger_head_hash", HEAD)
    kw.setdefault("now", NOW)
    kw.setdefault("allowed_signers_path", Path("/dev/null"))
    kw.setdefault("skip_crypto_verification", True)
    return verify_transition(req, sigs, signer_registry=registry, **kw)


# --- invariant: an influence increase requires validator AND operator -----


def test_increase_with_both_validator_and_operator_accepted() -> None:
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    transition = _verify(req, sigs)
    assert transition.applied_by_role == "validator+operator"
    assert transition.influence_after == InfluenceTier.T1
    assert transition.validator_signer_id == "validator-x"
    assert transition.operator_signer_id == "operator-y"


def test_increase_missing_validator_rejected() -> None:
    req = _request()
    sigs = [_sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_increase_missing_operator_rejected() -> None:
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "validator-x")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.MISSING_OPERATOR_SIGNATURE


def test_increase_with_validator_who_is_a_contributor_rejected() -> None:
    req = _request(contributors=["author-1", "validator-x"])
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.SIGNER_IS_CONTRIBUTOR_OR_AUTHOR


def test_increase_with_operator_who_is_the_author_rejected() -> None:
    req = _request(requested_by="operator-y")
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.SIGNER_IS_CONTRIBUTOR_OR_AUTHOR


# --- P0-2 exploit regressions (adversarial review finding C.2) -------------


def test_EXPLOIT_same_signer_relabelled_for_both_roles_is_rejected() -> None:
    """The review's exact repro: one principal ("lone-wolf"), registered
    under BOTH roles in the registry (the review's attacker controls their
    own registration in this scenario, or a misconfiguration grants it), and
    two Signature objects wrap the same signer_id under the two different
    role labels. Must be rejected: the SAME principal can never satisfy both
    required roles on one transition, regardless of what the registry or the
    self-declared labels say."""
    registry = _registry(_registered("lone-wolf", [SignerRole.VALIDATOR, SignerRole.OPERATOR]))
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "lone-wolf"), _sig(OPERATOR_ROLE, "lone-wolf")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.SAME_SIGNER_FOR_BOTH_ROLES


def test_EXPLOIT_same_key_wrapped_under_two_role_labels_is_rejected_even_with_real_crypto(
    tmp_path: Path,
) -> None:
    """Same repro as above, but with REAL ssh-keygen -Y sign/verify (not the
    skip_crypto_verification shortcut) -- one real keypair, one real
    signature, wrapped in two Signature objects under different role labels.
    This is the review's exact executed repro, reproduced here."""
    try:
        key = generate_throwaway_keypair(tmp_path / "lone-wolf", comment="lone-wolf")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available in this environment")

    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(allowed_signers, [("lone-wolf", [NAMESPACE], key.public_key_text)])
    registry = _registry(_registered("lone-wolf", [SignerRole.VALIDATOR, SignerRole.OPERATOR]))

    req = _request()
    payload = canonical_sha256_excluding(req.canonical_dict(), "request_sha256").encode("utf-8")
    sig_text = sign_bytes(payload, key.private_key_path, NAMESPACE)

    sigs = [
        Signature(signer_role=VALIDATOR_ROLE, signer_id="lone-wolf", namespace=NAMESPACE, signature_b64=sig_text, signed_at_utc=NOW),
        Signature(signer_role=OPERATOR_ROLE, signer_id="lone-wolf", namespace=NAMESPACE, signature_b64=sig_text, signed_at_utc=NOW),
    ]
    with pytest.raises(TransitionRejected) as exc:
        verify_transition(
            req, sigs, signer_registry=registry, current_ledger_head_hash=HEAD, now=NOW,
            allowed_signers_path=allowed_signers,
        )
    assert exc.value.reason == RejectionReason.SAME_SIGNER_FOR_BOTH_ROLES


def test_distinct_authorized_validator_and_operator_accepted() -> None:
    """The legitimate case: two DIFFERENT registered principals, each
    holding exactly the one role they claim. Must succeed."""
    registry = _registry(
        _registered("validator-x", [SignerRole.VALIDATOR]),
        _registered("operator-y", [SignerRole.OPERATOR]),
    )
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    transition = _verify(req, sigs, registry=registry)
    assert transition.applied_by_role == "validator+operator"
    assert transition.validator_signer_id == "validator-x"
    assert transition.operator_signer_id == "operator-y"


def test_unauthorized_signer_claiming_validator_role_is_rejected() -> None:
    """A principal NOT registered as a validator cannot satisfy the
    validator requirement merely by self-labelling a Signature
    signer_role="validator" -- the registry, not the label, decides."""
    registry = _registry(
        _registered("nobody", [SignerRole.AUTOMATON]),  # registered, but NOT as validator
        _registered("operator-y", [SignerRole.OPERATOR]),
    )
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "nobody"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_unauthorized_signer_claiming_operator_role_is_rejected() -> None:
    registry = _registry(
        _registered("validator-x", [SignerRole.VALIDATOR]),
        _registered("nobody", [SignerRole.AUTOMATON]),
    )
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "nobody")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.MISSING_OPERATOR_SIGNATURE


def test_completely_unregistered_signer_claiming_either_role_is_rejected() -> None:
    """A principal not in the registry AT ALL, claiming validator -- the
    registry lookup returns no roles, so this is indistinguishable from
    "missing"."""
    registry = _registry(_registered("operator-y", [SignerRole.OPERATOR]))
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "total-stranger"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_self_declared_role_label_is_never_trusted_even_when_it_happens_to_match() -> None:
    """A principal registered ONLY as operator, whose Signature is labelled
    signer_role="validator" -- must NOT count as a validator just because
    they are a legitimate, registered signer of some other role."""
    registry = _registry(
        _registered("operator-y", [SignerRole.OPERATOR]),  # NOT a validator
        _registered("operator-z", [SignerRole.OPERATOR]),
    )
    req = _request()
    # operator-y's Signature is mislabelled "validator" -- registry says otherwise.
    sigs = [_sig(VALIDATOR_ROLE, "operator-y"), _sig(OPERATOR_ROLE, "operator-z")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_three_distinct_signers_where_one_holds_both_roles_still_succeeds_via_the_others() -> None:
    """A registry misconfiguration (one principal holding both roles) does
    NOT break legitimate transitions when a genuinely distinct second
    signer is also present -- the fix finds any valid distinct pair, it
    doesn't rigidly reject the whole registry."""
    registry = _registry(
        _registered("dual-role", [SignerRole.VALIDATOR, SignerRole.OPERATOR]),
        _registered("operator-y", [SignerRole.OPERATOR]),
    )
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "dual-role"), _sig(OPERATOR_ROLE, "operator-y")]
    transition = _verify(req, sigs, registry=registry)
    assert transition.applied_by_role == "validator+operator"
    assert transition.validator_signer_id == "dual-role"
    assert transition.operator_signer_id == "operator-y"


# --- invariant: automaton can never sign an increase, even if "valid" -----


def test_automaton_cannot_authorise_increase_even_alongside_unrelated_valid_sig() -> None:
    req = _request()
    sigs = [_sig(AUTOMATON_ROLE, "automaton-1")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_automaton_signature_sufficient_for_a_decrease() -> None:
    req = _request(
        lifecycle_before=LifecycleState.SHADOW,
        lifecycle_after=LifecycleState.QUARANTINED,
        influence_before=InfluenceTier.T1,
        influence_after=InfluenceTier.T0,
    )
    sigs = [_sig(AUTOMATON_ROLE, "automaton-1")]
    transition = _verify(req, sigs)
    assert transition.applied_by_role == AUTOMATON_ROLE


def test_automaton_signature_sufficient_for_unchanged_influence() -> None:
    req = _request(
        lifecycle_before=LifecycleState.RESEARCH,
        lifecycle_after=LifecycleState.REPRODUCED,
        influence_before=InfluenceTier.T0,
        influence_after=InfluenceTier.T0,
    )
    sigs = [_sig(AUTOMATON_ROLE, "automaton-1")]
    transition = _verify(req, sigs)
    assert transition.applied_by_role == AUTOMATON_ROLE


def test_unregistered_principal_claiming_automaton_role_does_not_count() -> None:
    registry = _registry(_registered("validator-x", [SignerRole.VALIDATOR]))
    req = _request(
        lifecycle_before=LifecycleState.RESEARCH,
        lifecycle_after=LifecycleState.REPRODUCED,
        influence_before=InfluenceTier.T0,
        influence_after=InfluenceTier.T0,
    )
    sigs = [_sig(AUTOMATON_ROLE, "not-registered")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, registry=registry)
    assert exc.value.reason == RejectionReason.NO_VALID_SIGNATURE


# --- expired / wrong-namespace / replayed request --------------------------


def test_expired_request_rejected() -> None:
    req = _request(valid_before_utc=NOW + timedelta(hours=1))
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    later = NOW + timedelta(hours=2)
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, now=later)
    assert exc.value.reason == RejectionReason.EXPIRED


def test_wrong_namespace_signature_not_counted() -> None:
    req = _request()
    sigs = [
        _sig(VALIDATOR_ROLE, "validator-x", namespace="some-other-namespace"),
        _sig(OPERATOR_ROLE, "operator-y"),
    ]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.MISSING_VALIDATOR_SIGNATURE


def test_replayed_request_wrong_ledger_head_rejected() -> None:
    req = _request()
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs, current_ledger_head_hash="c" * 64)
    assert exc.value.reason == RejectionReason.WRONG_LEDGER_HEAD


def test_invalid_lifecycle_step_rejected_even_with_valid_signatures() -> None:
    req = _request(lifecycle_before=LifecycleState.RESEARCH, lifecycle_after=LifecycleState.SHADOW)
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(req, sigs)
    assert exc.value.reason == RejectionReason.INVALID_LIFECYCLE_TRANSITION


def test_tampered_request_body_fails_integrity_check() -> None:
    req = _request()
    tampered = req.model_copy(update={"reason": "a different reason entirely"})
    sigs = [_sig(VALIDATOR_ROLE, "validator-x"), _sig(OPERATOR_ROLE, "operator-y")]
    with pytest.raises(TransitionRejected) as exc:
        _verify(tampered, sigs)
    assert exc.value.reason == RejectionReason.REQUEST_INTEGRITY_MISMATCH


# --- end-to-end with REAL ssh-keygen -Y sign/verify (throwaway keys) ------


def test_end_to_end_real_signature_verification(tmp_path: Path) -> None:
    try:
        validator_key = generate_throwaway_keypair(tmp_path / "validator", comment="validator")
        operator_key = generate_throwaway_keypair(tmp_path / "operator", comment="operator")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available in this environment")

    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(
        allowed_signers,
        [
            ("validator-x", [NAMESPACE], validator_key.public_key_text),
            ("operator-y", [NAMESPACE], operator_key.public_key_text),
        ],
    )
    registry = _registry(
        _registered("validator-x", [SignerRole.VALIDATOR]),
        _registered("operator-y", [SignerRole.OPERATOR]),
    )

    req = _request()
    payload = canonical_sha256_excluding(req.canonical_dict(), "request_sha256").encode("utf-8")

    validator_sig_text = sign_bytes(payload, validator_key.private_key_path, NAMESPACE)
    operator_sig_text = sign_bytes(payload, operator_key.private_key_path, NAMESPACE)

    sigs = [
        Signature(signer_role=VALIDATOR_ROLE, signer_id="validator-x", namespace=NAMESPACE, signature_b64=validator_sig_text, signed_at_utc=NOW),
        Signature(signer_role=OPERATOR_ROLE, signer_id="operator-y", namespace=NAMESPACE, signature_b64=operator_sig_text, signed_at_utc=NOW),
    ]

    transition = verify_transition(
        req, sigs, signer_registry=registry, current_ledger_head_hash=HEAD, now=NOW,
        allowed_signers_path=allowed_signers,
    )
    assert transition.applied_by_role == "validator+operator"
    assert transition.validator_signer_id == "validator-x"
    assert transition.operator_signer_id == "operator-y"


def test_end_to_end_forged_signature_rejected(tmp_path: Path) -> None:
    try:
        validator_key = generate_throwaway_keypair(tmp_path / "validator", comment="validator")
        operator_key = generate_throwaway_keypair(tmp_path / "operator", comment="operator")
        attacker_key = generate_throwaway_keypair(tmp_path / "attacker", comment="attacker")
    except SigningToolUnavailableError:
        pytest.skip("ssh-keygen not available in this environment")

    allowed_signers = tmp_path / "allowed_signers"
    write_allowed_signers(
        allowed_signers,
        [
            ("validator-x", [NAMESPACE], validator_key.public_key_text),
            ("operator-y", [NAMESPACE], operator_key.public_key_text),
        ],
    )
    registry = _registry(
        _registered("validator-x", [SignerRole.VALIDATOR]),
        _registered("operator-y", [SignerRole.OPERATOR]),
    )

    req = _request()
    payload = canonical_sha256_excluding(req.canonical_dict(), "request_sha256").encode("utf-8")

    forged_sig_text = sign_bytes(payload, attacker_key.private_key_path, NAMESPACE)
    validator_sig_text = sign_bytes(payload, validator_key.private_key_path, NAMESPACE)

    sigs = [
        Signature(signer_role=VALIDATOR_ROLE, signer_id="validator-x", namespace=NAMESPACE, signature_b64=validator_sig_text, signed_at_utc=NOW),
        Signature(signer_role=OPERATOR_ROLE, signer_id="operator-y", namespace=NAMESPACE, signature_b64=forged_sig_text, signed_at_utc=NOW),
    ]

    with pytest.raises(TransitionRejected) as exc:
        verify_transition(
            req, sigs, signer_registry=registry, current_ledger_head_hash=HEAD, now=NOW,
            allowed_signers_path=allowed_signers,
        )
    assert exc.value.reason == RejectionReason.MISSING_OPERATOR_SIGNATURE
