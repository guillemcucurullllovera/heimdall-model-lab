"""AT-16 dogfooding controls — MODEL1_foundation_contracts.md §21.

"ESN control -> REJECTED; pure noise -> REJECTED; skewed noise -> REJECTED;
planted edge -> SHADOW; challenger identical to incumbent -> inertness
fail; deliberate leak -> caught; tampered ledger -> QUARANTINED."

Six of the seven controls live here (Gate v2 Branch S/F evidence-driven
scenarios + the leakage harness); "tampered ledger -> QUARANTINED" is
covered separately in `test_tampered_ledger_quarantine.py` (it exercises
the evidence ledger directly, not Gate v2).

Mapping from Gate v2 verdicts to lifecycle actions, per §11/§17's own state
machine (no new rule invented here):
- Verdict FAIL or INSUFFICIENT on a RESEARCH/candidate model's only
  evidence => the model does not advance; for a control explicitly expected
  to show no real edge, this is what "REJECTED" means in practice (the gate
  itself only ever returns SUFFICIENT/INSUFFICIENT/FAIL -- see gate_v2/
  verdict.py's own docstring: "the gate ... can never raise influence"; the
  REJECTED lifecycle side-state is a governance/registry decision made in
  response to a non-SUFFICIENT verdict, never asserted here).
- Verdict SUFFICIENT is the necessary (not sufficient by itself --
  §17 governance still applies) condition for a model to be eligible to
  enter SHADOW.
"""

from __future__ import annotations

import numpy as np
import pytest

from model_lab.eval.leakage import future_shuffle_test
from model_lab.gate_v2.branch_f import BranchFEvidence, evaluate_branch_f
from model_lab.gate_v2.branch_s import BranchSEvidence, evaluate_branch_s
from model_lab.gate_v2.contracts import Verdict
from model_lab.gate_v2.verdict import compute_verdict

pytestmark = pytest.mark.dogfood


def _branch_s_verdict(trial_matrix: np.ndarray, selected_idx: int, *, calendar_years: float = 2.4) -> Verdict:
    ev = BranchSEvidence(
        family_id="dogfood",
        lineage_id="dogfood",
        trial_matrix=trial_matrix,
        selected_trial_index=selected_idx,
        cost_model_id="flat",
        cost_multiplier=0.0001,
        stress_episode_windows=[(50, 100), (400, 450)],
        comparator_series={"ewma": trial_matrix[:, (selected_idx + 2) % trial_matrix.shape[1]]},
        calendar_years=calendar_years,
        target_sr_annualised=0.5,
    )
    verdict, _ = compute_verdict(evaluate_branch_s(ev))
    return verdict


def test_esn_control_rejected() -> None:
    """Stand-in for the estate's known ESN-baseline control: an
    Echo-State-Network-style reservoir model has no genuine forward-looking
    edge on data unrelated to its training signal, so its "best" trial
    among many registered configurations is indistinguishable from noise
    after multiplicity correction. Modelled here as pure Gaussian noise
    across M registered trial configurations (an ESN's actual output
    distribution is not reproduced; what matters for this control is that a
    baseline KNOWN to carry no real edge must not spuriously clear the
    gate)."""
    rng = np.random.RandomState(42)
    T, M = 800, 12
    trial_matrix = rng.randn(T, M) * 0.008  # no true mean anywhere
    best_idx = int(np.argmax([trial_matrix[:, j].mean() for j in range(M)]))
    verdict = _branch_s_verdict(trial_matrix, best_idx)
    assert verdict != Verdict.SUFFICIENT, f"ESN-control noise must not clear the gate (got {verdict})"


def test_pure_noise_rejected() -> None:
    rng = np.random.RandomState(7)
    T, M = 700, 10
    trial_matrix = rng.randn(T, M) * 0.01
    best_idx = int(np.argmax([trial_matrix[:, j].mean() / (trial_matrix[:, j].std() + 1e-9) for j in range(M)]))
    verdict = _branch_s_verdict(trial_matrix, best_idx)
    assert verdict != Verdict.SUFFICIENT


def test_skewed_noise_rejected() -> None:
    """Noise drawn from a skewed (chi-square, recentred) distribution --
    tests that the gate's DSR statistic (which explicitly corrects for
    skew/kurtosis, unlike v1) doesn't mistake distributional asymmetry for
    real edge."""
    rng = np.random.RandomState(11)
    T, M = 700, 10
    raw = rng.chisquare(df=3, size=(T, M))
    trial_matrix = (raw - raw.mean(axis=0)) * 0.004  # recentre to zero true mean, keep the skew
    best_idx = int(np.argmax([trial_matrix[:, j].mean() / (trial_matrix[:, j].std() + 1e-9) for j in range(M)]))
    verdict = _branch_s_verdict(trial_matrix, best_idx)
    assert verdict != Verdict.SUFFICIENT


def test_planted_edge_reaches_sufficient_verdict_eligible_for_shadow() -> None:
    """A genuine, consistent edge planted in one trial among a SMALL
    registered family (low multiplicity penalty) should clear every S0-S16
    gating component -- the necessary condition (not by itself sufficient;
    §17 governance still applies) for a model to be eligible to enter
    SHADOW."""
    rng = np.random.RandomState(3)
    T, M = 1500, 5  # small family: low N_eff multiplicity penalty
    trial_matrix = rng.randn(T, M) * 0.004
    trial_matrix[:, 2] += 0.0015  # the selected trial: a real, consistent edge
    trial_matrix[:, 1] += 0.0010  # +/-1-step "parameter neighbours" (S14) carry a
    trial_matrix[:, 3] += 0.0010  # weaker but same-direction edge, not a lucky spike
    # calendar_years must be consistent with T daily observations (S5's MinBTL
    # consistency check computes a required-years figure from the DSR inputs
    # and compares it against calendar_years -- an inconsistent value fails
    # S5 even when every other component is genuinely satisfied).
    verdict = _branch_s_verdict(trial_matrix, 2, calendar_years=T / 252.0)
    assert verdict == Verdict.SUFFICIENT, (
        "a genuine, consistently-planted edge in a small registered family should clear Branch S "
        "(necessary condition for SHADOW eligibility)"
    )


def test_challenger_identical_to_incumbent_is_inertness_fail() -> None:
    """A challenger whose decisions never differ from the incumbent's
    carries zero evidence of real difference -- F7 (inertness+power) must
    refuse to treat this as adequate evidence, so the verdict can never be
    SUFFICIENT no matter how the loss values themselves compare."""
    rng = np.random.RandomState(5)
    n = 400
    loss_incumbent = rng.randn(n) * 0.01
    loss_challenger = loss_incumbent.copy()  # byte-for-byte identical losses
    decisions = rng.randint(0, 3, size=n)  # same decision sequence for both
    ev = BranchFEvidence(
        loss_incumbent=loss_incumbent,
        loss_challenger=loss_challenger,
        loss_naive=rng.randn(n) * 0.012,
        is_nested=False,
        decision_incumbent=decisions,
        decision_challenger=decisions.copy(),
        era_labels=np.array([i // 100 for i in range(n)]),
    )
    components = evaluate_branch_f(ev)
    f7 = next(c for c in components if c.id == "F7")
    assert f7.passed is False, "F7 must detect zero decision-difference as inadequate evidence"
    verdict, _ = compute_verdict(components)
    assert verdict != Verdict.SUFFICIENT, "an inert (identical-decision) challenger must never reach SUFFICIENT"


def test_deliberate_leak_is_caught() -> None:
    """Reuses the M1-F leakage harness (eval/leakage.py, also exercised by
    AT-05) as the dogfooding control for 'deliberate leak -> caught': a
    feature function that peeks at a future checkpoint must fail the
    future-shuffle test."""

    def leaky_feature(data: np.ndarray) -> np.ndarray:
        # Deliberately uses a CENTRED window that includes points strictly
        # after t (a positional leak -- must use a shuffle-sensitive,
        # position-dependent leak; a permutation-invariant global statistic
        # like a plain mean would NOT be caught by a shuffle-based test,
        # since reordering the tail doesn't change its mean).
        window = 10
        out = np.full(len(data), np.nan)
        half = window // 2
        for t in range(len(data)):
            lo, hi = max(0, t - half), min(len(data), t + half + 1)
            out[t] = data[lo:hi].mean()
        return out

    rng = np.random.RandomState(9)
    data = rng.randn(200)
    caught = not future_shuffle_test(leaky_feature, data, checkpoint=50)
    assert caught, "the leakage harness must catch a feature function that uses future information"
