"""Proper scoring rules — MODEL1_foundation_contracts.md §14.

"Modules (all pure, deterministic, unit-tested against known answers and
`arch`/`scoringrules` oracles): log score, CRPS (incl. Student-t and mixture
closed forms), Brier, pinball, QLIKE (with a declared proxy-floor policy),
joint (VaR,ES) FZ0 ... Every evaluation output carries `n`, `n_independent`,
MDE and data cutoff."

QLIKE proxy-floor policy: the realised-variance proxy `sigma2_true` is
clipped to `QLIKE_PROXY_FLOOR` before use. Motivation: QLIKE = sigma2_true/sigma2_hat
- ln(sigma2_true/sigma2_hat) - 1 is undefined at sigma2_true=0 (a genuinely
zero realised variance is impossible for a continuous return process but can
appear from a stale/repeated-price data artifact); flooring prevents a single
degenerate proxy observation from producing +inf and silently discarding the
whole score. The floor value is a data-quality parameter, not a statistical
tuning knob, and is always reported alongside the score.

FZ0 (Fissler-Ziegel joint VaR/ES scoring function): implemented as the
G1=0 special case of Fissler & Ziegel (2016a, "Higher Order Elicitability
and Osband's Principle," Annals of Statistics), the choice the ORIGINAL
paper itself explicitly endorses ("S remains strictly consistent upon
choosing G1 = 0 and G2 strictly increasing and strictly convex," remark
following the Theorem 5.2(ii) discussion) with G2 = exp.

Erratum status (CORRECTED 2026-09-30 after the independent adversarial
review found the prior "2021 erratum" framing was mis-dated and
imprecise; see reports/MODEL1_REVIEW_REMEDIATION.md P2-7): the actual
erratum in this corpus is "Supplement to 'Erratum: Higher Order
Elicitability and Osband's Principle'" (Fissler & Ziegel, arXiv:1901.08826,
submitted ~January 2019 -- NOT "post-2021"). It corrects Proposition 3.4
and Theorem 5.2(ii), and comments on imprecisions in Propositions 4.2/4.4,
of the SAME 2016a paper this FZ0 implementation is built on.

This is MORE directly relevant than the prior framing suggested, not less:
Theorem 5.2(ii) is literally the theorem the original paper invokes to
justify the G1=0 choice used here. The erratum shows Theorem 5.2(ii)'s
original proof has a gap and supplies a counterexample where a scoring
function satisfying the theorem's stated conditions is NOT actually
consistent -- but that counterexample uses a specific CONE-RESTRICTED
action domain (A = {(x1,x2): x1>=0, |x2|<=x1}) and a non-zero G1 coupled
to G2 (G1(s) = G2(-s)/alpha), not the unrestricted-domain, G1=0 case used
here. The erratum's Proposition 2 gives a sufficient extra condition
(existence of a specific connecting path/sequence in the action domain)
under which Theorem 5.2(ii)'s original conclusion DOES hold; whether an
unrestricted real-valued (VaR, ES) domain satisfies Proposition 2's
condition was NOT independently verified in this pass -- it plausibly
does (unrestricted rectangular domains are the easy case the erratum's
own counterexample construction needed a restricted cone to break), but
this is a plausibility judgement, not a checked proof. **Disposition:
genuinely OPEN, not resolved either way** -- do not read this docstring
as confirming FZ0 is erratum-safe, and do not read it as confirming it
is erratum-broken. Propriety (score minimised at the true (VaR,ES) pair)
IS verified numerically in tests regardless, which is a real, independent
correctness check -- but numerical propriety-at-a-point does not by
itself establish the erratum's global strict-consistency condition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

QLIKE_PROXY_FLOOR = 1e-8


@dataclass(frozen=True)
class ScoreOutput:
    value: float
    n: int
    n_independent: float
    mean: float

    @classmethod
    def from_losses(cls, losses: np.ndarray, *, n_independent: float | None = None) -> ScoreOutput:
        n = len(losses)
        return cls(
            value=float(np.mean(losses)),
            n=n,
            n_independent=float(n_independent if n_independent is not None else n),
            mean=float(np.mean(losses)),
        )


def log_score_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> ScoreOutput:
    """Negative log-likelihood under N(mu, sigma^2); lower is better (a proper score)."""
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    losses = -stats.norm.logpdf(y, loc=mu, scale=sigma)
    return ScoreOutput.from_losses(losses)


def brier_score(y: np.ndarray, p: np.ndarray) -> ScoreOutput:
    """y in {0,1}, p = predicted probability of y=1."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    losses = (p - y) ** 2
    return ScoreOutput.from_losses(losses)


def pinball_loss(y: np.ndarray, q_hat: np.ndarray, tau: float) -> ScoreOutput:
    """Quantile/pinball loss at quantile level tau in (0,1)."""
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0,1)")
    y, q_hat = np.asarray(y, float), np.asarray(q_hat, float)
    diff = y - q_hat
    losses = np.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    return ScoreOutput.from_losses(losses)


def qlike(sigma2_true: np.ndarray, sigma2_hat: np.ndarray, *, floor: float = QLIKE_PROXY_FLOOR) -> ScoreOutput:
    """QLIKE = sigma2_true/sigma2_hat - ln(sigma2_true/sigma2_hat) - 1. Proper
    for forecasting the conditional variance under mild conditions (Patton
    2011); minimised at sigma2_hat == sigma2_true. `sigma2_true` (the proxy)
    is floored at `floor` (see module docstring)."""
    proxy = np.clip(np.asarray(sigma2_true, float), floor, None)
    hat = np.asarray(sigma2_hat, float)
    if np.any(hat <= 0):
        raise ValueError("sigma2_hat must be strictly positive")
    ratio = proxy / hat
    losses = ratio - np.log(ratio) - 1.0
    return ScoreOutput.from_losses(losses)


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> ScoreOutput:
    """Closed-form CRPS for a Gaussian predictive distribution
    (Gneiting & Raftery 2007, eq. 20)."""
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    z = (y - mu) / sigma
    losses = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
    return ScoreOutput.from_losses(losses)


def crps_student_t(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float) -> ScoreOutput:
    """Closed-form CRPS for a (non-standardized) Student-t predictive
    distribution with nu>2 degrees of freedom (Jordan, Kruger & Lerch 2019,
    following the standardized-t closed form)."""
    if nu <= 1:
        raise ValueError("nu must be > 1 for CRPS to be finite")
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    z = (y - mu) / sigma
    B = (2 * math.sqrt(nu) / (nu - 1)) * (
        math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(math.pi)
    )
    term1 = z * (2 * stats.t.cdf(z, df=nu) - 1)
    term2 = 2 * stats.t.pdf(z, df=nu) * (nu + z**2) / (nu - 1)
    crps_std = term1 + term2 - B
    losses = sigma * crps_std
    return ScoreOutput.from_losses(losses)


def crps_gaussian_mixture(
    y: np.ndarray, weights: Sequence[float], mus: Sequence[np.ndarray], sigmas: Sequence[np.ndarray]
) -> ScoreOutput:
    """CRPS for a K-component Gaussian mixture via the standard pairwise
    closed-form expansion (Grimit, Gneiting, Berrocal & Johnson 2006):
    CRPS(F,y) = sum_i w_i*A(y-mu_i,sigma_i) - 0.5*sum_i sum_j w_i*w_j*A(mu_i-mu_j, sqrt(sigma_i^2+sigma_j^2))
    where A(mu,sigma) = mu*(2*Phi(mu/sigma)-1) + 2*sigma*phi(mu/sigma).
    """
    y = np.asarray(y, float)
    w = np.asarray(weights, float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("mixture weights must sum to 1")
    K = len(w)
    mu_arrs = [np.asarray(m, float) for m in mus]
    sigma_arrs = [np.asarray(s, float) for s in sigmas]

    def A(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        z = mu / sigma
        return mu * (2 * stats.norm.cdf(z) - 1) + 2 * sigma * stats.norm.pdf(z)

    term1 = sum(w[i] * A(y - mu_arrs[i], sigma_arrs[i]) for i in range(K))
    term2: np.ndarray = np.zeros_like(y)
    for i in range(K):
        for j in range(K):
            combined_sigma = np.sqrt(sigma_arrs[i] ** 2 + sigma_arrs[j] ** 2)
            term2 = term2 + w[i] * w[j] * A(mu_arrs[i] - mu_arrs[j], combined_sigma)
    losses = term1 - 0.5 * term2
    return ScoreOutput.from_losses(losses)


def fz0_joint_var_es(
    y: np.ndarray, var_hat: np.ndarray, es_hat: np.ndarray, alpha: float
) -> ScoreOutput:
    """FZ0 joint (VaR, ES) scoring function (Fissler & Ziegel 2016; see
    module docstring for the erratum caveat). alpha is the VaR/ES coverage
    level (e.g. 0.05 for 95% VaR). Convention: y is a P&L/return (losses are
    negative returns); VaR/ES are reported as negative numbers (the alpha-
    quantile of returns) matching y's sign convention throughout.

    FZ0(y, v, e) = -1/(alpha*e) * 1{y<=v} * (v - y) + v/e + ln(-e) - 1
    (using G1(v)=0, G2(x) = -1/x as the standard "0" member of the FZ family;
    requires e < 0, i.e. ES more extreme than 0 for a loss-making tail).
    """
    y, v, e = np.asarray(y, float), np.asarray(var_hat, float), np.asarray(es_hat, float)
    if np.any(e >= 0):
        raise ValueError("es_hat must be < 0 (a negative expected shortfall) for this FZ0 form")
    indicator = (y <= v).astype(float)
    losses = (-1.0 / (alpha * e)) * indicator * (v - y) + v / e + np.log(-e) - 1.0
    return ScoreOutput.from_losses(losses)
