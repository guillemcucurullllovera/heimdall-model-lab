"""Bayesian Online Changepoint Detection (Adams & MacKay 2007) with a
Normal-Inverse-Gamma conjugate model for i.i.d.-within-regime Gaussian
observations (unknown mean AND variance) -- T-CHANGE's frozen `BOCPD`
incumbent. `[M]`: the NIG-BOCPD recursion and the Student-t predictive form
below are standard, independently re-derived from conjugate-Bayesian-
Gaussian first principles (Murphy's well-known "Conjugate Bayesian
analysis of the Gaussian distribution" note derives the same predictive
form), not copied from any single primary source in the corpus.

At each step t, maintains a distribution over the current "run length" r_t
(days since the last changepoint) and, for each hypothesized run length, a
posterior (mu, kappa, alpha, beta) over the regime's own (mean, variance).
`changepoint_prob[t] = P(r_t = 0 | x_1..t)` -- the posterior probability a
changepoint occurred exactly at t. Vectorized over the run-length axis at
every step; low-probability run-length hypotheses are pruned to keep the
per-step cost bounded (a standard, purely computational approximation --
does not affect causality/correctness, only which negligible-probability
hypotheses are tracked).

## Remediation (independent adversarial review, finding P1-3)

The original implementation's `beta0` (the NIG prior's scale
hyperparameter, used for the "fresh regime" hypothesis at every step)
defaulted to `np.var(x)` computed ONCE, over the ENTIRE input array,
before the online loop began -- i.e. the prior used at t=0 already
encoded information about every future observation. Quantified: on a
series with a large shock only at the very end, `changepoint_prob` at
early, SHARED indices differed by up to 4x between a full-series run and
a prefix-only run (max abs diff 0.025 against a ~0.0005-0.004 baseline
scale) -- a real, demonstrated violation of this module's own "purely
online/causal by construction" claim, not merely a theoretical concern.

Fixed: `beta0` is now either (a) a FIXED, externally-supplied constant,
independent of `x` entirely (trivially causal), or (b) by default, an
EXPANDING, Welford-updated running variance of `x[0:t]` -- computed
strictly BEFORE `x[t]` is incorporated at each step, using only
information a live system at time `t` could actually have. Before
`min_burnin_for_variance` observations exist, a small fixed constant
(`burnin_beta0`, never derived from data) is used instead. Prefix-
invariance (running the same series truncated at different points
produces IDENTICAL results up to the truncation point) is now an exact,
permanent regression test -- see `tests/unit/changepoint/test_bocpd.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

NAN = float("nan")


@dataclass(frozen=True)
class BocpdResult:
    changepoint_prob: np.ndarray  # P(run length resets to 0) at each t
    map_run_length: np.ndarray  # argmax_r P(run length = r) at each t, a diagnostic


def _student_t_logpdf(x: float, mu: np.ndarray, kappa: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    df = 2.0 * alpha
    scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
    return student_t.logpdf(x, df, loc=mu, scale=scale)


def bocpd_normal_inverse_gamma(
    x: np.ndarray,
    *,
    hazard: float = 1.0 / 250.0,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float | None = None,
    burnin_beta0: float = 1e-4,
    min_burnin_for_variance: int = 20,
    prune_threshold: float = 1e-6,
) -> BocpdResult:
    """`hazard`: constant per-step probability of a changepoint under the
    geometric-run-length prior (default 1/250 ~= "about one regime change
    per trading year", a defensible, documented default, not a tuned
    value).

    `beta0` (the NIG prior's scale hyperparameter, re-used for the "fresh
    regime" hypothesis at EVERY step) has two modes, BOTH strictly causal:
    - `beta0=<a float>`: a FIXED value used unchanged at every step,
      independent of `x` -- e.g. a domain-informed guess at the series'
      typical variance, supplied by the caller and disclosed as a
      hyperparameter choice, never derived from `x` itself.
    - `beta0=None` (default): an EXPANDING, Welford-updated running
      variance of `x[0:t]` -- STRICTLY BEFORE `x[t]` is incorporated --
      used as the prior scale at step t. Before `min_burnin_for_variance`
      observations exist, `burnin_beta0` (a small fixed constant, never
      derived from data) is used instead.
    """
    n = len(x)
    log_hazard = np.log(hazard)
    log_1m_hazard = np.log1p(-hazard)
    fixed_beta0 = beta0  # None => causal expanding estimate; float => fixed, data-independent

    # Causal running (mean, M2, count) for the expanding-variance beta0 mode
    # (Welford's algorithm). Updated AFTER use each step, so `_causal_beta0()`
    # at the START of step t reflects x[0:t] only -- never x[t] itself.
    running_mean = 0.0
    running_m2 = 0.0
    running_count = 0

    def _causal_beta0() -> float:
        if fixed_beta0 is not None:
            return fixed_beta0
        if running_count < min_burnin_for_variance:
            return burnin_beta0
        return max(running_m2 / running_count, 1e-12)

    # State: arrays indexed by current run-length hypothesis (grows each step,
    # index 0 always represents "changepoint just happened, r=0" of the CURRENT step).
    # Before any data is seen, the only sensible beta is the burn-in/fixed constant --
    # there is no "expanding variance of zero observations".
    initial_beta0 = fixed_beta0 if fixed_beta0 is not None else burnin_beta0
    log_r = np.array([0.0])  # log P(r_0 = 0) = log 1 (before seeing any data)
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([initial_beta0])

    changepoint_prob = np.zeros(n)
    map_run_length = np.zeros(n, dtype=int)

    for t in range(n):
        xt = x[t]
        current_beta0 = _causal_beta0()  # uses ONLY x[0:t] -- strictly before xt

        # Growth: r_t = r_{t-1} + 1 (no changepoint) -- x_t's likelihood under
        # EACH hypothesis's OWN fitted (mu, kappa, alpha, beta).
        log_pred_hyp = _student_t_logpdf(xt, mu, kappa, alpha, beta)
        log_growth = log_r + log_pred_hyp + log_1m_hazard

        # Changepoint: r_t = 0 -- x_t's likelihood under the FRESH PRIOR
        # (mu0, kappa0, alpha0, current_beta0), NOT any hypothesis's own
        # fitted params. This asymmetry (prior vs. fitted-regime predictive)
        # is what actually lets BOCPD detect a shift: data surprising to
        # every long-run-length hypothesis but unsurprising to a fresh/wide
        # prior pushes P(r_t=0) up. (A prior version of this function reused
        # `log_pred_hyp` for the changepoint branch too, which makes
        # P(r_t=0|x_1:t) collapse to EXACTLY `hazard` for every t regardless
        # of data -- see MODEL2B_2C_2D_COVARIANCE_CHANGEPOINT_REGIME.md.)
        log_pred_prior = float(_student_t_logpdf(xt, np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([current_beta0]))[0])
        log_cp = log_hazard + log_pred_prior + _logsumexp(log_r)

        new_log_r = np.concatenate(([log_cp], log_growth))
        new_log_r -= _logsumexp(new_log_r)  # normalize

        # NIG posterior update for EVERY hypothesis incorporating x_t: the new
        # r=0 hypothesis is the CAUSAL prior (current_beta0, known as of this
        # step) updated with x_t; each surviving r->r+1 hypothesis is ITS OWN
        # prior (mu,kappa,alpha,beta) updated with x_t.
        new_kappa = np.concatenate(([kappa0], kappa)) + 1.0
        prior_mu = np.concatenate(([mu0], mu))
        prior_kappa = np.concatenate(([kappa0], kappa))
        new_mu = (prior_kappa * prior_mu + xt) / new_kappa
        new_alpha = np.concatenate(([alpha0], alpha)) + 0.5
        prior_beta = np.concatenate(([current_beta0], beta))
        new_beta = prior_beta + prior_kappa * (xt - prior_mu) ** 2 / (2.0 * new_kappa)

        changepoint_prob[t] = float(np.exp(new_log_r[0]))
        map_run_length[t] = int(np.argmax(new_log_r))

        # Prune negligible-mass hypotheses (bounds per-step cost; does not
        # affect causality -- a pruned hypothesis' mass was already <=
        # prune_threshold, redistributed implicitly via renormalization next step).
        keep = new_log_r > np.log(prune_threshold)
        if not np.any(keep):
            keep[np.argmax(new_log_r)] = True
        log_r = new_log_r[keep] - _logsumexp(new_log_r[keep])
        mu, kappa, alpha, beta = new_mu[keep], new_kappa[keep], new_alpha[keep], new_beta[keep]

        # Update the causal running (mean, M2, count) stats AFTER using them
        # for this step (Welford's online algorithm) -- so step t+1's
        # `_causal_beta0()` reflects x[0:t+1], never x[t+1] itself.
        running_count += 1
        delta = xt - running_mean
        running_mean += delta / running_count
        delta2 = xt - running_mean
        running_m2 += delta * delta2

    return BocpdResult(changepoint_prob=changepoint_prob, map_run_length=map_run_length)


def _logsumexp(log_values: np.ndarray) -> float:
    m = np.max(log_values)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(log_values - m))))
