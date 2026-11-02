"""E-process / sequential testing utilities — Ramdas & Wang framework, as
used by Gate v2 Branch P (see `model_lab.gate_v2.branch_p`).

Definitions (bounded paired difference d_t = (L_incumbent_t - L_challenger_t)/b
in [-1,1], b fixed a priori as a property of the score function):

- Betting martingale (efficacy): E_t = prod_{s<=t} (1 + lambda_s * d_s), with
  lambda_s in [0,1] chosen PREDICTABLY (a function only of d_1..d_{s-1}) --
  this makes E_t a nonnegative supermartingale under H0: E[d_t | F_{t-1}]<=0,
  hence an e-process, and Ville's inequality gives
  P(exists t: E_t >= 1/alpha) <= alpha, so E_t may be inspected continuously
  (a "confidence sequence" / anytime-valid test).
- Betting scheme (GRAPA-like plug-in): lambda_t = clip(mu_hat_{t-1}/m2_hat_{t-1}, 0, 0.5),
  where mu_hat_{t-1}, m2_hat_{t-1} are the running mean/second-moment of
  d_1..d_{t-1}, SHRUNK toward 0 by 10 phantom zero-valued observations
  (n_eff = 10 + (t-1); phantom observations contribute 0 to both the sum and
  the sum-of-squares) -- this avoids an undefined/unstable bet before any
  real data has accrued and avoids overbetting on a tiny early sample.
- Harm process (non-inferiority guard): H0^h: E[d_t|F] >= -delta;
  E_t^h = prod (1 + lambda_s^h * (-delta - d_s)), lambda_s^h in [0, 1/(1+delta)].
- e-BH (Def 9.8): for K concurrent (final/stopped) e-values at level alpha,
  sort descending e_(1)>=...>=e_(K); reject the top k* values where
  k* = max{k in 1..K : k*e_(k)/K >= 1/alpha} (0 if no such k exists). Valid
  FDR control under ARBITRARY dependence among the e-values (Ramdas-Wang
  Thm 9.11) -- no independence/PRDS assumption needed, unlike ordinary BH.

Pinned regression vectors (from the verified gate-v2 research extraction):
- e = [80,30,25,4,1], K=5: alpha=0.05 -> none rejected; alpha=0.10 -> k*=3.
- d = [0.1,-0.05,0.2,0,0.15,-0.1,0.3,0.05] -> log E = 0.255474, E = 1.29107.

Citation re-verification (2026-09-30 remediation, reports/
MODEL1_REVIEW_REMEDIATION.md P2-6): the independent adversarial review
found these citations had only ever been checked against the FORMULAS'
numeric regression-test output, never against the actual Ramdas & Wang
primary text (in corpus, B2-02/TB-05, "Hypothesis Testing with E-values").
Independently re-checked directly against the book text during this
remediation pass -- all three are EXACT, CONFIRMED matches, both in
citation number and in content:
- Fact 7.6 (p.97): "If M is an e-process for P, then ... P(exists t in N:
  M_t >= 1/alpha) <= alpha for every P in P and alpha in [0,1]" -- exactly
  Ville's inequality as used above.
- Definition 9.8 (p.128): "k* := max{k in K: k*e_[k]/K >= 1/alpha}" --
  exactly the e-BH rejection rule implemented above (e_[k] = k-th order
  statistic, largest to smallest).
- Theorem 9.11 (p.128-129): "Any self-consistent e-testing procedure at
  level alpha ..., including the e-BH procedure, has FDR at most alpha.
  This claim holds ... regardless of the dependence structure" -- exactly
  the "arbitrary dependence" FDR-control claim cited above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PHANTOM_ZERO_OBSERVATIONS = 10
MAX_LAMBDA = 0.5


@dataclass(frozen=True)
class BettingProcessResult:
    e_values: np.ndarray  # E_1..E_t, the running process
    lambdas: np.ndarray  # lambda_1..lambda_t actually used
    final_e: float
    log_final_e: float
    n: int


def betting_martingale(d: np.ndarray, *, max_lambda: float = MAX_LAMBDA, phantom: int = PHANTOM_ZERO_OBSERVATIONS) -> BettingProcessResult:
    """The efficacy betting martingale E_t for a bounded paired-difference
    sequence `d` (each entry in [-1,1]), using the GRAPA-like plug-in
    scheme described in the module docstring."""
    d = np.asarray(d, float)
    if np.any(np.abs(d) > 1.0 + 1e-9):
        raise ValueError("d must be bounded in [-1, 1] (a property of the score bound b)")
    n = len(d)
    e_values = np.empty(n)
    lambdas = np.empty(n)
    running_sum = 0.0
    running_sumsq = 0.0
    e_prev = 1.0
    for t in range(n):
        n_eff = phantom + t  # history is d[0..t-1], i.e. t observations
        if n_eff == 0 or running_sumsq == 0.0:
            lam = 0.0
        else:
            mu_hat = running_sum / n_eff
            m2_hat = running_sumsq / n_eff
            lam = float(np.clip(mu_hat / m2_hat, 0.0, max_lambda))
        lambdas[t] = lam
        e_prev = e_prev * (1.0 + lam * d[t])
        e_values[t] = e_prev
        running_sum += d[t]
        running_sumsq += d[t] ** 2
    final_e = float(e_values[-1]) if n > 0 else 1.0
    return BettingProcessResult(e_values=e_values, lambdas=lambdas, final_e=final_e, log_final_e=math.log(final_e) if final_e > 0 else float("-inf"), n=n)


def ville_crossing_time(e_values: np.ndarray, threshold: float) -> int | None:
    """First t (1-indexed) at which E_t >= threshold (e.g. 1/alpha), or None
    if the process never crosses. Ville's inequality guarantees the
    probability of ever crossing 1/alpha under H0 is <= alpha, so this
    threshold may be monitored continuously without additional correction."""
    idx = np.argmax(np.asarray(e_values) >= threshold) if np.any(np.asarray(e_values) >= threshold) else None
    return int(idx) + 1 if idx is not None and e_values[idx] >= threshold else None


@dataclass(frozen=True)
class HarmProcessResult:
    e_values: np.ndarray
    lambdas: np.ndarray
    final_e: float
    degraded: bool  # final_e >= 5 (alpha'=0.2 eligibility threshold, per gate v2 spec)
    quarantined: bool  # final_e >= 20


def harm_process(
    d: np.ndarray, *, delta: float, alpha_prime: float = 0.2, phantom: int = PHANTOM_ZERO_OBSERVATIONS
) -> HarmProcessResult:
    """Non-inferiority harm-monitoring process, H0^h: E[d_t|F] >= -delta.
    lambda_s^h in [0, 1/(1+delta)]. Automatic DEGRADED at E^h>=5 (1/alpha'
    with alpha'=0.2 per gate v2 [P]), QUARANTINED at E^h>=20 (1/0.05)."""
    d = np.asarray(d, float)
    n = len(d)
    max_lambda_h = 1.0 / (1.0 + delta)
    e_values = np.empty(n)
    lambdas = np.empty(n)
    running_sum = 0.0
    running_sumsq = 0.0
    e_prev = 1.0
    harm_target = -delta
    for t in range(n):
        harm_d_t = harm_target - d[t]  # positive when d_t is worse than the -delta floor
        n_eff = phantom + t
        if n_eff == 0 or running_sumsq == 0.0:
            lam = 0.0
        else:
            mu_hat = running_sum / n_eff
            m2_hat = running_sumsq / n_eff
            lam = float(np.clip(mu_hat / m2_hat, 0.0, max_lambda_h))
        lambdas[t] = lam
        e_prev = max(e_prev * (1.0 + lam * harm_d_t), 0.0)
        e_values[t] = e_prev
        running_sum += harm_d_t
        running_sumsq += harm_d_t**2
    final_e = float(e_values[-1]) if n > 0 else 1.0
    return HarmProcessResult(e_values=e_values, lambdas=lambdas, final_e=final_e, degraded=final_e >= 5.0, quarantined=final_e >= 20.0)


@dataclass(frozen=True)
class EBHResult:
    rejected_indices: list[int]  # indices into the ORIGINAL (unsorted) e array
    k_star: int
    alpha: float


def e_bh(e_values: np.ndarray, *, alpha: float) -> EBHResult:
    """e-BH (Ramdas-Wang Def 9.8): valid FDR control at level alpha under
    arbitrary dependence among the e-values."""
    e = np.asarray(e_values, float)
    K = len(e)
    order = np.argsort(-e)  # descending
    e_sorted = e[order]
    k_star = 0
    for k in range(1, K + 1):
        if k * e_sorted[k - 1] / K >= 1.0 / alpha:
            k_star = k
    rejected_sorted_positions = list(range(k_star))
    rejected_indices = sorted(int(order[p]) for p in rejected_sorted_positions)
    return EBHResult(rejected_indices=rejected_indices, k_star=k_star, alpha=alpha)
