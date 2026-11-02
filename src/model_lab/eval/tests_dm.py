"""Forecast-comparison tests — MODEL1_foundation_contracts.md §14.

"Diebold-Mariano with Harvey-Leybourne-Newbold correction, Clark-West
(nested), Giacomini-White."

Classic DM over-rejects under overlapping/serially-correlated loss
differentials (the readiness pack's own R4 simulations found 8-14%
rejection at nominal 5%); the HLN small-sample correction and a Bartlett/HAC
long-run-variance estimator with lag h-1+ceil(n^(1/3)) are used throughout.

Citation status (independently verified against primary sources present in
the corpus, 2026-09-30 remediation -- see reports/MODEL1_REVIEW_REMEDIATION.md
P2-5; a prior pass had tagged all four of these `[M]`/from-memory despite
the papers being available and simply never opened):

- **Diebold & Mariano (1995), "Comparing Predictive Accuracy," JBES 13(3)**
  (in corpus, B2-07a): `[V]` for the test-statistic FORM -- S1 = d-bar /
  sqrt(2*pi*f_d_hat(0)/T) (paper's Eq., Sec 1.1) matches `diebold_mariano`'s
  dm_stat = d_bar/se exactly. `[D]` (deliberate, not literal) for the
  variance estimator: the paper's own f_d_hat(0) uses a UNIFORM/rectangular
  lag window with EXACT truncation S(T)=h-1 (justified by the paper's
  (h-1)-dependence argument for optimal h-step forecasts, Sec 1.1), not a
  Bartlett/triangular-declining-weight kernel and not lag=h-1+ceil(n^(1/3)).
  This implementation deliberately uses the latter (a standard, widely-used
  robustification against the real possibility the paper itself names --
  "(k-1)-dependence may be violated for a variety of reasons" -- not a
  literal transcription of the paper's exact S1 variance estimator). Do not
  read `[V]` here as "bitwise matches the paper's formula in every detail."
- **Harvey, Leybourne & Newbold (1997)**, the small-sample HLN correction:
  `[M]` UNVERIFIED -- genuinely absent from this research corpus (searched:
  no file, no corpus-manifest mention, and Diebold's own 2015 retrospective
  "Comparing Predictive Accuracy, Twenty Years Later" -- B2-07bs, in corpus
  -- does not cite Harvey/Leybourne/Newbold anywhere in its text either).
  The `hln_factor` formula below is implemented from the standard,
  widely-cited textbook form; it has NOT been checked against the primary
  paper because that paper could not be located.
- **Clark & West, "Approximately Normal Tests for Equal Predictive Accuracy
  in Nested Models," J. Econometrics 138 (2007)** (in corpus, B2-07d):
  `[V-CW07 Eq.(2.1)]` -- VERIFIED, exact match. The paper's Eq.(2.1),
  f_hat_{t+tau} = (y-yhat1)^2 - [(y-yhat2)^2 - (yhat1-yhat2)^2], is
  byte-for-byte what `clark_west`'s `f_t = e_r2 - (e_u2 - adj)` computes.
  The paper's one-sided test (reject for large positive t-stat, critical
  values +1.282/+1.645 for .10/.05) matches `p_value = 1 - norm.cdf(stat)`.
- **Giacomini & White, "Tests of Conditional Predictive Ability," Econometrica
  74(6) (2006)** (in corpus, B2-07c): `[V-GW06 Eq.(4)/Thm.1]` for the general
  Wald-statistic FORM -- T^h_{n,m} = n*Zbar'*Omega_hat^-1*Zbar (paper's
  Eq.(4)) matches `stat = n * h_bar @ S_inv @ h_bar` exactly, and both reject
  via a chi-square(q) upper tail. `[D]` (deliberate, not literal) for the
  variance estimator: the paper's own Theorem 1 proof notes that at the
  one-step (tau=1) horizon, the martingale-difference-sequence property
  under H0 means "the asymptotic variance can be consistently estimated by
  the sample variance" -- i.e. the UNCENTERED second moment
  Omega_hat = n^-1 * sum(Z_t Z_t'), no demeaning, NO HAC/lag-window
  correction. This implementation instead demeans (`hc = h - h_bar`) and
  always applies a Bartlett-HAC correction with lag=ceil(n^(1/3)) -- a
  deliberate generalization for robustness at horizons/settings where the
  pure one-step mds assumption may not hold, not a literal implementation
  of the paper's stated one-step estimator. `arch`'s size/power checks in
  `tests/unit/eval/test_tests_dm.py` are consistent with the generalized
  form still controlling size correctly; the paper's own Corollary 2/3
  regression-based alternative formulations were not additionally checked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def _bartlett_hac_lrv(d: np.ndarray, lag: int) -> float:
    """Bartlett-kernel HAC long-run variance of the mean of `d`."""
    n = len(d)
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    lrv = gamma0
    for k in range(1, lag + 1):
        if k >= n:
            break
        w = 1.0 - k / (lag + 1)
        gamma_k = float(np.dot(d[k:], d[:-k]) / n)
        lrv += 2 * w * gamma_k
    return lrv


def dm_hac_lag(h: int, n: int) -> int:
    """Bartlett HAC lag h-1+ceil(n^(1/3)) (§14, §20 cross-cutting rule)."""
    return int(h - 1 + math.ceil(n ** (1.0 / 3.0)))


@dataclass(frozen=True)
class DMResult:
    dm_stat: float
    p_value: float
    hln_dm_stat: float
    hln_p_value: float
    n: int
    lag: int
    mean_loss_diff: float


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, h: int, lag: int | None = None) -> DMResult:
    """DM test on d_t = loss_a_t - loss_b_t (H0: E[d_t]=0, i.e. equal
    predictive accuracy). Negative stat / d-bar favours model B. Returns both
    the classic DM statistic and the Harvey-Leybourne-Newbold small-sample
    corrected version: DM_HLN = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n).

    `[V-DM95 Eq.S1 form, D for the variance estimator]` / `[M HLN correction]`
    -- see module docstring for the full, independently-verified citation
    disposition.
    """
    loss_a, loss_b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    d = loss_a - loss_b
    n = len(d)
    if lag is None:
        lag = dm_hac_lag(h, n)
    lrv = _bartlett_hac_lrv(d, lag)
    d_bar = float(d.mean())
    se = math.sqrt(max(lrv, 1e-300) / n)
    dm_stat = d_bar / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    hln_factor = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    hln_stat = dm_stat * hln_factor
    hln_p = 2 * (1 - stats.t.cdf(abs(hln_stat), df=n - 1))

    return DMResult(
        dm_stat=dm_stat, p_value=p_value, hln_dm_stat=hln_stat, hln_p_value=hln_p, n=n, lag=lag, mean_loss_diff=d_bar
    )


def clark_west(
    y: np.ndarray, forecast_restricted: np.ndarray, forecast_unrestricted: np.ndarray, *, h: int, lag: int | None = None
) -> DMResult:
    """Clark-West (2007) test for comparing NESTED models (the unrestricted
    model nests the restricted one), which corrects DM's tendency to
    over-reject H0 (equal accuracy) in favour of the larger model due to
    parameter-estimation noise. Adjusted loss differential:
        f_t = (y_t - fcst_restricted_t)^2 - [(y_t - fcst_unrestricted_t)^2 - (fcst_restricted_t - fcst_unrestricted_t)^2]
    then a DM-style test on f_t.

    `[V-CW07 Eq.(2.1)]` -- exact match, independently verified against the
    primary source (see module docstring).
    """
    y = np.asarray(y, float)
    fr, fu = np.asarray(forecast_restricted, float), np.asarray(forecast_unrestricted, float)
    e_r2 = (y - fr) ** 2
    e_u2 = (y - fu) ** 2
    adj = (fr - fu) ** 2
    f_t = e_r2 - (e_u2 - adj)
    n = len(f_t)
    if lag is None:
        lag = dm_hac_lag(h, n)
    lrv = _bartlett_hac_lrv(f_t, lag)
    f_bar = float(f_t.mean())
    se = math.sqrt(max(lrv, 1e-300) / n)
    stat = f_bar / se if se > 0 else 0.0
    p_value = 1 - stats.norm.cdf(stat)  # one-sided: reject H0 if unrestricted model is better
    return DMResult(dm_stat=stat, p_value=p_value, hln_dm_stat=stat, hln_p_value=p_value, n=n, lag=lag, mean_loss_diff=f_bar)


@dataclass(frozen=True)
class GiacominiWhiteResult:
    stat: float
    p_value: float
    n: int
    n_instruments: int


def giacomini_white(loss_a: np.ndarray, loss_b: np.ndarray, instruments: np.ndarray) -> GiacominiWhiteResult:
    """Giacomini-White (2006) conditional predictive ability test: regress
    the loss differential d_t on a set of F_{t-1}-measurable instruments
    (e.g. a constant and lagged d_{t-1}) and test joint significance via a
    chi-square Wald statistic — this subsumes DM (unconditional PA) as the
    special case of a constant-only instrument set, and additionally tests
    whether relative performance is *predictable* from information available
    at t-1 (useful for detecting regime-dependent forecast superiority).

    `[V-GW06 Eq.(4)/Thm.1 general form, D for the variance estimator]` --
    see module docstring for the full, independently-verified citation
    disposition (the paper's own one-step estimator is undemeaned, non-HAC;
    this implementation deliberately generalizes it).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    Z = np.asarray(instruments, float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n, q = Z.shape
    if len(d) != n:
        raise ValueError("instruments must have the same length as the loss series")
    # h_t = Z_t * d_t; test E[h_t] = 0 via a HAC Wald statistic (Newey-West
    # type covariance of h_t), which is GW's asymptotic chi-square(q) test.
    h = Z * d[:, None]
    h_bar = h.mean(axis=0)
    lag = int(math.ceil(n ** (1.0 / 3.0)))
    hc = h - h_bar
    S = (hc.T @ hc) / n
    for k in range(1, lag + 1):
        if k >= n:
            break
        w = 1.0 - k / (lag + 1)
        gamma_k = (hc[k:].T @ hc[:-k]) / n
        S += w * (gamma_k + gamma_k.T)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    stat = float(n * h_bar @ S_inv @ h_bar)
    p_value = float(1 - stats.chi2.cdf(stat, df=q))
    return GiacominiWhiteResult(stat=stat, p_value=p_value, n=n, n_instruments=q)
