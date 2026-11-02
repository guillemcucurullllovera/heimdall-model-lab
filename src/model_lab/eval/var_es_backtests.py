"""VaR/ES backtests — MODEL1_foundation_contracts.md §14.

"VaR/ES backtests -- Kupiec-type LR_uc, Christoffersen LR_ind and LR_cc
(definitions verified in the supplied Christoffersen paper), Acerbi-Szekely-
type ES statistic."

Standard textbook forms (Christoffersen 1998; Kupiec 1995):
- hit sequence I_t = 1{loss_t > VaR_t} (a "violation")
- LR_uc = -2*ln[ (1-p)^(T-n1) * p^n1 / ((1-pi_hat)^(T-n1) * pi_hat^n1) ] ~ chi2(1)
  where p is the nominal violation rate (e.g. 0.05), pi_hat = n1/T the
  observed rate.
- LR_ind: first-order Markov independence test on the hit sequence, using
  transition counts n_ij (i,j in {0,1}) and transition probabilities
  pi_01 = n01/(n00+n01), pi_11 = n11/(n10+n11), pi = (n01+n11)/T.
  LR_ind = -2*ln[ L(pi) / L(pi_01, pi_11) ] ~ chi2(1)
- LR_cc = LR_uc + LR_ind ~ chi2(2) (Christoffersen 1998, conditional coverage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


@dataclass(frozen=True)
class LikelihoodRatioResult:
    lr_stat: float
    p_value: float
    df: int
    n: int
    n_violations: int
    observed_rate: float
    nominal_rate: float


def hit_sequence(losses: np.ndarray, var_forecast: np.ndarray) -> np.ndarray:
    """1 where a realised loss exceeds the (positive-valued) VaR forecast."""
    return (np.asarray(losses, float) > np.asarray(var_forecast, float)).astype(int)


def kupiec_lr_uc(hits: np.ndarray, p: float) -> LikelihoodRatioResult:
    """Kupiec (1995) unconditional-coverage LR test."""
    hits = np.asarray(hits, int)
    T = len(hits)
    n1 = int(hits.sum())
    pi_hat = n1 / T if T > 0 else 0.0
    log_l_null = (T - n1) * _safe_log(1 - p) + n1 * _safe_log(p)
    log_l_alt = (T - n1) * _safe_log(1 - pi_hat) + n1 * _safe_log(pi_hat)
    lr = -2.0 * (log_l_null - log_l_alt)
    lr = max(lr, 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return LikelihoodRatioResult(lr_stat=lr, p_value=p_value, df=1, n=T, n_violations=n1, observed_rate=pi_hat, nominal_rate=p)


def christoffersen_lr_ind(hits: np.ndarray) -> LikelihoodRatioResult:
    """Christoffersen (1998) independence LR test on the first-order Markov
    transition structure of the hit sequence."""
    hits = np.asarray(hits, int)
    T = len(hits)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        prev, cur = hits[t - 1], hits[t]
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1
    n0 = n00 + n01
    n1 = n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0

    log_l_restricted = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_l_unrestricted = n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01) + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    lr = -2.0 * (log_l_restricted - log_l_unrestricted)
    lr = max(lr, 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return LikelihoodRatioResult(
        lr_stat=lr, p_value=p_value, df=1, n=T, n_violations=int(hits.sum()), observed_rate=pi, nominal_rate=float("nan")
    )


def christoffersen_lr_cc(hits: np.ndarray, p: float) -> LikelihoodRatioResult:
    """LR_cc = LR_uc + LR_ind ~ chi2(2) (Christoffersen 1998 conditional
    coverage test)."""
    uc = kupiec_lr_uc(hits, p)
    ind = christoffersen_lr_ind(hits)
    lr = uc.lr_stat + ind.lr_stat
    p_value = 1 - stats.chi2.cdf(lr, df=2)
    return LikelihoodRatioResult(
        lr_stat=lr, p_value=p_value, df=2, n=uc.n, n_violations=uc.n_violations, observed_rate=uc.observed_rate, nominal_rate=p
    )


@dataclass(frozen=True)
class AcerbiSzekeleResult:
    z_stat: float
    n: int
    n_violations: int


def acerbi_szekely_es_statistic(losses: np.ndarray, var_forecast: np.ndarray, es_forecast: np.ndarray, p: float) -> AcerbiSzekeleResult:
    """Acerbi-Szekely (2014) -style ES backtest statistic Z1:
        Z1 = sum_t [ I_t * loss_t / (T*p*ES_t) ] - 1
    where I_t is the VaR-violation indicator. We use the "positive VaR/ES"
    convention consistent with kupiec_lr_uc/hit_sequence above (VaR/ES
    reported as positive loss magnitudes, not the original paper's signed
    P&L convention; the ratio-based statistic is invariant to this sign
    flip as long as it is applied consistently, which it is here).

    Under a correctly specified model, E[I_t*loss_t] = p*ES_t exactly (ES_t
    is defined as E[loss_t | loss_t > VaR_t], and P(loss_t > VaR_t) = p by
    construction of VaR_t as the p-quantile), so
    E[sum_t I_t*loss_t/(T*p*ES_t)] = sum_t (p*ES_t)/(T*p*ES_t) = 1, and
    E[Z1] = 1 - 1 = 0. Large positive values indicate ES underestimates
    (in magnitude) the realised tail losses.
    """
    losses = np.asarray(losses, float)
    var_forecast = np.asarray(var_forecast, float)
    es_forecast = np.asarray(es_forecast, float)
    if np.any(es_forecast <= 0):
        raise ValueError("es_forecast must be a positive loss magnitude in this convention")
    hits = hit_sequence(losses, var_forecast)
    T = len(losses)
    n1 = int(hits.sum())
    z1 = float(np.sum(hits * losses / (T * p * es_forecast))) - 1.0
    return AcerbiSzekeleResult(z_stat=z1, n=T, n_violations=n1)
