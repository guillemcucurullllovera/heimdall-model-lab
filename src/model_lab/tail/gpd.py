"""Peaks-Over-Threshold / Generalised Pareto Distribution tail estimation
(McNeil & Frey 2000 -- standard textbook POT-VaR/ES formula, `[M]`
independently re-derived here, not copied from any source in the corpus).

VaR_p = u + (beta/xi) * [ (n/Nu * p)^(-xi) - 1 ]
ES_p  = (VaR_p + beta - xi*u) / (1 - xi)      (requires xi < 1)

where `u` is the exceedance threshold, `(xi, beta)` the fitted GPD shape/
scale, `n` the total sample size, `Nu` the number of exceedances, and `p`
the tail probability (e.g. 0.05 for 95% VaR). Both outputs are POSITIVE
loss magnitudes (loss = -return), matching `eval/var_es_backtests.py`'s
convention throughout this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class GpdFit:
    u: float
    xi: float
    beta: float
    n: int
    n_exceedances: int


def fit_gpd_tail(values: np.ndarray, *, tail_quantile: float = 0.05) -> GpdFit:
    """Fits a GPD to the upper tail of `losses = -values` (values are
    returns; a large loss = a large negative return). `tail_quantile` sets
    the exceedance threshold `u` at the `1 - tail_quantile` empirical
    quantile of losses."""
    losses = -np.asarray(values, dtype=float)
    n = len(losses)
    u = float(np.quantile(losses, 1.0 - tail_quantile))
    exceedances = losses[losses > u] - u
    if len(exceedances) < 10:
        raise ValueError(f"only {len(exceedances)} exceedances above u={u:.6g} -- too few for a stable GPD fit")
    xi, _loc, beta = genpareto.fit(exceedances, floc=0.0)
    return GpdFit(u=u, xi=float(xi), beta=float(beta), n=n, n_exceedances=len(exceedances))


def gpd_var_es(fit: GpdFit, p: float) -> tuple[float, float]:
    """Returns (VaR_p, ES_p) as positive loss magnitudes for tail
    probability `p` (e.g. p=0.05 -> 95% VaR/ES). Requires `p` small enough
    that `(n/Nu)*p <= 1` (extrapolating INTO the tail, the POT formula's
    intended use) and `xi < 1` (finite-mean condition for ES)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if fit.xi >= 1.0:
        raise ValueError(f"xi={fit.xi} >= 1 -- ES is not finite under this fitted tail shape")
    ratio = (fit.n / fit.n_exceedances) * p
    if fit.xi == 0.0:
        var = fit.u - fit.beta * np.log(ratio)
        es = var + fit.beta
    else:
        var = fit.u + (fit.beta / fit.xi) * (ratio ** (-fit.xi) - 1.0)
        es = (var + fit.beta - fit.xi * fit.u) / (1.0 - fit.xi)
    return float(var), float(es)
