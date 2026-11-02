"""Two classical baselines named directly in T-VOL's own frozen task spec
(`tasks/registry.py`): `naive="trailing_realised_variance"` and
`incumbents=["EWMA_0.94", ...]`. Both use ONLY information strictly at or
before index `i` to forecast `targets.forward_realised_variance`'s
`target[i]` -- no look-ahead, verified by `tests/unit/vol/test_leakage.py`'s
prefix-invariance check (`eval/leakage.py`).

Multi-day-ahead convention: both baselines scale a ONE-DAY variance
estimate by `horizon` (flat/constant-variance-over-the-horizon assumption --
the simplest defensible choice for a NAIVE and an EWMA baseline; the
GARCH-family models in `garch_models.py` instead use the model's own
mean-reverting multi-step forecast, a genuinely different and usually
better-informed aggregation)."""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def naive_trailing_rv_forecast(log_returns: np.ndarray, horizon: int, *, window: int = 252) -> np.ndarray:
    """forecast[i] = horizon * mean(log_returns[i-window+1 : i+1] ** 2), for
    i >= window - 1; NaN (insufficient trailing history) otherwise."""
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    sq = r**2
    out = np.full(n, NAN, dtype=float)
    if n < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    for i in range(window - 1, n):
        trailing_mean = (csum[i + 1] - csum[i + 1 - window]) / window
        out[i] = horizon * trailing_mean
    return out


def ewma_variance(log_returns: np.ndarray, *, lam: float = 0.94, seed_window: int = 30) -> np.ndarray:
    """RiskMetrics-style recursive EWMA variance: var[t] = lam*var[t-1] +
    (1-lam)*r[t]**2. Seeded with the simple sample variance of the first
    `seed_window` observations (NaN before the series has that much
    history)."""
    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0, 1)")
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    out = np.full(n, NAN, dtype=float)
    if n < seed_window:
        return out
    var = float(np.mean(r[:seed_window] ** 2))
    out[seed_window - 1] = var
    for t in range(seed_window, n):
        var = lam * var + (1.0 - lam) * r[t] ** 2
        out[t] = var
    return out


def ewma_forecast(log_returns: np.ndarray, horizon: int, *, lam: float = 0.94, seed_window: int = 30) -> np.ndarray:
    """forecast[i] = horizon * ewma_variance[i] (flat scaling, see module
    docstring)."""
    var = ewma_variance(log_returns, lam=lam, seed_window=seed_window)
    return horizon * var
