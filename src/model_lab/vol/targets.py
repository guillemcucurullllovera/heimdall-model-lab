"""Forward realised-variance target construction, matching T-VOL's own
declared target (`tasks/registry.py`): `variable="forward_realised_variance"`,
`aggregation="sum_of_squared_returns_or_range_proxy"`. Built from daily
close-to-close log returns, since no genuine intraday realised-variance
measure is available (see `reports/MODEL2A_DATA_VOL_TAIL_BASELINES.md`
sec 5) -- the task spec's own aggregation clause explicitly permits this
("sum_of_squared_returns_OR_range_proxy" from daily data), so this is not a
mislabelled daily-close proxy standing in for something it isn't.
"""

from __future__ import annotations

import numpy as np


def forward_realised_variance(log_returns: np.ndarray, horizon: int) -> np.ndarray:
    """target[i] = sum(log_returns[i+1 : i+1+horizon] ** 2), for
    i = 0 .. len(log_returns) - horizon - 1. `target[i]` is the label for a
    forecast made standing at the close AFTER `log_returns[i]` was observed
    (i.e. using information through that point only) -- aligned 1:1 with
    every baseline/model forecast function in this package."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    m = n - horizon
    if m <= 0:
        return np.array([], dtype=float)
    sq = r**2
    # cumulative sum trick: sum(sq[i+1:i+1+h]) for each i
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    out = np.empty(m, dtype=float)
    for i in range(m):
        out[i] = csum[i + 1 + horizon] - csum[i + 1]
    return out
