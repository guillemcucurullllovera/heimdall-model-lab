"""Page's (1954) two-sided CUSUM on rolling-standardized returns --
T-CHANGE's frozen `naive="CUSUM"`. Detects a persistent shift in the MEAN
of the series relative to a TRAILING window's own mean/std (a classical,
purely online/causal detector: each day's statistic depends only on data
strictly before that day's own window, never on it or anything later)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NAN = float("nan")


@dataclass(frozen=True)
class CusumResult:
    g_plus: np.ndarray  # upward CUSUM statistic
    g_minus: np.ndarray  # downward CUSUM statistic
    alarm: np.ndarray  # bool, True where either statistic exceeds `threshold`


def cusum_detect(
    x: np.ndarray, *, window: int = 60, drift: float = 0.5, threshold: float = 5.0
) -> CusumResult:
    """`x[t]` is standardized using the mean/std of the TRAILING `window`
    observations strictly before `t` (`x[t-window:t]`) -- causal by
    construction, never including `x[t]` itself in its own baseline.
    `drift` (std-dev units) is Page's CUSUM slack (tolerable drift before a
    real shift is flagged); `threshold` (std-dev units) triggers an alarm,
    which resets both statistics to 0 so the detector can re-fire on a
    SUBSEQUENT shift rather than saturating."""
    n = len(x)
    g_plus = np.zeros(n)
    g_minus = np.zeros(n)
    alarm = np.zeros(n, dtype=bool)
    if n <= window:
        return CusumResult(g_plus, g_minus, alarm)

    gp = gm = 0.0
    for t in range(window, n):
        baseline = x[t - window : t]
        mean = float(np.mean(baseline))
        std = max(float(np.std(baseline)), 1e-12)
        z = (x[t] - mean) / std
        gp = max(0.0, gp + z - drift)
        gm = max(0.0, gm - z - drift)
        g_plus[t] = gp
        g_minus[t] = gm
        if gp > threshold or gm > threshold:
            alarm[t] = True
            gp = gm = 0.0
    return CusumResult(g_plus, g_minus, alarm)
