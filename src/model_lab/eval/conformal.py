"""Conformal prediction utilities — MODEL1_foundation_contracts.md §14.

"conformal utilities (volatility-normalised split conformal with an ACI/PID
guard)."

Volatility-normalised split conformal: nonconformity scores are studentized
by a volatility estimate (e.g. |y - point_forecast| / sigma_hat) so interval
width adapts to the local volatility regime rather than being a single fixed
width; the ACI (Adaptive Conformal Inference, Gibbs & Candes 2021) guard
online-updates the target miscoverage level in response to realised
coverage, keeping long-run coverage near nominal even under distribution
shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SplitConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    q_hat: float
    n_calibration: int
    alpha: float


def volatility_normalised_split_conformal(
    point_forecast_cal: np.ndarray,
    y_cal: np.ndarray,
    sigma_cal: np.ndarray,
    point_forecast_test: np.ndarray,
    sigma_test: np.ndarray,
    *,
    alpha: float = 0.1,
) -> SplitConformalResult:
    """Split conformal prediction with volatility-normalised nonconformity
    scores: s_i = |y_i - yhat_i| / sigma_i. q_hat is the finite-sample-
    corrected (1-alpha) quantile of the calibration scores (ceil((n+1)(1-
    alpha))/n rule, Vovk et al.), and test intervals are
    yhat_test +/- q_hat*sigma_test — width scales with local volatility."""
    point_forecast_cal = np.asarray(point_forecast_cal, float)
    y_cal = np.asarray(y_cal, float)
    sigma_cal = np.asarray(sigma_cal, float)
    n = len(y_cal)
    scores = np.abs(y_cal - point_forecast_cal) / sigma_cal
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, level, method="higher"))

    point_forecast_test = np.asarray(point_forecast_test, float)
    sigma_test = np.asarray(sigma_test, float)
    lower = point_forecast_test - q_hat * sigma_test
    upper = point_forecast_test + q_hat * sigma_test
    return SplitConformalResult(lower=lower, upper=upper, q_hat=q_hat, n_calibration=n, alpha=alpha)


@dataclass
class AciGuard:
    """Adaptive Conformal Inference (Gibbs & Candes 2021) online guard: after
    each new observation, nudge the working miscoverage target alpha_t based
    on whether the previous interval covered, so long-run empirical coverage
    tracks the nominal (1-alpha) level even under distribution shift.

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where err_t = 1 if the realised point missed the interval, else 0.
    """

    target_alpha: float
    gamma: float = 0.01
    alpha_t: float = field(init=False)
    history: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.alpha_t = self.target_alpha

    def update(self, covered: bool) -> float:
        err_t = 0.0 if covered else 1.0
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.target_alpha - err_t), 1e-6, 1 - 1e-6))
        self.history.append(self.alpha_t)
        return self.alpha_t
