"""Calibration / reliability utilities — MODEL1_foundation_contracts.md §14.

"calibration: reliability slope and ECE with CIs, PIT uniformity, coverage."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ReliabilitySlopeResult:
    slope: float
    intercept: float
    ci_lower: float
    ci_upper: float
    n: int


def reliability_slope(
    forecast_prob: np.ndarray, outcome: np.ndarray, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> ReliabilitySlopeResult:
    """OLS slope of outcome ~ a + b*forecast_prob (a well-calibrated
    forecaster has slope=1, intercept=0); percentile bootstrap CI for the
    slope."""
    x = np.asarray(forecast_prob, float)
    y = np.asarray(outcome, float)
    n = len(x)

    def _fit(xx: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
        A = np.vstack([xx, np.ones_like(xx)]).T
        b, a = np.linalg.lstsq(A, yy, rcond=None)[0]
        return float(b), float(a)

    slope, intercept = _fit(x, y)
    rng = np.random.RandomState(seed)
    boot_slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_slopes[i], _ = _fit(x[idx], y[idx])
    lo, hi = np.percentile(boot_slopes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return ReliabilitySlopeResult(slope=slope, intercept=intercept, ci_lower=float(lo), ci_upper=float(hi), n=n)


@dataclass(frozen=True)
class EceResult:
    ece: float
    ci_lower: float
    ci_upper: float
    n_bins: int
    n: int


def expected_calibration_error(
    forecast_prob: np.ndarray, outcome: np.ndarray, *, n_bins: int = 10, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> EceResult:
    """Standard equal-width-bin ECE with a percentile bootstrap CI (resampling
    (forecast, outcome) pairs)."""
    x = np.asarray(forecast_prob, float)
    y = np.asarray(outcome, float)
    n = len(x)

    def _ece(xx: np.ndarray, yy: np.ndarray) -> float:
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(xx, bins) - 1, 0, n_bins - 1)
        total = 0.0
        for b in range(n_bins):
            mask = idx == b
            if not np.any(mask):
                continue
            conf = xx[mask].mean()
            acc = yy[mask].mean()
            total += (mask.sum() / len(xx)) * abs(acc - conf)
        return total

    point = _ece(x, y)
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bidx = rng.randint(0, n, size=n)
        boots[i] = _ece(x[bidx], y[bidx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return EceResult(ece=point, ci_lower=float(lo), ci_upper=float(hi), n_bins=n_bins, n=n)


@dataclass(frozen=True)
class PitUniformityResult:
    ks_stat: float
    p_value: float
    n: int


def pit_uniformity_test(pit_values: np.ndarray) -> PitUniformityResult:
    """Kolmogorov-Smirnov test of the probability-integral-transform values
    against Uniform(0,1) — a correctly calibrated predictive distribution
    should produce PIT values that are uniform."""
    pit = np.asarray(pit_values, float)
    stat, p = stats.kstest(pit, "uniform")
    return PitUniformityResult(ks_stat=float(stat), p_value=float(p), n=len(pit))


@dataclass(frozen=True)
class CoverageResult:
    empirical_coverage: float
    nominal_coverage: float
    n: int
    ci_lower: float
    ci_upper: float


def interval_coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, *, nominal: float) -> CoverageResult:
    """Empirical coverage of a prediction interval, with a Wilson score CI."""
    lower, upper, y = np.asarray(lower, float), np.asarray(upper, float), np.asarray(y, float)
    covered = (y >= lower) & (y <= upper)
    n = len(y)
    k = int(covered.sum())
    p_hat = k / n
    z = stats.norm.ppf(0.975)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return CoverageResult(empirical_coverage=p_hat, nominal_coverage=nominal, n=n, ci_lower=float(center - half), ci_upper=float(center + half))
