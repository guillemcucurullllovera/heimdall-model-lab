"""Rolling, periodic-refit covariance estimation + Mahalanobis anomaly
scoring. Four estimators (matching T-ANOM's own named incumbents plus the
two more from the mission brief's C4 candidate list):

- `empirical` -- plain sample covariance (naive comparator).
- `ledoit_wolf` -- Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`).
- `oas` -- Oracle Approximating Shrinkage (`sklearn.covariance.OAS`).
- `min_cov_det` -- Minimum Covariance Determinant, a robust/high-breakdown
  estimator (`sklearn.covariance.MinCovDet`) -- T-ANOM's frozen
  `incumbents` name "robust covariance" generically; MinCovDet is the
  standard scikit-learn implementation of that family.

Refit once per calendar-year fold (same discipline as `vol`/`tail`): each
fold's (mean, covariance) come ONLY from strictly-prior training rows;
every test-window day's Mahalanobis distance uses that fixed fit, never
re-estimated on the day being scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import OAS, EmpiricalCovariance, LedoitWolf, MinCovDet

NAN = float("nan")

ESTIMATORS = {
    "empirical": EmpiricalCovariance,
    "ledoit_wolf": LedoitWolf,
    "oas": OAS,
    "min_cov_det": MinCovDet,
}


@dataclass(frozen=True)
class CovarianceAnomalyResult:
    model_id: str
    mahalanobis_d2: np.ndarray  # shape (n_dates,), NaN where no fold covers it
    fold_fits: list[dict]


def mahalanobis_d2_series(
    returns: np.ndarray, fold_train_ends: list[int], *, estimator: str, purge_days: int = 1
) -> CovarianceAnomalyResult:
    """`purge_days` (default 1, matching T-ANOM's own effective horizon of
    1 day in `anomaly.evaluate`): the covariance is estimated on
    `returns[:train_end_idx - purge_days]`, not the full unpurged prefix
    (independent adversarial review, finding P2-5; same fix and rationale
    as `vol.garch_models._run_ladder_model`)."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}, choose one of {sorted(ESTIMATORS)}")
    n, k = returns.shape
    out = np.full(n, NAN, dtype=float)
    fold_fits: list[dict] = []
    sorted_ends = sorted(set(fold_train_ends))
    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 5 * k or train_end_idx >= n:  # need enough rows to estimate a k x k covariance sensibly
            continue
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        fit_end_idx = max(train_end_idx - purge_days, 5 * k)
        train = returns[:fit_end_idx]
        est = ESTIMATORS[estimator]().fit(train)
        mean = np.asarray(est.location_, dtype=float)
        precision = np.asarray(est.get_precision(), dtype=float)
        fold_fits.append({"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "mean": mean.tolist()})
        for t in range(train_end_idx, next_boundary):
            diff = returns[t] - mean
            out[t] = float(diff @ precision @ diff)
    return CovarianceAnomalyResult(model_id=estimator, mahalanobis_d2=out, fold_fits=fold_fits)
