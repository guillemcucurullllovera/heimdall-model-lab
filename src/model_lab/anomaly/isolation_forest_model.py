"""Periodic-refit Isolation Forest anomaly scoring -- T-ANOM's frozen
`incumbents=["Ledoit_Wolf_Mahalanobis", "Isolation_Forest"]` second name.
Unlike the covariance/Mahalanobis family, Isolation Forest makes no
distributional assumption -- it isolates points via random recursive
partitioning, with FEWER partitions needed to isolate a point taken as
more anomalous. Refit once per calendar-year fold, same discipline as
`covariance_models.py`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

NAN = float("nan")


@dataclass(frozen=True)
class IsolationForestResult:
    model_id: str
    anomaly_score: np.ndarray  # higher = more anomalous (negated sklearn decision_function), NaN where uncovered
    fold_fits: list[dict]


def isolation_forest_score_series(
    returns: np.ndarray, fold_train_ends: list[int], *, n_estimators: int = 200, seed: int = 20260930, purge_days: int = 1
) -> IsolationForestResult:
    """`purge_days` (default 1): the forest is trained on
    `returns[:train_end_idx - purge_days]`, not the full unpurged prefix
    (independent adversarial review, finding P2-5)."""
    n, k = returns.shape
    out = np.full(n, NAN, dtype=float)
    fold_fits: list[dict] = []
    sorted_ends = sorted(set(fold_train_ends))
    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 5 * k or train_end_idx >= n:
            continue
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        fit_end_idx = max(train_end_idx - purge_days, 5 * k)
        train = returns[:fit_end_idx]
        clf = IsolationForest(n_estimators=n_estimators, random_state=seed, contamination="auto").fit(train)
        # decision_function: higher = more "normal" (sklearn convention) -- negate so
        # higher-here = more anomalous, consistent with the Mahalanobis d^2 convention.
        scores = -clf.decision_function(returns[train_end_idx:next_boundary])
        out[train_end_idx:next_boundary] = scores
        fold_fits.append({"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "n_train": int(fit_end_idx)})
    return IsolationForestResult(model_id="Isolation_Forest", anomaly_score=out, fold_fits=fold_fits)
