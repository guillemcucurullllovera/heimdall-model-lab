"""Applies CUSUM (naive) and BOCPD (incumbent) to a real return series and
runs the SAME illustrative (non-validating) stress-date cross-check
`anomaly.evaluate` uses -- see that module's docstring for the explicit
caveat, which applies identically here: T-CHANGE's real blocker
(`market_state_features`) is unresolved, and this cross-check is NOT a
substitute for a pre-registered changepoint benchmark.

Both detectors are purely online/causal by construction (each day's
statistic depends only on data at or before that day) -- no walk-forward
fold machinery is needed here, unlike `vol`/`tail`/`regime`, which refit
periodically."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from model_lab.anomaly.evaluate import ILLUSTRATIVE_STRESS_DATES
from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma
from model_lab.changepoint.cusum import cusum_detect
from model_lab.vol.data import ReturnSeries


def _fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ChangepointRunResult:
    ticker: str
    dataset_id: str
    data_fingerprint: str
    config_fingerprints: dict
    n_obs: int
    n_cusum_alarms: int
    n_bocpd_dates_covered: int
    bocpd_top_decile_hits: int
    bocpd_top_5pct_hits: int
    cusum_alarm_within_5d_of_illustrative: int
    bocpd_score_percentile_of_illustrative_dates: list[float]


def run_changepoint_detectors(series: ReturnSeries) -> ChangepointRunResult:
    r = series.log_returns
    dates = series.dates[1:]

    cusum_res = cusum_detect(r, window=60, drift=0.5, threshold=5.0)
    bocpd_res = bocpd_normal_inverse_gamma(r, hazard=1.0 / 250.0)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    covered = [d for d in ILLUSTRATIVE_STRESS_DATES if d in date_to_idx]
    bocpd_pcts: list[float] = []
    top_decile = top_5pct = 0
    for d in covered:
        i = date_to_idx[d]
        pct = float((bocpd_res.changepoint_prob <= bocpd_res.changepoint_prob[i]).mean())
        bocpd_pcts.append(pct)
        top_decile += pct >= 0.90
        top_5pct += pct >= 0.95

    cusum_hits = 0
    for d in covered:
        i = date_to_idx[d]
        window = cusum_res.alarm[max(0, i - 5) : i + 6]
        cusum_hits += int(np.any(window))

    return ChangepointRunResult(
        ticker=series.ticker,
        dataset_id=series.dataset_id,
        data_fingerprint=series.data_fingerprint,
        config_fingerprints={
            "cusum": _fingerprint({"window": 60, "drift": 0.5, "threshold": 5.0}),
            "bocpd": _fingerprint({"hazard": 1.0 / 250.0, "model": "NIG"}),
        },
        n_obs=len(r),
        n_cusum_alarms=int(cusum_res.alarm.sum()),
        n_bocpd_dates_covered=len(covered),
        bocpd_top_decile_hits=top_decile,
        bocpd_top_5pct_hits=top_5pct,
        cusum_alarm_within_5d_of_illustrative=cusum_hits,
        bocpd_score_percentile_of_illustrative_dates=bocpd_pcts,
    )
