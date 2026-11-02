"""Capture step: freeze the information set at the frozen dataset's
current last date and produce candidate + benchmark forecasts for the
NEXT (unknown) trading day. Writes an immutable `PendingObservation`.

Idempotent by construction: capturing twice for the same (ticker,
information_cutoff_date) produces an IDENTICAL record (same hash), since
every input is deterministic given the frozen dataset -- callers should
still check `already_captured` before writing to avoid a duplicate file,
but nothing breaks if they don't."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from model_lab.prospective.records import (
    ForecastEntry,
    PendingObservation,
    compute_pending_observation_hash,
)
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.shadow.build_combo import build_combo_shadow_records
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.data import ReturnSeries
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

CANDIDATE_TASK_ID = "T-VOL"
HORIZON = 1


def capture_pending_observation(series: ReturnSeries) -> PendingObservation:
    """Builds the candidate's shadow `ModelOutput` (via the SAME
    `build_combo_shadow_records` the shadow-snapshot pipeline uses -- no
    second implementation of the candidate) plus each benchmark's own
    forecast at the identical origin, and bundles them into one immutable
    `PendingObservation` whose `information_cutoff_date` is exactly
    `series.dates[-1]` (the frozen dataset's own last known date)."""
    dates = series.dates[1:]
    train_ends, _ = expanding_fold_train_ends(dates, horizon=HORIZON, min_train_groups=2)
    _, candidate_output = build_combo_shadow_records(series, fold_train_ends=train_ends)

    last_idx = len(series.log_returns) - 1
    naive_fc = naive_trailing_rv_forecast(series.log_returns, HORIZON)
    ewma_fc = ewma_forecast(series.log_returns, HORIZON, lam=0.94)
    garch_fc = gjr_garch_t_forecast(series.log_returns, HORIZON, train_ends).variance_forecast
    egarch_fc = egarch_t_forecast(series.log_returns, HORIZON, train_ends).variance_forecast

    benchmark_forecasts = [
        ForecastEntry(model_id="naive_trailing_rv", point_estimate=str(float(naive_fc[last_idx]))),
        ForecastEntry(model_id="EWMA_0.94", point_estimate=str(float(ewma_fc[last_idx]))),
        ForecastEntry(model_id="GARCH_t", point_estimate=str(float(garch_fc[last_idx]))),
        ForecastEntry(model_id="EGARCH_t", point_estimate=str(float(egarch_fc[last_idx]))),
    ]
    information_cutoff_date = series.dates[-1]
    information_cutoff_close = str(float(series.closes[-1]))
    captured_at = datetime.now(UTC)
    pending_hash = compute_pending_observation_hash(
        ticker=series.ticker,
        task_id=CANDIDATE_TASK_ID,
        information_cutoff_date=information_cutoff_date,
        information_cutoff_close=information_cutoff_close,
        candidate_output=candidate_output,
        benchmark_forecasts=benchmark_forecasts,
        data_fingerprint=series.data_fingerprint,
    )
    return PendingObservation(
        ticker=series.ticker,
        task_id=CANDIDATE_TASK_ID,
        information_cutoff_date=information_cutoff_date,
        information_cutoff_close=information_cutoff_close,
        observation_captured_at_utc=captured_at,
        candidate_output=candidate_output,
        benchmark_forecasts=benchmark_forecasts,
        data_fingerprint=series.data_fingerprint,
        status="PENDING",
        pending_observation_hash=pending_hash,
    )


def pending_observation_path(store_dir: Path, ticker: str, information_cutoff_date: str) -> Path:
    return store_dir / f"pending_{ticker}_{information_cutoff_date}.json"
