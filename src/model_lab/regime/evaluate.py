"""Does the 2-state regime read add forecasting value beyond the existing
GARCH filter? (Mission brief: "quantify what information it adds beyond
existing state.") Compares the regime-blended one-step-AHEAD variance
forecast (`p_high * var_high + (1-p_high) * var_low`, where `p_high` is now
the REAL one-step-transitioned probability -- see `regime.markov_switching`
for the fix and its rationale) against the SAME naive/EWMA/GARCH_t/
EGARCH_t baselines `vol.evaluate` already scores, using the identical
QLIKE + Diebold-Mariano machinery, at horizon=1 (T-STATE/T-DOWNSTREAM's own
declared horizon).

## Remediation (independent adversarial review, finding P1-2)

This module's forecast used to rely on the SAME-DAY filtered probability
directly, justified by a claim ("persistence >= 0.99 throughout every
fold") that was independently checked and found FALSE -- real fitted
persistence ranges from 0.00 (a degenerate GLD fold, now separately
excluded -- finding P2-7) to the high 0.9s, with many folds well below
0.99. That justification is retracted, not merely reworded: the actual
fix is in `regime.markov_switching.fit_and_filter_2state`, which now
propagates the filtered distribution forward exactly one step via the
model's own real transition matrix before using it as a forecast -- the
mathematically correct treatment regardless of how persistent any given
fold's regimes turn out to be."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from model_lab.eval.calibration import reliability_slope
from model_lab.eval.scores import QLIKE_PROXY_FLOOR, qlike
from model_lab.eval.tests_dm import diebold_mariano
from model_lab.regime.markov_switching import fit_and_filter_2state
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.data import ReturnSeries
from model_lab.vol.garch_models import gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

MIN_TRAIN_YEARS = 2


def _fingerprint(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _qlike_loss_vec(y_true: np.ndarray, sigma2_hat: np.ndarray) -> np.ndarray:
    ratio = np.clip(np.asarray(y_true, float), QLIKE_PROXY_FLOOR, None) / np.clip(sigma2_hat, QLIKE_PROXY_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


@dataclass(frozen=True)
class RegimeValueAddResult:
    ticker: str
    dataset_id: str
    data_fingerprint: str
    n_folds: int
    n_converged_folds: int
    n_well_identified_folds: int  # converged AND not flagged degenerate (finding P2-7)
    n_oos: int
    qlike_by_model: dict  # model_id -> mean QLIKE
    dm_regime_vs_model_p: dict  # model_id -> DM p-value (regime vs that model)
    dm_regime_vs_model_stat: dict
    regime_mz_slope: float
    fold_fits: list


def regime_value_add(series: ReturnSeries, *, min_train_groups: int = MIN_TRAIN_YEARS) -> RegimeValueAddResult:
    r = series.log_returns
    dates = series.dates[1:]
    n = len(r)

    train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=min_train_groups)
    if not train_ends:
        raise ValueError(f"{series.ticker}: not enough history for {min_train_groups} warm-up year(s)")
    oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - 1})

    regime_res = fit_and_filter_2state(r, train_ends)
    target = forward_realised_variance(r, 1)

    forecasts = {
        "naive_trailing_rv": naive_trailing_rv_forecast(r, 1),
        "EWMA_0.94": ewma_forecast(r, 1, lam=0.94),
        "GARCH_t": gjr_garch_t_forecast(r, 1, train_ends).variance_forecast,
        "regime_2state": regime_res.regime_variance_forecast,
    }

    valid_idx = [i for i in oos_indices if all(not np.isnan(forecasts[m][i]) for m in forecasts)]
    if not valid_idx:
        raise ValueError(f"{series.ticker}: no OOS point has forecasts from every model")
    idx_arr = np.array(valid_idx)
    y = target[idx_arr]

    qlike_by_model = {m: qlike(y, fc[idx_arr]).value for m, fc in forecasts.items()}
    regime_loss = _qlike_loss_vec(y, forecasts["regime_2state"][idx_arr])
    dm_p: dict[str, float] = {}
    dm_stat: dict[str, float] = {}
    for m, fc in forecasts.items():
        if m == "regime_2state":
            continue
        other_loss = _qlike_loss_vec(y, fc[idx_arr])
        dm = diebold_mariano(regime_loss, other_loss, h=1)
        dm_p[m] = dm.p_value
        dm_stat[m] = dm.dm_stat
    cal = reliability_slope(forecasts["regime_2state"][idx_arr], y, n_boot=500)

    n_converged = sum(1 for f in regime_res.fold_fits if f.converged)
    n_well_identified = sum(1 for f in regime_res.fold_fits if f.converged and f.well_identified)
    return RegimeValueAddResult(
        ticker=series.ticker,
        dataset_id=series.dataset_id,
        data_fingerprint=series.data_fingerprint,
        n_folds=len(train_ends),
        n_converged_folds=n_converged,
        n_well_identified_folds=n_well_identified,
        n_oos=len(valid_idx),
        qlike_by_model=qlike_by_model,
        dm_regime_vs_model_p=dm_p,
        dm_regime_vs_model_stat=dm_stat,
        regime_mz_slope=cal.slope,
        fold_fits=[asdict(f) for f in regime_res.fold_fits],
    )
