"""Hansen SPA + Model Confidence Set across the volatility ladder's four
models' QLIKE loss series, per (ticker, horizon). Recomputes the same
forecasts `vol.evaluate.run_vol_ladder` already scored (same functions,
same fold boundaries) -- this module adds the multiple-comparison-aware
layer on top, it does not re-derive the forecasts differently."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_lab.eval.mcs_spa import McsResult, SpaResult, hansen_spa, model_confidence_set
from model_lab.eval.scores import QLIKE_PROXY_FLOOR
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.data import ReturnSeries
from model_lab.vol.evaluate import COMBO_MODEL_ID
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

MIN_TRAIN_YEARS = 2


def _qlike_loss_vec(y_true: np.ndarray, sigma2_hat: np.ndarray) -> np.ndarray:
    ratio = np.clip(np.asarray(y_true, float), QLIKE_PROXY_FLOOR, None) / np.clip(sigma2_hat, QLIKE_PROXY_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


@dataclass(frozen=True)
class VolModelComparisonResult:
    ticker: str
    horizon: int
    n_oos: int
    model_ids: list[str]
    spa_benchmark: str
    spa_p_value_consistent: float
    spa_benchmark_not_rejected: bool
    mcs_superior_set: list[str]
    mcs_alpha: float


def compare_vol_models(
    series: ReturnSeries, horizon: int, *, min_train_groups: int = MIN_TRAIN_YEARS, spa_reps: int = 1000, mcs_reps: int = 1000
) -> VolModelComparisonResult:
    r = series.log_returns
    dates = series.dates[1:]
    n = len(r)
    target = forward_realised_variance(r, horizon)

    train_ends, test_index_lists = expanding_fold_train_ends(dates, horizon, min_train_groups=min_train_groups)
    if not train_ends:
        raise ValueError(f"{series.ticker} h={horizon}: not enough history")
    oos_indices = sorted({i for lst in test_index_lists for i in lst if i < n - horizon})

    forecasts = {
        "naive_trailing_rv": naive_trailing_rv_forecast(r, horizon),
        "EWMA_0.94": ewma_forecast(r, horizon, lam=0.94),
        "GARCH_t": gjr_garch_t_forecast(r, horizon, train_ends).variance_forecast,
        "EGARCH_t": egarch_t_forecast(r, horizon, train_ends).variance_forecast,
    }
    forecasts[COMBO_MODEL_ID] = median_variance_combination(
        forecasts["EWMA_0.94"], forecasts["GARCH_t"], forecasts["EGARCH_t"]
    )
    valid_idx = [i for i in oos_indices if all(not np.isnan(forecasts[m][i]) for m in forecasts)]
    if not valid_idx:
        raise ValueError(f"{series.ticker} h={horizon}: no common OOS points across all 4 models")
    idx_arr = np.array(valid_idx)
    y = target[idx_arr]

    model_ids = list(forecasts.keys())
    loss_matrix = np.column_stack([_qlike_loss_vec(y, forecasts[m][idx_arr]) for m in model_ids])

    bench_idx = model_ids.index("naive_trailing_rv")
    bench_losses = loss_matrix[:, bench_idx]
    challenger_losses = np.delete(loss_matrix, bench_idx, axis=1)
    spa: SpaResult = hansen_spa(bench_losses, challenger_losses, reps=spa_reps, seed=20260930)

    mcs: McsResult = model_confidence_set(loss_matrix, alpha=0.10, reps=mcs_reps, seed=20260930)

    return VolModelComparisonResult(
        ticker=series.ticker,
        horizon=horizon,
        n_oos=len(valid_idx),
        model_ids=model_ids,
        spa_benchmark="naive_trailing_rv",
        spa_p_value_consistent=spa.p_value_consistent,
        spa_benchmark_not_rejected=spa.benchmark_not_rejected,
        mcs_superior_set=[model_ids[i] for i in mcs.included_model_indices],
        mcs_alpha=0.10,
    )
