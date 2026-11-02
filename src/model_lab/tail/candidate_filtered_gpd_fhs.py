"""Phase 7 research probe (downstream economic/risk value, isolated from
AMC): does the MODEL-2G candidate's vol-forecasting edge (`vol/
candidate_spec.py`) translate into better VaR/ES coverage when used as the
conditional-volatility SCALE in T-TAIL's existing Filtered Historical
Simulation pipeline (`tail/conditional.py::gjr_t_filtered_gpd_fhs_forecast`)
-- i.e. a single, isolated ablation: same GPD tail shape (fit on GARCH_t's
own standardized residuals, unchanged), but each day's VaR/ES is scaled by
the candidate's median-combination forecast instead of GARCH_t's own.

This directly answers the mission's "does improved volatility forecasting
actually improve something economically/risk-wise?" question for the one
candidate this project has: if the candidate predicts h=1 variance better
for TLT/IEF/GLD (its own demonstrated edge, `candidate_spec.ELIGIBLE_
CELLS`), does scaling the SAME tail shape by that better variance forecast
produce measurably better VaR/ES coverage for those same tickers?

Not connected to AMC in any way; not a new tail-risk incumbent; purely a
research-only ablation to test whether a QLIKE improvement has ANY
downstream consequence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_lab.tail.conditional import _fold_standardized_residuals
from model_lab.tail.gpd import GpdFit, fit_gpd_tail, gpd_var_es
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

NAN = float("nan")


@dataclass(frozen=True)
class CandidateTailResult:
    model_id: str
    config: dict
    var_forecast: dict[float, np.ndarray]
    es_forecast: dict[float, np.ndarray]
    fold_fits: list[dict]


def combo_filtered_gpd_fhs_forecast(
    log_returns: np.ndarray,
    fold_train_ends: list[int],
    *,
    tail_levels: tuple[float, ...] = (0.05, 0.025, 0.01),
    tail_quantile: float = 0.10,
    purge_days: int = 1,
) -> CandidateTailResult:
    """Identical procedure to `gjr_t_filtered_gpd_fhs_forecast`, with ONE
    substitution: `sigma_one_step` comes from the MODEL-2G candidate
    (median of EWMA_0.94/GARCH_t/EGARCH_t) instead of `GARCH_t` alone. The
    GPD tail shape is fit on the SAME GARCH_t standardized residuals as the
    incumbent, unchanged -- isolating the ablation to exactly one variable
    (which model's vol scales the quantile into VaR/ES)."""
    n = len(log_returns)
    ewma_fc = ewma_forecast(log_returns, 1, lam=0.94)
    garch_fc = gjr_garch_t_forecast(log_returns, 1, fold_train_ends).variance_forecast
    egarch_fc = egarch_t_forecast(log_returns, 1, fold_train_ends).variance_forecast
    valid = ~np.isnan(ewma_fc) & ~np.isnan(garch_fc) & ~np.isnan(egarch_fc)
    sigma2_combo = np.full(n, NAN)
    sigma2_combo[valid] = median_variance_combination(ewma_fc[valid], garch_fc[valid], egarch_fc[valid])
    sigma_one_step = np.sqrt(np.clip(sigma2_combo, 0.0, None))

    var_out = {p: np.full(n, NAN, dtype=float) for p in tail_levels}
    es_out = {p: np.full(n, NAN, dtype=float) for p in tail_levels}
    fold_fits: list[dict] = []
    sorted_ends = sorted(set(fold_train_ends))
    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 60 or train_end_idx >= n:
            continue
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        fit_end_idx = max(train_end_idx - purge_days, 60)
        std_resid = _fold_standardized_residuals(log_returns, fit_end_idx)
        fit: GpdFit = fit_gpd_tail(std_resid, tail_quantile=tail_quantile)
        fold_fits.append(
            {"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "gpd_on_std_resid": {"u": fit.u, "xi": fit.xi, "beta": fit.beta, "n": fit.n, "n_exceedances": fit.n_exceedances}}
        )
        for p in tail_levels:
            var_z, es_z = gpd_var_es(fit, p)
            for t in range(train_end_idx, min(next_boundary, n)):
                if np.isnan(sigma_one_step[t]):
                    continue
                var_out[p][t] = sigma_one_step[t] * var_z
                es_out[p][t] = sigma_one_step[t] * es_z
    return CandidateTailResult(
        model_id="Combo_median_filtered_GPD_FHS",
        config={"tail_quantile": tail_quantile, "purge_days": purge_days, "vol_source": "Combo_median_EWMA_GARCH_EGARCH"},
        var_forecast=var_out,
        es_forecast=es_out,
        fold_fits=fold_fits,
    )
