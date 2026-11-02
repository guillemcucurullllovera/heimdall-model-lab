"""T-TAIL's `incumbents=["GJR_t_filtered_GPD_FHS"]`: GJR-GARCH-t conditional
volatility (reuses `vol.garch_models.gjr_garch_t_forecast` at horizon=1 --
same fits, same fold boundaries, no duplicated GARCH estimation) combined
with a GPD tail fitted to the TRAINING window's standardized residuals
(Filtered Historical Simulation: the tail shape is estimated once per fold
from `z_t = r_t / sigma_t`, then scaled day-by-day by the day's own
conditional volatility forecast -- `VaR_{t+1} = sigma_{t+1} * VaR_z`,
`ES_{t+1} = sigma_{t+1} * ES_z`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arch import arch_model

from model_lab.tail.gpd import GpdFit, fit_gpd_tail, gpd_var_es
from model_lab.vol.garch_models import RETURN_SCALE, gjr_garch_t_forecast

NAN = float("nan")


@dataclass(frozen=True)
class ConditionalTailResult:
    model_id: str
    config: dict
    var_forecast: dict[float, np.ndarray]
    es_forecast: dict[float, np.ndarray]
    fold_fits: list[dict]


def _fold_standardized_residuals(log_returns: np.ndarray, fit_end_idx: int) -> np.ndarray:
    r_pct = log_returns[:fit_end_idx] * RETURN_SCALE
    am = arch_model(r_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t")
    res = am.fit(disp="off", show_warning=False)
    return np.asarray(res.std_resid, dtype=float)


def gjr_t_filtered_gpd_fhs_forecast(
    log_returns: np.ndarray,
    fold_train_ends: list[int],
    *,
    tail_levels: tuple[float, ...] = (0.05, 0.025, 0.01),
    tail_quantile: float = 0.10,
    purge_days: int = 1,
) -> ConditionalTailResult:
    """`purge_days` (default 1, matching T-TAIL's own `horizon_days=[1]`):
    both the GARCH conditional-variance fit (via `gjr_garch_t_forecast`,
    which purges internally -- see its own remediation note) and the
    auxiliary GARCH fit used ONLY to extract standardized residuals for the
    GPD tail shape are estimated on the purged window
    `log_returns[:train_end_idx - purge_days]`, not the full unpurged
    prefix (independent adversarial review, finding P2-5)."""
    n = len(log_returns)
    sigma2_one_step = gjr_garch_t_forecast(log_returns, 1, fold_train_ends).variance_forecast  # h=1 cumulative == one-step variance
    sigma_one_step = np.sqrt(np.clip(sigma2_one_step, 0.0, None))

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
    return ConditionalTailResult(
        model_id="GJR_t_filtered_GPD_FHS",
        config={"garch_vol": "GARCH", "o": 1, "dist": "t", "tail_quantile": tail_quantile, "tail_levels": list(tail_levels)},
        var_forecast=var_out,
        es_forecast=es_out,
        fold_fits=fold_fits,
    )
