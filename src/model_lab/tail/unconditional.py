"""T-TAIL's `naive="unconditional_GPD_POT"`: a single, unconditional GPD tail
fit (no volatility conditioning) refit once per calendar-year fold on
training data only -- VaR/ES are FLAT across a fold's whole test window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_lab.tail.gpd import GpdFit, fit_gpd_tail, gpd_var_es

NAN = float("nan")


@dataclass(frozen=True)
class UnconditionalTailResult:
    model_id: str
    config: dict
    var_forecast: dict[float, np.ndarray]  # p -> array aligned to log_returns index, NaN where no fold covers it
    es_forecast: dict[float, np.ndarray]
    fold_fits: list[dict]  # [{"train_end_idx": int, "gpd": GpdFit-as-dict}]


def unconditional_gpd_pot_forecast(
    log_returns: np.ndarray,
    fold_train_ends: list[int],
    *,
    tail_levels: tuple[float, ...] = (0.05, 0.025, 0.01),
    tail_quantile: float = 0.10,
    purge_days: int = 1,
) -> UnconditionalTailResult:
    """`purge_days` (default 1, matching T-TAIL's own `horizon_days=[1]`):
    the GPD fit is estimated on `log_returns[:train_end_idx - purge_days]`,
    not the full unpurged prefix -- see `vol.garch_models._run_ladder_model`
    for the identical remediation and its full rationale (independent
    adversarial review, finding P2-5). The VaR/ES assignment window
    (`[train_end_idx, next_boundary)`) is unchanged -- only what data may be
    used to ESTIMATE the fit shrinks."""
    n = len(log_returns)
    var_out = {p: np.full(n, NAN, dtype=float) for p in tail_levels}
    es_out = {p: np.full(n, NAN, dtype=float) for p in tail_levels}
    fold_fits: list[dict] = []
    sorted_ends = sorted(set(fold_train_ends))
    for idx, train_end_idx in enumerate(sorted_ends):
        if train_end_idx < 60 or train_end_idx >= n:
            continue
        next_boundary = sorted_ends[idx + 1] if idx + 1 < len(sorted_ends) else n
        fit_end_idx = max(train_end_idx - purge_days, 60)
        fit: GpdFit = fit_gpd_tail(log_returns[:fit_end_idx], tail_quantile=tail_quantile)
        fold_fits.append({"train_end_idx": train_end_idx, "fit_end_idx": fit_end_idx, "gpd": {"u": fit.u, "xi": fit.xi, "beta": fit.beta, "n": fit.n, "n_exceedances": fit.n_exceedances}})
        for p in tail_levels:
            var_p, es_p = gpd_var_es(fit, p)
            var_out[p][train_end_idx:next_boundary] = var_p
            es_out[p][train_end_idx:next_boundary] = es_p
    return UnconditionalTailResult(
        model_id="unconditional_GPD_POT",
        config={"tail_quantile": tail_quantile, "tail_levels": list(tail_levels)},
        var_forecast=var_out,
        es_forecast=es_out,
        fold_fits=fold_fits,
    )
