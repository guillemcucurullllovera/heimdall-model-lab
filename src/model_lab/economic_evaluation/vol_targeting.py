"""Volatility-targeting overlay backtest. See package docstring for why
this -- and only this -- is the legitimate economic use of a variance
forecast built this pass (Moreira & Muir 2017).

Mechanics (`run_vol_targeting_backtest`): at each origin `t` (an already-
causal forecast, reused unchanged from the vol ladder), set
`weight[t] = clip(target_daily_vol / sqrt(variance_forecast[t]), 0,
leverage_cap)`. That weight is applied to the REALIZED return on day
`t+1` (`weight[t] * (exp(log_returns[t+1]) - 1)`) -- identical forecast-
origin convention to every other walk-forward evaluation in this project
(a forecast stored at `t` predicts/sizes for `t+1`, never `t` itself).
Transaction cost is charged on `|weight[t] - weight[t-1]|` at a disclosed
bps rate. No new model is fit here -- every `variance_forecast` input is
an array this project's own GARCH/EWMA/regime functions already produced
and already proved causal elsewhere.
"""

from __future__ import annotations

import numpy as np

from model_lab.contracts.economic_evaluation import VolTargetingAssumptions, VolTargetingResult

TRADING_DAYS_PER_YEAR = 252


def _annualized_return(daily_simple_returns: np.ndarray) -> float:
    if len(daily_simple_returns) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + daily_simple_returns)
    total_return = equity[-1]
    years = len(daily_simple_returns) / TRADING_DAYS_PER_YEAR
    if total_return <= 0 or years <= 0:
        return float("nan")
    return float(total_return ** (1.0 / years) - 1.0)


def _max_drawdown(daily_simple_returns: np.ndarray) -> float:
    if len(daily_simple_returns) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + daily_simple_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def _sharpe(daily_simple_returns: np.ndarray) -> float:
    """Regression note: a near-constant series' `np.std` is not exactly
    0.0 in floating point (e.g. `std(full(100, 0.001))` ~= 2.17e-19, not
    0.0) -- an exact `== 0` guard is fragile; a relative epsilon against
    the series' own scale is used instead."""
    if len(daily_simple_returns) < 2:
        return float("nan")
    std = np.std(daily_simple_returns)
    scale = max(np.abs(np.mean(daily_simple_returns)), 1e-12)
    if std < 1e-9 * scale:
        return float("nan")
    return float(np.mean(daily_simple_returns) / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _metrics_for_weight_series(
    weights: np.ndarray, log_returns_next: np.ndarray, cost_bps: float
) -> dict:
    simple_returns_next = np.exp(log_returns_next) - 1.0
    gross_daily = weights * simple_returns_next
    turnover = np.abs(np.diff(weights, prepend=weights[0]))
    cost_daily = turnover * (cost_bps / 10_000.0)
    net_daily = gross_daily - cost_daily
    return {
        "gross_annualized_return": _annualized_return(gross_daily),
        "net_annualized_return": _annualized_return(net_daily),
        "realized_annualized_volatility": float(np.std(net_daily) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "sharpe_gross": _sharpe(gross_daily),
        "sharpe_net": _sharpe(net_daily),
        "max_drawdown": _max_drawdown(net_daily),
        "turnover_mean_daily": float(np.mean(turnover)),
        "exposure_mean": float(np.mean(weights)),
        "exposure_std": float(np.std(weights)),
        "worst_day_return": float(np.min(net_daily)),
        "cvar_5pct_daily": float(np.mean(net_daily[net_daily <= np.percentile(net_daily, 5)])),
    }


def run_vol_targeting_backtest(
    *,
    model_id: str,
    ticker: str,
    log_returns: np.ndarray,
    variance_forecast: np.ndarray,
    naive_variance_forecast: np.ndarray | None,
    oos_indices: list[int],
    target_annual_vol: float = 0.10,
    leverage_cap: float = 3.0,
    cost_bps_per_unit_turnover: float = 5.0,
) -> VolTargetingResult | None:
    """`oos_indices` must already exclude any index with `i+1 >=
    len(log_returns)` (no target to realize) -- callers pass the same
    index sets the vol ladder itself uses. Returns `None` if fewer than
    30 valid (non-NaN forecast) observations exist -- a controlled
    refusal, not a fabricated result from too little data."""
    valid = [i for i in oos_indices if not np.isnan(variance_forecast[i]) and i + 1 < len(log_returns)]
    if len(valid) < 30:
        return None
    idx = np.array(valid)
    target_daily_vol = target_annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    weights = np.clip(target_daily_vol / np.sqrt(np.clip(variance_forecast[idx], 1e-12, None)), 0.0, leverage_cap)
    log_returns_next = log_returns[idx + 1]

    model_metrics = _metrics_for_weight_series(weights, log_returns_next, cost_bps_per_unit_turnover)
    buy_hold_weights = np.ones_like(weights)
    buy_hold_metrics = _metrics_for_weight_series(buy_hold_weights, log_returns_next, 0.0)

    naive_metrics = None
    incremental_vs_naive = None
    if naive_variance_forecast is not None and model_id != "naive_trailing_rv":
        naive_valid = [i for i in valid if not np.isnan(naive_variance_forecast[i])]
        if len(naive_valid) == len(valid):
            naive_weights = np.clip(target_daily_vol / np.sqrt(np.clip(naive_variance_forecast[idx], 1e-12, None)), 0.0, leverage_cap)
            naive_metrics = _metrics_for_weight_series(naive_weights, log_returns_next, cost_bps_per_unit_turnover)
            incremental_vs_naive = {
                "sharpe_net_delta": model_metrics["sharpe_net"] - naive_metrics["sharpe_net"],
                "max_drawdown_delta": model_metrics["max_drawdown"] - naive_metrics["max_drawdown"],
                "net_annualized_return_delta": model_metrics["net_annualized_return"] - naive_metrics["net_annualized_return"],
            }

    incremental_vs_buy_hold = {
        "sharpe_net_delta": model_metrics["sharpe_net"] - buy_hold_metrics["sharpe_net"],
        "max_drawdown_delta": model_metrics["max_drawdown"] - buy_hold_metrics["max_drawdown"],
        "net_annualized_return_delta": model_metrics["net_annualized_return"] - buy_hold_metrics["net_annualized_return"],
        "realized_vol_vs_target": model_metrics["realized_annualized_volatility"] - target_annual_vol,
    }

    assumptions = VolTargetingAssumptions(
        target_annualized_volatility=str(target_annual_vol), leverage_cap=str(leverage_cap),
        cost_bps_per_unit_turnover=str(cost_bps_per_unit_turnover),
    )
    return VolTargetingResult(
        model_id=model_id, ticker=ticker, assumptions=assumptions, n_observations=len(valid),
        gross_annualized_return=str(model_metrics["gross_annualized_return"]),
        net_annualized_return=str(model_metrics["net_annualized_return"]),
        realized_annualized_volatility=str(model_metrics["realized_annualized_volatility"]),
        sharpe_gross=str(model_metrics["sharpe_gross"]), sharpe_net=str(model_metrics["sharpe_net"]),
        max_drawdown=str(model_metrics["max_drawdown"]), turnover_mean_daily=str(model_metrics["turnover_mean_daily"]),
        exposure_mean=str(model_metrics["exposure_mean"]), exposure_std=str(model_metrics["exposure_std"]),
        worst_day_return=str(model_metrics["worst_day_return"]), cvar_5pct_daily=str(model_metrics["cvar_5pct_daily"]),
        benchmark_buy_and_hold={k: str(v) for k, v in buy_hold_metrics.items()},
        benchmark_naive_targeted={k: str(v) for k, v in naive_metrics.items()} if naive_metrics else None,
        incremental_vs_buy_and_hold={k: str(v) for k, v in incremental_vs_buy_hold.items()},
        incremental_vs_naive_targeted={k: str(v) for k, v in incremental_vs_naive.items()} if incremental_vs_naive else None,
        walk_forward_note=(
            "Weights computed from the SAME already-causal, already-prefix-invariance-tested "
            "forecast arrays the vol ladder uses; realized return is strictly the NEXT day's "
            "already-realized log return (i -> i+1 convention, unchanged from every other "
            "walk-forward evaluation in this project). No new model fitting; purely a sizing "
            "overlay on top of already-proven-causal forecasts."
        ),
        limitations=[
            "Financing cost on leverage above 1x is not modeled (disclosed simplification).",
            "Risk-free rate assumed 0 (no risk-free series wired into this evaluation).",
            f"Transaction cost ({cost_bps_per_unit_turnover}bps/unit turnover) and leverage cap "
            f"({leverage_cap}x) are disclosed, pre-registered, round-number assumptions, NOT "
            "measured AMC-specific costs.",
        ],
    )
