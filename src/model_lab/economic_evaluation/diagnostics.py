"""Model-native diagnostic charts -- generated from REAL model outputs
and real data, never decorative. Two new diagnostic TYPES this pass,
complementing (not replacing) the 15 model-vs-benchmark/event/coverage
charts already built in `command_center_export/registry_builder.py`:

1. **Predicted vs realized** (`build_predicted_vs_realized_chart`): real
   per-day h=1 forecasts (re-derived by calling the SAME already-tested
   forecast functions the vol ladder uses -- no new model fitting) vs
   the real realized variance target, downsampled to monthly means for
   chart readability (a disclosed summarization, not a fabrication).

2. **Calibration (Mincer-Zarnowitz slope)** (`build_calibration_chart`):
   reuses the `mz_slope` ALREADY committed in `reports/vol_ladder/*.json`
   -- no new computation. A slope of 1.0 is perfect calibration; this
   chart makes that comparison visible per model per ticker, which no
   existing chart in this project does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.chart_spec import (
    Causality,
    ChartSeries,
    ChartSeriesPoint,
    ChartSpec,
    DataAuthority,
)
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast
from model_lab.vol.targets import forward_realised_variance

VOL_MODEL_ROLES = {
    "naive_trailing_rv": "naive", "EWMA_0.94": "comparator", "GARCH_t": "incumbent",
    "EGARCH_t": "comparator", "Combo_median_EWMA_GARCH_EGARCH": "challenger",
}


def _chart(chart_id: str, question: str, incumbent_id: str, series: list[ChartSeries], n: int | None) -> ChartSpec:
    payload = {
        "chart_id": chart_id, "decision_question": question, "incumbent_id": incumbent_id,
        "series": [s.model_dump(mode="json") for s in series], "n": n,
        "causality": "NOT_APPLICABLE", "data_authority": "PROVISIONAL",
    }
    chart_sha = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return ChartSpec(
        chart_id=chart_id, decision_question=question, incumbent_id=incumbent_id, series=series, n=n,
        causality=Causality.NOT_APPLICABLE, data_authority=DataAuthority.PROVISIONAL, chart_sha256=chart_sha,
    )


def _monthly_mean(dates: list[str], values: np.ndarray, idx: np.ndarray) -> dict[str, float]:
    by_month: dict[str, list[float]] = {}
    for i in idx:
        month = dates[i][:7]  # 'YYYY-MM'
        by_month.setdefault(month, []).append(float(values[i]))
    return {m: float(np.mean(v)) for m, v in sorted(by_month.items())}


def build_predicted_vs_realized_chart(registry, ticker: str) -> ChartSpec:
    """Real per-day h=1 forecasts vs the real realized target, re-derived
    from the vol ladder's own already-tested functions (no new fitting),
    downsampled to monthly means."""
    from model_lab.vol.combination import median_variance_combination

    series_obj = load_return_series(ticker, registry)
    r = series_obj.log_returns
    dates = series_obj.dates[1:]
    train_ends, test_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
    oos_idx = sorted({i for lst in test_lists for i in lst if i < len(r) - 1})

    naive_fc = naive_trailing_rv_forecast(r, 1)
    ewma_fc = ewma_forecast(r, 1, lam=0.94)
    garch_fc = gjr_garch_t_forecast(r, 1, train_ends).variance_forecast
    egarch_fc = egarch_t_forecast(r, 1, train_ends).variance_forecast
    valid = [i for i in oos_idx if not any(np.isnan(x[i]) for x in (naive_fc, ewma_fc, garch_fc, egarch_fc))]
    idx = np.array(valid)
    combo_fc = np.full(len(r), np.nan)
    combo_fc[idx] = median_variance_combination(ewma_fc[idx], garch_fc[idx], egarch_fc[idx])
    target = forward_realised_variance(r, 1)

    forecasts = {"naive_trailing_rv": naive_fc, "EWMA_0.94": ewma_fc, "GARCH_t": garch_fc, "EGARCH_t": egarch_fc, "Combo_median_EWMA_GARCH_EGARCH": combo_fc}
    series: list[ChartSeries] = []
    for model_id, role in VOL_MODEL_ROLES.items():
        monthly = _monthly_mean(dates, forecasts[model_id], idx)
        series.append(ChartSeries(label=f"{model_id}_forecast", role=role, points=[ChartSeriesPoint(x_label=m, value=str(v)) for m, v in monthly.items()]))
    realized_monthly = _monthly_mean(dates, target, idx)
    series.append(ChartSeries(label="realized_variance", role="comparator", points=[ChartSeriesPoint(x_label=m, value=str(v)) for m, v in realized_monthly.items()]))

    return _chart(
        f"predicted_vs_realized_{ticker}",
        f"How closely do each T-VOL model's monthly-mean h=1 forecasts track {ticker}'s real realized variance?",
        "GARCH_t", series, len(valid),
    )


def build_calibration_chart(reports_dir: Path, tickers: tuple[str, ...]) -> ChartSpec:
    """Mincer-Zarnowitz slope (ideal=1.0) per model per ticker -- real,
    already-committed values, reused directly, no recomputation."""
    series_by_model: dict[str, list[ChartSeriesPoint]] = {m: [] for m in VOL_MODEL_ROLES}
    n_total = 0
    for ticker in tickers:
        d = json.loads((reports_dir / "vol_ladder" / f"{ticker}_h1.json").read_text())
        for model_id in VOL_MODEL_ROLES:
            series_by_model[model_id].append(ChartSeriesPoint(x_label=ticker, value=str(d["results"][model_id]["mz_slope"])))
        n_total += d["results"]["GARCH_t"]["n_oos"]
    series = [ChartSeries(label=m, role=r, points=series_by_model[m]) for m, r in VOL_MODEL_ROLES.items()]
    return _chart(
        "mz_calibration_slope_by_ticker", "Is each T-VOL model's h=1 forecast well-calibrated (Mincer-Zarnowitz slope near 1.0)?",
        "GARCH_t", series, n_total,
    )
