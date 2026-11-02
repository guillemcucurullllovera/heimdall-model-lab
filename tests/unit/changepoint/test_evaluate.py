from __future__ import annotations

from model_lab.changepoint.evaluate import run_changepoint_detectors
from model_lab.vol.data import ReturnSeries


def _fake_series(n: int = 400) -> ReturnSeries:
    from datetime import date, timedelta

    import numpy as np

    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, n)
    r[200] = -0.08  # a deliberate shock
    dates = []
    d = date(2018, 2, 1)
    while len(dates) < n + 1:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return ReturnSeries(
        ticker="FAKE", dataset_id="fake_raw", dates=dates, closes=closes, highs=closes, lows=closes,
        opens=closes, log_returns=r, data_fingerprint="f" * 64,
    )


def test_run_changepoint_detectors_produces_sane_structure() -> None:
    series = _fake_series()
    result = run_changepoint_detectors(series)
    assert result.ticker == "FAKE"
    assert result.n_obs == len(series.log_returns)
    assert result.n_cusum_alarms >= 0
    assert 0 <= result.bocpd_top_decile_hits <= result.n_bocpd_dates_covered
    assert all(0.0 <= p <= 1.0 for p in result.bocpd_score_percentile_of_illustrative_dates)
