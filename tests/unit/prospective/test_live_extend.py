from __future__ import annotations

import pytest

from model_lab.prospective.live_extend import extend_series_with_live_bars
from model_lab.vol.data import ReturnSeries


def _base_series() -> ReturnSeries:
    import numpy as np

    dates = ["2026-09-28", "2026-09-29"]
    closes = np.array([100.0, 101.0])
    return ReturnSeries(
        ticker="FAKE", dataset_id="fake_raw", dates=dates, closes=closes, highs=closes, lows=closes,
        opens=closes, log_returns=np.log(closes[1:] / closes[:-1]), data_fingerprint="e" * 64,
    )


def test_returns_none_when_no_new_bar_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.live_extend as mod

    monkeypatch.setattr(mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [])
    assert extend_series_with_live_bars(_base_series()) is None


def test_extends_with_a_real_new_bar_and_computes_the_log_return(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.live_extend as mod
    from model_lab.prospective.live_fetch import LiveBar

    monkeypatch.setattr(
        mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [LiveBar(trading_date="2026-09-30", close="102.0")]
    )
    series = _base_series()
    extended = extend_series_with_live_bars(series)
    assert extended is not None
    assert extended.dates[-1] == "2026-09-30"
    assert extended.closes[-1] == 102.0
    import math

    assert extended.log_returns[-1] == pytest.approx(math.log(102.0 / 101.0), rel=1e-9)
    # the original series' own data must be untouched (new object, not mutated in place)
    assert series.dates == ["2026-09-28", "2026-09-29"]
    assert len(series.closes) == 2


def test_extended_series_preserves_ticker_and_dataset_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.live_extend as mod
    from model_lab.prospective.live_fetch import LiveBar

    monkeypatch.setattr(
        mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [LiveBar(trading_date="2026-09-30", close="102.0")]
    )
    series = _base_series()
    extended = extend_series_with_live_bars(series)
    assert extended is not None
    assert extended.ticker == series.ticker
    assert extended.dataset_id == series.dataset_id
    assert extended.data_fingerprint == series.data_fingerprint
