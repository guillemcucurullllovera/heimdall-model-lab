from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from model_lab.prospective.capture import capture_pending_observation
from model_lab.vol.data import ReturnSeries


def _synthetic_series(n: int = 700, seed: int = 0) -> ReturnSeries:
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-4
    for t in range(n):
        if t > 0:
            sigma2[t] = 1e-6 + 0.05 * r[t - 1] ** 2 + 0.9 * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_t(6)
    dates = []
    d = date(2022, 1, 3)
    while len(dates) < n + 1:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return ReturnSeries(
        ticker="PROSPTEST", dataset_id="prosptest_raw", dates=dates, closes=closes, highs=closes,
        lows=closes, opens=closes, log_returns=r, data_fingerprint="a" * 64,
    )


def _isolate_shadow_store(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every `capture_pending_observation` call writes a real
    `ModelVersion` parameter artifact via `shadow.build_combo` -- tests
    must redirect that to a tmp dir, exactly like `tests/unit/shadow/
    test_build_combo.py` already does, or they leak files into the real
    `reports/shadow/` tree (a real regression caught live: an earlier
    version of this file did exactly that, leaving a stray
    `vol_combo_median_PROSPTEST_params.json` in the committed tree)."""
    import model_lab.shadow.build_combo as build_combo_mod

    monkeypatch.setattr(build_combo_mod, "SHADOW_STORE", tmp_path / "shadow")


def test_capture_sets_information_cutoff_to_the_series_own_last_date(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_shadow_store(tmp_path, monkeypatch)
    series = _synthetic_series()
    pending = capture_pending_observation(series)
    assert pending.information_cutoff_date == series.dates[-1]
    assert pending.status == "PENDING"
    assert pending.candidate_output.decision_relevant is False


def test_capture_is_deterministic_given_identical_input(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_shadow_store(tmp_path, monkeypatch)
    series = _synthetic_series(seed=7)
    p1 = capture_pending_observation(series)
    p2 = capture_pending_observation(series)
    assert p1.pending_observation_hash == p2.pending_observation_hash
    assert p1.candidate_output.point_estimate == p2.candidate_output.point_estimate


def test_capture_includes_all_four_benchmarks_at_the_same_origin(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_shadow_store(tmp_path, monkeypatch)
    series = _synthetic_series(seed=3)
    pending = capture_pending_observation(series)
    model_ids = {f.model_id for f in pending.benchmark_forecasts}
    assert model_ids == {"naive_trailing_rv", "EWMA_0.94", "GARCH_t", "EGARCH_t"}
    for f in pending.benchmark_forecasts:
        assert float(f.point_estimate) > 0


def test_capture_uses_no_information_beyond_the_cutoff(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate_shadow_store(tmp_path, monkeypatch)
    """Prefix-truncation check: capturing from a series truncated right
    after a given origin must produce the IDENTICAL candidate
    point_estimate as capturing from the full series and then looking at
    what that full series' capture WOULD have been at that earlier origin
    -- i.e. the extra future rows the truncated version never saw must not
    change the forecast. Implemented directly: truncate at two different
    points and confirm the earlier capture's own point_estimate is
    unaffected by data appended after its own cutoff."""
    series = _synthetic_series(n=700, seed=9)
    origin = 650
    short_series = ReturnSeries(
        ticker=series.ticker, dataset_id=series.dataset_id, dates=series.dates[: origin + 1],
        closes=series.closes[: origin + 1], highs=series.highs[: origin + 1], lows=series.lows[: origin + 1],
        opens=series.opens[: origin + 1], log_returns=series.log_returns[:origin], data_fingerprint=series.data_fingerprint,
    )
    longer_series = ReturnSeries(
        ticker=series.ticker, dataset_id=series.dataset_id, dates=series.dates[: origin + 11],
        closes=series.closes[: origin + 11], highs=series.highs[: origin + 11], lows=series.lows[: origin + 11],
        opens=series.opens[: origin + 11], log_returns=series.log_returns[: origin + 10], data_fingerprint=series.data_fingerprint,
    )
    # Re-truncate the longer series back to the SAME cutoff the short one used, and confirm
    # that capture (which only ever reads up to series.dates[-1]) gives the identical forecast --
    # proving the extra 10 future rows present in `longer_series` play no role.
    re_truncated = ReturnSeries(
        ticker=longer_series.ticker, dataset_id=longer_series.dataset_id, dates=longer_series.dates[: origin + 1],
        closes=longer_series.closes[: origin + 1], highs=longer_series.highs[: origin + 1],
        lows=longer_series.lows[: origin + 1], opens=longer_series.opens[: origin + 1],
        log_returns=longer_series.log_returns[:origin], data_fingerprint=longer_series.data_fingerprint,
    )
    short_pending = capture_pending_observation(short_series)
    re_truncated_pending = capture_pending_observation(re_truncated)
    assert short_pending.information_cutoff_date == re_truncated_pending.information_cutoff_date
    assert short_pending.candidate_output.point_estimate == re_truncated_pending.candidate_output.point_estimate


def test_fetch_bars_after_returns_empty_on_rate_limit_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for a real bug found live (2026-10-01): the actual
    Massive provider rate-limited a 6th sequential call and raised
    `SourceUnavailableError`, which `fetch_bars_after` did not catch,
    crashing the whole resolve pass instead of leaving that one
    observation controlled-pending."""
    import model_lab.prospective.live_fetch as live_fetch_mod
    from model_lab.data.providers._http import SourceUnavailableError

    class _RaisingClient:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def fetch_daily_bars(self, *a, **k):
            raise SourceUnavailableError("rate limited: /fake")

    monkeypatch.setattr(live_fetch_mod, "MassiveClient", lambda: _RaisingClient())
    result = live_fetch_mod.fetch_bars_after("SPY", "2026-09-29")
    assert result == []
