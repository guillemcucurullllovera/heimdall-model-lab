from __future__ import annotations

from datetime import UTC, datetime

import pytest

from model_lab.contracts.model_output import ModelOutput
from model_lab.prospective.records import ForecastEntry, PendingObservation
from model_lab.prospective.resolve import resolve_pending_observation


def _fake_pending(candidate_point_estimate: str = "0.0001") -> PendingObservation:
    candidate_output = ModelOutput(
        model_version_hash="a" * 64,
        task_id="T-VOL",
        as_of_utc=datetime(2026, 9, 29, tzinfo=UTC),
        produced_at_utc=datetime.now(UTC),
        target_variable="forward_realised_variance",
        horizon_days=1,
        point_estimate=candidate_point_estimate,
        input_hashes=["b" * 64],
        decision_relevant=False,
    )
    benchmarks = [
        ForecastEntry(model_id="naive_trailing_rv", point_estimate="0.00012"),
        ForecastEntry(model_id="EWMA_0.94", point_estimate="0.00011"),
        ForecastEntry(model_id="GARCH_t", point_estimate="0.0001"),
        ForecastEntry(model_id="EGARCH_t", point_estimate="0.00009"),
    ]
    from model_lab.prospective.records import compute_pending_observation_hash

    pending_hash = compute_pending_observation_hash(
        ticker="FAKE", task_id="T-VOL", information_cutoff_date="2026-09-29",
        information_cutoff_close="100.0", candidate_output=candidate_output,
        benchmark_forecasts=benchmarks, data_fingerprint="c" * 64,
    )
    return PendingObservation(
        ticker="FAKE", task_id="T-VOL", information_cutoff_date="2026-09-29",
        information_cutoff_close="100.0", observation_captured_at_utc=datetime.now(UTC),
        candidate_output=candidate_output, benchmark_forecasts=benchmarks,
        data_fingerprint="c" * 64, status="PENDING", pending_observation_hash=pending_hash,
    )


def test_resolve_returns_none_when_no_new_bar_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [])
    pending = _fake_pending()
    result = resolve_pending_observation(pending)
    assert result is None, "no new real-world data means the observation must stay PENDING, not be fabricated"


def test_resolve_computes_realized_outcome_from_the_live_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.resolve as resolve_mod
    from model_lab.prospective.live_fetch import LiveBar

    monkeypatch.setattr(
        resolve_mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [LiveBar(trading_date="2026-09-30", close="101.0")]
    )
    pending = _fake_pending()
    resolved = resolve_pending_observation(pending)
    assert resolved is not None
    assert resolved.realized_target_date == "2026-09-30"
    assert resolved.pending_observation_hash == pending.pending_observation_hash
    assert resolved.is_historical_replay is False
    assert set(resolved.benchmark_forecast_errors.keys()) == {"naive_trailing_rv", "EWMA_0.94", "GARCH_t", "EGARCH_t"}
    import math

    expected_log_return = math.log(101.0 / 100.0)
    assert float(resolved.realized_log_return) == pytest.approx(expected_log_return, rel=1e-9)
    assert float(resolved.realized_variance_proxy) == pytest.approx(expected_log_return**2, rel=1e-9)


def test_resolve_flags_a_bar_not_strictly_after_cutoff_as_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defense-in-depth: even if `live_fetch`'s own filter were ever
    weakened by a future edit, resolve.py's own guard must still catch and
    flag a non-prospective (replay) bar rather than silently treating it as
    genuine prospective evidence."""
    import model_lab.prospective.resolve as resolve_mod
    from model_lab.prospective.live_fetch import LiveBar

    monkeypatch.setattr(
        resolve_mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [LiveBar(trading_date="2026-09-29", close="100.5")]
    )
    pending = _fake_pending()
    resolved = resolve_pending_observation(pending)
    assert resolved is not None
    assert resolved.is_historical_replay is True


def test_resolve_never_mutates_the_pending_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    import model_lab.prospective.resolve as resolve_mod
    from model_lab.prospective.live_fetch import LiveBar

    monkeypatch.setattr(
        resolve_mod, "fetch_bars_after", lambda ticker, since, lookahead_days=10: [LiveBar(trading_date="2026-09-30", close="101.0")]
    )
    pending = _fake_pending()
    pending_before = pending.model_dump_json()
    resolve_pending_observation(pending)
    assert pending.model_dump_json() == pending_before, "resolving must never mutate the original pending record (frozen=True enforces this structurally too)"
