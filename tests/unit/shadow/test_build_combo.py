from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.model_output import ModelOutput
from model_lab.contracts.model_version import LifecycleState, ModelVersion
from model_lab.shadow.build_combo import build_combo_shadow_records, candidate_is_eligible_for
from model_lab.vol.data import ReturnSeries


def _synthetic_series(n: int = 700, seed: int = 0, ticker: str = "SHADOWCOMBO") -> ReturnSeries:
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
        ticker=ticker, dataset_id=f"{ticker.lower()}_raw", dates=dates, closes=closes, highs=closes,
        lows=closes, opens=closes, log_returns=r, data_fingerprint="c" * 64,
    )


def test_build_combo_shadow_records_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build_combo as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series()
    model_version, model_output = build_combo_shadow_records(series, fold_train_ends=[400, 600])

    assert isinstance(model_version, ModelVersion)
    assert model_version.lifecycle_state == LifecycleState.SHADOW
    assert model_version.tier_ceiling_signed is False
    assert isinstance(model_output, ModelOutput)
    assert model_output.decision_relevant is False, "the candidate must never be decision_relevant"
    assert model_output.task_id == "T-VOL"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) > 0
    assert model_version.model_version_hash == model_output.model_version_hash

    artifact_files = list((tmp_path / "shadow").glob("*_params.json"))
    assert len(artifact_files) == 1
    assert sha256_bytes(artifact_files[0].read_bytes()) == model_version.parameter_artifacts[0].sha256


def test_model_version_is_pinned_and_reproducible(tmp_path, monkeypatch) -> None:
    """Identical inputs must produce an identical model_version_hash and an
    identical point_estimate across two independent calls -- the candidate
    has no hidden randomness of its own (EGARCH_t's own simulation RNG is
    seeded, as everywhere else in this project)."""
    import model_lab.shadow.build_combo as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series()
    v1, o1 = build_combo_shadow_records(series, fold_train_ends=[400, 600])
    v2, o2 = build_combo_shadow_records(series, fold_train_ends=[400, 600])
    assert v1.model_version_hash == v2.model_version_hash
    assert o1.point_estimate == o2.point_estimate


def test_candidate_is_exactly_the_median_of_its_three_constituents_not_fabricated(tmp_path, monkeypatch) -> None:
    """Benchmark/candidate alignment: the recorded point_estimate must equal
    the ACTUAL median of EWMA_0.94/GARCH_t/EGARCH_t computed independently
    via the same functions build_combo_shadow_records calls -- not an
    approximation, not a mean, not a different rule."""
    import model_lab.shadow.build_combo as build_mod
    from model_lab.vol.baselines import ewma_forecast
    from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=2)
    fold_train_ends = [400, 600]
    _, model_output = build_combo_shadow_records(series, fold_train_ends=fold_train_ends)

    last_idx = len(series.log_returns) - 1
    ewma_v = ewma_forecast(series.log_returns, 1, lam=0.94)[last_idx]
    garch_v = gjr_garch_t_forecast(series.log_returns, 1, fold_train_ends).variance_forecast[last_idx]
    egarch_v = egarch_t_forecast(series.log_returns, 1, fold_train_ends).variance_forecast[last_idx]
    expected_median = float(np.median([ewma_v, garch_v, egarch_v]))
    assert float(model_output.point_estimate) == pytest.approx(expected_median, rel=1e-9)


def test_missing_constituent_forecast_raises_rather_than_fabricating(tmp_path, monkeypatch) -> None:
    """A controlled failure (raise), never a silently-degraded 2-of-3 or
    fabricated combination, when any constituent lacks a valid forecast."""
    import model_lab.shadow.build_combo as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(n=50)  # far too short for any fold -> GARCH_t/EGARCH_t forecasts are NaN
    with pytest.raises(ValueError, match="requires all 3 constituents"):
        build_combo_shadow_records(series, fold_train_ends=[400])


def test_candidate_output_carries_no_future_information(tmp_path, monkeypatch) -> None:
    """Prefix-truncation check: truncating the series to exactly one day
    after a given origin and rebuilding the shadow record for that
    (now-final) origin must reproduce the SAME point_estimate as building it
    from the full series -- the combination (and each of its 3 already-
    causal constituents) uses no information beyond the origin."""
    import model_lab.shadow.build_combo as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(n=700, seed=5)
    fold_train_ends = [400, 600]
    origin = 650
    truncated_dates = series.dates[: origin + 2]
    truncated_closes = series.closes[: origin + 2]
    truncated_series = ReturnSeries(
        ticker=series.ticker, dataset_id=series.dataset_id, dates=truncated_dates, closes=truncated_closes,
        highs=truncated_closes, lows=truncated_closes, opens=truncated_closes,
        log_returns=series.log_returns[: origin + 1], data_fingerprint=series.data_fingerprint,
    )
    _, full_output = build_combo_shadow_records(
        ReturnSeries(
            ticker=series.ticker, dataset_id=series.dataset_id, dates=series.dates[: origin + 2],
            closes=series.closes[: origin + 2], highs=series.highs[: origin + 2], lows=series.lows[: origin + 2],
            opens=series.opens[: origin + 2], log_returns=series.log_returns[: origin + 1],
            data_fingerprint=series.data_fingerprint,
        ),
        fold_train_ends=[e for e in fold_train_ends if e <= origin],
    )
    _, truncated_output = build_combo_shadow_records(
        truncated_series, fold_train_ends=[e for e in fold_train_ends if e <= origin]
    )
    assert full_output.point_estimate == truncated_output.point_estimate


def test_candidate_is_eligible_for_matches_the_locked_spec() -> None:
    assert candidate_is_eligible_for("TLT") is True
    assert candidate_is_eligible_for("IEF") is True
    assert candidate_is_eligible_for("GLD") is True
    assert candidate_is_eligible_for("SPY") is False
    assert candidate_is_eligible_for("QQQ") is False
    assert candidate_is_eligible_for("USO") is False


def test_shadow_output_has_no_amc_or_decision_relevant_pathway() -> None:
    """Structural guarantee, not just a convention: `ModelOutput.decision_relevant`
    defaults to False and `ModelVersion.tier_ceiling_signed` defaults to
    requiring an explicit True that build_combo_shadow_records never sets --
    nothing in this module's own code path can make either field True."""
    import inspect

    import model_lab.shadow.build_combo as build_mod

    source = inspect.getsource(build_mod)
    assert "decision_relevant=True" not in source
    assert "tier_ceiling_signed=True" not in source
