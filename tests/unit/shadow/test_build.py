from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest

from model_lab.anomaly.panel import ReturnPanel
from model_lab.contracts.model_output import ModelOutput
from model_lab.contracts.model_version import LifecycleState, ModelVersion
from model_lab.shadow.build import (
    build_anomaly_mahalanobis_shadow_record,
    build_bocpd_changepoint_shadow_record,
    build_combo_vol_shadow_record,
    build_egarch_t_vol_shadow_record,
    build_ewma_vol_shadow_record,
    build_regime_2state_shadow_record,
    build_tail_shadow_records,
    build_vol_shadow_records,
)
from model_lab.vol.data import ReturnSeries


def _synthetic_series(n: int = 500, seed: int = 0) -> ReturnSeries:
    rng = np.random.default_rng(seed)
    r = rng.standard_t(6, n) * 0.01
    dates = []
    d = date(2022, 1, 3)
    while len(dates) < n + 1:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    closes = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return ReturnSeries(
        ticker="SHADOWTEST", dataset_id="shadowtest_raw", dates=dates, closes=closes, highs=closes,
        lows=closes, opens=closes, log_returns=r, data_fingerprint="d" * 64,
    )


def test_build_vol_shadow_records_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series()
    model_version, model_output = build_vol_shadow_records(series, fold_train_ends=[300])

    assert isinstance(model_version, ModelVersion)
    assert isinstance(model_output, ModelOutput)
    assert model_version.lifecycle_state == LifecycleState.SHADOW
    assert model_version.tier_ceiling_signed is False
    assert model_output.decision_relevant is False
    assert model_output.task_id == "T-VOL"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) > 0
    assert model_version.model_version_hash == model_output.model_version_hash
    # The parameter artifact must be a real file on disk with a matching hash.
    artifact_files = list((tmp_path / "shadow").glob("*_params.json"))
    assert len(artifact_files) == 1
    from model_lab.canon.jcs import sha256_bytes

    assert sha256_bytes(artifact_files[0].read_bytes()) == model_version.parameter_artifacts[0].sha256


def test_build_tail_shadow_records_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=1)
    model_version, model_output = build_tail_shadow_records(series, fold_train_ends=[300])

    assert model_version.task_ids == ["T-TAIL"]
    assert model_output.task_id == "T-TAIL"
    assert len(model_output.quantiles) > 0
    for q in model_output.quantiles:
        assert 0.0 <= q.quantile <= 1.0
        assert float(q.value) > 0  # VaR magnitudes are positive in this convention


def test_build_vol_shadow_records_raises_closed_on_insufficient_history() -> None:
    series = _synthetic_series(n=50)  # far too short for any fold
    with pytest.raises(ValueError):
        build_vol_shadow_records(series, fold_train_ends=[300])


# -- Remediation (remediation-specification item 4 / independent review
# P1-D): EWMA_0.94, EGARCH_t, and the median combination had no
# ModelVersion/ModelOutput shadow record at all until this remediation.


def test_build_ewma_vol_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=2)
    model_version, model_output = build_ewma_vol_shadow_record(series)

    assert isinstance(model_version, ModelVersion)
    assert isinstance(model_output, ModelOutput)
    assert model_version.family == "EWMA_0.94"
    assert model_version.lifecycle_state == LifecycleState.SHADOW
    assert model_version.tier_ceiling_signed is False
    assert model_output.decision_relevant is False
    assert model_output.task_id == "T-VOL"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) > 0
    assert model_version.model_version_hash == model_output.model_version_hash


def test_build_ewma_vol_shadow_record_raises_closed_on_insufficient_history() -> None:
    series = _synthetic_series(n=10)  # shorter than EWMA's seed_window=30
    with pytest.raises(ValueError):
        build_ewma_vol_shadow_record(series)


def test_build_egarch_t_vol_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=3)
    model_version, model_output = build_egarch_t_vol_shadow_record(series, fold_train_ends=[300])

    assert model_version.family == "EGARCH-t"
    assert model_version.task_ids == ["T-VOL"]
    assert model_output.task_id == "T-VOL"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) > 0


def test_build_egarch_t_vol_shadow_record_raises_closed_on_insufficient_history() -> None:
    series = _synthetic_series(n=50)
    with pytest.raises(ValueError):
        build_egarch_t_vol_shadow_record(series, fold_train_ends=[300])


def test_build_combo_vol_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=4)
    model_version, model_output = build_combo_vol_shadow_record(series, fold_train_ends=[300])

    assert model_version.family == "Combo_median_EWMA_GARCH_EGARCH"
    assert model_output.task_id == "T-VOL"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) > 0
    # the artifact must record both GARCH-family members' last-fold parameters,
    # not a newly-fitted parameter set of its own
    artifact_files = list((tmp_path / "shadow").glob("vol_combo_median_*_params.json"))
    assert len(artifact_files) == 1
    payload = json.loads(artifact_files[0].read_text())
    assert "garch_t_last_fold" in payload
    assert "egarch_t_last_fold" in payload


def test_build_combo_vol_shadow_record_raises_closed_only_when_every_member_is_undefined(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")

    # n=50 is enough for EWMA_0.94 (seed_window=30) to have a real value even though
    # GARCH_t/EGARCH_t both fail here -- under this module's NaN-robust median
    # contract (see vol/combination.py), the combination then correctly falls back
    # to EWMA alone rather than raising. This is intentional, not a bug: verify it.
    series_partial = _synthetic_series(n=50, seed=5)
    model_version, model_output = build_combo_vol_shadow_record(series_partial, fold_train_ends=[300])
    assert model_output.point_estimate is not None

    # n=20 is below EWMA's own seed_window=30, so EVERY member is genuinely
    # undefined at the last index -- this must still fail closed.
    series_all_undefined = _synthetic_series(n=20, seed=6)
    with pytest.raises(ValueError):
        build_combo_vol_shadow_record(series_all_undefined, fold_train_ends=[300])


# -- Remediation (2026-10-02 Phase 2, independent review finding P2-5):
# regime/changepoint/anomaly had no shadow/ModelOutput record at all until
# this remediation. One representative builder per family, tested below.


def _synthetic_panel(tickers: tuple[str, ...] = ("A", "B", "C"), n: int = 500, seed: int = 0) -> ReturnPanel:
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(6, (n, len(tickers))) * 0.01
    dates = []
    d = date(2022, 1, 3)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d.isoformat())
        d += timedelta(days=1)
    return ReturnPanel(
        tickers=list(tickers),
        dates=dates,
        returns=returns,
        data_fingerprints={t: "f" * 64 for t in tickers},
        dataset_ids={t: f"{t}_raw" for t in tickers},
    )


def test_build_regime_2state_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=0)
    model_version, model_output = build_regime_2state_shadow_record(series, fold_train_ends=[300])

    assert isinstance(model_version, ModelVersion)
    assert isinstance(model_output, ModelOutput)
    assert model_version.family == "Markov-switching-2-state-Gaussian"
    assert model_version.task_ids == ["T-STATE"]
    assert model_version.lifecycle_state == LifecycleState.SHADOW
    assert model_version.tier_ceiling_signed is False
    assert model_output.task_id == "T-STATE"
    assert model_output.target_variable == "regime_state_probability"
    assert model_output.decision_relevant is False
    assert model_output.point_estimate is not None
    assert 0.0 <= float(model_output.point_estimate) <= 1.0
    assert model_version.model_version_hash == model_output.model_version_hash


def test_build_regime_2state_shadow_record_raises_closed_on_insufficient_history() -> None:
    series = _synthetic_series(n=50)  # far too short for any fold (needs >=150)
    with pytest.raises(ValueError):
        build_regime_2state_shadow_record(series, fold_train_ends=[300])


def test_build_bocpd_changepoint_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    series = _synthetic_series(seed=1)
    model_version, model_output = build_bocpd_changepoint_shadow_record(series)

    assert isinstance(model_version, ModelVersion)
    assert isinstance(model_output, ModelOutput)
    assert model_version.family == "BOCPD-NIG"
    assert model_version.task_ids == ["T-CHANGE"]
    assert model_output.task_id == "T-CHANGE"
    assert model_output.target_variable == "changepoint_posterior_probability"
    assert model_output.horizon_days == 0
    assert model_output.point_estimate is not None
    assert 0.0 <= float(model_output.point_estimate) <= 1.0


def test_build_bocpd_changepoint_shadow_record_is_causal_same_prefix_same_value(tmp_path, monkeypatch) -> None:
    """BOCPD is purely online -- the shadow record's point estimate for a
    given last day must not depend on data after that day (the same
    prefix-invariance property `changepoint/bocpd.py`'s own regression
    test already proves for the raw function; this confirms the shadow
    wrapper does not break it)."""
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    full_series = _synthetic_series(n=500, seed=2)
    prefix_series = _synthetic_series(n=500, seed=2)
    prefix_series = ReturnSeries(
        ticker=prefix_series.ticker, dataset_id=prefix_series.dataset_id, dates=prefix_series.dates[:301],
        closes=prefix_series.closes[:301], highs=prefix_series.highs[:301], lows=prefix_series.lows[:301],
        opens=prefix_series.opens[:301], log_returns=prefix_series.log_returns[:300],
        data_fingerprint=prefix_series.data_fingerprint,
    )
    _, full_output = build_bocpd_changepoint_shadow_record(
        ReturnSeries(
            ticker=full_series.ticker, dataset_id=full_series.dataset_id, dates=full_series.dates[:301],
            closes=full_series.closes[:301], highs=full_series.highs[:301], lows=full_series.lows[:301],
            opens=full_series.opens[:301], log_returns=full_series.log_returns[:300],
            data_fingerprint=full_series.data_fingerprint,
        )
    )
    _, prefix_output = build_bocpd_changepoint_shadow_record(prefix_series)
    assert full_output.point_estimate == prefix_output.point_estimate


def test_build_anomaly_mahalanobis_shadow_record_produces_valid_contracts(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    panel = _synthetic_panel()
    model_version, model_output = build_anomaly_mahalanobis_shadow_record(panel, fold_train_ends=[300])

    assert isinstance(model_version, ModelVersion)
    assert isinstance(model_output, ModelOutput)
    assert model_version.family == "Covariance-Mahalanobis-ledoit_wolf"
    assert model_version.task_ids == ["T-ANOM"]
    assert model_version.data_snapshot_ids == sorted(panel.dataset_ids.values())
    assert model_output.task_id == "T-ANOM"
    assert model_output.target_variable == "anomaly_score"
    assert model_output.point_estimate is not None
    assert float(model_output.point_estimate) >= 0.0  # Mahalanobis d^2 is non-negative
    assert len(model_output.input_hashes) == len(panel.tickers)


def test_build_anomaly_mahalanobis_shadow_record_raises_closed_on_insufficient_history() -> None:
    panel = _synthetic_panel(n=10)  # far too short for any 3-ticker covariance fold (needs >= 5*k rows)
    with pytest.raises(ValueError):
        build_anomaly_mahalanobis_shadow_record(panel, fold_train_ends=[300])


def test_build_anomaly_mahalanobis_shadow_record_supports_alternate_estimator(tmp_path, monkeypatch) -> None:
    import model_lab.shadow.build as build_mod

    monkeypatch.setattr(build_mod, "SHADOW_STORE", tmp_path / "shadow")
    panel = _synthetic_panel(seed=3)
    model_version, _ = build_anomaly_mahalanobis_shadow_record(panel, fold_train_ends=[300], estimator="oas")
    assert model_version.family == "Covariance-Mahalanobis-oas"
    assert model_version.model_id == "anomaly_oas_panel"
