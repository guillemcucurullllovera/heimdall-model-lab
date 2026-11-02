"""Builds real, validating `ModelVersion`/`ModelOutput` records for one
ticker's LATEST available forecast from the vol/tail ladders. See package
docstring for what this deliberately does NOT do."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from model_lab.anomaly.covariance_models import mahalanobis_d2_series
from model_lab.anomaly.panel import ReturnPanel
from model_lab.canon.jcs import sha256_bytes
from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma
from model_lab.contracts.model_output import ModelOutput, QuantilePoint
from model_lab.contracts.model_version import (
    LifecycleState,
    ModelVersion,
    ParameterArtifact,
    RefitLane,
    compute_model_version_hash,
)
from model_lab.contracts.task import InfluenceTier
from model_lab.regime.markov_switching import MIN_PERSISTENCE, fit_and_filter_2state
from model_lab.tail.conditional import gjr_t_filtered_gpd_fhs_forecast
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.data import ReturnSeries
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

SHADOW_STORE = Path(__file__).resolve().parents[3] / "reports" / "shadow"
_SRC_ROOT = Path(__file__).resolve().parents[1]


def _code_tree_hash(*relative_paths: str) -> str:
    """Real hash of the actual source files that produced this output --
    not a placeholder. Concatenates file bytes in a fixed (sorted) order."""
    data = b""
    for rel in sorted(relative_paths):
        data += (_SRC_ROOT / rel).read_bytes()
    return sha256_bytes(data)


def _write_parameter_artifact(model_id: str, ticker: str, payload: dict) -> ParameterArtifact:
    SHADOW_STORE.mkdir(parents=True, exist_ok=True)
    out_path = SHADOW_STORE / f"{model_id}_{ticker}_params.json"
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(content, encoding="utf-8")
    digest = sha256_bytes(content.encode("utf-8"))
    return ParameterArtifact(path=str(out_path.relative_to(SHADOW_STORE.parents[1])), sha256=digest)


def build_vol_shadow_records(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """T-VOL shadow records for the GJR-GARCH-t (`GARCH_t`) challenger,
    the one-day-ahead variance forecast for the LAST available day in
    `series`."""
    garch_res = gjr_garch_t_forecast(series.log_returns, 1, fold_train_ends)
    last_idx = len(series.log_returns) - 1
    forecast_value = garch_res.variance_forecast[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(f"{series.ticker}: no valid GARCH_t forecast at the latest available index")

    last_fold = garch_res.fold_params[-1]
    artifact = _write_parameter_artifact("vol_garch_t", series.ticker, last_fold)
    code_hash = _code_tree_hash("vol/garch_models.py", "vol/data.py", "vol/targets.py")
    spec_hash = sha256_bytes(json.dumps(garch_res.config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"vol_garch_t::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"vol_garch_t_{series.ticker}",
        model_version_hash=version_hash,
        family="GJR-GARCH-t",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-VOL"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-VOL",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="forward_realised_variance",
        horizon_days=1,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_ewma_vol_shadow_record(series: ReturnSeries) -> tuple[ModelVersion, ModelOutput]:
    """T-VOL shadow records for `EWMA_0.94`, the one-day-ahead variance
    forecast for the LAST available day in `series`.

    Remediation (remediation-specification item 4, following the
    independent review's finding P1-D: 'no vol-ladder challenger besides
    two has a canonical, versioned ModelVersion/ModelOutput record'). EWMA
    has no per-fold FIT the way GARCH_t/EGARCH_t do (`ewma_variance`'s
    recursion is a fixed formula, not an estimated parameter set) -- its
    `parameter_artifact` records that fixed configuration (`lam`,
    `seed_window`) rather than a fitted-parameter dict, which is the
    correct analogue here: the artifact is "what fully determines this
    model's output," not "what required estimation."
    """
    r = series.log_returns
    forecast = ewma_forecast(r, 1, lam=0.94)
    last_idx = len(r) - 1
    forecast_value = forecast[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(f"{series.ticker}: no valid EWMA_0.94 forecast at the latest available index")

    config = {"lam": 0.94, "seed_window": 30, "scaling": "flat_h_x_ewma_var"}
    artifact = _write_parameter_artifact("vol_ewma", series.ticker, config)
    code_hash = _code_tree_hash("vol/baselines.py", "vol/data.py")
    spec_hash = sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"vol_ewma::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"vol_ewma_{series.ticker}",
        model_version_hash=version_hash,
        family="EWMA_0.94",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-VOL"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.FAST,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-VOL",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="forward_realised_variance",
        horizon_days=1,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_egarch_t_vol_shadow_record(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """T-VOL shadow records for `EGARCH_t`, the one-day-ahead variance
    forecast for the LAST available day in `series`. `egarch_t_forecast`'s
    own plausibility filter (`vol/garch_models.py`) already converts an
    implausible simulated value to NaN before it reaches this function, so
    the NaN check below is also, in effect, "was the latest forecast
    detected-and-excluded by that filter" -- see the independent review's
    finding P1-B/P2-1 for why that filter's exclusions are disclosed
    separately, not silently masked here."""
    egarch_res = egarch_t_forecast(series.log_returns, 1, fold_train_ends)
    last_idx = len(series.log_returns) - 1
    forecast_value = egarch_res.variance_forecast[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(
            f"{series.ticker}: no valid EGARCH_t forecast at the latest available index "
            "(either no fold covers it, or it was excluded by the plausibility filter)"
        )

    last_fold = egarch_res.fold_params[-1]
    artifact = _write_parameter_artifact("vol_egarch_t", series.ticker, last_fold)
    code_hash = _code_tree_hash("vol/garch_models.py", "vol/data.py", "vol/targets.py")
    spec_hash = sha256_bytes(json.dumps(egarch_res.config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"vol_egarch_t::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"vol_egarch_t_{series.ticker}",
        model_version_hash=version_hash,
        family="EGARCH-t",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-VOL"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-VOL",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="forward_realised_variance",
        horizon_days=1,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_combo_vol_shadow_record(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """T-VOL shadow records for `Combo_median_EWMA_GARCH_EGARCH`, the
    one-day-ahead variance forecast for the LAST available day in
    `series`. Recomputes the same three member forecasts the vol ladder
    uses (same functions, same fold boundaries) and combines them with
    `median_variance_combination` -- this performs no estimation of its
    own (see `vol/combination.py`'s docstring), so its `parameter_artifact`
    records the three members' own last-fold parameters plus the
    combination rule, not a newly-fitted parameter set."""
    ewma = ewma_forecast(series.log_returns, 1, lam=0.94)
    garch_res = gjr_garch_t_forecast(series.log_returns, 1, fold_train_ends)
    egarch_res = egarch_t_forecast(series.log_returns, 1, fold_train_ends)
    combo = median_variance_combination(ewma, garch_res.variance_forecast, egarch_res.variance_forecast)
    last_idx = len(series.log_returns) - 1
    forecast_value = combo[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(f"{series.ticker}: no valid combination forecast at the latest available index")

    payload = {
        "method": "median_variance_combination",
        "members": ["EWMA_0.94", "GARCH_t", "EGARCH_t"],
        # Either member's own last fold may be absent if that model alone
        # could not produce a forecast at this index (e.g. no fold covers
        # it, or EGARCH_t's plausibility filter excluded it) while the
        # OTHER members still could -- the NaN-robust combination (see
        # `vol/combination.py`) can still produce a real value from
        # whichever members remain, so this must not assume both are present.
        "garch_t_last_fold": garch_res.fold_params[-1] if garch_res.fold_params else None,
        "egarch_t_last_fold": egarch_res.fold_params[-1] if egarch_res.fold_params else None,
    }
    artifact = _write_parameter_artifact("vol_combo_median", series.ticker, payload)
    code_hash = _code_tree_hash("vol/combination.py", "vol/garch_models.py", "vol/baselines.py", "vol/data.py")
    spec_hash = sha256_bytes(json.dumps(
        {"method": "median_variance_combination", "members": ["EWMA_0.94", "GARCH_t", "EGARCH_t"]}, sort_keys=True
    ).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"vol_combo_median::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"vol_combo_median_{series.ticker}",
        model_version_hash=version_hash,
        family="Combo_median_EWMA_GARCH_EGARCH",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-VOL"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-VOL",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="forward_realised_variance",
        horizon_days=1,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_tail_shadow_records(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """T-TAIL shadow records for the GJR-t-filtered-GPD-FHS challenger,
    95/97.5/99% VaR for the last available day, as ModelOutput quantiles."""
    tail_res = gjr_t_filtered_gpd_fhs_forecast(series.log_returns, fold_train_ends, tail_levels=(0.05, 0.025, 0.01))
    last_idx = len(series.log_returns) - 1
    quantiles = []
    for p in (0.05, 0.025, 0.01):
        v = tail_res.var_forecast[p][last_idx]
        if v == v:  # not NaN
            quantiles.append(QuantilePoint(quantile=1.0 - p, value=str(float(v))))
    if not quantiles:
        raise ValueError(f"{series.ticker}: no valid tail forecast at the latest available index")

    last_fold = tail_res.fold_fits[-1]
    artifact = _write_parameter_artifact("tail_gjr_fhs", series.ticker, last_fold)
    code_hash = _code_tree_hash("tail/conditional.py", "tail/gpd.py", "vol/garch_models.py")
    spec_hash = sha256_bytes(json.dumps(tail_res.config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"tail_gjr_fhs::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"tail_gjr_fhs_{series.ticker}",
        model_version_hash=version_hash,
        family="GJR-t-filtered-GPD-FHS",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-TAIL"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-TAIL",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="var_es_forecast",
        horizon_days=1,
        quantiles=quantiles,
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


# -- Remediation (2026-10-02 Phase 2, independent review finding P2-5:
# "regime/changepoint/anomaly challengers have no shadow/ModelOutput path
# at all"). The three builders below extend this module's existing
# vol/tail pattern to one representative challenger per remaining
# MODEL-2B-2D family. None of this changes any model's own evidence
# (MODEL-2D's regime-loses-to-GARCH_t negative finding, T-ANOM/T-CHANGE's
# BLOCKED status) -- it gives each family's representative challenger a
# real, hashed, versioned identity so the governance machinery can see it
# and (eventually) accumulate prospective evidence against it, which is a
# PREREQUISITE for, not a claim toward, any future promotion.


def build_regime_2state_shadow_record(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """T-STATE shadow record for the 2-state Markov-switching regime
    blend (`regime.markov_switching.fit_and_filter_2state`), the one-
    step-ahead high-vol-regime probability for the LAST available day in
    `series`. Regime's own periodic-refit discipline (annual fold, like
    GARCH_t) makes `refit_lane=SLOW` the correct analogue to
    `build_vol_shadow_records`, not `build_ewma_vol_shadow_record`'s FAST.

    If the fold covering the latest day was flagged `well_identified=
    False` (a degenerate/boundary MLE solution -- see the module's own
    `MIN_PERSISTENCE` guard, independent review finding P2-7 of the FIRST
    review), `high_vol_prob` is NaN there and this raises, exactly like
    every other NaN-at-latest-index case in this module -- it does not
    paper over a degenerate fold with a fabricated forecast.
    """
    regime_res = fit_and_filter_2state(series.log_returns, fold_train_ends)
    last_idx = len(series.log_returns) - 1
    forecast_value = regime_res.high_vol_prob[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(
            f"{series.ticker}: no valid regime forecast at the latest available index "
            "(either no fold covers it, or that fold was flagged well_identified=False)"
        )

    last_fold = asdict(regime_res.fold_fits[-1])
    artifact = _write_parameter_artifact("regime_2state", series.ticker, last_fold)
    code_hash = _code_tree_hash("regime/markov_switching.py", "vol/data.py")
    config = {
        "k_regimes": 2, "trend": "c", "switching_variance": True,
        "purge_days": 1, "min_persistence": MIN_PERSISTENCE,
    }
    spec_hash = sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"regime_2state::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"regime_2state_{series.ticker}",
        model_version_hash=version_hash,
        family="Markov-switching-2-state-Gaussian",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-STATE"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-STATE",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="regime_state_probability",
        horizon_days=1,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_bocpd_changepoint_shadow_record(series: ReturnSeries) -> tuple[ModelVersion, ModelOutput]:
    """T-CHANGE shadow record for the BOCPD (Normal-Inverse-Gamma)
    changepoint detector, the same-day changepoint posterior probability
    for the LAST available day in `series`. BOCPD is purely online/causal
    (`changepoint.bocpd`'s own docstring) -- there is no periodic-refit
    fold here, so `parameter_artifacts` records the fixed hyperparameters
    (hazard rate, NIG prior, Welford expanding-variance mode) rather than
    any fitted/estimated values, the same rationale
    `build_ewma_vol_shadow_record` already uses for `EWMA_0.94`.

    `target_variable` is deliberately NOT T-CHANGE's own frozen task
    target (`changepoint_detection_delay`, a days-to-detection metric that
    requires labelled changepoint events to compute against --
    `tasks/registry.py` already names that label set as BLOCKED). What
    this function actually outputs is BOCPD's own posterior `P(run length
    resets to 0 at t | x_1..t)`, named accordingly so this record never
    silently overstates itself as the blocked, labelled-event metric.
    """
    r = series.log_returns
    if len(r) == 0:
        raise ValueError(f"{series.ticker}: no return observations to run BOCPD on")

    hazard = 1.0 / 250.0
    config = {
        "hazard": hazard, "model": "NIG", "mu0": 0.0, "kappa0": 1.0, "alpha0": 1.0,
        "beta0_mode": "expanding_welford", "min_burnin_for_variance": 20,
    }
    bocpd_res = bocpd_normal_inverse_gamma(r, hazard=hazard)
    last_idx = len(r) - 1
    forecast_value = bocpd_res.changepoint_prob[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(f"{series.ticker}: no valid BOCPD changepoint probability at the latest available index")

    artifact = _write_parameter_artifact("changepoint_bocpd", series.ticker, config)
    code_hash = _code_tree_hash("changepoint/bocpd.py", "vol/data.py")
    spec_hash = sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(f"changepoint_bocpd::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"changepoint_bocpd_{series.ticker}",
        model_version_hash=version_hash,
        family="BOCPD-NIG",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=["T-CHANGE"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.FAST,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-CHANGE",
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="changepoint_posterior_probability",
        horizon_days=0,
        point_estimate=str(float(forecast_value)),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def build_anomaly_mahalanobis_shadow_record(
    panel: ReturnPanel, *, fold_train_ends: list[int], estimator: str = "ledoit_wolf"
) -> tuple[ModelVersion, ModelOutput]:
    """T-ANOM shadow record for a covariance/Mahalanobis anomaly scorer
    (`anomaly.covariance_models.mahalanobis_d2_series`), the same-day
    anomaly score for the LAST available common date across `panel`.
    Defaults to `ledoit_wolf`, matching T-ANOM's own frozen incumbent name
    `Ledoit_Wolf_Mahalanobis` (`tasks/registry.py`).

    Unlike every other builder in this module, this model's own unit of
    analysis is the CROSS-SECTIONAL panel, not one ticker's own series
    (the Mahalanobis distance is computed from a mean/covariance FIT
    ACROSS ALL TICKERS AT ONCE) -- so `model_id`/`data_snapshot_ids`/
    `input_hashes` here are panel-level, not per-ticker, reflecting what
    the model actually conditions on, rather than forcing it into the
    per-ticker shape every other T-VOL/T-TAIL/T-STATE/T-CHANGE builder
    uses.
    """
    res = mahalanobis_d2_series(panel.returns, fold_train_ends, estimator=estimator)
    last_idx = len(panel.dates) - 1
    forecast_value = res.mahalanobis_d2[last_idx]
    if not (forecast_value == forecast_value):  # NaN check without importing numpy here
        raise ValueError(f"panel: no valid {estimator} anomaly score at the latest available index")

    last_fold = res.fold_fits[-1]
    artifact = _write_parameter_artifact(f"anomaly_{estimator}", "panel", last_fold)
    code_hash = _code_tree_hash("anomaly/covariance_models.py", "anomaly/panel.py")
    config = {"estimator": estimator, "purge_days": 1, "tickers": panel.tickers}
    spec_hash = sha256_bytes(json.dumps(config, sort_keys=True).encode("utf-8"))
    experiment_fingerprint = sha256_bytes(
        f"anomaly_{estimator}::panel::{'|'.join(sorted(panel.dataset_ids.values()))}".encode()
    )
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"anomaly_{estimator}_panel",
        model_version_hash=version_hash,
        family=f"Covariance-Mahalanobis-{estimator}",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=sorted(panel.dataset_ids.values()),
        task_ids=["T-ANOM"],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id="T-ANOM",
        as_of_utc=datetime.fromisoformat(panel.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="anomaly_score",
        horizon_days=0,
        point_estimate=str(float(forecast_value)),
        input_hashes=sorted(panel.data_fingerprints.values()),
        decision_relevant=False,
    )
    return model_version, model_output
