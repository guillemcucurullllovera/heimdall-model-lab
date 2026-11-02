"""Shadow records for the MODEL-2G candidate (`Combo_median_EWMA_GARCH_EGARCH`,
see `vol/candidate_spec.py` for its locked definition). Follows the EXACT
same `ModelVersion`/`ModelOutput` construction pattern `shadow/build.py`
already established for `GARCH_t`/`GJR_t_filtered_GPD_FHS` -- not a new ad
hoc identity scheme.

Horizon is fixed at 1 (matching the existing `run_shadow_snapshot.py`'s own
"latest available day" daily-cadence design). The candidate is built for
EVERY ticker, not only the 3 (TLT/IEF/GLD) where it has a demonstrated edge
at h=1 -- `is_eligible_cell` is recorded as metadata on the output rather
than used to refuse construction, so prospective monitoring (Phase 5) can
honestly observe the candidate's behaviour everywhere, not only where it
already won, which would otherwise bias future evidence collection toward
confirming the existing result rather than genuinely testing it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.contracts.model_output import ModelOutput
from model_lab.contracts.model_version import (
    LifecycleState,
    ModelVersion,
    ParameterArtifact,
    RefitLane,
    compute_model_version_hash,
)
from model_lab.contracts.task import InfluenceTier
from model_lab.vol.baselines import ewma_forecast
from model_lab.vol.candidate_spec import (
    CANDIDATE_COMBINATION_RULE,
    CANDIDATE_CONSTITUENTS,
    CANDIDATE_MODEL_ID,
    CANDIDATE_TASK_ID,
    is_eligible_cell,
)
from model_lab.vol.combination import median_variance_combination
from model_lab.vol.data import ReturnSeries
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

SHADOW_STORE = Path(__file__).resolve().parents[3] / "reports" / "shadow"
_SRC_ROOT = Path(__file__).resolve().parents[1]
HORIZON = 1


def _code_tree_hash(*relative_paths: str) -> str:
    data = b""
    for rel in sorted(relative_paths):
        data += (_SRC_ROOT / rel).read_bytes()
    return sha256_bytes(data)


def _write_parameter_artifact(ticker: str, payload: dict) -> ParameterArtifact:
    SHADOW_STORE.mkdir(parents=True, exist_ok=True)
    out_path = SHADOW_STORE / f"vol_combo_median_{ticker}_params.json"
    content = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    out_path.write_text(content, encoding="utf-8")
    digest = sha256_bytes(content.encode("utf-8"))
    return ParameterArtifact(path=str(out_path.relative_to(SHADOW_STORE.parents[1])), sha256=digest)


def build_combo_shadow_records(series: ReturnSeries, *, fold_train_ends: list[int]) -> tuple[ModelVersion, ModelOutput]:
    """Shadow `ModelVersion`/`ModelOutput` for the median-combination
    candidate's 1-day-ahead variance forecast at the LAST available index
    in `series`. Raises (fails closed) if any of the three constituents
    lacks a valid forecast there -- never fabricates a combination from
    fewer than 3 members."""
    garch_res = gjr_garch_t_forecast(series.log_returns, HORIZON, fold_train_ends)
    egarch_res = egarch_t_forecast(series.log_returns, HORIZON, fold_train_ends)
    ewma_fc = ewma_forecast(series.log_returns, HORIZON, lam=0.94)

    last_idx = len(series.log_returns) - 1
    members = {
        "EWMA_0.94": ewma_fc[last_idx],
        "GARCH_t": garch_res.variance_forecast[last_idx],
        "EGARCH_t": egarch_res.variance_forecast[last_idx],
    }
    missing = [m for m, v in members.items() if v != v]  # NaN check without importing numpy
    if missing:
        raise ValueError(
            f"{series.ticker}: candidate requires all 3 constituents to have a valid forecast at "
            f"the latest available index; missing: {missing}"
        )
    combo_value = float(median_variance_combination(*[members[m] for m in CANDIDATE_CONSTITUENTS]))

    last_fold_garch = garch_res.fold_params[-1]
    last_fold_egarch = egarch_res.fold_params[-1]
    artifact_payload = {
        "combination_rule": CANDIDATE_COMBINATION_RULE,
        "constituents": {m: str(members[m]) for m in CANDIDATE_CONSTITUENTS},
        "garch_t_fold_params": last_fold_garch,
        "egarch_t_fold_params": last_fold_egarch,
    }
    artifact = _write_parameter_artifact(series.ticker, artifact_payload)
    code_hash = _code_tree_hash(
        "vol/combination.py", "vol/candidate_spec.py", "vol/garch_models.py", "vol/baselines.py", "vol/data.py"
    )
    spec_hash = sha256_bytes(
        json.dumps({"rule": CANDIDATE_COMBINATION_RULE, "constituents": CANDIDATE_CONSTITUENTS}, sort_keys=True).encode()
    )
    experiment_fingerprint = sha256_bytes(f"{CANDIDATE_MODEL_ID}::{series.ticker}::{series.dataset_id}".encode())
    version_hash = compute_model_version_hash(experiment_fingerprint, [artifact.sha256])

    model_version = ModelVersion(
        model_id=f"{CANDIDATE_MODEL_ID}_{series.ticker}",
        model_version_hash=version_hash,
        family="median-forecast-combination",
        spec_hash=spec_hash,
        code_tree_hash=code_hash,
        parameter_artifacts=[artifact],
        data_snapshot_ids=[series.dataset_id],
        task_ids=[CANDIDATE_TASK_ID],
        lifecycle_state=LifecycleState.SHADOW,
        influence_tier=InfluenceTier.T0,
        tier_ceiling_signed=False,
        refit_lane=RefitLane.SLOW,
    )
    now = datetime.now(UTC)
    model_output = ModelOutput(
        model_version_hash=version_hash,
        task_id=CANDIDATE_TASK_ID,
        as_of_utc=datetime.fromisoformat(series.dates[-1]).replace(tzinfo=UTC),
        produced_at_utc=now,
        target_variable="forward_realised_variance",
        horizon_days=HORIZON,
        point_estimate=str(combo_value),
        input_hashes=[series.data_fingerprint],
        decision_relevant=False,
    )
    return model_version, model_output


def candidate_is_eligible_for(ticker: str) -> bool:
    """Whether (ticker, HORIZON=1) is one of the 9 cells with a demonstrated
    edge -- metadata only, never used to gate construction (see module
    docstring)."""
    return is_eligible_cell(ticker, HORIZON)
