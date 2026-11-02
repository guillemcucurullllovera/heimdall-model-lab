"""Builds the full `EmpiricalEvaluationExport` -- the Model Lab Empirical
Evaluation / Research Validation Layer. Runs REAL evaluations (vol-
targeting backtests, tail capital-efficiency, risk-reduction utility)
against every implemented model; literature candidates are carried
forward as `not_evaluable` (no implementation to evaluate at all).

Reuses `model_registry_data.STATIC_ENTRIES` for descriptive text (what a
model estimates, its inputs/target/horizon) rather than re-authoring it a
third time -- single source of truth for that content remains `model_
registry_data.py`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from model_lab.anomaly.covariance_models import mahalanobis_d2_series
from model_lab.anomaly.isolation_forest_model import isolation_forest_score_series
from model_lab.anomaly.panel import build_return_panel
from model_lab.changepoint.bocpd import bocpd_normal_inverse_gamma
from model_lab.changepoint.cusum import cusum_detect
from model_lab.command_center_export.model_registry_data import STATIC_ENTRIES, StaticEntry
from model_lab.contracts.economic_evaluation import (
    EmpiricalEvaluationExport,
    EvidenceDiscipline,
    EvidenceDisciplineFlag,
    ModelEmpiricalEvaluation,
)
from model_lab.contracts.model_lab_registry import ResearchState
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.economic_evaluation.diagnostics import (
    build_calibration_chart,
    build_predicted_vs_realized_chart,
)
from model_lab.economic_evaluation.risk_reduction_utility import (
    anomaly_risk_reduction_utility,
    changepoint_risk_reduction_utility,
)
from model_lab.economic_evaluation.tail_capital_efficiency import (
    compute_tail_capital_efficiency,
    summarize_capital_efficiency,
)
from model_lab.economic_evaluation.vol_targeting import run_vol_targeting_backtest
from model_lab.research.folds import expanding_fold_train_ends
from model_lab.tasks.registry import SIX_ETFS
from model_lab.vol.baselines import ewma_forecast, naive_trailing_rv_forecast
from model_lab.vol.data import load_return_series
from model_lab.vol.garch_models import egarch_t_forecast, gjr_garch_t_forecast

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"
VOL_MODEL_IDS = ("naive_trailing_rv", "EWMA_0.94", "GARCH_t", "EGARCH_t", "Combo_median_EWMA_GARCH_EGARCH")


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _vol_forecast_for(model_id: str, r: np.ndarray, train_ends: list[int]):
    from model_lab.vol.combination import median_variance_combination

    if model_id == "naive_trailing_rv":
        return naive_trailing_rv_forecast(r, 1)
    if model_id == "EWMA_0.94":
        return ewma_forecast(r, 1, lam=0.94)
    if model_id == "GARCH_t":
        return gjr_garch_t_forecast(r, 1, train_ends).variance_forecast
    if model_id == "EGARCH_t":
        return egarch_t_forecast(r, 1, train_ends).variance_forecast
    if model_id == "Combo_median_EWMA_GARCH_EGARCH":
        ewma_fc = ewma_forecast(r, 1, lam=0.94)
        garch_fc = gjr_garch_t_forecast(r, 1, train_ends).variance_forecast
        egarch_fc = egarch_t_forecast(r, 1, train_ends).variance_forecast
        out = np.full(len(r), np.nan)
        valid = ~np.isnan(ewma_fc) & ~np.isnan(garch_fc) & ~np.isnan(egarch_fc)
        out[valid] = median_variance_combination(ewma_fc[valid], garch_fc[valid], egarch_fc[valid])
        return out
    raise ValueError(model_id)


def _evaluate_vol_or_regime_model(static: StaticEntry, registry: DatasetSnapshotRegistry) -> ModelEmpiricalEvaluation:
    vol_targeting_results = []
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        train_ends, test_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        oos_idx = sorted({i for lst in test_lists for i in lst})

        if static.model_id == "regime_2state":
            from model_lab.regime.markov_switching import fit_and_filter_2state

            variance_forecast = fit_and_filter_2state(r, train_ends).regime_variance_forecast
        else:
            variance_forecast = _vol_forecast_for(static.model_id, r, train_ends)
        naive_forecast = naive_trailing_rv_forecast(r, 1) if static.model_id != "naive_trailing_rv" else None

        result = run_vol_targeting_backtest(
            model_id=static.model_id, ticker=ticker, log_returns=r, variance_forecast=variance_forecast,
            naive_variance_forecast=naive_forecast, oos_indices=oos_idx,
        )
        if result is not None:
            vol_targeting_results.append(result)

    n_beats_naive = sum(
        1 for res in vol_targeting_results
        if res.incremental_vs_naive_targeted and float(res.incremental_vs_naive_targeted["sharpe_net_delta"]) > 0
    )
    n_with_naive_comparison = sum(1 for res in vol_targeting_results if res.incremental_vs_naive_targeted)
    economically_useful = n_with_naive_comparison > 0 and n_beats_naive >= max(1, n_with_naive_comparison // 2)

    chart_ids = []
    if static.model_id in VOL_MODEL_IDS:
        chart_ids = [f"predicted_vs_realized_{t}" for t in SIX_ETFS] + ["mz_calibration_slope_by_ticker"]

    return ModelEmpiricalEvaluation(
        model_id=static.model_id,
        what_it_estimates=static.methodology.split(".")[0] + ".",
        input_data=static.inputs_data,
        target_variable=static.target_variable,
        horizon_description=f"{list(static.horizons_days)} trading day(s)" if static.horizons_days else "1 trading day",
        output_description="Cumulative h-day variance forecast (or, for regime_2state, the regime-blended variance forecast).",
        available_data_range="2016-01-04 through 2026-09-29 (frozen research dataset), 6 ETFs",
        methodology=(
            "Calendar-year expanding walk-forward folds (2016/2017 warm-up, test years 2018-2026), "
            "purge=horizon; the SAME fold boundaries and forecasts this model's own entry in "
            "reports/vol_ladder or reports/regime_value_add already uses -- no new fitting."
        ),
        benchmark_description="naive_trailing_rv-targeted overlay (trivial/classical benchmark) and a fixed weight=1 buy-and-hold.",
        incremental_value_question=(
            f"Does scaling a buy-and-hold single-ETF position's exposure by "
            f"target_vol/forecast_vol using {static.model_id}'s own forecast achieve a higher net Sharpe "
            f"than the same overlay using the naive trailing-RV forecast, or than plain buy-and-hold?"
        ),
        appropriate_economic_use=(
            "Volatility-targeting position-sizing overlay (Moreira & Muir 2017) -- the one legitimate, "
            "literature-grounded economic use of a variance forecast; no directional signal is implied "
            "or required."
        ),
        transaction_cost_assumptions="5bps per unit of daily weight-change (one-way), a disclosed round-number assumption -- not a measured AMC-specific cost.",
        vol_targeting_results=vol_targeting_results,
        evidence_discipline=EvidenceDiscipline(
            implemented=EvidenceDisciplineFlag(value=True, evidence=f"{', '.join(static.implementation_files)}"),
            empirically_evaluated=EvidenceDisciplineFlag(value=bool(vol_targeting_results), evidence=f"{len(vol_targeting_results)}/6 tickers produced a valid vol-targeting backtest"),
            oos_validated=EvidenceDisciplineFlag(value=True, evidence="Statistical OOS evidence exists in reports/vol_ladder or reports/regime_value_add (walk-forward, not in-sample)."),
            economically_useful=EvidenceDisciplineFlag(value=economically_useful, evidence=f"Net Sharpe beats the naive-targeted benchmark in {n_beats_naive}/{n_with_naive_comparison} tickers with a valid comparison."),
            shadow_prospective=EvidenceDisciplineFlag(value=static.model_id == "Combo_median_EWMA_GARCH_EGARCH", evidence="Only MODEL-2 has a real shadow/prospective record." if static.model_id != "Combo_median_EWMA_GARCH_EGARCH" else "6 resolved non-replay + 6 pending observations (reports/prospective/*.json)."),
            amc_eligible=EvidenceDisciplineFlag(value=False, evidence="No AMC decision mapping exists (reports/MODEL_LAB_AMC_DECISION_MAPPING_INVESTIGATION.md)."),
        ),
        diagnostic_chart_ids=chart_ids,
        limitations=list(static.limitations) + ["Vol-targeting is ONE candidate economic use, not the only one, and not an AMC decision mapping."],
        reproducibility_command="uv run python scripts/run_empirical_evaluation.py",
        source_data_refs=[f"reports/vol_ladder/{t}_h1.json" for t in SIX_ETFS] if static.model_id in VOL_MODEL_IDS else [f"reports/regime_value_add/{t}.json" for t in SIX_ETFS],
    )


def _combo_tail_violation_reduction(reports_dir: Path) -> tuple[int, int]:
    """`Combo_median_filtered_GPD_FHS` has NO `fz0_mean` entry anywhere in
    reports/tail_ladder/*.json (confirmed: only unconditional_GPD_POT and
    GJR_t_filtered_GPD_FHS are keyed there) -- reusing the GJR_t-vs-
    unconditional FZ0 comparison for THIS model would misattribute GJR_t's
    own evidence to a different model. Its real, distinct evidence is a
    VaR-violation-count ablation vs GARCH_t in reports/downstream_value_
    probe/_summary.json (`incumbent_GARCH_t_scaled` vs
    `candidate_Combo_median_scaled`, every ticker that has the candidate
    key, all 3 levels)."""
    path = reports_dir / "downstream_value_probe" / "_summary.json"
    if not path.exists():
        return (0, 0)
    d = json.loads(path.read_text())
    n_improved = n_total = 0
    for ticker_data in d.values():
        if "candidate_Combo_median_scaled" not in ticker_data:
            continue
        for level in ("0.05", "0.025", "0.01"):
            inc = ticker_data["incumbent_GARCH_t_scaled"][level]["n_violations"]
            cand = ticker_data["candidate_Combo_median_scaled"][level]["n_violations"]
            n_total += 1
            if cand < inc:
                n_improved += 1
    return (n_improved, n_total)


def _evaluate_tail_model(static: StaticEntry) -> ModelEmpiricalEvaluation:
    is_combo = static.model_id == "Combo_median_filtered_GPD_FHS"
    is_conditional = static.model_id != "unconditional_GPD_POT"

    if is_combo:
        all_results: list = []
        n_improved, n_total = _combo_tail_violation_reduction(REPORTS_DIR)
        benchmark_description = "GARCH_t-scaled FHS (GJR_t_filtered_GPD_FHS's own vol input) -- this model's real, distinct comparator; it has no fz0_mean entry in reports/tail_ladder, so it is NOT compared via the FZ0 capital-efficiency metric used for the other two T-TAIL models."
        incremental_value_question = "Does scaling each day's VaR/ES by the MODEL-2 candidate's own (combo) vol forecast, instead of GARCH_t's alone, reduce the number of VaR violations at the same nominal level (a real, already-computed ablation, reports/downstream_value_probe/_summary.json)?"
        economically_useful = n_total > 0 and n_improved >= max(1, n_total * 2 // 3)
        empirically_evaluated = n_total > 0
        economically_useful_evidence = f"Reduces VaR violation counts in {n_improved}/{n_total} (ticker, level) cells vs GARCH_t-scaled FHS." if n_total else "No downstream_value_probe data found."
        empirically_evaluated_evidence = f"{n_total} (ticker, level) cells evaluated via violation-count ablation (reports/downstream_value_probe/_summary.json)."
        source_data_refs = ["reports/downstream_value_probe/_summary.json"]
    else:
        all_results = []
        for ticker in SIX_ETFS:
            all_results.extend(compute_tail_capital_efficiency(REPORTS_DIR, ticker))
        summary = summarize_capital_efficiency(all_results)
        benchmark_description = "unconditional_GPD_POT (T-TAIL's own naive comparator)."
        incremental_value_question = "For the SAME real, already-backtested coverage, does this model require a smaller (more capital-efficient) VaR/ES buffer than the unconditional model?"
        economically_useful = is_conditional and summary.get("n_cells_conditional_more_efficient", 0) > 9
        empirically_evaluated = bool(all_results)
        economically_useful_evidence = (
            f"Conditional model is more capital-efficient (lower FZ0) in {summary.get('n_cells_conditional_more_efficient', 0)}/18 cells."
            if is_conditional else "Unconditional model is the comparator, not the challenger, for this metric."
        )
        empirically_evaluated_evidence = f"{len(all_results)}/18 (ticker, level) cells evaluated via FZ0 capital-efficiency comparison."
        source_data_refs = [f"reports/tail_ladder/{t}.json" for t in SIX_ETFS]

    return ModelEmpiricalEvaluation(
        model_id=static.model_id,
        what_it_estimates=static.methodology.split(".")[0] + ".",
        input_data=static.inputs_data, target_variable=static.target_variable,
        horizon_description="1 trading day",
        output_description="VaR/ES at the 95%/97.5%/99% levels.",
        available_data_range="2016-01-04 through 2026-09-29, 6 ETFs",
        methodology="Same 9 calendar-year expanding folds as reports/tail_ladder/*.json -- no new fitting.",
        benchmark_description=benchmark_description,
        incremental_value_question=incremental_value_question,
        appropriate_economic_use="Capital-at-risk sizing for a risk desk -- NOT a directional P&L strategy; VaR/ES models estimate loss magnitude at a confidence level, not a trading signal.",
        transaction_cost_assumptions=None,
        tail_capital_efficiency=all_results,
        evidence_discipline=EvidenceDiscipline(
            implemented=EvidenceDisciplineFlag(value=True, evidence=f"{', '.join(static.implementation_files)}"),
            empirically_evaluated=EvidenceDisciplineFlag(value=empirically_evaluated, evidence=empirically_evaluated_evidence),
            oos_validated=EvidenceDisciplineFlag(value=True, evidence="reports/tail_ladder/*.json or reports/downstream_value_probe/*.json -- real walk-forward Kupiec/Christoffersen backtests."),
            economically_useful=EvidenceDisciplineFlag(value=economically_useful, evidence=economically_useful_evidence),
            shadow_prospective=EvidenceDisciplineFlag(value=False, evidence="No prospective pipeline wired for T-TAIL models."),
            amc_eligible=EvidenceDisciplineFlag(value=False, evidence="No AMC decision mapping exists."),
        ),
        diagnostic_chart_ids=[f"tail_coverage_{t}" for t in SIX_ETFS],
        limitations=list(static.limitations),
        reproducibility_command="uv run python scripts/run_empirical_evaluation.py",
        source_data_refs=source_data_refs,
    )


def _evaluate_anomaly_model(static: StaticEntry, registry: DatasetSnapshotRegistry, panel_cache: dict) -> ModelEmpiricalEvaluation:
    if "panel" not in panel_cache:
        panel_cache["panel"] = build_return_panel(list(SIX_ETFS), registry)
        panel_cache["train_ends"], test_lists = expanding_fold_train_ends(panel_cache["panel"].dates, horizon=1, min_train_groups=2)
        panel_cache["oos"] = sorted({i for lst in test_lists for i in lst})
        panel_cache["panel_returns"] = panel_cache["panel"].returns.mean(axis=1)
    panel, train_ends, oos, panel_returns = panel_cache["panel"], panel_cache["train_ends"], panel_cache["oos"], panel_cache["panel_returns"]

    if static.model_id == "Isolation_Forest":
        scores = isolation_forest_score_series(panel.returns, train_ends).anomaly_score
    else:
        scores = mahalanobis_d2_series(panel.returns, train_ends, estimator=static.model_id).mahalanobis_d2

    utility = anomaly_risk_reduction_utility(static.model_id, scores, panel_returns, oos)

    return ModelEmpiricalEvaluation(
        model_id=static.model_id,
        what_it_estimates=static.methodology.split(".")[0] + ".",
        input_data=static.inputs_data, target_variable=static.target_variable,
        horizon_description="0 (same-day score)",
        output_description="Cross-sectional anomaly score (Mahalanobis distance or Isolation Forest score) for the six-ETF panel.",
        available_data_range="2016-01-04 through 2026-09-29, 6-ETF panel (inner-joined dates)",
        methodology="Same periodic-refit walk-forward as reports/anomaly_ladder/six_etf_panel.json -- no new fitting.",
        benchmark_description="The other 4 real covariance/IF estimators (T-ANOM's own naive, OCSVM_legacy, has no implementation).",
        incremental_value_question="If exposure were cut to zero on this model's own top-decile-flagged days, would the equal-weight panel's forward return have been worse on those days than on unflagged days (a real risk-avoidance signal)?",
        appropriate_economic_use="Cross-sectional risk-reduction flag (exposure cut, not a directional trade) -- NOT a P&L strategy; an anomaly score has no sign/direction.",
        transaction_cost_assumptions=None,
        risk_reduction_utility=utility,
        evidence_discipline=EvidenceDiscipline(
            implemented=EvidenceDisciplineFlag(value=True, evidence=f"{', '.join(static.implementation_files)}"),
            empirically_evaluated=EvidenceDisciplineFlag(value=utility is not None, evidence="anomaly_risk_reduction_utility computed against the real equal-weight panel return." if utility else "Insufficient flagged/unflagged days."),
            oos_validated=EvidenceDisciplineFlag(value=True, evidence="reports/anomaly_ladder/six_etf_panel.json -- real walk-forward scores (T-ANOM itself remains data-BLOCKED for a validated benchmark)."),
            economically_useful=EvidenceDisciplineFlag(
                value=bool(utility and float(utility.mean_forward_return_flagged_days) < float(utility.mean_forward_return_unflagged_days)),
                evidence=utility.interpretation if utility else "No utility result.",
            ),
            shadow_prospective=EvidenceDisciplineFlag(value=False, evidence="No prospective pipeline wired for T-ANOM models."),
            amc_eligible=EvidenceDisciplineFlag(value=False, evidence="No AMC decision mapping exists."),
        ),
        diagnostic_chart_ids=["anomaly_illustrative_dates"],
        limitations=list(static.limitations) + ["Risk-reduction utility uses a descriptive (non-causal) top-decile threshold over the full OOS distribution -- same convention as the existing illustrative-date check, not a forecast claim."],
        reproducibility_command="uv run python scripts/run_empirical_evaluation.py",
        source_data_refs=["reports/anomaly_ladder/six_etf_panel.json"],
    )


def _evaluate_changepoint_model(static: StaticEntry, registry: DatasetSnapshotRegistry) -> ModelEmpiricalEvaluation:
    all_flagged_returns: list[float] = []
    all_unflagged_returns: list[float] = []
    for ticker in SIX_ETFS:
        series = load_return_series(ticker, registry)
        r = series.log_returns
        dates = series.dates[1:]
        train_ends, test_lists = expanding_fold_train_ends(dates, horizon=1, min_train_groups=2)
        oos = sorted({i for lst in test_lists for i in lst})
        if static.model_id == "CUSUM":
            values = cusum_detect(r, window=60, drift=0.5, threshold=5.0).alarm
            is_prob = False
        else:
            values = bocpd_normal_inverse_gamma(r, hazard=1.0 / 250.0).changepoint_prob
            is_prob = True
        result = changepoint_risk_reduction_utility(static.model_id, values, r, oos, is_probability=is_prob)
        if result is None:
            continue
        valid = [i for i in oos if i + 1 < len(r) and not np.isnan(values[i])]
        idx = np.array(valid)
        if is_prob:
            thr = np.percentile(values[idx], 90)
            flagged = values[idx] >= thr
        else:
            flagged = values[idx].astype(bool)
        forward = r[idx + 1]
        all_flagged_returns.extend(forward[flagged].tolist())
        all_unflagged_returns.extend(forward[~flagged].tolist())

    utility = None
    if len(all_flagged_returns) >= 5 and len(all_unflagged_returns) >= 5:
        from model_lab.contracts.economic_evaluation import RiskReductionUtilityResult

        mean_f, mean_u = float(np.mean(all_flagged_returns)), float(np.mean(all_unflagged_returns))
        utility = RiskReductionUtilityResult(
            model_id=static.model_id, mean_forward_return_flagged_days=str(mean_f), mean_forward_return_unflagged_days=str(mean_u),
            n_flagged=len(all_flagged_returns), n_unflagged=len(all_unflagged_returns),
            hypothetical_avoided_loss_if_flat_on_flagged_days=str(float(np.sum(np.clip(all_flagged_returns, None, 0.0)))),
            interpretation=(
                f"Pooled across all 6 tickers: on {len(all_flagged_returns)} flagged days, mean next-day "
                f"log return was {mean_f:.5f}, vs {mean_u:.5f} on {len(all_unflagged_returns)} unflagged days. "
                + ("Flagged days are worse on average -- consistent with a risk-avoidance use." if mean_f < mean_u
                   else "Flagged days are NOT worse on average in this sample -- no risk-avoidance value demonstrated here.")
            ),
        )

    return ModelEmpiricalEvaluation(
        model_id=static.model_id,
        what_it_estimates=static.methodology.split(".")[0] + ".",
        input_data=static.inputs_data, target_variable=static.target_variable,
        horizon_description="0 (same-day online statistic)",
        output_description="CUSUM alarm flag (boolean) or BOCPD changepoint probability, per ticker per day.",
        available_data_range="2016-01-04 through 2026-09-29, 6 ETFs (evaluated individually, pooled for this result)",
        methodology="Purely online/causal detector (no fold machinery); same real series as reports/changepoint_ladder/*.json.",
        benchmark_description="The other changepoint detector (CUSUM vs BOCPD) -- both T-CHANGE-relevant.",
        incremental_value_question="Pooled across all 6 tickers, are days this detector flags followed by worse-than-average next-day returns (a real risk-avoidance signal)?",
        appropriate_economic_use="Single-ticker risk-reduction flag (exposure cut, not a directional trade) -- NOT a P&L strategy.",
        transaction_cost_assumptions=None,
        risk_reduction_utility=utility,
        evidence_discipline=EvidenceDiscipline(
            implemented=EvidenceDisciplineFlag(value=True, evidence=f"{', '.join(static.implementation_files)}"),
            empirically_evaluated=EvidenceDisciplineFlag(value=utility is not None, evidence="Pooled risk-reduction utility computed across all 6 tickers." if utility else "Insufficient flagged/unflagged days."),
            oos_validated=EvidenceDisciplineFlag(value=True, evidence="reports/changepoint_ladder/*.json -- real, purely-causal online detection (T-CHANGE itself remains data-BLOCKED for a validated benchmark)."),
            economically_useful=EvidenceDisciplineFlag(
                value=bool(utility and float(utility.mean_forward_return_flagged_days) < float(utility.mean_forward_return_unflagged_days)),
                evidence=utility.interpretation if utility else "No utility result.",
            ),
            shadow_prospective=EvidenceDisciplineFlag(value=False, evidence="No prospective pipeline wired for T-CHANGE models."),
            amc_eligible=EvidenceDisciplineFlag(value=False, evidence="No AMC decision mapping exists."),
        ),
        diagnostic_chart_ids=["changepoint_bocpd_vs_cusum"],
        limitations=list(static.limitations) + ["Pooled across tickers for statistical power -- per-ticker results vary (see reports/MODEL_LAB_EMPIRICAL_EVALUATION.md)."],
        reproducibility_command="uv run python scripts/run_empirical_evaluation.py",
        source_data_refs=[f"reports/changepoint_ladder/{t}.json" for t in SIX_ETFS],
    )


def build_empirical_evaluation_export() -> EmpiricalEvaluationExport:
    registry = DatasetSnapshotRegistry.load_from_directory()
    evaluations: dict[str, ModelEmpiricalEvaluation] = {}
    not_evaluable: dict[str, str] = {}
    panel_cache: dict = {}

    anomaly_ids = {"empirical", "ledoit_wolf", "oas", "min_cov_det", "Isolation_Forest"}
    tail_ids = {"unconditional_GPD_POT", "GJR_t_filtered_GPD_FHS", "Combo_median_filtered_GPD_FHS"}
    changepoint_ids = {"CUSUM", "BOCPD"}
    vol_regime_ids = set(VOL_MODEL_IDS) | {"regime_2state"}

    for static in STATIC_ENTRIES:
        if static.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
            not_evaluable[static.model_id] = (
                "No implementation exists -- there is no model output to empirically evaluate. "
                "See reports/MODEL_LAB_FORENSIC_INVENTORY.md."
            )
            continue
        if static.model_id in vol_regime_ids:
            evaluations[static.model_id] = _evaluate_vol_or_regime_model(static, registry)
        elif static.model_id in tail_ids:
            evaluations[static.model_id] = _evaluate_tail_model(static)
        elif static.model_id in anomaly_ids:
            evaluations[static.model_id] = _evaluate_anomaly_model(static, registry, panel_cache)
        elif static.model_id in changepoint_ids:
            evaluations[static.model_id] = _evaluate_changepoint_model(static, registry)
        else:
            not_evaluable[static.model_id] = "Not yet mapped to an empirical-evaluation family in this increment."

    charts = [build_predicted_vs_realized_chart(registry, t) for t in SIX_ETFS]
    charts.append(build_calibration_chart(REPORTS_DIR, tuple(SIX_ETFS)))

    payload = {
        "git_commit": _git_commit(),
        "evaluations": {k: v.model_dump(mode="json") for k, v in evaluations.items()},
        "not_evaluable": not_evaluable,
        "charts": [c.model_dump(mode="json") for c in charts],
    }
    from model_lab.canon.jcs import sha256_bytes

    export_sha = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))

    return EmpiricalEvaluationExport(
        git_commit=_git_commit(), evaluations=evaluations, not_evaluable=not_evaluable,
        charts=charts, export_sha256=export_sha,
    )
