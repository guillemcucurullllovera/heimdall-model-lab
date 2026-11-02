"""Task/1 registry — MODEL1_foundation_contracts.md §8, populated from the
readiness pack's §4 data-readiness matrix and §19 preregistration drafts
(MODEL1_READINESS_PACK.md, MODEL1_data_readiness.md).

Eight frozen task specs: T-VOL (C1), T-TAIL (C2), T-STATE and T-DOWNSTREAM
(C3), T-ANOM (C4), T-CHANGE (C3 evidence), T-CAL (cross-cutting), T-VRP (C5).

## P0-1/P0-4 remediation (adversarial review findings C.1, C.4)

Every `Task` here is now built via `build_task(dataset_registry=registry,
...)` (see contracts/task.py) — direct `Task(**kwargs)` construction is
refused. `build_default_task_registry(registry=...)` defaults to
`DatasetSnapshotRegistry.load_from_directory()` (the REAL, on-disk data
estate) so a task's `DataRequirement.dataset_id`/`research_state` claims are
always cross-checked against actual `DatasetSnapshot`s, never hand-typed.

ETF bar requirements are now PER-TICKER (six separate `DataRequirement`s,
not one bundled "etf_daily_ohlcv" role) because the real data's status
genuinely differs per ticker: five tickers (SPY, QQQ, TLT, IEF, GLD) are
safe to use raw (zero corporate actions in-window, confirmed by a real
`corporate_actions_<TICKER>` snapshot); USO specifically required an actual
split adjustment and resolves to the derived `uso_daily_split_adjusted`
snapshot instead of the raw (contaminated) one — see
`ETF_COVERAGE_MATRIX.md`'s P0-4 section. A single bundled requirement could
not express this per-instrument distinction mechanically.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from model_lab.contracts.task import (
    DataRequirement,
    DataRequirementStatus,
    GuardSpec,
    InfluenceTier,
    MdeSpec,
    ScoreSpec,
    SplitterScheme,
    SplitterSpec,
    Task,
    TaskTarget,
    TaskUniverse,
    build_task,
    derive_promotion_class,
    derive_task_status,
)
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry

SIX_ETFS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "USO"]

FROZEN_AT = datetime(2026, 9, 30, tzinfo=UTC)

# Real, on-disk snapshot_ids (heimdall-model-lab-data/snapshots/) that back
# the six-ETF universe today. USO alone points at the derived, split-adjusted
# snapshot (P0-4); the other five use their raw Alpaca snapshot directly,
# each backed by a companion corporate-actions snapshot confirming zero
# events in-window (see DataRequirement.adjustment_dataset_id).
ETF_PRIMARY_DATASET_ID = {
    "SPY": "alpaca_SPY_daily_raw",
    "QQQ": "alpaca_QQQ_daily_raw",
    "TLT": "alpaca_TLT_daily_raw",
    "IEF": "alpaca_IEF_daily_raw",
    "GLD": "alpaca_GLD_daily_raw",
    "USO": "uso_daily_split_adjusted",
}
ETF_CORPORATE_ACTIONS_DATASET_ID = {ticker: f"corporate_actions_{ticker}" for ticker in SIX_ETFS}

MACRO_PIT_DATASET_IDS = {
    "growth": "macro_pit_INDPRO",
    "inflation_headline": "macro_pit_CPIAUCSL",
    "inflation_breakeven": "macro_pit_T10YIE",
    "labor_unemployment": "macro_pit_UNRATE",
    "labor_sahm": "macro_pit_SAHMREALTIME",
    "policy_rate": "macro_pit_DFF",
    "yield_3m": "macro_pit_DGS3MO",
    "yield_2y": "macro_pit_DGS2",
    "yield_10y": "macro_pit_DGS10",
    "financial_conditions": "macro_pit_NFCI",
    "liquidity_m2": "macro_pit_M2SL",
}


def _spec_hash(task_id: str, salt: str) -> str:
    return hashlib.sha256(f"{task_id}::{salt}".encode()).hexdigest()


def _etf_requirement(
    ticker: str, registry: DatasetSnapshotRegistry, *, min_depth_years: float = 10.0
) -> DataRequirement:
    """One real, per-ticker ETF-bar requirement, registry-resolved. Falls
    back to an UNRESOLVED (dataset_id=None, BLOCKED) requirement if the
    expected snapshot isn't actually registered (e.g. a stale data estate) --
    never fabricates a status against a missing snapshot."""
    dataset_id = ETF_PRIMARY_DATASET_ID[ticker]
    ca_dataset_id = ETF_CORPORATE_ACTIONS_DATASET_ID[ticker]
    resolved = registry.resolve(dataset_id)
    ca_resolved = registry.resolve(ca_dataset_id)

    if resolved is None:
        return DataRequirement(
            dataset_role=f"etf_daily_ohlcv_{ticker}",
            dataset_id=None,
            min_depth_years=min_depth_years,
            preferred_depth_years=25.0,
            frequency="daily",
            pit_required=False,
            adjustments=["corporate_actions_snapshot"],
            status=DataRequirementStatus.BLOCKED,
            research_state=None,
            blockers=[f"expected snapshot {dataset_id!r} is not registered in the data estate"],
        )

    return DataRequirement(
        dataset_role=f"etf_daily_ohlcv_{ticker}",
        dataset_id=dataset_id,
        min_depth_years=min_depth_years,
        preferred_depth_years=25.0,
        frequency="daily",
        pit_required=False,
        adjustments=["corporate_actions_snapshot"],
        adjustment_dataset_id=ca_dataset_id if ca_resolved is not None else None,
        # P1 remediation (MODEL1_REMEDIATION_REREVIEW.md sec I): the companion's
        # zero-event claim is now content-checked, scoped to this exact ticker.
        instrument=ticker if ca_resolved is not None else None,
        status=DataRequirementStatus.READY,
        research_state=resolved.research_state,
        blockers=[],
    )


def build_default_task_registry(registry: DatasetSnapshotRegistry | None = None) -> dict[str, Task]:
    registry = registry if registry is not None else DatasetSnapshotRegistry.load_from_directory()
    tasks: dict[str, Task] = {}

    # --- T-VOL (C1): volatility forecast ladder ----------------------------
    reqs = [_etf_requirement(t, registry) for t in SIX_ETFS] + [
        DataRequirement(
            dataset_role="implied_vol_regressors",
            min_depth_years=10.0,
            frequency="daily",
            pit_required=False,
            status=DataRequirementStatus.BLOCKED,
            blockers=["SPX/VIX family RESTRICTED/UNRESOLVED (Cboe/S&P); Yahoo PROHIBITED"],
        ),
    ]
    status = derive_task_status(reqs)
    tasks["T-VOL"] = build_task(
        dataset_registry=registry,
        task_id="T-VOL",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-VOL", "v2"),
        target=TaskTarget(
            variable="forward_realised_variance",
            horizon_days=[1, 5, 21, 45],
            aggregation="sum_of_squared_returns_or_range_proxy",
            label_end_time_rule="t+h close",
            knowledge_time_rule="close_plus_latency",
            units="variance (per-day)",
        ),
        universe=TaskUniverse(instruments=SIX_ETFS),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="cpcv_6_2", scheme=SplitterScheme.CPCV, embargo_rule="max(h, publication_lag)",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=["EWMA_0.94", "PDV", "GARCH_t"],
        naive="trailing_realised_variance",
        primary_score=ScoreSpec(name="QLIKE", orientation="lower_is_better", proper=True),
        guards=[GuardSpec(name="calibration_slope", margin=0.2)],
        secondary_scores=[ScoreSpec(name="log_score", orientation="higher_is_better", proper=True)],
        mde=MdeSpec(
            table_id="R4_5a_T-VOL",
            assumptions="10y single-ETF gaps >=2-5% (h=1) / >=7.5-21% (h=5) resolvable at 80% power",
            n_available=0.0,
            n_required_80pct=None,
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    # --- T-TAIL (C2): conditional EVT / FHS --------------------------------
    reqs = [_etf_requirement(t, registry) for t in SIX_ETFS]
    status = derive_task_status(reqs)
    tasks["T-TAIL"] = build_task(
        dataset_registry=registry,
        task_id="T-TAIL",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-TAIL", "v2"),
        target=TaskTarget(
            variable="var_es_forecast",
            horizon_days=[1],
            aggregation="quantile",
            label_end_time_rule="t+1 return",
            knowledge_time_rule="close_plus_latency",
            units="return quantile",
        ),
        universe=TaskUniverse(instruments=SIX_ETFS),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="expanding_daily", scheme=SplitterScheme.EXPANDING, embargo_rule="1d",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=["GJR_t_filtered_GPD_FHS"],
        naive="unconditional_GPD_POT",
        primary_score=ScoreSpec(name="FZ0", orientation="lower_is_better", proper=True),
        guards=[GuardSpec(name="LR_cc_rejection_rate", margin=0.25)],
        secondary_scores=[],
        mde=MdeSpec(
            table_id="R4_5a_T-TAIL",
            assumptions="Kupiec 1.5x violation rate SUFFICIENT at 95/97.5% with 0.8-1.5k days; "
            "99% MARGINAL at ~3.8k days; independence test at 99% BLOCKED below ~9k days",
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    # --- T-STATE / T-DOWNSTREAM (C3) ---------------------------------------
    def _macro_pit_requirement() -> DataRequirement:
        """Bundled macro PIT requirement -- kept as ONE requirement (not 11
        per-field ones) for Task-level purposes, but its status is now
        MECHANICALLY derived from each field's real, registered
        `DatasetSnapshot.pit_class` (P1-3 remediation, adversarial review
        finding D.3: "No code path connects the 11 per-field... flags to
        this bundled requirement's status"). `pit_class` is a structural
        contract field (contracts/dataset_snapshot.py), not a parsed report
        -- `AS_RELEASED` means genuine multi-vintage PIT depth was obtained;
        `SNAPSHOT_ONLY` (the FRED 2000-vintage-cap degradation case, finding
        D.4) means only a single, current vintage exists for that field.

        Rule: BLOCKED if any field's snapshot is missing; PARTIAL if all 11
        resolve but at least one is SNAPSHOT_ONLY (not every field has
        genuine historical PIT depth); READY only if all 11 are AS_RELEASED.

        Known limitation (documented, not hidden): this requirement's own
        `dataset_id`/`research_state` stay None because it represents 11
        underlying datasets, not one -- `DataRequirement` has no multi-
        dataset construct yet, so this bundle currently cannot itself
        contribute to `derive_promotion_class`'s PROMOTION_READY check even
        though every field is genuinely CLEAR/promotion_eligible. This has
        no effect on T-STATE/T-DOWNSTREAM's current BLOCKED status (driven
        by the separate, still-BLOCKED market_state_features requirement
        regardless), but would need a real fix (e.g. a `dataset_ids: list`
        field) before this bundle could ever gate a task's promotion class
        on its own.
        """
        snapshots_by_field = {field: registry.get_snapshot(sid) for field, sid in MACRO_PIT_DATASET_IDS.items()}
        missing = [f for f, s in snapshots_by_field.items() if s is None]
        if missing:
            return DataRequirement(
                dataset_role="macro_pit_panel",
                dataset_id=None,
                min_depth_years=5.0,
                frequency="mixed_daily_monthly",
                pit_required=True,
                status=DataRequirementStatus.BLOCKED,
                blockers=[f"missing macro PIT snapshot(s) for field(s): {missing}"],
            )
        snapshot_only_fields = [
            f for f, s in snapshots_by_field.items() if s is not None and s.pit_class == "SNAPSHOT_ONLY"
        ]
        if snapshot_only_fields:
            return DataRequirement(
                dataset_role="macro_pit_panel",
                dataset_id=None,
                min_depth_years=5.0,
                frequency="mixed_daily_monthly",
                pit_required=True,
                status=DataRequirementStatus.PARTIAL,
                blockers=[
                    f"{len(snapshot_only_fields)}/11 fields have pit_class=SNAPSHOT_ONLY (single current "
                    f"vintage only, no genuine historical PIT depth): {sorted(snapshot_only_fields)}"
                ],
            )
        return DataRequirement(
            dataset_role="macro_pit_panel",
            dataset_id=None,
            min_depth_years=5.0,
            frequency="mixed_daily_monthly",
            pit_required=True,
            status=DataRequirementStatus.READY,
            blockers=[],
        )

    macro_pit_req = _macro_pit_requirement()
    market_state_req = DataRequirement(
        dataset_role="market_state_features",
        min_depth_years=10.0,
        frequency="daily",
        pit_required=False,
        status=DataRequirementStatus.BLOCKED,
        blockers=["VIX/VIX3M/VVIX/SKEW RESTRICTED; HY OAS PROHIBITED; BAA10Y UNRESOLVED"],
    )
    for tid, target_var in (("T-STATE", "regime_state_probability"), ("T-DOWNSTREAM", "downstream_vol_tail_improvement")):
        reqs = [macro_pit_req, market_state_req]
        status = derive_task_status(reqs)
        tasks[tid] = build_task(
            dataset_registry=registry,
            task_id=tid,
            version=1,
            frozen_at=FROZEN_AT,
            spec_sha256=_spec_hash(tid, "v2"),
            target=TaskTarget(
                variable=target_var,
                horizon_days=[1],
                aggregation="filtered_probability" if tid == "T-STATE" else "delta_qlike_fz0",
                label_end_time_rule="t close",
                knowledge_time_rule="filtered_only_no_smoothed",
                units="probability" if tid == "T-STATE" else "score delta",
            ),
            universe=TaskUniverse(instruments=["SPX_proxy"] + SIX_ETFS),
            data_requirements=reqs,
            splitter=SplitterSpec(
                config_id="expanding_daily", scheme=SplitterScheme.EXPANDING, embargo_rule="1d",
                purge_rule="afml_3cond_label_span",
            ),
            incumbents=["GJR_t_filter_nonregime"],
            naive="unconditional_student_t",
            primary_score=ScoreSpec(name="delta_QLIKE_FZ0", orientation="higher_is_better", proper=False),
            guards=[GuardSpec(name="ARI_across_restarts", margin=0.8)],
            secondary_scores=[],
            mde=MdeSpec(
                table_id="R4_5a_T-STATE",
                assumptions="exit-probability calibration BLOCKED (115-2400 spells needed vs 42-51 available); "
                "TVTP with observable covariate SUFFICIENT at ~3k days",
            ),
            status=status,
            promotion_class=derive_promotion_class(status, reqs),
            amc_seams_allowed=[],
            influence_ceiling=InfluenceTier.T0,
        )

    # --- T-ANOM (C4) ---------------------------------------------------------
    reqs = [
        DataRequirement(
            dataset_role="cross_asset_feature_panel",
            min_depth_years=20.0,
            frequency="daily",
            pit_required=False,
            status=DataRequirementStatus.BLOCKED,
            blockers=[
                "six-ETF panel is technically buildable today from already-acquired bars "
                "(P2 correction, adversarial review finding E.4 -- stale 'impossible today' wording "
                "removed); not yet built in this remediation pass"
            ],
        ),
        DataRequirement(
            dataset_role="frozen_stress_event_labels",
            min_depth_years=20.0,
            frequency="daily",
            pit_required=False,
            status=DataRequirementStatus.BLOCKED,
            blockers=["no pre-registered financial anomaly benchmark with labelled events exists"],
        ),
    ]
    status = derive_task_status(reqs)
    tasks["T-ANOM"] = build_task(
        dataset_registry=registry,
        task_id="T-ANOM",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-ANOM", "v2"),
        target=TaskTarget(
            variable="anomaly_score",
            horizon_days=[0],
            aggregation="mahalanobis_distance",
            label_end_time_rule="t close",
            knowledge_time_rule="close_plus_latency",
            units="score",
        ),
        universe=TaskUniverse(instruments=SIX_ETFS),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="expanding_daily", scheme=SplitterScheme.EXPANDING, embargo_rule="1d",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=["Ledoit_Wolf_Mahalanobis", "Isolation_Forest"],
        naive="OCSVM_legacy",
        primary_score=ScoreSpec(name="AUPRC", orientation="higher_is_better", proper=False, bounds=(0.0, 1.0)),
        guards=[GuardSpec(name="false_alarms_per_year", margin=0.05)],
        secondary_scores=[],
        mde=MdeSpec(
            table_id="R4_5a_T-ANOM",
            assumptions="VIX-spike detection SUFFICIENT at 5y for 0.05 AUPRC effect; other 3 event sets need 31-37y",
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    # --- T-CHANGE (C3 evidence) ---------------------------------------------
    reqs = [market_state_req]
    status = derive_task_status(reqs)
    tasks["T-CHANGE"] = build_task(
        dataset_registry=registry,
        task_id="T-CHANGE",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-CHANGE", "v2"),
        target=TaskTarget(
            variable="changepoint_detection_delay",
            horizon_days=[0],
            aggregation="delay_days",
            label_end_time_rule="event date",
            knowledge_time_rule="causal_filtered_only",
            units="days",
        ),
        universe=TaskUniverse(instruments=["SPX_proxy"]),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="expanding_daily", scheme=SplitterScheme.EXPANDING, embargo_rule="1d",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=["BOCPD", "Fearnhead"],
        naive="CUSUM",
        primary_score=ScoreSpec(name="detection_delay", orientation="lower_is_better", proper=False),
        guards=[],
        secondary_scores=[],
        mde=MdeSpec(
            table_id="R4_5a_T-CHANGE",
            assumptions="detector-vs-detector delay differences BLOCKED (54-150 labelled changes needed vs 6-12)",
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    # --- T-CAL (cross-cutting) ------------------------------------------------
    reqs = [
        DataRequirement(
            dataset_role="resolved_llm_forecast_outcomes",
            min_depth_years=1.0,
            frequency="per_cycle",
            pit_required=False,
            status=DataRequirementStatus.BLOCKED,
            blockers=["0 resolved forecasts as of 2026-09-30; accrual ~1-2 independent forecasts/week"],
        )
    ]
    status = derive_task_status(reqs)
    tasks["T-CAL"] = build_task(
        dataset_registry=registry,
        task_id="T-CAL",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-CAL", "v2"),
        target=TaskTarget(
            variable="llm_confidence_calibration",
            horizon_days=[0],
            aggregation="ece",
            label_end_time_rule="outcome resolution",
            knowledge_time_rule="resolution_time",
            units="ECE",
        ),
        universe=TaskUniverse(instruments=["AMC_journal"]),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="expanding_daily", scheme=SplitterScheme.EXPANDING, embargo_rule="1d",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=[],
        naive="uniform_confidence",
        primary_score=ScoreSpec(name="ECE", orientation="lower_is_better", proper=False, bounds=(0.0, 1.0)),
        guards=[],
        secondary_scores=[],
        mde=MdeSpec(
            table_id="R4_5a_T-CAL",
            assumptions="ECE<0.05 resolution needs ~400 independent forecasts; slope 0.8 vs 1 needs ~3200 (~41y at current accrual)",
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    # --- T-VRP (C5) ------------------------------------------------------------
    reqs = [
        DataRequirement(
            dataset_role="option_full_chains_stress_windows",
            min_depth_years=14.0,
            frequency="eod_plus_intraday_snapshots",
            pit_required=True,
            status=DataRequirementStatus.BLOCKED,
            blockers=[
                "archive holds 1,472 leg-records from a single date (2026-09-08), 3 of 6 ETFs -- "
                "~2 independent snapshots, not meaningful time-series history (adversarial review "
                "finding M); nowhere near stress-window full chains"
            ],
        )
    ]
    status = derive_task_status(reqs)
    tasks["T-VRP"] = build_task(
        dataset_registry=registry,
        task_id="T-VRP",
        version=1,
        frozen_at=FROZEN_AT,
        spec_sha256=_spec_hash("T-VRP", "v2"),
        target=TaskTarget(
            variable="variance_risk_premium",
            horizon_days=[21],
            aggregation="implied_minus_forecast_variance",
            label_end_time_rule="option expiry",
            knowledge_time_rule="vendor_quote_time_or_fetch_time_only",
            units="variance points",
        ),
        universe=TaskUniverse(instruments=SIX_ETFS),
        data_requirements=reqs,
        splitter=SplitterSpec(
            config_id="expanding_weekly", scheme=SplitterScheme.EXPANDING, embargo_rule="21d",
            purge_rule="afml_3cond_label_span",
        ),
        incumbents=["model_free_surface_summaries"],
        naive="BS_european_approx",
        primary_score=ScoreSpec(name="QLIKE_forecast_vs_implied", orientation="lower_is_better", proper=True),
        guards=[GuardSpec(name="arbitrage_stability_failure_rate", margin=0.05)],
        secondary_scores=[],
        mde=MdeSpec(
            table_id="R4_5a_T-VRP",
            assumptions="mean VRP level SUFFICIENT (479wk needed vs 1836 held on SPX/VIX proxy); "
            "cheapness signal worth <50% mean VRP INSUFFICIENT (15-135y depending on correlation/size)",
        ),
        status=status,
        promotion_class=derive_promotion_class(status, reqs),
        amc_seams_allowed=[],
        influence_ceiling=InfluenceTier.T0,
    )

    return tasks
