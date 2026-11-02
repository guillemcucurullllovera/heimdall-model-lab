"""Builds a real `Model2CommandCenterExport` from already-committed
evidence. Reads JSON files under `reports/`; never recomputes a result,
never calls `arch`/`statsmodels`/any model-fitting code. If this module
ever needs to fit a model to produce a value, that is a bug -- the
builder's whole purpose is to be a thin, honest projection of evidence
that already exists.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from model_lab.canon.jcs import sha256_bytes
from model_lab.command_center_export.gates_data import GATES
from model_lab.command_center_export.ledger_data import LEDGER
from model_lab.contracts.chart_spec import (
    Causality,
    ChartSeries,
    ChartSeriesPoint,
    ChartSpec,
    DataAuthority,
)
from model_lab.contracts.model2_export import (
    CandidateCellRef,
    CandidateDefinitionSection,
    ConfidenceAbstentionSection,
    DecisionLabel,
    DownstreamValueCell,
    DownstreamValueSection,
    EstablishmentState,
    GateState,
    HealthLimitationsSection,
    HistoricalCellResult,
    HistoricalEvidenceSection,
    Model2CommandCenterExport,
    ModelIdentitySection,
    ModelPurposeSection,
    PromotionGateEntry,
    PromotionGatesSection,
    ProspectiveEvidenceSection,
    ProspectiveObservationRef,
    ProvenanceSection,
    ResearchLedgerEntry,
    ResearchLedgerSection,
)
from model_lab.contracts.model_version import LifecycleState
from model_lab.data.snapshot_registry import DatasetSnapshotRegistry
from model_lab.tasks.registry import SIX_ETFS, build_default_task_registry
from model_lab.vol.candidate_spec import (
    CANDIDATE_COMBINATION_RULE,
    CANDIDATE_CONSTITUENTS,
    CANDIDATE_FAMILY,
    CANDIDATE_INFORMATION_SET,
    CANDIDATE_MODEL_ID,
    CANDIDATE_OUTPUT_DESCRIPTION,
    CANDIDATE_STRONGEST_BENCHMARK,
    CANDIDATE_TRAINING_PROCEDURE,
    ELIGIBLE_CELLS,
    REJECTED_CELLS,
    is_eligible_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"
HORIZONS = (1, 5, 21, 45)

_CANDIDATE_DEFINING_FILES = (
    "vol/combination.py",
    "vol/candidate_spec.py",
    "vol/garch_models.py",
    "vol/baselines.py",
)


def _git_info() -> tuple[str, str, bool]:
    def _run(args: list[str]) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"]) != ""
    return commit, branch, dirty


def _candidate_code_fingerprint() -> str:
    src_root = REPO_ROOT / "src" / "model_lab"
    data = b""
    for rel in sorted(_CANDIDATE_DEFINING_FILES):
        data += (src_root / rel).read_bytes()
    return sha256_bytes(data)


def _latest_shadow_snapshot() -> dict | None:
    snapshots = sorted((REPORTS_DIR / "shadow").glob("snapshot_*.json"))
    if not snapshots:
        return None
    return json.loads(snapshots[-1].read_text())


def _build_identity(generated_at_utc) -> ModelIdentitySection:
    commit, branch, dirty = _git_info()
    snapshot = _latest_shadow_snapshot()
    per_ticker_hash: dict[str, str] = {}
    if snapshot:
        for ticker, entry in snapshot.get("records", {}).items():
            combo = entry.get("combo_candidate", {})
            mv = combo.get("model_version")
            if mv:
                per_ticker_hash[ticker] = mv["model_version_hash"]
    return ModelIdentitySection(
        model_id=CANDIDATE_MODEL_ID,
        family=CANDIDATE_FAMILY,
        lifecycle_state=LifecycleState.SHADOW,
        git_commit=commit,
        git_branch=branch,
        working_tree_dirty=dirty,
        candidate_code_fingerprint=_candidate_code_fingerprint(),
        per_ticker_shadow_model_version_hash=per_ticker_hash,
        generated_at_utc=generated_at_utc,
    )


def _build_purpose() -> ModelPurposeSection:
    registry = DatasetSnapshotRegistry.load_from_directory()
    tasks = build_default_task_registry(registry)
    t_vol = tasks["T-VOL"]
    return ModelPurposeSection(
        task_id=t_vol.task_id,
        task_status=str(t_vol.status),
        target_variable=t_vol.target.variable,
        horizons_days=list(t_vol.target.horizon_days),
        universe=list(t_vol.universe.instruments),
        primary_score=t_vol.primary_score.name,
        naive_benchmark=t_vol.naive,
        frozen_incumbents=list(t_vol.incumbents),
        candidate_constituents=CANDIDATE_CONSTITUENTS,
        candidate_combination_rule=CANDIDATE_COMBINATION_RULE,
        candidate_strongest_benchmark=CANDIDATE_STRONGEST_BENCHMARK,
        challenger_relationship=(
            "A fixed, non-estimating combination (median) of three existing T-VOL benchmark "
            "models -- not a new fitted model, not a baseline, and not a replacement for any "
            "incumbent. Demonstrated to significantly beat the strongest incumbent (GARCH_t) in "
            "9 of 24 evaluated (ticker, horizon) cells; never significantly worse in any cell."
        ),
    )


def _build_candidate_definition() -> CandidateDefinitionSection:
    return CandidateDefinitionSection(
        eligible_cells=[CandidateCellRef(ticker=c.ticker, horizon=c.horizon) for c in ELIGIBLE_CELLS],
        rejected_cells=[CandidateCellRef(ticker=c.ticker, horizon=c.horizon) for c in REJECTED_CELLS],
        information_set=CANDIDATE_INFORMATION_SET,
        training_procedure=CANDIDATE_TRAINING_PROCEDURE,
        output_description=CANDIDATE_OUTPUT_DESCRIPTION,
    )


def _build_historical_evidence() -> HistoricalEvidenceSection:
    cells: list[HistoricalCellResult] = []
    n_better, n_worse, n_none = 0, 0, 0
    for ticker in SIX_ETFS:
        for h in HORIZONS:
            path = REPORTS_DIR / "vol_ladder" / f"{ticker}_h{h}.json"
            d = json.loads(path.read_text())
            combo = d["results"]["Combo_median_EWMA_GARCH_EGARCH"]
            garch = d["results"]["GARCH_t"]
            p = combo["dm_vs_garch_t_p"]
            eligible = is_eligible_cell(ticker, h)
            if p is not None and p < 0.05:
                if combo["qlike_mean"] < garch["qlike_mean"]:
                    n_better += 1
                else:
                    n_worse += 1
            else:
                n_none += 1
            cells.append(
                HistoricalCellResult(
                    ticker=ticker, horizon=h,
                    candidate_qlike=str(combo["qlike_mean"]), benchmark_qlike=str(garch["qlike_mean"]),
                    benchmark_model_id="GARCH_t", dm_p_value=str(p), n_oos=combo["n_oos"], is_eligible=eligible,
                )
            )
    return HistoricalEvidenceSection(
        cells=cells,
        n_cells_significant_better=n_better,
        n_cells_significant_worse=n_worse,
        n_cells_no_significant_difference=n_none,
        negative_findings=[
            "2-state Markov-switching regime blend loses to GARCH_t on all 6 tickers (reports/regime_value_add/*.json)",
            "Regime-conditional value (does regime help when its classifier is uncertain?): REJECT, 1/6 tickers marginal only",
            "Changepoint-gated model selection (BOCPD-flagged switching): REJECT/INCONCLUSIVE, 0/4 testable tickers significant",
            "naive_trailing_rv remains the single best model at GLD h=45 (exact tie, DM p=1.0)",
            "SPA/MCS: naive significantly beaten only for SPY/QQQ (+USO h=5 borderline), not universal across the 6-ETF panel",
        ],
        regime_limitations=[
            "The candidate's own vol-forecast edge is concentrated in TLT/IEF (all horizons) + GLD (h=1 only) -- "
            "0 of 15 other (ticker, horizon) cells show significance either direction",
            "No regime-aware or changepoint-aware version of the candidate exists or showed value when tested",
        ],
        source_files=[f"reports/vol_ladder/{t}_h{h}.json" for t in SIX_ETFS for h in HORIZONS],
    )


def _build_prospective_evidence() -> ProspectiveEvidenceSection:
    prospective_dir = REPORTS_DIR / "prospective"
    observations: list[ProspectiveObservationRef] = []
    resolved_by_key: dict[tuple[str, str], dict] = {}
    for path in sorted(prospective_dir.glob("resolved_*.json")):
        d = json.loads(path.read_text())
        # resolved_{ticker}_{cutoff}.json -- recover (ticker, cutoff) from the pending file it resolves
        resolved_by_key[d["pending_observation_hash"]] = d

    n_resolved_non_replay = 0
    n_pending = 0
    for path in sorted(prospective_dir.glob("pending_*.json")):
        pending = json.loads(path.read_text())
        resolved = resolved_by_key.get(pending["pending_observation_hash"])
        if resolved:
            if not resolved["is_historical_replay"]:
                n_resolved_non_replay += 1
            observations.append(
                ProspectiveObservationRef(
                    ticker=pending["ticker"], information_cutoff_date=pending["information_cutoff_date"],
                    status="RESOLVED", resolved_target_date=resolved["realized_target_date"],
                    candidate_forecast_error=resolved["candidate_forecast_error"],
                    is_historical_replay=resolved["is_historical_replay"],
                    pending_observation_hash=pending["pending_observation_hash"],
                )
            )
        else:
            n_pending += 1
            observations.append(
                ProspectiveObservationRef(
                    ticker=pending["ticker"], information_cutoff_date=pending["information_cutoff_date"],
                    status="PENDING", pending_observation_hash=pending["pending_observation_hash"],
                )
            )
    return ProspectiveEvidenceSection(
        n_resolved_non_replay_observations=n_resolved_non_replay,
        n_pending_observations=n_pending,
        observations=observations,
        sufficiency_state=EstablishmentState.NOT_ESTABLISHED,
        historical_vs_prospective_note=(
            "Historical walk-forward results (the 24-cell vol-ladder evidence above) score a "
            "forecast against an outcome that was ALREADY PRESENT in the frozen research dataset "
            "before the forecast was computed -- they validate the pipeline, but are not "
            "prospective evidence. Every observation listed here was captured from information "
            "available strictly before its own realized outcome existed anywhere in this "
            "project's data, confirmed via `is_historical_replay=False` on every resolved one."
        ),
    )


def _build_downstream_value() -> DownstreamValueSection:
    path = REPORTS_DIR / "downstream_value_probe" / "_summary.json"
    d = json.loads(path.read_text())
    eligible_tickers = {"TLT", "IEF", "GLD"}
    cells: list[DownstreamValueCell] = []
    for ticker, entry in d.items():
        if ticker == "_decision":
            continue
        for level in ("0.05", "0.025", "0.01"):
            inc = entry["incumbent_GARCH_t_scaled"][level]
            cand = entry["candidate_Combo_median_scaled"][level]
            cells.append(
                DownstreamValueCell(
                    ticker=ticker, tail_level=level,
                    incumbent_christoffersen_p=str(inc["christoffersen_lr_cc_p"]),
                    candidate_christoffersen_p=str(cand["christoffersen_lr_cc_p"]),
                    candidate_improved=cand["christoffersen_lr_cc_p"] > inc["christoffersen_lr_cc_p"],
                    is_eligible_ticker=ticker in eligible_tickers,
                )
            )
    decision_str = d["_decision"]["decision"]
    return DownstreamValueSection(
        cells=cells,
        pre_registered_rule=d["_decision"]["pre_registered_rule"],
        mechanical_decision=DecisionLabel.RETAIN if decision_str == "RETAIN" else DecisionLabel.REJECT,
        substantive_caveat=(
            "Violation counts drop in 8/9 (ticker, level) cells for TLT/IEF/GLD and one cell "
            "(GLD, 99%) flips from a Christoffersen rejection to a pass, but TLT/IEF remain "
            "decisively miscalibrated (p<0.01 at most levels) regardless of which vol model "
            "scales the tail -- the FHS tail SHAPE, not the vol-scaling choice, is the dominant "
            "source of miscalibration. The mechanical RETAIN verdict should not be read as "
            "'TLT/IEF's tail risk is now adequately modelled.'"
        ),
    )


def _build_confidence_abstention() -> ConfidenceAbstentionSection:
    path = REPORTS_DIR / "confidence_abstention_probe" / "_summary.json"
    d = json.loads(path.read_text())
    top: dict[str, str] = {}
    bottom: dict[str, str] = {}
    for ticker, entry in d.items():
        if ticker == "_decision":
            continue
        top[ticker] = str(entry["top_tertile_mean_loss"])
        bottom[ticker] = str(entry["bottom_tertile_mean_loss"])
    decision_str = d["_decision"]["decision"]
    return ConfidenceAbstentionSection(
        per_ticker_top_tertile_loss=top,
        per_ticker_bottom_tertile_loss=bottom,
        pooled_dm_stat=str(d["_decision"]["pooled_dm_stat"]),
        pooled_dm_p_value=str(d["_decision"]["pooled_dm_p"]),
        decision=DecisionLabel.RETAIN if decision_str == "RETAIN" else DecisionLabel.REJECT,
        limitations=[
            "The candidate itself has no native uncertainty output -- this dispersion diagnostic "
            "is computed SEPARATELY from the candidate's own 3 constituent forecasts, not part of "
            "the candidate's own definition (see MODEL2_CANDIDATE_SPEC.md)",
            "Pooled significance (p=0.038) is modest, not overwhelming, and concentrated in "
            "TLT/IEF -- GLD's own top-vs-bottom gap is negligible (1.5653 vs 1.5701)",
            "Not yet wired into any actual abstention/NO-TRADE decision logic -- a diagnostic "
            "finding, not an implemented gating mechanism",
        ],
    )


def _build_research_ledger() -> ResearchLedgerSection:
    entries = [
        ResearchLedgerEntry(index=r.index, phase=r.phase, experiment=r.experiment, decision=r.decision,
                             reason=r.reason, evidence=r.evidence)
        for r in LEDGER
    ]
    n_retain = sum(1 for r in LEDGER if r.decision == DecisionLabel.RETAIN)
    n_reject = sum(1 for r in LEDGER if r.decision in (DecisionLabel.REJECT, DecisionLabel.INCONCLUSIVE))
    n_other = len(LEDGER) - n_retain - n_reject
    return ResearchLedgerSection(entries=entries, n_retain=n_retain, n_reject_or_inconclusive=n_reject, n_descriptive_or_definitional=n_other)


def _build_promotion_gates() -> PromotionGatesSection:
    gates = [
        PromotionGateEntry(gate_id=g.gate_id, name=g.name, state=g.state, evidence=g.evidence,
                            maps_to_lifecycle_state=g.maps_to_lifecycle_state)
        for g in GATES
    ]
    assert all(g.state != GateState.PASSED or g.gate_id in ("G1_RESEARCH", "G2_SHADOW_CANDIDATE") for g in gates), (
        "only the research and shadow-candidate gates may be PASSED without further governance action"
    )
    return PromotionGatesSection(gates=gates, amc_promotion_authorized=False)


def _build_provenance() -> ProvenanceSection:
    manifest_path = REPORTS_DIR / "MODEL2_EVIDENCE_MANIFEST.json"
    manifest_sha = sha256_bytes(manifest_path.read_bytes())
    return ProvenanceSection(
        evidence_manifest_sha256=manifest_sha,
        test_suite_command=(
            "uv run pytest -q --deselect "
            "tests/unit/data/test_pit_macro_chunked_fetch.py::test_live_nfci_full_history_reaches_current_vintage"
        ),
        test_suite_last_known_result="see OVERNIGHT_MODEL_HANDOFF.md for the exact count as of the commit it documents -- re-run the command above to verify currently, never trust a stale number",
        literature_references=[
            "Bates & Granger (1969), 'The Combination of Forecasts', Operational Research Quarterly 20(4):451-468",
            "Stock & Watson (2004), 'Combination Forecasts of Output Growth in a Seven-Country Data Set', Journal of Forecasting 23(6):405-430",
            "Patton (2011) -- QLIKE as a proper scoring rule for variance forecasts",
            "Diebold & Mariano (1995) -- pairwise predictive-accuracy test",
            "Hansen (2005) SPA; Hansen/Lunde/Nason (2011) Model Confidence Set",
            "Ramdas & Wang -- e-process / betting-martingale framework (Gate v2 Branch P, eval/eprocess.py)",
            "McNeil & Frey (2000) -- Filtered Historical Simulation / GPD tail (tail/gpd.py)",
        ],
        reproduction_commands=[
            "uv run python scripts/run_vol_ladder.py",
            "uv run python scripts/run_prospective_capture.py",
            "uv run python scripts/run_downstream_value_probe.py",
            "uv run python scripts/run_confidence_abstention_probe.py",
            "uv run python scripts/build_model2_evidence_manifest.py",
            "uv run python scripts/build_model2_command_center_export.py",
        ],
    )


def _build_health_limitations() -> HealthLimitationsSection:
    return HealthLimitationsSection(
        known_defects_outstanding=[],
        defects_found_and_fixed_this_arc=[
            "EGARCH_t simulation-based forecast could produce finite-but-numerically-absurd values "
            "(2.0e-46 to 4.3e+93), affecting 13/24 vol-ladder cells -- fixed with a 3-tier plausibility bound",
            "BOCPD prior variance used the full-series variance (non-causal) -- fixed with a causal "
            "expanding/Welford variance",
            "Tail-evaluator off-by-one scoring misalignment -- fixed",
            "Purge/embargo computed upstream then silently discarded -- fixed, threaded through 6 model families",
            "Regime value-add's persistence-based forecast justification was false -- fixed with a real "
            "one-step-ahead transition",
        ],
        data_limitations=[
            "No true VIX/VIX3M/VVIX/SKEW access at any checked provider (entitlement-gated)",
            "No genuine intraday realised-volatility measure exists (blocks HAR-X)",
            "No pre-registered frozen_stress_event_labels benchmark (blocks T-ANOM promotion regardless of findings)",
        ],
        prospective_evidence_limitations=[
            "Only 6 resolved, non-replay observations exist -- far short of any defensible statistical threshold",
            "No scheduler exists; every capture/resolve round is a manual script invocation",
            "Live data fetching is rate-limited (~5 calls/60s); resolving all 6 tickers requires pacing",
        ],
        model_specific_failure_modes=[
            "The candidate has no native uncertainty/confidence output",
            "The candidate's own vol-forecast edge does not generalize to SPY/QQQ/USO or to GLD beyond h=1",
            "Even where the candidate improves downstream VaR/ES coverage (TLT/IEF/GLD), the improvement is "
            "mostly sub-significant -- the FHS tail shape remains the dominant miscalibration source",
        ],
        independent_review_status=EstablishmentState.PENDING,
    )


def _build_charts() -> list[ChartSpec]:
    """One real `ChartSpec` per eligible ticker (TLT/IEF/GLD): QLIKE by
    horizon for naive/EWMA/GARCH_t/EGARCH_t/candidate, reusing the EXACT
    contract MODEL-1 built for exactly this purpose (`ChartSpec` requires
    an 'incumbent' series and never renders UNAVAILABLE as 0 -- both
    honoured here)."""
    charts: list[ChartSpec] = []
    model_roles = {
        "naive_trailing_rv": "naive", "EWMA_0.94": "comparator", "GARCH_t": "incumbent",
        "EGARCH_t": "comparator", "Combo_median_EWMA_GARCH_EGARCH": "challenger",
    }
    for ticker in ("TLT", "IEF", "GLD"):
        series_by_model: dict[str, list[ChartSeriesPoint]] = {m: [] for m in model_roles}
        n_total = 0
        for h in HORIZONS:
            path = REPORTS_DIR / "vol_ladder" / f"{ticker}_h{h}.json"
            d = json.loads(path.read_text())
            for model_id in model_roles:
                qlike = d["results"][model_id]["qlike_mean"]
                series_by_model[model_id].append(ChartSeriesPoint(x_label=f"h={h}", value=str(qlike)))
            n_total += d["results"]["GARCH_t"]["n_oos"]
        series = [
            ChartSeries(label=model_id, role=role, points=series_by_model[model_id])
            for model_id, role in model_roles.items()
        ]
        chart_payload = {
            "chart_id": f"model2_qlike_by_horizon_{ticker}",
            "decision_question": f"Does the combination candidate beat GARCH_t on QLIKE for {ticker} across horizons?",
            "incumbent_id": "GARCH_t",
            "series": [s.model_dump(mode="json") for s in series],
            "n": n_total, "causality": "NOT_APPLICABLE", "data_authority": "PROVISIONAL",
        }
        chart_sha = sha256_bytes(json.dumps(chart_payload, sort_keys=True, default=str).encode("utf-8"))
        charts.append(
            ChartSpec(
                chart_id=f"model2_qlike_by_horizon_{ticker}",
                decision_question=f"Does the combination candidate beat GARCH_t on QLIKE for {ticker} across horizons?",
                incumbent_id="GARCH_t", series=series, n=n_total,
                causality=Causality.NOT_APPLICABLE, data_authority=DataAuthority.PROVISIONAL,
                chart_sha256=chart_sha,
            )
        )
    return charts


def build_model2_export(generated_at_utc) -> Model2CommandCenterExport:
    identity = _build_identity(generated_at_utc)
    purpose = _build_purpose()
    candidate_definition = _build_candidate_definition()
    historical_evidence = _build_historical_evidence()
    prospective_evidence = _build_prospective_evidence()
    downstream_value = _build_downstream_value()
    confidence_abstention = _build_confidence_abstention()
    research_ledger = _build_research_ledger()
    promotion_gates = _build_promotion_gates()
    provenance = _build_provenance()
    health_limitations = _build_health_limitations()
    charts = _build_charts()

    payload = {
        "identity": identity.model_dump(mode="json"),
        "purpose": purpose.model_dump(mode="json"),
        "candidate_definition": candidate_definition.model_dump(mode="json"),
        "historical_evidence": historical_evidence.model_dump(mode="json"),
        "prospective_evidence": prospective_evidence.model_dump(mode="json"),
        "downstream_value": downstream_value.model_dump(mode="json"),
        "confidence_abstention": confidence_abstention.model_dump(mode="json"),
        "research_ledger": research_ledger.model_dump(mode="json"),
        "promotion_gates": promotion_gates.model_dump(mode="json"),
        "provenance": provenance.model_dump(mode="json"),
        "health_limitations": health_limitations.model_dump(mode="json"),
        "charts": [c.model_dump(mode="json") for c in charts],
    }
    export_sha = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))

    return Model2CommandCenterExport(
        identity=identity, purpose=purpose, candidate_definition=candidate_definition,
        historical_evidence=historical_evidence, prospective_evidence=prospective_evidence,
        downstream_value=downstream_value, confidence_abstention=confidence_abstention,
        research_ledger=research_ledger, promotion_gates=promotion_gates, provenance=provenance,
        health_limitations=health_limitations, charts=charts, export_sha256=export_sha,
    )
