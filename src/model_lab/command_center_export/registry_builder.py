"""Builds the full `ModelLabRegistryExport` by enriching
`model_registry_data.STATIC_ENTRIES` with REAL evidence read from
committed JSON under `reports/`. Never recomputes a result; the one
exception, `scripts/export_regime_state_series.py`, is a separate,
explicit, already-run precompute step (see that script's own docstring
for why it is not a "recomputation" in the sense this module forbids --
it is a bounded, real export of an already-tested function's own output,
run once and committed, same as every other `reports/*.json` artifact
this builder reads)."""

from __future__ import annotations

import json

from model_lab.canon.jcs import sha256_bytes
from model_lab.command_center_export.amc_promotion_evaluator import evaluate_model_for_amc_promotion
from model_lab.command_center_export.builder import REPO_ROOT, REPORTS_DIR, _git_info
from model_lab.command_center_export.model_registry_data import STATIC_ENTRIES, StaticEntry
from model_lab.contracts.chart_spec import (
    Causality,
    ChartSeries,
    ChartSeriesPoint,
    ChartSpec,
    DataAuthority,
)
from model_lab.contracts.model2_export import EstablishmentState
from model_lab.contracts.model_lab_registry import (
    ModelLabRegistryExport,
    ModelRegistryEntry,
    RegistryBucket,
    ResearchState,
    ResearchSummaryCounts,
)
from model_lab.vol.candidate_spec import ELIGIBLE_CELLS

SIX_ETFS = ("SPY", "QQQ", "TLT", "IEF", "GLD", "USO")
VOL_HORIZONS = (1, 5, 21, 45)
TAIL_LEVELS = ("0.05", "0.025", "0.01")
ELIGIBLE_TAIL_TICKERS = {"TLT", "IEF", "GLD"}


def _load(*parts: str) -> dict:
    return json.loads((REPORTS_DIR / "/".join(parts)).read_text())


def _code_fingerprint(*relative_paths: str) -> str:
    src_root = REPO_ROOT / "src" / "model_lab"
    data = b""
    for rel in sorted(relative_paths):
        data += (src_root / rel).read_bytes()
    return sha256_bytes(data)


# ---- per-family enrichment --------------------------------------------------


def _enrich_vol_model(s: StaticEntry) -> dict:
    sample_ticker, sample_h = "TLT", 1
    cell = _load("vol_ladder", f"{sample_ticker}_h{sample_h}.json")
    model_result = cell["results"][s.model_id]
    n_cells = len(SIX_ETFS) * len(VOL_HORIZONS)
    return dict(
        historical_evidence_summary=f"Evaluated across all {n_cells} (ticker, horizon) cells of the T-VOL walk-forward ladder (QLIKE, Diebold-Mariano vs naive).",
        historical_evidence_refs=[f"reports/vol_ladder/{t}_h{h}.json" for t in SIX_ETFS for h in VOL_HORIZONS],
        actual_outputs_sample={
            "ticker": sample_ticker, "horizon_days": sample_h, "qlike_mean": model_result["qlike_mean"],
            "n_oos": model_result["n_oos"], "mz_slope": model_result["mz_slope"],
        },
        benchmark_comparison_summary=f"dm_vs_naive_p={model_result['dm_vs_naive_p']} at ({sample_ticker}, h={sample_h})",
        code_fingerprint=_code_fingerprint("vol/baselines.py") if "EWMA" in s.model_id or "naive" in s.model_id else _code_fingerprint("vol/garch_models.py"),
        chart_ids=[f"vol_qlike_by_horizon_{t}" for t in SIX_ETFS],
    )


def _enrich_tail_model(s: StaticEntry) -> dict:
    sample = _load("tail_ladder", "TLT.json")
    model_result = sample["results"][s.model_id]["0.05"] if s.model_id in sample["results"] else None
    refs = [f"reports/tail_ladder/{t}.json" for t in SIX_ETFS]
    sample_out: dict = {}
    if model_result:
        sample_out = {
            "ticker": "TLT", "level": "0.05", "observed_rate": model_result["observed_rate"],
            "nominal_rate": model_result["nominal_rate"], "n_violations": model_result["n_violations"],
            "christoffersen_lr_cc_p": model_result["christoffersen_lr_cc_p"],
        }
    elif s.model_id == "Combo_median_filtered_GPD_FHS":
        dv = _load("downstream_value_probe", "_summary.json")
        cand = dv["TLT"]["candidate_Combo_median_scaled"]["0.05"]
        sample_out = {
            "ticker": "TLT", "level": "0.05", "observed_rate": cand["observed_rate"],
            "nominal_rate": cand["p"], "n_violations": cand["n_violations"],
            "christoffersen_lr_cc_p": cand["christoffersen_lr_cc_p"],
        }
        refs = ["reports/downstream_value_probe/_summary.json"]
    return dict(
        historical_evidence_summary="Evaluated via Kupiec LR_uc / Christoffersen LR_cc / Acerbi-Szekely VaR-ES backtests across all 6 tickers x 3 tail levels.",
        historical_evidence_refs=refs,
        actual_outputs_sample=sample_out,
        chart_ids=[f"tail_coverage_{t}" for t in SIX_ETFS],
    )


def _enrich_anomaly_model(s: StaticEntry) -> dict:
    panel = _load("anomaly_ladder", "six_etf_panel.json")
    result = panel["results"][s.model_id]
    from model_lab.anomaly.evaluate import ILLUSTRATIVE_STRESS_DATES

    sample_pairs = dict(zip(ILLUSTRATIVE_STRESS_DATES[:3], result["score_percentile_of_illustrative_dates"][:3], strict=False))
    return dict(
        historical_evidence_summary="Descriptive-only cross-check: real anomaly score percentile at 14 illustrative historical stress dates, across the six-ETF panel. T-ANOM itself remains BLOCKED (no pre-registered benchmark), so this is NOT a validated detection-power claim.",
        historical_evidence_refs=["reports/anomaly_ladder/six_etf_panel.json"],
        actual_outputs_sample={
            "n_scored": result["n_scored"], "illustrative_dates_in_top_decile": result["illustrative_dates_in_top_decile"],
            "illustrative_dates_in_top_5pct": result["illustrative_dates_in_top_5pct"],
            "sample_percentile_by_date": sample_pairs,
        },
        code_fingerprint=_code_fingerprint("anomaly/covariance_models.py") if s.model_id != "Isolation_Forest" else _code_fingerprint("anomaly/isolation_forest_model.py"),
        chart_ids=["anomaly_illustrative_dates"],
    )


def _enrich_changepoint_model(s: StaticEntry) -> dict:
    sample = _load("changepoint_ladder", "SPY.json")
    key_map = {"BOCPD": ("bocpd_top_decile_hits", "bocpd_score_percentile_of_illustrative_dates"), "CUSUM": ("n_cusum_alarms", None)}
    count_key, pct_key = key_map[s.model_id]
    out: dict = {"ticker": "SPY", count_key: sample[count_key]}
    if pct_key:
        from model_lab.anomaly.evaluate import ILLUSTRATIVE_STRESS_DATES

        out["sample_percentile_by_date"] = dict(zip(ILLUSTRATIVE_STRESS_DATES[:3], sample[pct_key][:3], strict=False))
    return dict(
        historical_evidence_summary="Descriptive-only cross-check against 14 illustrative historical stress dates, across all 6 tickers.",
        historical_evidence_refs=[f"reports/changepoint_ladder/{t}.json" for t in SIX_ETFS],
        actual_outputs_sample=out,
        code_fingerprint=_code_fingerprint(f"changepoint/{s.model_id.lower()}.py"),
        chart_ids=["changepoint_bocpd_vs_cusum"],
    )


def _enrich_regime_model(s: StaticEntry) -> dict:
    sample = _load("regime_value_add", "TLT.json")
    series = _load("regime_value_add", "TLT_recent_state_series.json")
    recent_n = len(series["dates"])
    return dict(
        historical_evidence_summary="Value-add test (does the regime blend beat GARCH_t at h=1?) run across all 6 tickers; REJECTED on every ticker, re-confirmed after a methodology fix.",
        historical_evidence_refs=[f"reports/regime_value_add/{t}.json" for t in SIX_ETFS],
        actual_outputs_sample={
            "ticker": "TLT", "qlike_regime_2state": sample["qlike_by_model"]["regime_2state"],
            "qlike_GARCH_t": sample["qlike_by_model"]["GARCH_t"],
            "n_well_identified_folds": sample["n_well_identified_folds"], "n_folds": sample["n_folds"],
            "recent_state_series_sample": {
                "window": series["window_description"],
                "last_date": series["dates"][-1], "last_high_vol_prob": series["high_vol_prob"][-1],
                "n_days_in_window": recent_n,
            },
        },
        benchmark_comparison_summary="regime_2state loses to GARCH_t on QLIKE for all 6 tickers (see chart).",
        code_fingerprint=_code_fingerprint("regime/markov_switching.py"),
        chart_ids=["regime_vs_garch_qlike"],
        result_artifact_references=[f"reports/regime_value_add/{t}_recent_state_series.json" for t in SIX_ETFS],
    )


def _enrich_composite_model(s: StaticEntry) -> dict:
    manifest_path = REPORTS_DIR / "MODEL2_EVIDENCE_MANIFEST.json"
    sample = _load("vol_ladder", "TLT_h1.json")
    combo = sample["results"][s.model_id]
    snapshots = sorted((REPORTS_DIR / "shadow").glob("snapshot_*.json"))
    n_prospective_resolved = len(list((REPORTS_DIR / "prospective").glob("resolved_*.json")))
    n_prospective_pending = len(list((REPORTS_DIR / "prospective").glob("pending_*.json"))) - n_prospective_resolved
    return dict(
        historical_evidence_summary=f"Significantly beats GARCH_t (DM p<0.05) in exactly {len(ELIGIBLE_CELLS)}/24 (ticker, horizon) cells; never significantly worse in any.",
        historical_evidence_refs=[f"reports/vol_ladder/{t}_h{h}.json" for t in SIX_ETFS for h in VOL_HORIZONS],
        oos_evidence_summary="Same 9 calendar-year expanding walk-forward folds as every other T-VOL model; see MODEL2_CANDIDATE_SPEC.md for the exact eligible-cell table.",
        prospective_evidence_summary=f"{n_prospective_resolved} resolved, non-replay observations; {n_prospective_pending} genuinely PENDING (not yet resolvable).",
        prospective_evidence_refs=sorted(str(p.relative_to(REPO_ROOT)) for p in (REPORTS_DIR / "prospective").glob("*.json")),
        actual_outputs_sample={
            "ticker": "TLT", "horizon_days": 1, "qlike_mean": combo["qlike_mean"], "n_oos": combo["n_oos"],
            "latest_shadow_point_estimate_by_ticker": (
                {t: e.get("combo_candidate", {}).get("model_output", {}).get("point_estimate") for t, e in json.loads(snapshots[-1].read_text())["records"].items()}
                if snapshots else {}
            ),
        },
        incremental_value_summary="Downstream-value ablation (Phase 7): reduces T-TAIL VaR violations in 8/9 cells for TLT/IEF/GLD; confidence/abstention probe (Phase 8): constituent dispersion modestly predicts own forecast error (pooled DM p=0.038).",
        diagnostics=list(s.diagnostics) + ["See reports/MODEL2_CAPSTONE.md for the full 20-section evidence package."],
        code_fingerprint=_code_fingerprint("vol/combination.py", "vol/candidate_spec.py"),
        lifecycle_state="SHADOW",
        chart_ids=[f"vol_qlike_by_horizon_{t}" for t in SIX_ETFS],
        result_artifact_references=["reports/MODEL2_EVIDENCE_MANIFEST.json"] if manifest_path.exists() else [],
    )


def _enrich_literature_candidate(s: StaticEntry) -> dict:
    return dict(
        historical_evidence_summary="No implementation exists -- no historical evidence to report.",
        actual_outputs_sample={},
    )


_ENRICH_BY_FAMILY_PREFIX = (
    ("T-VOL volatility forecasting (composite", _enrich_composite_model),
    ("T-VOL volatility forecasting", _enrich_vol_model),
    ("T-TAIL tail risk", _enrich_tail_model),
    ("T-ANOM cross-sectional anomaly", _enrich_anomaly_model),
    ("T-CHANGE changepoint detection", _enrich_changepoint_model),
    ("T-STATE regime detection", _enrich_regime_model),
)


def _enrich(s: StaticEntry) -> dict:
    if s.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
        return _enrich_literature_candidate(s)
    for prefix, fn in _ENRICH_BY_FAMILY_PREFIX:
        if s.family.startswith(prefix):
            return fn(s)
    return _enrich_literature_candidate(s)


# ---- charts (non-decorative: every one plots real, already-committed evidence) --------------


def _chart(chart_id: str, question: str, incumbent_id: str, series: list[ChartSeries], n: int | None = None) -> ChartSpec:
    payload = {
        "chart_id": chart_id, "decision_question": question, "incumbent_id": incumbent_id,
        "series": [sr.model_dump(mode="json") for sr in series], "n": n,
        "causality": "NOT_APPLICABLE", "data_authority": "PROVISIONAL",
    }
    chart_sha = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return ChartSpec(
        chart_id=chart_id, decision_question=question, incumbent_id=incumbent_id, series=series, n=n,
        causality=Causality.NOT_APPLICABLE, data_authority=DataAuthority.PROVISIONAL, chart_sha256=chart_sha,
    )


def _build_vol_charts() -> list[ChartSpec]:
    roles = {"naive_trailing_rv": "naive", "EWMA_0.94": "comparator", "GARCH_t": "incumbent",
             "EGARCH_t": "comparator", "Combo_median_EWMA_GARCH_EGARCH": "challenger"}
    charts = []
    for ticker in SIX_ETFS:
        series_pts: dict[str, list[ChartSeriesPoint]] = {m: [] for m in roles}
        n_total = 0
        for h in VOL_HORIZONS:
            cell = _load("vol_ladder", f"{ticker}_h{h}.json")
            for model_id in roles:
                series_pts[model_id].append(ChartSeriesPoint(x_label=f"h={h}", value=str(cell["results"][model_id]["qlike_mean"])))
            n_total += cell["results"]["GARCH_t"]["n_oos"]
        series = [ChartSeries(label=m, role=r, points=series_pts[m]) for m, r in roles.items()]
        charts.append(_chart(f"vol_qlike_by_horizon_{ticker}", f"Does the combination candidate beat GARCH_t on QLIKE for {ticker} across horizons?", "GARCH_t", series, n_total))
    return charts


def _build_tail_charts() -> list[ChartSpec]:
    charts = []
    for ticker in SIX_ETFS:
        d = _load("tail_ladder", f"{ticker}.json")
        series_defs = [("unconditional_GPD_POT", "naive"), ("GJR_t_filtered_GPD_FHS", "incumbent")]
        series = []
        n_total = 0
        for model_id, role in series_defs:
            pts = [ChartSeriesPoint(x_label=f"p={p}", value=str(d["results"][model_id][p]["observed_rate"])) for p in TAIL_LEVELS]
            series.append(ChartSeries(label=model_id, role=role, points=pts))
            n_total = max(n_total, d["results"][model_id]["0.05"]["n_oos"])
        if ticker in ELIGIBLE_TAIL_TICKERS:
            dv = _load("downstream_value_probe", "_summary.json")
            cand_pts = [ChartSeriesPoint(x_label=f"p={p}", value=str(dv[ticker]["candidate_Combo_median_scaled"][p]["observed_rate"])) for p in TAIL_LEVELS]
            series.append(ChartSeries(label="Combo_median_filtered_GPD_FHS", role="challenger", points=cand_pts))
        nominal_pts = [ChartSeriesPoint(x_label=f"p={p}", value=p) for p in TAIL_LEVELS]
        series.append(ChartSeries(label="nominal_rate", role="comparator", points=nominal_pts))
        charts.append(_chart(f"tail_coverage_{ticker}", f"Is {ticker}'s observed VaR violation rate close to the nominal rate at each level?", "GJR_t_filtered_GPD_FHS", series, n_total))
    return charts


def _build_anomaly_chart() -> ChartSpec:
    from model_lab.anomaly.evaluate import ILLUSTRATIVE_STRESS_DATES

    panel = _load("anomaly_ladder", "six_etf_panel.json")
    roles = {"empirical": "comparator", "ledoit_wolf": "incumbent", "oas": "comparator",
             "min_cov_det": "comparator", "Isolation_Forest": "incumbent"}
    series = []
    for model_id, role in roles.items():
        pcts = panel["results"][model_id]["score_percentile_of_illustrative_dates"]
        pts = [ChartSeriesPoint(x_label=d, value=str(p)) for d, p in zip(ILLUSTRATIVE_STRESS_DATES, pcts, strict=True)]
        series.append(ChartSeries(label=model_id, role=role, points=pts))
    return _chart("anomaly_illustrative_dates", "Do the anomaly estimators place known historical stress dates in their own top score percentile?", "ledoit_wolf", series, panel["results"]["ledoit_wolf"]["n_scored"])


def _build_changepoint_chart() -> ChartSpec:
    bocpd_pts, cusum_pts = [], []
    for ticker in SIX_ETFS:
        d = _load("changepoint_ladder", f"{ticker}.json")
        bocpd_pts.append(ChartSeriesPoint(x_label=ticker, value=str(d["bocpd_top_decile_hits"])))
        cusum_pts.append(ChartSeriesPoint(x_label=ticker, value=str(d["cusum_alarm_within_5d_of_illustrative"])))
    series = [ChartSeries(label="BOCPD", role="incumbent", points=bocpd_pts), ChartSeries(label="CUSUM", role="comparator", points=cusum_pts)]
    return _chart("changepoint_bocpd_vs_cusum", "How many of the 14 illustrative stress dates does each changepoint detector flag, per ticker?", "BOCPD", series, 14)


def _build_regime_chart() -> ChartSpec:
    garch_pts, regime_pts = [], []
    n_total = 0
    for ticker in SIX_ETFS:
        d = _load("regime_value_add", f"{ticker}.json")
        garch_pts.append(ChartSeriesPoint(x_label=ticker, value=str(d["qlike_by_model"]["GARCH_t"])))
        regime_pts.append(ChartSeriesPoint(x_label=ticker, value=str(d["qlike_by_model"]["regime_2state"])))
        n_total += d["n_oos"]
    series = [ChartSeries(label="GARCH_t", role="incumbent", points=garch_pts), ChartSeries(label="regime_2state", role="challenger", points=regime_pts)]
    return _chart("regime_vs_garch_qlike", "Does the 2-state regime blend beat GARCH_t on QLIKE, per ticker?", "GARCH_t", series, n_total)


def _build_all_charts() -> list[ChartSpec]:
    return [*_build_vol_charts(), *_build_tail_charts(), _build_anomaly_chart(), _build_changepoint_chart(), _build_regime_chart()]


# ---- top-level assembly ------------------------------------------------------


def _independent_review_state(s: StaticEntry) -> EstablishmentState:
    if s.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE:
        return EstablishmentState.NOT_ESTABLISHED
    return EstablishmentState.PENDING


def build_model_lab_registry(generated_at_utc) -> ModelLabRegistryExport:
    commit, branch, _dirty = _git_info()
    charts = _build_all_charts()

    entries: list[ModelRegistryEntry] = []
    for s in STATIC_ENTRIES:
        evidence = _enrich(s)
        entries.append(
            ModelRegistryEntry(
                model_id=s.model_id, display_name=s.display_name, family=s.family,
                research_state=s.research_state, bucket=s.bucket, task_id=s.task_id,
                constituent_of=s.constituent_of, constituents=list(s.constituents),
                target_variable=s.target_variable, universe=list(s.universe), horizons_days=list(s.horizons_days),
                methodology=s.methodology, inputs_data=s.inputs_data,
                implementation_files=list(s.implementation_files), code_fingerprint=evidence.get("code_fingerprint"),
                source_references=list(s.source_references),
                lifecycle_state=evidence.get("lifecycle_state"),
                independent_review_state=_independent_review_state(s),
                amc_promotion_authorized=False,
                historical_evidence_summary=evidence.get("historical_evidence_summary", "No implementation exists -- no historical evidence to report."),
                historical_evidence_refs=evidence.get("historical_evidence_refs", []),
                oos_evidence_summary=evidence.get("oos_evidence_summary"),
                prospective_evidence_summary=evidence.get("prospective_evidence_summary"),
                prospective_evidence_refs=evidence.get("prospective_evidence_refs", []),
                actual_outputs_sample=evidence.get("actual_outputs_sample", {}),
                benchmark_comparison_summary=evidence.get("benchmark_comparison_summary"),
                incremental_value_summary=evidence.get("incremental_value_summary"),
                diagnostics=evidence.get("diagnostics", list(s.diagnostics)),
                limitations=list(s.limitations),
                governance_provenance=list(s.governance_provenance),
                chart_ids=evidence.get("chart_ids", []),
                result_artifact_references=evidence.get("result_artifact_references", []),
            )
        )

    task_hierarchy: dict[str, list[str]] = {}
    status_buckets: dict[str, list[str]] = {b.value: [] for b in RegistryBucket}
    for e in entries:
        if e.task_id:
            task_hierarchy.setdefault(e.task_id, []).append(e.model_id)
        status_buckets[e.bucket.value if hasattr(e.bucket, "value") else e.bucket].append(e.model_id)

    static_by_id = {s.model_id: s for s in STATIC_ENTRIES}
    amc_promotion_evaluations = {
        e.model_id: evaluate_model_for_amc_promotion(e, static_by_id[e.model_id]) for e in entries
    }

    n_historical = sum(1 for e in entries if e.historical_evidence_refs)
    n_prospective = sum(1 for e in entries if e.prospective_evidence_refs)
    n_with_charts = sum(1 for e in entries if e.chart_ids)
    summary = ResearchSummaryCounts(
        n_implemented_reproducible=sum(1 for e in entries if e.research_state == ResearchState.IMPLEMENTED_REPRODUCIBLE),
        n_implemented_validation_pending=sum(1 for e in entries if e.research_state == ResearchState.IMPLEMENTED_VALIDATION_PENDING),
        n_shadow_research=sum(1 for e in entries if e.research_state == ResearchState.SHADOW_RESEARCH),
        n_literature_research_candidates=sum(1 for e in entries if e.research_state == ResearchState.LITERATURE_RESEARCH_CANDIDATE),
        n_rejected_or_retired=sum(1 for e in entries if e.research_state == ResearchState.REJECTED_OR_RETIRED),
        n_task_families_represented=len(task_hierarchy),
        n_models_with_historical_evidence=n_historical,
        n_models_with_prospective_evidence=n_prospective,
        n_models_with_charts=n_with_charts,
        n_models_amc_eligible=0,
        n_models_amc_not_evaluable=sum(1 for ev in amc_promotion_evaluations.values() if ev.overall_status == "NOT_EVALUABLE"),
        n_models_amc_evaluated_failed=sum(1 for ev in amc_promotion_evaluations.values() if ev.overall_status == "EVALUATED_FAILED"),
    )

    payload = {
        "generated_at_utc": str(generated_at_utc), "git_commit": commit, "git_branch": branch,
        "entries": [e.model_dump(mode="json") for e in entries],
        "amc_promotion_evaluations": {k: v.model_dump(mode="json") for k, v in amc_promotion_evaluations.items()},
        "task_hierarchy": task_hierarchy, "status_buckets": status_buckets,
        "charts": [c.model_dump(mode="json") for c in charts], "research_summary": summary.model_dump(mode="json"),
    }
    export_sha = sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))

    return ModelLabRegistryExport(
        generated_at_utc=generated_at_utc, git_commit=commit, git_branch=branch,
        entries=entries, task_hierarchy=task_hierarchy, status_buckets=status_buckets,
        amc_promotion_evaluations=amc_promotion_evaluations,
        charts=charts, research_summary=summary, export_sha256=export_sha,
    )
