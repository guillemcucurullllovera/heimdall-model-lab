"""Structured (machine-readable) twin of `reports/MODEL2_RESEARCH_LEDGER.md`'s
own table -- same canonical-source relationship as `gates_data.py` has
with `MODEL2_PROMOTION_GATES.md`. Every row here must match that
document's table exactly; if they diverge, fix this module to match the
markdown (the human-reviewed source), not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_lab.contracts.model2_export import DecisionLabel


@dataclass(frozen=True)
class LedgerRow:
    index: int
    phase: str
    experiment: str
    decision: DecisionLabel
    reason: str
    evidence: str


LEDGER: tuple[LedgerRow, ...] = (
    LedgerRow(1, "2A", "Vol ladder: naive/EWMA/GARCH_t/EGARCH_t, 6 tickers x 4 horizons, QLIKE+DM",
              DecisionLabel.MIXED, "Mixed, ticker-heterogeneous (see row 8 for the combination follow-up)",
              "reports/vol_ladder/*.json, MODEL2A_...md sec9-10a"),
    LedgerRow(2, "2A", "Tail ladder: unconditional GPD/POT vs GJR-t-filtered-GPD-FHS, Kupiec/Christoffersen",
              DecisionLabel.RETAIN, "Conditional model better in 17/18 cells, not uniformly passing",
              "reports/tail_ladder/*.json, MODEL2A_...md sec9-10"),
    LedgerRow(3, "2B", "Anomaly ladder: 5 covariance/IF estimators vs 14 illustrative stress dates",
              DecisionLabel.DESCRIPTIVE_ONLY, "Not a validated benchmark, T-ANOM blocked",
              "reports/anomaly_ladder/six_etf_panel.json"),
    LedgerRow(4, "2C", "Changepoint ladder: CUSUM vs BOCPD vs 14 illustrative dates",
              DecisionLabel.DESCRIPTIVE_ONLY, "Weaker signal than row 3's cross-sectional approach",
              "reports/changepoint_ladder/*.json"),
    LedgerRow(5, "2D", "Regime value-add: 2-state Markov blend vs GARCH_t, h=1, 6 tickers",
              DecisionLabel.REJECT, "Regime loses on all 6 tickers; re-confirmed MORE robustly after remediation",
              "reports/regime_value_add/*.json, MODEL2B_2C_2D_...md"),
    LedgerRow(6, "2E/2G", "SPA/MCS model comparison, SPY-only then all 6 tickers x 4 horizons",
              DecisionLabel.MIXED, "Naive significantly beaten only for SPY/QQQ (+USO h=5 borderline); NOT universal",
              "reports/vol_model_comparison/*.json"),
    LedgerRow(7, "2F", "Shadow record infrastructure (ModelVersion/ModelOutput for GARCH_t, GJR-t-FHS)",
              DecisionLabel.CAPABILITY_BUILT, "Capability built, not a hypothesis test",
              "reports/shadow/, MODEL2E_2F_...md"),
    LedgerRow(8, "2G", "Forecast combination: median(EWMA, GARCH_t, EGARCH_t) vs GARCH_t",
              DecisionLabel.RETAIN, "Significant in 9/24 cells (TLT/IEF all horizons, GLD h=1), never significantly worse",
              "reports/vol_ladder/*.json, MODEL2G_...md sec1, MODEL2_CANDIDATE_SPEC.md"),
    LedgerRow(9, "2G", "Regime-conditional value: does regime help when its own classifier is uncertain?",
              DecisionLabel.REJECT, "1/6 tickers marginal (does not survive multiple-comparison correction)",
              "reports/regime_conditional_value/*.json, MODEL2G_...md sec3"),
    LedgerRow(10, "2G", "Changepoint-gated model selection: BOCPD-flagged days -> switch GARCH_t->EWMA_0.94?",
              DecisionLabel.INCONCLUSIVE, "0/4 testable tickers significant; 2 had too few flagged days",
              "reports/changepoint_gated_model_choice/*.json, MODEL2G_...md sec4"),
    LedgerRow(11, "2G", "EGARCH_t numerical-defect discovery (simulation blow-up, finite-but-absurd values)",
              DecisionLabel.DEFECT_FOUND_AND_FIXED, "13/24 cells affected pre-fix",
              "vol/garch_models.py's 3-tier plausibility bound, MODEL2G_...md sec0"),
    LedgerRow(12, "2H-P1", "Evidence manifest reconciliation (dataset identity, leakage-proof, fingerprint, n_oos cross-checks)",
              DecisionLabel.VERIFIED_CLEAN, "PASSED, no defect found this pass",
              "reports/MODEL2_EVIDENCE_MANIFEST.json, .md"),
    LedgerRow(13, "2H-P2", "Candidate definition locked",
              DecisionLabel.DEFINITIONAL, "Definitional, not a hypothesis test",
              "reports/MODEL2_CANDIDATE_SPEC.md, vol/candidate_spec.py"),
    LedgerRow(14, "2H-P4", "Shadow challenger built for the locked candidate",
              DecisionLabel.CAPABILITY_BUILT, "Capability built, not a hypothesis test",
              "shadow/build_combo.py, tests/unit/shadow/test_build_combo.py"),
    LedgerRow(15, "2H-P5", "Genuine prospective evidence pipeline: capture + live-resolve",
              DecisionLabel.CAPABILITY_BUILT,
              "Mechanism built AND exercised: 6 genuinely non-replay resolved observations; a 2nd round is honestly PENDING",
              "reports/prospective/*.json, MODEL2_PROMOTION_GATES.md"),
    LedgerRow(16, "2H-P6", "Promotion gate definition (mapped to LifecycleState + Gate v2 Branch P)",
              DecisionLabel.DEFINITIONAL, "Governance mapping; sufficiency NOT met -- 6 observations far too few for Branch P's thresholds",
              "reports/MODEL2_PROMOTION_GATES.md"),
    LedgerRow(17, "2H-P7", "Downstream value: does the candidate's vol edge improve VaR/ES coverage via FHS for TLT/IEF/GLD?",
              DecisionLabel.RETAIN,
              "RETAIN per the pre-registered rule (2/3 tickers improved) but weak in substance -- "
              "violations drop in 8/9 cells, ONE cell (GLD 99%) flips reject->pass; TLT/IEF remain "
              "miscalibrated at every level regardless of vol-scaling choice",
              "reports/downstream_value_probe/_summary.json"),
    LedgerRow(18, "2H-P8", "Confidence/abstention: does dispersion among the candidate's 3 constituents predict its own forecast error?",
              DecisionLabel.RETAIN,
              "RETAIN (modest) -- 3/3 eligible tickers show top-dispersion-tertile loss > bottom-tertile loss; "
              "pooled DM p=0.038; effect concentrated in TLT/IEF, negligible for GLD",
              "reports/confidence_abstention_probe/_summary.json"),
)
