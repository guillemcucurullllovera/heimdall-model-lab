"""Static (hand-authored) classification and metadata for every Model Lab
model entry -- the canonical source `registry_builder.py` enriches with
REAL evidence read from committed JSON at build time. This module never
contains a number or statistic that could drift from the real evidence;
only qualitative facts that do not change run-to-run (what a model IS,
what task it belongs to, what papers it cites, what its real limitations
are) live here. See `reports/MODEL_LAB_FORENSIC_INVENTORY.md` for the
forensic basis of every classification below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_lab.contracts.model_lab_registry import RegistryBucket, ResearchState


@dataclass(frozen=True)
class StaticEntry:
    model_id: str
    display_name: str
    family: str
    research_state: ResearchState
    bucket: RegistryBucket
    task_id: str | None
    constituent_of: str | None = None
    constituents: tuple[str, ...] = ()
    target_variable: str | None = None
    universe: tuple[str, ...] = ()
    horizons_days: tuple[int, ...] = ()
    methodology: str = ""
    inputs_data: str = ""
    implementation_files: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    governance_provenance: tuple[str, ...] = field(default_factory=tuple)


SIX_ETF_UNIVERSE = ("SPY", "QQQ", "TLT", "IEF", "GLD", "USO")
VOL_HORIZONS = (1, 5, 21, 45)

# ---- T-VOL ------------------------------------------------------------------

STATIC_ENTRIES: tuple[StaticEntry, ...] = (
    StaticEntry(
        model_id="naive_trailing_rv", display_name="Naive Trailing Realised Variance", family="T-VOL volatility forecasting",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-VOL",
        target_variable="forward_realised_variance", universe=SIX_ETF_UNIVERSE, horizons_days=VOL_HORIZONS,
        methodology="252-day trailing mean of squared log returns, scaled by horizon. T-VOL's own frozen naive -- no estimation, purely causal.",
        inputs_data="Daily OHLC closes (EOD, no intraday data).",
        implementation_files=("vol/baselines.py",),
        source_references=(),
        diagnostics=("Zero estimation error by construction (no fitted parameters).",),
        limitations=("Flat/unresponsive to genuine regime shifts within the trailing window.",),
    ),
    StaticEntry(
        model_id="EWMA_0.94", display_name="EWMA (λ=0.94, RiskMetrics-style)", family="T-VOL volatility forecasting",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-VOL",
        constituent_of="Combo_median_EWMA_GARCH_EGARCH",
        target_variable="forward_realised_variance", universe=SIX_ETF_UNIVERSE, horizons_days=VOL_HORIZONS,
        methodology="Recursive exponentially-weighted variance (RiskMetrics λ=0.94), seeded on the first 30 observations' sample variance. T-VOL's own frozen incumbent; also a constituent of the MODEL-2 combination candidate.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("vol/baselines.py",),
        source_references=("J.P. Morgan RiskMetrics Technical Document (1996) -- the standard EWMA/λ=0.94 convention",),
        diagnostics=("Dominant single model for the two bond ETFs (TLT/IEF) at every horizon.",),
        limitations=("No leverage/asymmetry term -- cannot distinguish up-moves from down-moves of equal magnitude.",),
    ),
    StaticEntry(
        model_id="GARCH_t", display_name="GJR-GARCH(1,1)-t", family="T-VOL volatility forecasting",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-VOL",
        constituent_of="Combo_median_EWMA_GARCH_EGARCH",
        target_variable="forward_realised_variance", universe=SIX_ETF_UNIVERSE, horizons_days=VOL_HORIZONS,
        methodology="GJR-GARCH(1,1) with Student-t innovations (asymmetric leverage term), refit once per calendar-year expanding fold on the purged training window, analytic multi-step forecast via `arch`. T-VOL's own frozen incumbent, this project's strongest single benchmark; also feeds T-TAIL's FHS tail model's own conditional-vol scale and MODEL-2's combination candidate.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("vol/garch_models.py",),
        source_references=(
            "Glosten, Jagannathan & Runkle (1993) -- the GJR-GARCH asymmetric leverage term",
            "Bollerslev (1986) -- GARCH(1,1)",
        ),
        diagnostics=("Dominant single model for SPY/QQQ at shorter horizons; strongest single T-VOL benchmark overall.",),
        limitations=("Analytic multi-step forecast assumes a fixed conditional-variance recursion; does not model jumps.",),
    ),
    StaticEntry(
        model_id="EGARCH_t", display_name="EGARCH(1,1)-t", family="T-VOL volatility forecasting",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-VOL",
        constituent_of="Combo_median_EWMA_GARCH_EGARCH",
        target_variable="forward_realised_variance", universe=SIX_ETF_UNIVERSE, horizons_days=VOL_HORIZONS,
        methodology="EGARCH(1,1) with Student-t innovations, refit once per fold, multi-step forecast via 500-path Monte Carlo simulation (no closed-form multi-step EGARCH forecast exists). Added as the mission brief's 4th T-VOL rung (not one of T-VOL's own 3 frozen incumbents); also a MODEL-2 constituent.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("vol/garch_models.py",),
        source_references=("Nelson (1991) -- EGARCH",),
        diagnostics=("Outright best single model in 2/24 (ticker, horizon) cells -- 0/24 before a numerical defect fix this arc.",),
        limitations=(
            "A real, previously-undisclosed defect was found and fixed this research arc: the simulation-based forecast could "
            "produce finite-but-numerically-absurd values (2.0e-46 to 4.3e+93), now excluded via a 3-tier plausibility bound "
            "rather than silently scored. No closed-form multi-step forecast exists for this family at all, unlike GARCH_t.",
        ),
    ),
    StaticEntry(
        model_id="Combo_median_EWMA_GARCH_EGARCH", display_name="MODEL-2: Median Forecast Combination", family="T-VOL volatility forecasting (composite challenger)",
        research_state=ResearchState.SHADOW_RESEARCH, bucket=RegistryBucket.COMPOSITE_CHALLENGER, task_id="T-VOL",
        constituents=("EWMA_0.94", "GARCH_t", "EGARCH_t"),
        target_variable="forward_realised_variance", universe=SIX_ETF_UNIVERSE, horizons_days=VOL_HORIZONS,
        methodology="Elementwise MEDIAN (not mean) of the three constituents' own variance forecasts -- a fixed, non-estimating combination rule. Median chosen specifically for robustness to EGARCH_t's own occasional numerical outliers (a mean combination was tried first and found materially corrupted by exactly this).",
        inputs_data="Daily OHLC closes (identical to its three constituents -- no estimation of its own).",
        implementation_files=("vol/combination.py", "vol/candidate_spec.py"),
        source_references=(
            "Bates & Granger (1969) -- The Combination of Forecasts, Operational Research Quarterly 20(4):451-468",
            "Stock & Watson (2004) -- Combination Forecasts of Output Growth in a Seven-Country Data Set, Journal of Forecasting 23(6):405-430",
        ),
        diagnostics=("Significantly beats GARCH_t (DM p<0.05) in exactly 9/24 (ticker, horizon) cells; never significantly worse in any of the other 15.",),
        limitations=(
            "The 9-cell edge is concentrated entirely in TLT/IEF (all horizons) + GLD (h=1 only) -- no demonstrated edge for "
            "SPY/QQQ/USO or GLD beyond h=1.",
            "No native uncertainty/confidence output of its own.",
            "Only 6 resolved, non-replay prospective observations exist -- far short of any defensible statistical sufficiency bar.",
        ),
        governance_provenance=(
            "lifecycle_state=SHADOW on every ModelVersion record; decision_relevant=False, tier_ceiling_signed=False, "
            "amc_promotion_authorized=False -- all structurally enforced by tests, not merely convention.",
        ),
    ),
    StaticEntry(
        model_id="PDV", display_name="PDV (Path-Dependent Volatility)", family="T-VOL volatility forecasting",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-VOL",
        methodology="Named as one of T-VOL's 3 frozen incumbents (`tasks/registry.py`). No implementation exists in this repository -- an explicit scope decision recorded since MODEL-2A, not an oversight.",
        inputs_data="N/A -- not implemented.",
        limitations=("No code, no evidence, no charts. Explicit scope decision, not a data blocker.",),
    ),
    StaticEntry(
        model_id="HAR-X", display_name="HAR-X (Heterogeneous Autoregressive, exogenous RV)", family="T-VOL volatility forecasting",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-VOL",
        methodology="Named in the original MODEL-2 mission brief as a candidate T-VOL rung. No implementation exists.",
        inputs_data="Requires a genuine intraday realised-volatility measure, which this project's data estate does not have.",
        limitations=("Blocked on a genuine data gap (no intraday RV), not a code decision -- re-confirmed unchanged throughout this project's whole arc.",),
    ),

    # ---- T-TAIL ------------------------------------------------------------
    StaticEntry(
        model_id="unconditional_GPD_POT", display_name="Unconditional GPD / Peaks-Over-Threshold", family="T-TAIL tail risk",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-TAIL",
        target_variable="VaR / ES", universe=SIX_ETF_UNIVERSE, horizons_days=(1,),
        methodology="Generalized Pareto Distribution fitted once per fold to the unconditional (unfiltered) upper tail of returns, periodic-refit walk-forward. T-TAIL's own naive comparator.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("tail/unconditional.py", "tail/gpd.py"),
        source_references=("McNeil & Frey (2000) -- Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series",),
        limitations=("Ignores conditional (time-varying) volatility entirely -- a static tail shape over the whole fold.",),
    ),
    StaticEntry(
        model_id="GJR_t_filtered_GPD_FHS", display_name="GJR-t-Filtered GPD (Filtered Historical Simulation)", family="T-TAIL tail risk",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-TAIL",
        target_variable="VaR / ES", universe=SIX_ETF_UNIVERSE, horizons_days=(1,),
        methodology="GPD fitted once per fold to GARCH_t's own standardized residuals (the tail SHAPE), then scaled day-by-day by GARCH_t's own conditional-volatility forecast (Filtered Historical Simulation). T-TAIL's own frozen incumbent.",
        inputs_data="Daily OHLC closes; reuses GARCH_t's own conditional-variance recursion internally.",
        implementation_files=("tail/conditional.py", "tail/gpd.py"),
        source_references=("McNeil & Frey (2000) -- the same primary source as the unconditional model, this is its conditional/FHS variant",),
        diagnostics=("Strictly better (or tied) Christoffersen LR_cc coverage than the unconditional model in 17/18 (ticker, level) cells.",),
        limitations=("TLT/IEF/GLD still REJECT Christoffersen LR_cc at every VaR level even for this better-performing model -- 'better than the alternative' is not 'well-calibrated'.",),
    ),
    StaticEntry(
        model_id="Combo_median_filtered_GPD_FHS", display_name="Combo-Scaled FHS (downstream-value ablation)", family="T-TAIL tail risk",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-TAIL",
        constituent_of="Combo_median_EWMA_GARCH_EGARCH",
        target_variable="VaR / ES", universe=SIX_ETF_UNIVERSE, horizons_days=(1,),
        methodology="IDENTICAL GPD tail shape to GJR_t_filtered_GPD_FHS (fit on GARCH_t's own standardized residuals, unchanged), with ONE substitution: each day's VaR/ES is scaled by the MODEL-2 candidate's own vol forecast instead of GARCH_t's alone -- an isolated ablation testing whether the candidate's vol-forecast edge has downstream tail-risk value.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("tail/candidate_filtered_gpd_fhs.py",),
        diagnostics=(
            "Reduces VaR violation counts in 8/9 (ticker, level) cells for the candidate's own eligible tickers (TLT/IEF/GLD); "
            "one cell (GLD, 99%) flips from a Christoffersen rejection to a pass.",
        ),
        limitations=(
            "TLT/IEF remain decisively miscalibrated at every level regardless of vol-scaling choice -- the FHS tail SHAPE, not "
            "the vol-scaling choice, is the dominant source of miscalibration.",
            "Not wrapped in its own shadow ModelVersion record; not independently reviewed.",
        ),
    ),

    # ---- T-STATE / T-DOWNSTREAM ---------------------------------------------
    StaticEntry(
        model_id="regime_2state", display_name="2-State Gaussian Markov-Switching (Gaussian HMM)", family="T-STATE regime detection",
        research_state=ResearchState.REJECTED_OR_RETIRED, bucket=RegistryBucket.REJECTED_RETIRED, task_id="T-STATE",
        target_variable="regime_state_probability", universe=SIX_ETF_UNIVERSE, horizons_days=(1,),
        methodology="2-state Gaussian Hamilton-filter Markov-switching mean+variance model, refit once per calendar-year fold, filtered (never smoothed) regime probability propagated forward exactly one step via the model's own real transition matrix.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("regime/markov_switching.py",),
        source_references=("Hamilton (1989) -- A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle",),
        diagnostics=(
            "Real regime-probability/state-history output exists and is exposed (see actual_outputs_sample) -- the REJECT "
            "verdict is about its VALUE-ADD versus GARCH_t, not about whether the implementation works.",
            "A degenerate-fold safeguard (well_identified flag) correctly excludes folds where the 2-state model is not "
            "actually identified (one GLD fold, persistence=0.0).",
        ),
        limitations=(
            "Value-add test (does this blend beat GARCH_t at h=1?) REJECTED on all 6 tickers, re-confirmed MORE robustly "
            "after fixing a false justification for an earlier, simpler version of the forecast rule.",
            "A conditional re-test (does it help specifically when the regime classifier itself is uncertain?) also REJECTED "
            "(1/6 tickers marginal, does not survive multiple-comparison correction).",
            "Gaussian-only (no Student-t regime innovations); no TVTP (no defensible covariate identified); no finite HSMM.",
        ),
    ),

    # ---- T-ANOM --------------------------------------------------------------
    StaticEntry(
        model_id="empirical", display_name="Mahalanobis Distance (Empirical Covariance)", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-ANOM",
        target_variable="anomaly_score", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Mahalanobis distance of the six-ETF return panel from a periodic-refit empirical covariance estimate.",
        inputs_data="Daily OHLC closes, six-ETF panel (inner-joined on common dates).",
        implementation_files=("anomaly/covariance_models.py", "anomaly/panel.py"),
        limitations=("T-ANOM remains BLOCKED on `frozen_stress_event_labels` -- no pre-registered benchmark exists, so no AUPRC validation is possible for any estimator here.",),
    ),
    StaticEntry(
        model_id="ledoit_wolf", display_name="Mahalanobis Distance (Ledoit-Wolf Shrinkage)", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-ANOM",
        target_variable="anomaly_score", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Mahalanobis distance from a periodic-refit Ledoit-Wolf shrinkage covariance estimate. T-ANOM's own frozen incumbent.",
        inputs_data="Daily OHLC closes, six-ETF panel.",
        implementation_files=("anomaly/covariance_models.py", "anomaly/panel.py"),
        source_references=("Ledoit & Wolf (2004) -- A well-conditioned estimator for large-dimensional covariance matrices",),
        limitations=("Same T-ANOM validation blocker as `empirical`.",),
    ),
    StaticEntry(
        model_id="oas", display_name="Mahalanobis Distance (Oracle Approximating Shrinkage)", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-ANOM",
        target_variable="anomaly_score", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Mahalanobis distance from a periodic-refit OAS covariance estimate.",
        inputs_data="Daily OHLC closes, six-ETF panel.",
        implementation_files=("anomaly/covariance_models.py", "anomaly/panel.py"),
        source_references=("Chen, Wiesel, Eldar & Hero (2010) -- Shrinkage Algorithms for MMSE Covariance Estimation (OAS)",),
        limitations=("Same T-ANOM validation blocker as `empirical`.",),
    ),
    StaticEntry(
        model_id="min_cov_det", display_name="Mahalanobis Distance (Robust MinCovDet)", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-ANOM",
        target_variable="anomaly_score", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Mahalanobis distance from a periodic-refit robust (Minimum Covariance Determinant) covariance estimate.",
        inputs_data="Daily OHLC closes, six-ETF panel.",
        implementation_files=("anomaly/covariance_models.py", "anomaly/panel.py"),
        source_references=("Rousseeuw & Van Driessen (1999) -- A Fast Algorithm for the Minimum Covariance Determinant Estimator",),
        limitations=("Same T-ANOM validation blocker as `empirical`.",),
    ),
    StaticEntry(
        model_id="Isolation_Forest", display_name="Isolation Forest", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.IMPLEMENTED_VALIDATION_PENDING, bucket=RegistryBucket.VALIDATION_PENDING, task_id="T-ANOM",
        target_variable="anomaly_score", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="sklearn IsolationForest fitted periodically on the six-ETF return panel. T-ANOM's own frozen incumbent.",
        inputs_data="Daily OHLC closes, six-ETF panel.",
        implementation_files=("anomaly/isolation_forest_model.py", "anomaly/panel.py"),
        source_references=("Liu, Ting & Zhou (2008) -- Isolation Forest",),
        limitations=("Same T-ANOM validation blocker as the covariance estimators.",),
    ),
    StaticEntry(
        model_id="OCSVM_legacy", display_name="One-Class SVM (legacy naive)", family="T-ANOM cross-sectional anomaly",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-ANOM",
        methodology="Named as T-ANOM's own frozen NAIVE baseline in `tasks/registry.py`. No implementation exists anywhere in this repository.",
        inputs_data="N/A -- not implemented.",
        limitations=(
            "Newly surfaced by this checkpoint's forensic inventory: every anomaly-ladder report to date compared the 5 real "
            "estimators only against each other and against 14 illustrative dates, never against T-ANOM's own declared naive.",
        ),
    ),

    # ---- T-CHANGE --------------------------------------------------------------
    StaticEntry(
        model_id="CUSUM", display_name="CUSUM (Page, 1954)", family="T-CHANGE changepoint detection",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-CHANGE",
        target_variable="changepoint indicator", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Two-sided CUSUM on a rolling-standardized return series -- purely online/causal, no fold machinery needed.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("changepoint/cusum.py",),
        source_references=("Page (1954) -- Continuous Inspection Schemes",),
        limitations=("Weaker signal than the cross-sectional anomaly approach (T-ANOM) on the same illustrative-date check -- a real, disclosed difference, not a defect.",),
    ),
    StaticEntry(
        model_id="BOCPD", display_name="Bayesian Online Changepoint Detection", family="T-CHANGE changepoint detection",
        research_state=ResearchState.IMPLEMENTED_REPRODUCIBLE, bucket=RegistryBucket.TASK_MODEL, task_id="T-CHANGE",
        target_variable="changepoint indicator", universe=SIX_ETF_UNIVERSE, horizons_days=(0,),
        methodology="Normal-Inverse-Gamma conjugate BOCPD, purely online/causal. T-CHANGE's own frozen incumbent #1.",
        inputs_data="Daily OHLC closes.",
        implementation_files=("changepoint/bocpd.py",),
        source_references=("Adams & MacKay (2007) -- Bayesian Online Changepoint Detection",),
        diagnostics=(
            "A follow-up Layer-C probe (does BOCPD's changepoint probability usefully gate a GARCH_t->EWMA_0.94 model "
            "switch?) was REJECTED/INCONCLUSIVE (0/4 testable tickers significant) -- BOCPD itself remains a working, "
            "real implementation; only that one downstream use of it was tested and found wanting.",
        ),
        limitations=(
            "A real, previously-undisclosed non-causal defect (prior variance used the full-series variance) was found and "
            "fixed this project's arc -- now a causal expanding/Welford variance.",
            "Weaker standalone detector than the cross-sectional anomaly approach on the same illustrative-date check.",
        ),
    ),
    StaticEntry(
        model_id="Fearnhead", display_name="Fearnhead Exact/Particle Multiple-Changepoint", family="T-CHANGE changepoint detection",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-CHANGE",
        methodology="T-CHANGE's own frozen incumbent #2. No implementation exists -- explicit scope decision ('materially more complex product-partition-model algorithm').",
        inputs_data="N/A -- not implemented.",
        source_references=(
            "Fearnhead (2006) -- Exact and Efficient Bayesian Inference for Multiple Changepoint Problems (in the research corpus)",
            "Related: reversible-jump MCMC changepoint papers in the same corpus",
        ),
        limitations=("Explicit scope decision, not a data blocker -- directly corpus-backed if pursued later.",),
    ),

    # ---- T-CAL / T-VRP: zero implementation ---------------------------------
    StaticEntry(
        model_id="uniform_confidence", display_name="Uniform Confidence (T-CAL naive)", family="T-CAL LLM confidence calibration",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-CAL",
        methodology="T-CAL's own frozen naive baseline name. Zero implementation of any kind exists for T-CAL in this repository.",
        inputs_data="N/A -- T-CAL's own data requirements are fully BLOCKED.",
        limitations=("No code exists for this task at all -- not even a naive baseline -- data fully blocked, not a code decision.",),
    ),
    StaticEntry(
        model_id="model_free_surface_summaries", display_name="Model-Free Surface Summaries (T-VRP incumbent)", family="T-VRP variance risk premium",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-VRP",
        methodology="T-VRP's own frozen incumbent name. Zero implementation exists -- option archive too thin (1,472 leg-records from a single date, 3 of 6 ETFs).",
        inputs_data="Requires full option chains across stress windows; this project's option archive is nowhere near sufficient.",
        source_references=("Arbitrage-free SVI volatility surfaces.pdf", "The extended SSVI volatility surface (Hendriks & Martini) -- ssrn-2971502.pdf"),
        limitations=("Blocked on a genuine, re-confirmed data gap, not a code decision.",),
    ),
    StaticEntry(
        model_id="jump_stochastic_vol_research", display_name="Jump-Diffusion / Stochastic Volatility (literature)", family="T-VOL volatility forecasting (unimplemented frontier)",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-VOL",
        methodology="Asymmetric/jump-aware volatility beyond GJR/EGARCH's existing leverage terms -- explicitly identified as an unpursued Layer-A frontier (MODEL2G research continuation, §7) but not attempted: larger implementation effort, and the existing GJR/EGARCH asymmetry terms already capture most of the leverage effect this literature targets.",
        inputs_data="N/A -- not implemented.",
        source_references=(
            "Bates-style jump-diffusion exchange-rate options paper (research corpus)",
            "Eraker & Johannes -- The Impact of Jumps in Volatility and Returns (ssrn-249764.pdf)",
        ),
        limitations=("No implementation; identified as a real candidate direction, not pursued this arc.",),
    ),
    StaticEntry(
        model_id="bayesian_latent_state_inference_research", display_name="Advanced Bayesian Latent-State Inference (literature programme)", family="Regime/changepoint inference research (not yet mapped to a frozen task implementation)",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id="T-STATE",
        methodology=(
            "A broad literature programme covering finite/nonparametric Hidden Semi-Markov Models, sticky HDP-HMM variants, "
            "particle Gibbs / conditional-SMC methods (PGAS, CBPF), reversible-jump MCMC, stochastic-gradient MCMC for "
            "HMMs/state-space models, and discrete-space Langevin/locally-balanced samplers (DULA/DMALA, Path Auxiliary "
            "Sampling). None of this has been implemented. Directly relevant to T-STATE's own disclosed 'finite HSMM... "
            "NOT implemented' and 'TVTP... no defensible covariate identified' frontiers (MODEL2B_2C_2D report) -- a more "
            "sophisticated latent-state model than the current 2-state Gaussian HMM would plausibly draw on this literature."
        ),
        inputs_data="N/A -- not implemented.",
        source_references=(
            "Bayesian Nonparametric Hidden Semi-Markov Models.pdf", "Bayesian Nonparametric Inference of Switching Dynamic Linear Models.pdf",
            "Disentangled Sticky Hierarchical Dirichlet Process Hidden Markov Model.pdf", "The Recurrent Sticky Hierarchical Dirichlet Process Hidden Markov Model.pdf",
            "Hidden semi-Markov models Shun-Zheng Yu.html", "Particle Gibbs with Ancestor Sampling.pdf", "COUPLED CONDITIONAL BACKWARD SAMPLING PARTICLE FILTER.pdf",
            "Grid Particle Gibbs with Ancestor Sampling for State-Space Models.pdf", "On particle Gibbs sampling.pdf", "Particle Gibbs without the Gibbs bit.pdf",
            "Particle Markov chain Monte Carlo methods.pdf", "Resampling in conditional SMC algorithms.pdf", "Mixing time of the conditional backward sampling particle filter.pdf",
            "CONDITIONAL PARTICLE FILTERS WITH BRIDGE BACKWARD SAMPLING.pdf", "Inference for Diffusion Processes via Controlled Sequential Monte Carlo and Splitting Schemes.pdf",
            "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination.pdf",
            "An extension to reversible jump Markov chain Monte Carlo for change point problems with heterogeneous temporal dynamics.pdf",
            "Exact and Efficient Bayesian Inference for Multiple Changepoint problems.pdf",
            "Stochastic Gradient MCMC for State Space Models.pdf", "Stochastic Gradient MCMC Methods for Hidden Markov Models.pdf",
            "SUPPLEMENTARY MATERIALS- STOCHASTIC GRADIENT MCMC FOR STATE SPACE MODELS.pdf",
            "Auxiliary MCMC samplers for parallelisable inference in high-dimensional latent dynamical systems.pdf",
            "A Langevin-like Sampler for Discrete Distributions.pdf", "ANY-SCALE BALANCED SAMPLERS FOR DISCRETE SPACES.pdf",
            "Discrete Langevin Samplers via Wasserstein Gradient Flow.pdf", "DISCS- A Benchmark for Discrete Sampling.pdf",
            "Informed proposals for local MCMC in discrete spaces.pdf", "LSB- Local Self-Balancing MCMC in Discrete Spaces.pdf",
            "Oops I Took A Gradient- Scalable Sampling for Discrete Distributions.pdf",
            "Optimal design of the Barker proposal and other locally-balanced Metropolis-Hastings algorithms.pdf",
            "Optimal Scaling for Locally Balanced Proposals in Discrete Spaces.pdf", "PATH AUXILIARY PROPOSAL FOR MCMC IN DISCRETE SPACE.pdf",
            "B A Y E S I A N I N F E R E N C E F O R S T O C H A S T I C P R O C E S S E S.pdf (general reference)",
        ),
        limitations=(
            "Zero implementation, zero evidence, zero charts -- a literature programme, not a working model.",
            "Not yet mapped to any single frozen task's specific incumbent name (unlike Fearnhead, PDV, HAR-X, OCSVM).",
        ),
    ),
    StaticEntry(
        model_id="reservoir_computing_research", display_name="Echo State Networks / Reservoir Computing (literature programme)", family="Unmapped research programme",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id=None,
        methodology=(
            "Classical echo-state-network / reservoir-computing literature. Explicitly named in the ORIGINAL MODEL-2 mission "
            "brief as 'ESN / Quantum Lab coordination -- NOT started -- out of scope this session' (preserved verbatim in "
            "OVERNIGHT_MODEL_HANDOFF.md's own models-attempted table). No implementation exists. Not mapped to any frozen task."
        ),
        inputs_data="N/A -- not implemented.",
        source_references=("ESN_course.pdf", "The_echo_state_approach_to_analysing_and_training_.pdf (Jaeger, foundational ESN paper)", "Information Processing Capacity of Dynamical Systems (srep00514.pdf)"),
        limitations=("Zero implementation; explicitly out of scope per the original mission brief, not merely unattempted.",),
    ),
    StaticEntry(
        model_id="quantum_reservoir_computing_research", display_name="Quantum Reservoir Computing (literature, boundary-flagged)", family="Unmapped research programme",
        research_state=ResearchState.LITERATURE_RESEARCH_CANDIDATE, bucket=RegistryBucket.RESEARCH_CANDIDATE, task_id=None,
        methodology=(
            "Quantum reservoir computing / QAOA-adjacent literature present in the Model Lab's own corpus directory. "
            "BOUNDARY NOTE: this theme overlaps with the separately-governed Quantum Lab, which this repository's own "
            "operating boundaries explicitly forbid touching. Listed here only because the papers exist in this corpus "
            "directory -- this entry makes NO claim that this work belongs to or should be implemented by Model Lab; that "
            "determination is an explicit governance decision outside this inventory's scope, not resolved by implementing "
            "anything."
        ),
        inputs_data="N/A -- not implemented; no quantum-computing library of any kind is a dependency of this project.",
        source_references=(
            "Boosting computational power through spatial multiplexing in quantum reservoir computing.pdf",
            "Controllable Quantum Memory Capacity in Quantum Reservoir Networks with Tunable partial-SWAPs.pdf",
            "Fisher-Orthogonal Memory in Quantum Reservoir Computing.pdf",
            "Information processing capacity of spin-based quantum reservoir computing systems.pdf",
            "Novel Approach to Multivariate Forecasting with Quantum Reservoirs.pdf",
            "Recurrent Quantum Feature Maps for Reservoir Computing.pdf",
            "Non-Markovianity and memory enhancement in quantum reservoir computing (s41534-026-01257-4.pdf)",
        ),
        limitations=(
            "Zero implementation. Governance boundary explicitly unresolved -- this entry does not assert Model Lab ownership "
            "of this research direction.",
        ),
    ),
)
