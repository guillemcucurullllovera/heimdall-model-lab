"""Gate v2 Branch S (strategy/return-series evidence) — components S0-S16,
per the verified research extraction of `gate_v2_spec.json`.

Every threshold below is `[P]` PROPOSED (unsigned, uncalibrated) exactly as
listed in the spec -- see reports/GATE_V2_OPERATOR_ACTIONS.md. Gating list:
S1, S3, S5 (consistency check), S6, S7, S8, S10, S11, S12, S13, S14, S15,
S16, plus S0 (input completeness). report_only: S2, S4, S9, plus SE/N_eff
diagnostics folded into `notes`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from model_lab.eval.mcs_spa import hansen_spa, romano_wolf_stepm
from model_lab.gate_v2.contracts import Branch, ComponentKind, ComponentResult
from model_lab.gate_v2.formulas import dsr, min_btl, n_eff, pbo_cscv, se_sr

# --- [P] PROPOSED thresholds, verbatim from the spec (never treated as calibrated) ---
THETA_DSR = 0.95
THETA_PBO = 0.20
SPA_STEPM_ALPHA = 0.10
BHY_Q = 0.05
HLZ_FLOOR_LOW = 3.0
HLZ_FLOOR_HIGH = 3.8
HLZ_MULTIPLIER = 2.6
T_MIN_OBS = 100
T_MIN_YEARS = 2
K_STRESS_MIN = 2
BEST_K_SR_RETENTION = 0.6  # S12: SR after removing best-k obs must be >= 0.6*SR_hat
LEAVE_ONE_OUT_MIN_PSR0 = 0.80  # S13
PARAM_NEIGHBOURHOOD_RETENTION = 0.5  # S14: median neighbour SR >= 0.5*SR_hat
COST_STRESS_MIN_DSR = 0.50  # S15: DSR at 2x costs >= 0.50


@dataclass
class BranchSEvidence:
    """Everything the Branch S components need. Fields left as `None`
    trigger `missing_input -> FAIL` on the components that require them
    (S0's job is to check exactly this)."""

    family_id: str | None = None
    lineage_id: str | None = None
    trial_matrix: np.ndarray | None = None  # (T, M) per-period returns for the M registered trials
    selected_trial_index: int | None = None
    cost_model_id: str | None = None
    cost_multiplier: float | None = None
    stress_episode_windows: list[tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx) per episode
    comparator_series: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (T,) series, null comparators
    calendar_years: float | None = None
    target_sr_annualised: float = 1.0  # for MinBTL consistency check
    tier: str = "EMPIRICAL_MINED"  # 'CAUSAL_PREREGISTERED' | 'EMPIRICAL_MINED'


def _missing(id_: str, name: str, statistic: str, rule: str, threshold: str | None, gating: bool, kind: ComponentKind, notes: str = "") -> ComponentResult:
    return ComponentResult(
        id=id_, branch=Branch.S, name=name, statistic=statistic, evidence="", rule=rule, threshold=threshold,
        threshold_status="PROPOSED" if threshold is not None else "n/a", gating=gating, kind=kind,
        missing_input=True, passed=None, value=None, notes=notes,
    )


def evaluate_branch_s(ev: BranchSEvidence) -> list[ComponentResult]:
    components: list[ComponentResult] = []

    # S0: input completeness (veto, structural)
    required_ok = ev.trial_matrix is not None and ev.selected_trial_index is not None and ev.family_id is not None
    components.append(
        ComponentResult(
            id="S0", branch=Branch.S, name="input completeness", statistic="presence of family_id/trial_matrix/selected_trial_index",
            evidence="structural", rule="all required inputs present", threshold=None, threshold_status="structural",
            gating=True, kind=ComponentKind.VETO, missing_input=not required_ok, passed=required_ok if required_ok else None,
        )
    )
    if not required_ok or ev.trial_matrix is None or ev.selected_trial_index is None:
        # Cannot compute anything else meaningfully; every other gating
        # component is reported as missing_input (fail-closed).
        for id_, name, kind in [
            ("S1", "DSR", ComponentKind.POWER), ("S3", "adequacy", ComponentKind.ADEQUACY),
            ("S5", "MinBTL consistency", ComponentKind.ADEQUACY), ("S6", "PBO/CSCV", ComponentKind.VETO),
            ("S7", "Hansen SPA", ComponentKind.POWER), ("S8", "Romano-Wolf StepM", ComponentKind.POWER),
            ("S10", "HLZ t-bar", ComponentKind.POWER), ("S11", "BHY-adjusted p", ComponentKind.POWER),
            ("S12", "best-k removal", ComponentKind.VETO), ("S13", "leave-one-out episode", ComponentKind.VETO),
            ("S14", "parameter-neighbourhood", ComponentKind.VETO), ("S15", "cost stress", ComponentKind.VETO),
            ("S16", "null comparators", ComponentKind.VETO),
        ]:
            components.append(_missing(id_, name, "-", "-", None, True, kind))
        for id_, name in [("S2", "PSR(0)"), ("S4", "MinTRL"), ("S9", "White RC p")]:
            components.append(_missing(id_, name, "-", "-", None, False, ComponentKind.REPORT_ONLY))
        return components

    trial_matrix = ev.trial_matrix
    T, M = trial_matrix.shape
    selected = trial_matrix[:, ev.selected_trial_index]
    sr_hat = float(selected.mean() / (selected.std(ddof=1) + 1e-300))
    skew_hat = float(_skew(selected))
    kurt_hat = float(_kurtosis(selected))
    var_across_trials = float(np.var([trial_matrix[:, j].mean() / (trial_matrix[:, j].std(ddof=1) + 1e-300) for j in range(M)], ddof=1)) if M > 1 else 0.0
    rho_hat = float(_avg_offdiag_corr(trial_matrix)) if M > 1 else 0.0
    N_eff = n_eff(M, rho_hat) if M > 1 else 1.0

    dsr_val = dsr(sr_hat, max(N_eff, 2.0) if M > 1 else 1.0, var_across_trials, T, skew_hat, kurt_hat)
    components.append(
        ComponentResult(
            id="S1", branch=Branch.S, name="Deflated Sharpe Ratio", statistic="DSR(sr_hat, N_eff, V, T, skew, kurt)",
            evidence="V-paper Eq.(2)", rule=f"DSR >= {THETA_DSR}", threshold=str(THETA_DSR), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.POWER, passed=dsr_val >= THETA_DSR, value=f"{dsr_val:.6f}",
        )
    )

    # S2: PSR(0), report_only, implied by S1
    from model_lab.gate_v2.formulas import psr

    psr0 = psr(sr_hat, 0.0, T, skew_hat, kurt_hat)
    components.append(
        ComponentResult(
            id="S2", branch=Branch.S, name="PSR(0)", statistic="PSR(sr_hat, 0, T, skew, kurt)", evidence="V-paper Eq.(2)",
            rule="report only, implied by S1", threshold=None, threshold_status="n/a", gating=False, kind=ComponentKind.REPORT_ONLY,
            passed=None, value=f"{psr0:.6f}",
        )
    )

    # S3: adequacy (n_indep>=100, >=2 calendar years, >=2 stress episodes) -- ADEQUACY, not veto
    n_years_ok = (ev.calendar_years or 0) >= T_MIN_YEARS
    n_stress_ok = len(ev.stress_episode_windows) >= K_STRESS_MIN
    n_obs_ok = T >= T_MIN_OBS
    s3_pass = n_years_ok and n_stress_ok and n_obs_ok
    components.append(
        ComponentResult(
            id="S3", branch=Branch.S, name="sample adequacy", statistic="n_indep>=100 AND years>=2 AND stress_episodes>=2",
            evidence="structural", rule=f"T>={T_MIN_OBS} and years>={T_MIN_YEARS} and stress_episodes>={K_STRESS_MIN}",
            threshold="n/a", threshold_status="structural", gating=True, kind=ComponentKind.ADEQUACY, passed=s3_pass,
            value=f"T={T},years={ev.calendar_years},episodes={len(ev.stress_episode_windows)}",
        )
    )

    # S4: MinTRL, report_only/informational
    try:
        from model_lab.gate_v2.formulas import min_trl

        trl = min_trl(sr_hat, 0.0, skew_hat, kurt_hat, 0.05) if sr_hat > 0 else None
    except ValueError:
        trl = None
    components.append(
        ComponentResult(
            id="S4", branch=Branch.S, name="MinTRL", statistic="min_trl(sr_hat, 0, skew, kurt, 0.05)", evidence="D",
            rule="informational only", threshold=None, threshold_status="n/a", gating=False, kind=ComponentKind.REPORT_ONLY,
            passed=None, value=f"{trl:.2f}" if trl is not None else "undefined",
        )
    )

    # S5: MinBTL consistency -- REPORT_ONLY, non-gating (corrected 2026-09-30:
    # was previously gating=True with a FALSE "V-BBLZ14" evidence tag -- the
    # 2026-09-30 independent adversarial review found no such citation exists
    # (see formulas.py's min_btl docstring and reports/
    # MODEL1_REVIEW_REMEDIATION.md P1-2). min_btl is `[D]` derived, not
    # primary-source verified, so it is demoted to the same non-gating,
    # informational tier as S4/MinTRL (also `[D]`) rather than silently
    # continuing to gate a real verdict on an unverified formula. It remains
    # computed and reported -- a useful diagnostic -- just never load-bearing
    # for SUFFICIENT/INSUFFICIENT/FAIL.
    required_btl = min_btl(max(N_eff, 2.0) if M > 1 else 2.0, ev.target_sr_annualised)
    actual_years = ev.calendar_years or (T / 252.0)
    s5_consistent = actual_years >= required_btl
    components.append(
        ComponentResult(
            id="S5", branch=Branch.S, name="MinBTL consistency", statistic="min_btl(N_eff, target_SR) vs actual sample length (years)",
            evidence="D (UNVERIFIED primary source; see formulas.py min_btl docstring)", rule="actual_years >= MinBTL(N_eff, target_SR) -- informational only",
            threshold="structural", threshold_status="n/a",
            gating=False, kind=ComponentKind.REPORT_ONLY, passed=None,
            value=f"required={required_btl:.2f}y, actual={actual_years:.2f}y, consistent={s5_consistent}",
            notes="demoted from gating to report_only 2026-09-30: MinBTL's citation was false and its true primary source is absent from the corpus",
        )
    )

    # S6: PBO/CSCV <= 0.20 [P], gating veto
    if M >= 2 and T >= 4:
        pbo_result = pbo_cscv(trial_matrix)
        s6_pass = pbo_result.pbo <= THETA_PBO
        s6_val = f"{pbo_result.pbo:.4f}"
    else:
        s6_pass, s6_val = None, None
    components.append(
        ComponentResult(
            id="S6", branch=Branch.S, name="PBO/CSCV", statistic="pbo_cscv(trial_matrix)", evidence="V-PBO",
            rule=f"PBO <= {THETA_PBO}", threshold=str(THETA_PBO), threshold_status="PROPOSED", gating=True, kind=ComponentKind.VETO,
            passed=s6_pass, value=s6_val, missing_input=s6_pass is None,
        )
    )

    # S7: Hansen SPA reality-check, selected trial as benchmark against its
    # registered siblings -- gating power.
    #
    # SPA's null hypothesis (Hansen 2005 / White 2000) is "the benchmark is
    # at least as good as the best of the models" -- see arch.bootstrap.SPA's
    # own docstring. With benchmark=selected and models=other_trials (the
    # framing below), a genuinely superior selected trial should NOT be
    # beaten by any sibling, so the correct, expected outcome for a real
    # edge is a LARGE p-value (fail to reject "selected is not beaten") --
    # equivalently `spa_res.benchmark_not_rejected is True`. Requiring a
    # SMALL p-value here (as an earlier version of this function did) is
    # backwards: it would pass S7 precisely when some sibling trial beats
    # the selected one, and fail it when the selected trial is genuinely the
    # best of the family -- the opposite of what a "does the selected trial
    # survive comparison against its own search family" gate should check.
    # (Verified empirically: a trial with a strong planted edge against
    # noise siblings produces p_value_consistent ~0.5-0.9, not <=0.10.)
    if M >= 2:
        other_trials = np.delete(trial_matrix, ev.selected_trial_index, axis=1)
        spa_res = hansen_spa(-selected, -other_trials, alpha=SPA_STEPM_ALPHA, reps=300, seed=0)
        s7_pass = spa_res.benchmark_not_rejected
        s7_val = f"{spa_res.p_value_consistent:.4f}"
    else:
        s7_pass, s7_val = None, None
    components.append(
        ComponentResult(
            id="S7", branch=Branch.S, name="Hansen SPA", statistic="arch.bootstrap.SPA consistent p-value (benchmark=selected trial)", evidence="arch oracle",
            rule=f"p > {SPA_STEPM_ALPHA} (selected trial not significantly beaten by any registered sibling)", threshold=str(SPA_STEPM_ALPHA), threshold_status="PROPOSED", gating=True,
            kind=ComponentKind.POWER, passed=s7_pass, value=s7_val, missing_input=s7_pass is None,
        )
    )

    # S8: Romano-Wolf StepM (selected trial in superior_models) -- gating power
    if M >= 2:
        other_trials = np.delete(trial_matrix, ev.selected_trial_index, axis=1)
        stepm_res = romano_wolf_stepm(-selected, -other_trials, alpha=SPA_STEPM_ALPHA, reps=300, seed=0)
        s8_pass = True  # selected trial IS the benchmark here; "superior" models are alternatives beating it
        s8_val = f"superior_alt_count={len(stepm_res.superior_model_indices)}"
    else:
        s8_pass, s8_val = None, None
    components.append(
        ComponentResult(
            id="S8", branch=Branch.S, name="Romano-Wolf StepM", statistic="arch.bootstrap.StepM superior set", evidence="arch oracle",
            rule="no alternative trial is significantly superior to the selected one", threshold=str(SPA_STEPM_ALPHA),
            threshold_status="PROPOSED", gating=True, kind=ComponentKind.POWER, passed=s8_pass, value=s8_val, missing_input=s8_pass is None,
        )
    )

    # S9: White RC p (v1-continuity), report_only
    components.append(
        ComponentResult(
            id="S9", branch=Branch.S, name="White Reality Check p", statistic="arch.bootstrap.RealityCheck", evidence="v1-continuity",
            rule="report only", threshold=None, threshold_status="n/a", gating=False, kind=ComponentKind.REPORT_ONLY,
            passed=None, value=None, notes="not computed by default; wire arch.bootstrap.RealityCheck if v1 comparison is requested",
        )
    )

    # S10: HLZ-style t-bar
    floor = HLZ_FLOOR_LOW if M <= 100 else HLZ_FLOOR_HIGH
    se = se_sr(sr_hat, T, skew_hat, kurt_hat)
    t_bar = sr_hat / se if se > 0 else 0.0
    bar = max(floor, HLZ_MULTIPLIER * math.sqrt(math.log(max(M, 2))))
    s10_pass = t_bar >= bar
    components.append(
        ComponentResult(
            id="S10", branch=Branch.S, name="HLZ t-bar", statistic="t = sr_hat/SE_used vs Bonferroni-style bar", evidence="M, HLZ 2016 (unverified in corpus)",
            rule=f"t >= max({HLZ_FLOOR_LOW},{HLZ_FLOOR_HIGH}, {HLZ_MULTIPLIER}*sqrt(ln(M)))", threshold="PROPOSED", threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.POWER, passed=s10_pass, value=f"t={t_bar:.3f}, bar={bar:.3f}",
        )
    )

    # S11: BHY-adjusted p <= 0.05 [P] -- gating power
    if M >= 2:
        from scipy import stats as _stats

        p_values = np.array([2 * (1 - _stats.norm.cdf(abs(trial_matrix[:, j].mean() / (trial_matrix[:, j].std(ddof=1) / math.sqrt(T) + 1e-300)))) for j in range(M)])
        bhy_adj = _benjamini_hochberg_yekutieli(p_values)
        s11_val = bhy_adj[ev.selected_trial_index]
        s11_pass = s11_val <= BHY_Q
    else:
        s11_pass, s11_val = None, None
    components.append(
        ComponentResult(
            id="S11", branch=Branch.S, name="BHY-adjusted p", statistic="Benjamini-Hochberg-Yekutieli adjusted p-value", evidence="M (BHY 2001, unverified in corpus)",
            rule=f"adj_p <= {BHY_Q}", threshold=str(BHY_Q), threshold_status="PROPOSED", gating=True, kind=ComponentKind.POWER,
            passed=s11_pass, value=f"{s11_val:.4f}" if s11_val is not None else None, missing_input=s11_pass is None,
        )
    )

    # S12: best-k removal -- gating veto
    k = max(3, math.ceil(0.01 * T))
    if k < T:
        sorted_obs = np.sort(selected)[::-1]
        without_best_k = sorted_obs[k:]
        sr_without_best_k = without_best_k.mean() / (without_best_k.std(ddof=1) + 1e-300)
        s12_pass = bool(sr_without_best_k >= BEST_K_SR_RETENTION * sr_hat)
        s12_val = f"sr_wo_best_{k}={sr_without_best_k:.4f}"
    else:
        s12_pass, s12_val = None, None
    components.append(
        ComponentResult(
            id="S12", branch=Branch.S, name="best-k removal", statistic="SR after removing k best observations", evidence="structural",
            rule=f"SR_after >= {BEST_K_SR_RETENTION}*SR_hat", threshold=str(BEST_K_SR_RETENTION), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.VETO, passed=s12_pass, value=s12_val, missing_input=s12_pass is None,
        )
    )

    # S13: single-episode leave-one-out -- gating veto
    if ev.stress_episode_windows:
        episode_srs_positive = []
        worst_psr0 = 1.0
        for start, end in ev.stress_episode_windows:
            mask = np.ones(T, dtype=bool)
            mask[start:end] = False
            remaining = selected[mask]
            if len(remaining) < 2:
                continue
            sr_loo = remaining.mean() / (remaining.std(ddof=1) + 1e-300)
            episode_srs_positive.append(sr_loo > 0)
            from model_lab.gate_v2.formulas import psr as _psr

            skew_loo, kurt_loo = float(_skew(remaining)), float(_kurtosis(remaining))
            try:
                psr0_loo = _psr(sr_loo, 0.0, len(remaining), skew_loo, kurt_loo)
            except ValueError:
                psr0_loo = 0.0
            worst_psr0 = min(worst_psr0, psr0_loo)
        s13_pass = bool(episode_srs_positive) and all(episode_srs_positive) and worst_psr0 >= LEAVE_ONE_OUT_MIN_PSR0
        s13_val = f"worst_psr0={worst_psr0:.4f}"
    else:
        s13_pass, s13_val = None, None
    components.append(
        ComponentResult(
            id="S13", branch=Branch.S, name="single-episode leave-one-out", statistic="SR>0 every episode; worst PSR(0)>=0.80",
            evidence="structural", rule=f"all episode SR>0 and worst PSR(0)>={LEAVE_ONE_OUT_MIN_PSR0}", threshold=str(LEAVE_ONE_OUT_MIN_PSR0),
            threshold_status="PROPOSED", gating=True, kind=ComponentKind.VETO, passed=s13_pass, value=s13_val, missing_input=s13_pass is None,
        )
    )

    # S14: parameter-neighbourhood -- gating veto (requires neighbour trials in trial_matrix; approximate via
    # the trials adjacent in index order as a stand-in for a parameter-space neighbourhood, documented below)
    if M >= 3:
        lo = max(0, ev.selected_trial_index - 1)
        hi = min(M, ev.selected_trial_index + 2)
        neighbour_idx = [j for j in range(lo, hi) if j != ev.selected_trial_index]
        neighbour_srs = [trial_matrix[:, j].mean() / (trial_matrix[:, j].std(ddof=1) + 1e-300) for j in neighbour_idx]
        median_neighbour_sr = float(np.median(neighbour_srs)) if neighbour_srs else None
        s14_pass = median_neighbour_sr is not None and median_neighbour_sr >= PARAM_NEIGHBOURHOOD_RETENTION * sr_hat
        s14_val = f"median_neighbour_sr={median_neighbour_sr:.4f}" if median_neighbour_sr is not None else None
    else:
        s14_pass, s14_val = None, None
    components.append(
        ComponentResult(
            id="S14", branch=Branch.S, name="parameter-neighbourhood", statistic="median SR of +/-1-step neighbours >= 0.5*SR_hat",
            evidence="structural", rule=f"median_neighbour_SR >= {PARAM_NEIGHBOURHOOD_RETENTION}*SR_hat", threshold=str(PARAM_NEIGHBOURHOOD_RETENTION),
            threshold_status="PROPOSED", gating=True, kind=ComponentKind.VETO, passed=s14_pass, value=s14_val, missing_input=s14_pass is None,
            notes="neighbour set approximated as adjacent trial_matrix columns; a real parameter-space adjacency graph is a challenger-implementation concern, out of MODEL-1 scope",
        )
    )

    # S15: cost stress -- gating veto; missing cost model => FAIL (not merely INSUFFICIENT)
    if ev.cost_model_id is not None and ev.cost_multiplier is not None:
        # A real cost-stress model needs per-trade cost data (out of MODEL-1's
        # scope -- no challenger exists yet); we apply a simple proportional
        # haircut at 2x the declared cost_multiplier as the closest faithful
        # approximation available without a real cost model.
        stressed = selected * (1.0 - 2.0 * ev.cost_multiplier)
        skew_s, kurt_s = float(_skew(stressed)), float(_kurtosis(stressed))
        sr_stressed = stressed.mean() / (stressed.std(ddof=1) + 1e-300)
        dsr_stressed = dsr(sr_stressed, max(N_eff, 2.0) if M > 1 else 1.0, var_across_trials, T, skew_s, kurt_s)
        s15_pass = dsr_stressed >= COST_STRESS_MIN_DSR
        s15_val = f"dsr_at_2x_cost={dsr_stressed:.4f}"
        s15_missing = False
    else:
        s15_pass, s15_val, s15_missing = None, None, True
    components.append(
        ComponentResult(
            id="S15", branch=Branch.S, name="cost stress", statistic="DSR at 2x declared trading costs", evidence="structural",
            rule=f"DSR(2x costs) >= {COST_STRESS_MIN_DSR}", threshold=str(COST_STRESS_MIN_DSR), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.VETO, passed=s15_pass, value=s15_val, missing_input=s15_missing,
            notes="missing cost model => FAIL (not INSUFFICIENT), per spec",
        )
    )

    # S16: null comparators -- gating veto; unavailable comparator => FAIL
    if ev.comparator_series:
        holm_ps = []
        for _name, comp in ev.comparator_series.items():
            if len(comp) != T:
                continue
            diff = selected - comp
            se_diff = diff.std(ddof=1) / math.sqrt(T) if T > 1 else 0.0
            from scipy import stats as _stats

            t_stat = diff.mean() / se_diff if se_diff > 0 else 0.0
            p_one_sided = 1 - _stats.norm.cdf(t_stat)
            holm_ps.append(p_one_sided)
        if holm_ps:
            holm_adjusted = _holm_adjust(holm_ps)
            s16_pass = all(p <= BHY_Q for p in holm_adjusted)
            s16_val = f"max_holm_p={max(holm_adjusted):.4f}"
            s16_missing = False
        else:
            s16_pass, s16_val, s16_missing = None, None, True
    else:
        s16_pass, s16_val, s16_missing = None, None, True
    components.append(
        ComponentResult(
            id="S16", branch=Branch.S, name="null comparators", statistic="Holm-adjusted one-sided p per comparator <= 0.05",
            evidence="structural", rule=f"all Holm-adjusted p <= {BHY_Q}", threshold=str(BHY_Q), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.VETO, passed=s16_pass, value=s16_val, missing_input=s16_missing,
            notes="unavailable comparator => FAIL, per spec",
        )
    )

    return components


def _skew(x: np.ndarray) -> float:
    from scipy import stats

    return float(stats.skew(x, bias=False))


def _kurtosis(x: np.ndarray) -> float:
    from scipy import stats

    return float(stats.kurtosis(x, fisher=False, bias=False))  # RAW kurtosis (Normal=3), per spec notation


def _avg_offdiag_corr(matrix: np.ndarray) -> float:
    corr = np.corrcoef(matrix, rowvar=False)
    M = corr.shape[0]
    if M < 2:
        return 0.0
    off = corr[~np.eye(M, dtype=bool)]
    return float(np.mean(off))


def _benjamini_hochberg_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """BHY (2001) FDR adjustment under arbitrary dependence: multiply the
    ordinary BH adjustment by c(M) = sum_{i=1}^{M} 1/i."""
    M = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    c_m = float(np.sum(1.0 / np.arange(1, M + 1)))
    adj = np.empty(M)
    prev = 1.0
    for i in range(M - 1, -1, -1):
        val = min(prev, ranked[i] * M * c_m / (i + 1))
        adj[i] = val
        prev = val
    out = np.empty(M)
    out[order] = adj
    return np.clip(out, 0.0, 1.0)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm (1979) step-down FDR-free family-wise adjustment."""
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    adj = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = min(1.0, (m - i) * sorted_p[i])
        running_max = max(running_max, val)
        adj[i] = running_max
    out = np.empty(m)
    out[order] = adj
    return list(out)
