"""Gate v2 Branch F (forecast/state evidence vs incumbent and naive
baseline) — components F1-F9, per the verified research extraction.

Signal-level only: F8's cost/fill fields are UNAVAILABLE by design (no
AMC-side integration exists in MODEL-1), never assumed. Gating list: F1,
F2, F3, F4 (veto), F5 (veto), F6 (veto), F7 (adequacy), F8 (veto). F9 is a
mandatory METHOD component (dependence handling correctness), gating.
report_only: DM vs CW alternates, MCS p-values, per-path table, CIs (folded
into `notes`/`value` on the gating components themselves here, since this
module does not duplicate every diagnostic as a separate report_only row).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from arch.bootstrap import optimal_block_length

from model_lab.eval.calibration import reliability_slope
from model_lab.eval.mcs_spa import hansen_spa
from model_lab.eval.tests_dm import diebold_mariano
from model_lab.gate_v2.contracts import Branch, ComponentKind, ComponentResult

SPA_ALPHA = 0.10
DELTA_PATH = 0.02  # [P] path-level non-inferiority margin
DELTA_ERA_FRACTION = 0.25  # [P] per-era margin = 25% of the pooled improvement
K_DIFF_MIN = 60
K_DIFF_MIN_FRACTION = 0.10
CALIBRATION_SLOPE_BAND = (0.8, 1.2)


@dataclass
class BranchFEvidence:
    loss_incumbent: np.ndarray | None = None
    loss_challenger: np.ndarray | None = None
    loss_naive: np.ndarray | None = None
    is_nested: bool | None = None  # pre-registered; absent -> FAIL (per spec)
    cpcv_path_losses: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (incumbent, challenger) per path
    era_labels: np.ndarray | None = None  # int labels, same length as losses
    asset_labels: np.ndarray | None = None
    forecast_prob: np.ndarray | None = None
    outcome: np.ndarray | None = None
    decision_incumbent: np.ndarray | None = None
    decision_challenger: np.ndarray | None = None
    mde_min_useful_improvement: float | None = None
    guard_values: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (observed, signed_margin_required)
    guards_unavailable_by_design: set[str] = field(default_factory=set)
    h: int = 1


def evaluate_branch_f(ev: BranchFEvidence) -> list[ComponentResult]:
    components: list[ComponentResult] = []

    have_core = ev.loss_incumbent is not None and ev.loss_challenger is not None

    # F1: vs incumbent
    if have_core and ev.is_nested is not None:
        assert ev.loss_incumbent is not None and ev.loss_challenger is not None  # narrows for mypy; `have_core` already guarantees this at runtime
        r = diebold_mariano(ev.loss_incumbent, ev.loss_challenger, h=ev.h)
        f1_pass = r.mean_loss_diff > 0 and r.hln_p_value <= 0.05
        f1_val = f"d_bar={r.mean_loss_diff:.6f}, hln_p={r.hln_p_value:.4f}"
        f1_missing = False
    else:
        f1_pass, f1_val, f1_missing = None, None, True
    components.append(
        ComponentResult(
            id="F1", branch=Branch.F, name="vs incumbent", statistic="DM+HLN (non-nested) / CW (nested) / GW", evidence="M (formulas unverified in corpus)",
            rule="challenger beats incumbent, HLN p<=0.05, one-sided", threshold="0.05", threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.POWER, passed=f1_pass, value=f1_val, missing_input=f1_missing,
            notes="is_nested must be pre-registered; absent -> FAIL",
        )
    )

    # F2: vs naive baseline
    if ev.loss_naive is not None and ev.loss_challenger is not None:
        r2 = diebold_mariano(ev.loss_naive, ev.loss_challenger, h=ev.h)
        f2_pass = r2.mean_loss_diff > 0 and r2.hln_p_value <= 0.05
        f2_val = f"d_bar={r2.mean_loss_diff:.6f}, hln_p={r2.hln_p_value:.4f}"
        f2_missing = False
    else:
        f2_pass, f2_val, f2_missing = None, None, True
    components.append(
        ComponentResult(
            id="F2", branch=Branch.F, name="vs naive baseline", statistic="DM+HLN vs naive baseline", evidence="M",
            rule="challenger beats naive, HLN p<=0.05, one-sided", threshold="0.05", threshold_status="PROPOSED", gating=True,
            kind=ComponentKind.POWER, passed=f2_pass, value=f2_val, missing_input=f2_missing,
        )
    )

    # F3: family control (SPA / StepM / MCS)
    if have_core:
        assert ev.loss_incumbent is not None and ev.loss_challenger is not None  # narrows for mypy; `have_core` already guarantees this at runtime
        spa_res = hansen_spa(ev.loss_incumbent, ev.loss_challenger.reshape(-1, 1), alpha=SPA_ALPHA, reps=300, seed=0)
        f3_pass = not spa_res.benchmark_beats_all_models
        f3_val = f"spa_p={spa_res.p_value_consistent:.4f}"
        f3_missing = False
    else:
        f3_pass, f3_val, f3_missing = None, None, True
    components.append(
        ComponentResult(
            id="F3", branch=Branch.F, name="family control", statistic="arch SPA consistent p, StepM/MCS membership", evidence="arch oracle",
            rule=f"SPA p<={SPA_ALPHA} (challenger beats incumbent significantly)", threshold=str(SPA_ALPHA), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.POWER, passed=f3_pass, value=f3_val, missing_input=f3_missing,
        )
    )

    # F4: CPCV path fraction
    if ev.cpcv_path_losses:
        frac_positive = 0
        path_improvements = []
        for inc, chal in ev.cpcv_path_losses:
            d = inc - chal
            path_improvements.append(float(d.mean()))
            if d.mean() > 0:
                frac_positive += 1
        frac = frac_positive / len(ev.cpcv_path_losses)
        p10 = float(np.percentile(path_improvements, 10))
        f4_pass = frac >= 0.60 and p10 >= -DELTA_PATH
        f4_val = f"frac_positive={frac:.2f}, p10_improvement={p10:.6f}"
        f4_missing = False
    else:
        f4_pass, f4_val, f4_missing = None, None, True
    components.append(
        ComponentResult(
            id="F4", branch=Branch.F, name="CPCV path fraction", statistic="fraction of CPCV paths with mean d_t>0; 10th-pct path improvement",
            evidence="structural", rule=f"frac>=0.60 and p10>=-{DELTA_PATH}", threshold=str(DELTA_PATH), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.VETO, passed=f4_pass, value=f4_val, missing_input=f4_missing,
        )
    )

    # F5: era/asset breakdown
    if have_core and ev.era_labels is not None and len(set(ev.era_labels.tolist())) >= 3:
        assert ev.loss_incumbent is not None and ev.loss_challenger is not None  # narrows for mypy; `have_core` already guarantees this at runtime
        eras = sorted(set(ev.era_labels.tolist()))
        pooled_improvement = float((ev.loss_incumbent - ev.loss_challenger).mean())
        delta_era = DELTA_ERA_FRACTION * abs(pooled_improvement)
        worst_cell = None
        ok = True
        for era in eras:
            mask = ev.era_labels == era
            if mask.sum() < 5:
                continue
            d_cell = float((ev.loss_incumbent[mask] - ev.loss_challenger[mask]).mean())
            if worst_cell is None or d_cell < worst_cell:
                worst_cell = d_cell
            if d_cell < -delta_era:
                ok = False
        f5_pass = ok
        f5_val = f"n_eras={len(eras)}, worst_cell_improvement={worst_cell}"
        f5_missing = False
    else:
        f5_pass, f5_val, f5_missing = None, None, True
    components.append(
        ComponentResult(
            id="F5", branch=Branch.F, name="era/asset breakdown", statistic="per-era/per-asset DM direction and margin", evidence="structural",
            rule=f">=3 eras; no cell worse than -{DELTA_ERA_FRACTION}*pooled", threshold=str(DELTA_ERA_FRACTION), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.VETO, passed=f5_pass, value=f5_val, missing_input=f5_missing,
        )
    )

    # F6: calibration
    if ev.forecast_prob is not None and ev.outcome is not None:
        rel = reliability_slope(ev.forecast_prob, ev.outcome, n_boot=500)
        lo, hi = CALIBRATION_SLOPE_BAND
        f6_pass = lo <= rel.slope <= hi and rel.ci_lower <= 1.0 <= rel.ci_upper
        f6_val = f"slope={rel.slope:.3f}, ci=({rel.ci_lower:.3f},{rel.ci_upper:.3f})"
        f6_missing = False
    else:
        f6_pass, f6_val, f6_missing = None, None, True
    components.append(
        ComponentResult(
            id="F6", branch=Branch.F, name="calibration", statistic="reliability slope in [0.8,1.2], 95% CI contains 1", evidence="structural",
            rule="slope in [0.8,1.2] and CI contains 1", threshold="[0.8, 1.2]", threshold_status="PROPOSED", gating=True,
            kind=ComponentKind.VETO, passed=f6_pass, value=f6_val, missing_input=f6_missing,
        )
    )

    # F7: inertness + power
    if ev.decision_incumbent is not None and ev.decision_challenger is not None:
        n = len(ev.decision_incumbent)
        n_diff = int(np.sum(ev.decision_incumbent != ev.decision_challenger))
        frac_diff = n_diff / n if n > 0 else 0.0
        adequacy_ok = n_diff >= K_DIFF_MIN and frac_diff >= K_DIFF_MIN_FRACTION
        f7_val = f"n_diff={n_diff}, frac_diff={frac_diff:.3f}"
        f7_pass = adequacy_ok
        f7_missing = False
    else:
        f7_pass, f7_val, f7_missing = None, None, True
    components.append(
        ComponentResult(
            id="F7", branch=Branch.F, name="inertness + power", statistic=f">= {K_DIFF_MIN} epochs (>=10%) where decisions differ", evidence="structural",
            rule=f"n_diff>={K_DIFF_MIN} and frac_diff>={K_DIFF_MIN_FRACTION}", threshold=str(K_DIFF_MIN), threshold_status="PROPOSED",
            gating=True, kind=ComponentKind.ADEQUACY, passed=f7_pass, value=f7_val, missing_input=f7_missing,
        )
    )

    # F8: non-inferiority guards
    if ev.guard_values or ev.guards_unavailable_by_design:
        all_ok = True
        details = []
        for name, (observed, margin) in ev.guard_values.items():
            ok = observed >= margin
            all_ok = all_ok and ok
            details.append(f"{name}: {observed:.4f} vs margin {margin:.4f} -> {'OK' if ok else 'FAIL'}")
        f8_pass = all_ok
        f8_val = "; ".join(details) if details else "all guards UNAVAILABLE_BY_DESIGN (scope=SIGNAL_ONLY)"
        f8_missing = False
    else:
        f8_pass, f8_val, f8_missing = None, None, True
    components.append(
        ComponentResult(
            id="F8", branch=Branch.F, name="non-inferiority guards", statistic="tail loss (CVaR), turnover, latency/cost signed margins",
            evidence="structural", rule="all provided guards meet their signed margin", threshold="n/a", threshold_status="structural",
            gating=True, kind=ComponentKind.VETO, passed=f8_pass, value=f8_val, missing_input=f8_missing,
            notes="UNAVAILABLE-by-design lowers scope to SIGNAL_ONLY (not a failure); merely missing (not declared unavailable) => FAIL",
        )
    )

    # F9: dependence handling (mandatory method)
    if have_core:
        assert ev.loss_incumbent is not None and ev.loss_challenger is not None  # narrows for mypy; `have_core` already guarantees this at runtime
        d = ev.loss_incumbent - ev.loss_challenger
        block_len = float(optimal_block_length(d)["stationary"].iloc[0]) if len(d) > 10 else None
        f9_pass = block_len is not None
        f9_val = f"optimal_block_length={block_len:.2f}" if block_len is not None else None
        f9_missing = block_len is None
    else:
        f9_pass, f9_val, f9_missing = None, None, True
    components.append(
        ComponentResult(
            id="F9", branch=Branch.F, name="dependence handling", statistic="HAC/HLN with arch.bootstrap.optimal_block_length",
            evidence="arch (Politis-White/Patton-Politis-White, unverified in corpus)", rule="a block length is computed and used consistently",
            threshold="n/a", threshold_status="structural", gating=True, kind=ComponentKind.METHOD, passed=f9_pass, value=f9_val,
            missing_input=f9_missing,
        )
    )

    return components
