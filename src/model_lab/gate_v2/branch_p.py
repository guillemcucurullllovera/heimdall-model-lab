"""Gate v2 Branch P (prospective e-process monitoring) — components P1-P4,
per the verified research extraction (Ramdas-Wang framework). Formula
machinery lives in `model_lab.eval.eprocess`; this module wires it to the
closed-list component/verdict structure.

Thresholds [P]: E>=20 (alpha=.05, tier T1); E>=100 (alpha=.01, tier
T2/T3); harm eligibility E^h<5 (alpha'=0.2); automatic DEGRADED at
E^h>=5, QUARANTINED at E^h>=20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_lab.eval.eprocess import betting_martingale, harm_process, ville_crossing_time
from model_lab.gate_v2.contracts import Branch, ComponentKind, ComponentResult

E_THRESHOLD_T1 = 20.0  # 1/alpha, alpha=0.05
E_THRESHOLD_T2_T3 = 100.0  # 1/alpha, alpha=0.01
HARM_ALPHA_PRIME = 0.2


@dataclass
class BranchPEvidence:
    freeze_signed: bool = False
    freeze_model_version_hash: str | None = None
    freeze_bound_b: float | None = None
    d: np.ndarray | None = None  # bounded paired differences, one per non-overlapping epoch
    delta: float | None = None  # harm non-inferiority margin
    tier: str = "T1"  # 'T1' | 'T2' | 'T3'
    epochs_overlapping: bool = False  # P4: must be non-overlapping
    n_min: int | None = None


def evaluate_branch_p(ev: BranchPEvidence) -> list[ComponentResult]:
    components: list[ComponentResult] = []

    # P4: thinning and freeze integrity (veto) -- checked first since P1/P2 depend on it
    p4_pass = ev.freeze_signed and not ev.epochs_overlapping and ev.freeze_bound_b is not None
    components.append(
        ComponentResult(
            id="P4", branch=Branch.P, name="thinning and freeze integrity", statistic="signed freeze precedes first epoch; non-overlapping epochs; bound b fixed a priori",
            evidence="structural", rule="freeze_signed AND NOT epochs_overlapping AND bound_b is not None", threshold="n/a",
            threshold_status="structural", gating=True, kind=ComponentKind.VETO, passed=p4_pass,
            value=f"freeze_signed={ev.freeze_signed}, overlapping={ev.epochs_overlapping}, b={ev.freeze_bound_b}",
            missing_input=ev.d is None,
        )
    )

    if ev.d is None or not p4_pass:
        for id_, name in [("P1", "betting martingale efficacy"), ("P2", "harm process")]:
            components.append(
                ComponentResult(
                    id=id_, branch=Branch.P, name=name, statistic="-", evidence="", rule="-", threshold=None,
                    threshold_status="PROPOSED", gating=True, kind=ComponentKind.VETO if id_ == "P2" else ComponentKind.POWER,
                    missing_input=True, passed=None,
                )
            )
        components.append(
            ComponentResult(
                id="P3", branch=Branch.P, name="e-BH (concurrent challengers)", statistic="-", evidence="", rule="report only",
                threshold=None, threshold_status="n/a", gating=False, kind=ComponentKind.REPORT_ONLY, missing_input=True, passed=None,
            )
        )
        return components

    # P1: betting martingale efficacy (power)
    bet = betting_martingale(ev.d)
    threshold = E_THRESHOLD_T1 if ev.tier == "T1" else E_THRESHOLD_T2_T3
    crossing_t = ville_crossing_time(bet.e_values, threshold)
    p1_pass = crossing_t is not None
    components.append(
        ComponentResult(
            id="P1", branch=Branch.P, name="betting martingale efficacy", statistic="E_t (GRAPA-like plug-in betting martingale)",
            evidence="V-RW (Ville's inequality, Fact 7.6)", rule=f"E_t crosses {threshold:.0f} at some t (Ville-valid, monitorable continuously)",
            threshold=str(threshold), threshold_status="PROPOSED", gating=True, kind=ComponentKind.POWER, passed=p1_pass,
            value=f"final_E={bet.final_e:.4f}, crossing_t={crossing_t}",
        )
    )

    # P2: harm process (non-inferiority guard) -- veto
    if ev.delta is not None:
        harm = harm_process(ev.d, delta=ev.delta, alpha_prime=HARM_ALPHA_PRIME)
        p2_pass = not harm.quarantined and harm.final_e < 5.0  # eligibility requires E^h < 5
        p2_val = f"final_Eh={harm.final_e:.4f}, degraded={harm.degraded}, quarantined={harm.quarantined}"
        p2_missing = False
    else:
        p2_pass, p2_val, p2_missing = None, None, True
    components.append(
        ComponentResult(
            id="P2", branch=Branch.P, name="harm process", statistic="E_t^h (non-inferiority harm martingale)", evidence="V-RW",
            rule="eligibility requires E^h < 5 at all t (alpha'=0.2); auto DEGRADED>=5, QUARANTINED>=20", threshold="5 / 20",
            threshold_status="PROPOSED", gating=True, kind=ComponentKind.VETO, passed=p2_pass, value=p2_val, missing_input=p2_missing,
        )
    )

    # P3: e-BH (report only at the single-challenger level; becomes gating
    # only in a multi-challenger concurrent-testing context, out of scope
    # for a single Branch-P evaluation)
    components.append(
        ComponentResult(
            id="P3", branch=Branch.P, name="e-BH (concurrent challengers)", statistic="e-BH Def 9.8 across K concurrent e-values",
            evidence="V-RW (Thm 9.11, arbitrary dependence)", rule="report only at single-challenger evaluation time", threshold=None,
            threshold_status="n/a", gating=False, kind=ComponentKind.REPORT_ONLY, passed=None, value=f"final_e={bet.final_e:.4f}",
            notes="apply eval.eprocess.e_bh across the full concurrent-challenger roster at the program level, not per single Branch P run",
        )
    )

    return components
