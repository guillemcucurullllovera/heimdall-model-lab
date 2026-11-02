"""Gate v2 core formulas — from `gate_v2_spec.json` / `D7_gate_v2_spec.md`
(verified against rasterised pages of the DSR paper: Eq.1, 2, 5, 6; App A.3
Eq.7-9; PBO/CSCV Alg 2.3). MinBTL is `[D]` derived, not primary-source
verified — see `min_btl`'s own docstring; a prior version of this module
cited "BBLZ14 Thm 3.1, eq (3.2)" for it, which is a FALSE citation (caught
by the 2026-09-30 independent adversarial review, `reports/
MODEL1_REVIEW_REMEDIATION.md` P1-2): no such theorem or equation exists in
Bailey/Borwein/Lopez de Prado/Zhu's "The Probability of Backtest
Overfitting" (the actual 22-page paper, independently re-confirmed absent
of any "Theorem 3.1" or "(3.2)" by direct full-text search during
remediation). That paper mentions "MinBTL" exactly once, in its closing
sentence, without deriving it — it only names the concept and points to a
companion paper (its own Reference [1], Bailey/Borwein/Lopez de
Prado/Zhu, "Pseudo-Mathematics and Financial Charlatanism," Notices of the
AMS 61(5) (2014), 458-471) which is NOT present in this research corpus.
See `min_btl`'s docstring for the full disposition.

Notation (matching the spec): gamma = Euler-Mascheroni constant, Z = std
normal CDF, Z^-1 its inverse, SR_hat = per-observation (NON-annualised)
Sharpe, T = sample length, skew_hat = skewness, kurt_hat = RAW kurtosis
(Normal = 3, not excess).

Every threshold/parameter default in this module that the spec marks `[P]`
PROPOSED is documented as such in its docstring; none is treated as
calibrated/frozen (Gate v2 calibration is explicitly NO-GO per the operator
ruling until the 11 open questions are answered and an independent formula
review completes — see reports/GATE_V2_OPERATOR_ACTIONS.md).

Frequency convention (§2.12 of the research extraction, not stated in the
DSR paper itself): SR_hat, V, and SR0 must all be expressed in the SAME
native (non-annualised) frequency; annualised numbers are display-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def expected_max_z(N: float) -> float:
    """E[max_N] for N iid standard normals -- Eq.(5), p.12 of the DSR paper
    (also PBO-companion Prop 2.1 eq 2.4):
        E[max_N] ~= (1-gamma)*Z^-1[1 - 1/N] + gamma*Z^-1[1 - 1/(N*e)]
    Domain: N >= 2 (Z^-1[1-1/N] = -inf at N=1); MODEL-1 clips N_eff >= 2
    upstream (see n_eff below) unless the lineage has exactly one
    pre-registered configuration ever, in which case SR0 = 0 by convention
    (§ per the research extraction's "N_eff cumulation" note).
    """
    if N < 2:
        raise ValueError("expected_max_z requires N >= 2 (Z^-1[1-1/N] diverges at N=1)")
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * math.e))
    return (1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2


def sr0(N: float, variance_across_trials: float) -> float:
    """SR0, the selection/rejection threshold -- Eq.(2), p.8:
        SR0_hat = sqrt(V[{SR_n}]) * E[max_N]
    `variance_across_trials` MUST be the variance ACROSS the trials'
    estimated Sharpe ratios (V[{SR_n}]), never 1/T -- substituting 1/T is
    exactly v1's Defect (documented in the v1 freeze manifest's
    `v1_semantics` block: v1 never estimates V[{SR_n}], it implicitly fixes
    it at 1/T, the single-trial null variance).
    """
    if variance_across_trials < 0:
        raise ValueError("variance_across_trials must be >= 0")
    if N == 1:
        return 0.0  # single pre-registered configuration ever: no selection bias to deflate
    return math.sqrt(variance_across_trials) * expected_max_z(N)


def se_sr(sr_hat: float, T: float, skew_hat: float, kurt_hat: float) -> float:
    """SE(SR_hat) with skew/kurtosis correction -- Eq.(2) denominator:
        SE(SR_hat) = sqrt( (1 - skew*SR + ((kurt-1)/4)*SR^2) / (T-1) )
    """
    inner = 1.0 - skew_hat * sr_hat + ((kurt_hat - 1.0) / 4.0) * sr_hat**2
    if inner < 0:
        raise ValueError(
            f"SE(SR_hat) radicand is negative ({inner!r}) -- skew/kurtosis/SR_hat combination is "
            "inconsistent with a valid variance estimate"
        )
    return math.sqrt(inner / (T - 1.0))


def psr(sr_hat: float, sr_star: float, T: float, skew_hat: float, kurt_hat: float) -> float:
    """PSR(SR*) -- Probabilistic Sharpe Ratio, Eq.(2) general form:
        PSR_hat(SR*) = Z[ (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) ]
    A probability in [0,1].
    """
    denom_inner = 1.0 - skew_hat * sr_hat + ((kurt_hat - 1.0) / 4.0) * sr_hat**2
    if denom_inner <= 0:
        raise ValueError(f"PSR denominator radicand must be > 0, got {denom_inner!r}")
    z = (sr_hat - sr_star) * math.sqrt(T - 1.0) / math.sqrt(denom_inner)
    return float(stats.norm.cdf(z))


def dsr(sr_hat: float, N: float, variance_across_trials: float, T: float, skew_hat: float, kurt_hat: float) -> float:
    """DSR (Deflated Sharpe Ratio) = PSR evaluated at SR0 -- Eq.(2), p.8.
    A probability in [0,1] (this is the central fix over v1, which produced
    an unbounded z-like statistic instead -- see v1_freeze.py /
    v1_semantics for the documented v1 defect)."""
    sr_star = sr0(N, variance_across_trials)
    return psr(sr_hat, sr_star, T, skew_hat, kurt_hat)


def n_eff(M: int, rho_hat: float) -> float:
    """N_eff (effective independent trials for M DEPENDENT trials) --
    App A.3, Eq.(7)-(9), pp.14-15:
        N_hat = rho_hat + (1 - rho_hat) * M
    `rho_hat` is the average off-diagonal correlation of the trial matrix.
    Flagged by the paper itself as an interpolation heuristic; for M > T the
    correlation matrix is singular and rho_hat "may itself be overfit"
    (p.15) -- callers should treat N_eff as approximate in that regime.
    """
    if not (-1.0 <= rho_hat <= 1.0):
        raise ValueError("rho_hat must be in [-1, 1]")
    return rho_hat + (1.0 - rho_hat) * M


def min_trl(sr_hat: float, sr_star: float, skew_hat: float, kurt_hat: float, alpha: float) -> float:
    """MinTRL (Minimum Track Record Length) -- NOT in the DSR paper itself;
    derived `[D]` by inverting Eq.(2) for T at PSR(T) = 1-alpha:
        T* = 1 + (1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) * (Z^-1[1-alpha] / (SR_hat - SR*))^2
    Verified numerically against the pinned vector SR=0.1, SR*=0, skew=-1,
    kurt=6, alpha=0.05 -> T* ~= 301.99 (see tests/unit/gate_v2/test_formulas.py).
    The PUBLISHED form (Bailey & Lopez de Prado 2012) is `[M]` -- not in the
    corpus, unverified; this derivation is checked by confirming
    PSR(T*) == 1-alpha to numerical precision, which is a self-consistent
    correctness check independent of whether it matches the published paper
    verbatim.
    """
    if sr_hat <= sr_star:
        raise ValueError("MinTRL is undefined when sr_hat <= sr_star (no track record ever achieves significance)")
    z_alpha = stats.norm.ppf(1.0 - alpha)
    inner = 1.0 - skew_hat * sr_hat + ((kurt_hat - 1.0) / 4.0) * sr_hat**2
    if inner <= 0:
        raise ValueError(f"MinTRL radicand must be > 0, got {inner!r}")
    return 1.0 + inner * (z_alpha / (sr_hat - sr_star)) ** 2


def min_btl(N: float, target_sr_annualised: float) -> float:
    """MinBTL (Minimum Backtest Length), in YEARS -- `[D]` DERIVED, NOT
    primary-source verified (corrected 2026-09-30; see module docstring for
    the false-citation finding this replaces):
        MinBTL ~= ( E[max_N] / target_SR )^2

    Derivation logic (self-consistent, not copied from a verified source):
    a backtest of T years over which N independent configurations are
    searched has, under the null of no true edge, an expected best-of-N
    Sharpe ratio of approximately E[max_N]/sqrt(T) (the same `E[max_N]`
    scaling already used, VERIFIED, in `sr0`/`dsr` elsewhere in this
    module). Setting this equal to a target annualised Sharpe and solving
    for T gives the formula above. This is the same style of derivation
    already used, and already honestly self-flagged `[D]`, for `min_trl`
    (see that function) -- MinBTL is demoted to the same tier here, not a
    novel exception.

    Known true primary source (per the PBO paper's own Reference [1], see
    module docstring), NOT in this corpus and therefore NOT checked line-
    by-line: Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and
    Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-
    Sample Performance," Notices of the AMS 61(5) (2014), 458-471. Until
    that paper is obtained and checked, this formula is USABLE ONLY AS A
    REPORT-ONLY DIAGNOSTIC, never as a gating/veto input -- see
    `gate_v2/branch_s.py`'s S5 component, which was changed from
    `gating=True` to `report_only` specifically because of this finding
    (mirroring how S4/MinTRL already treats its own `[D]`-tier formula).
    The paper's own informational upper bound MinBTL < 2*ln(N)/target_SR^2
    is exposed separately as `min_btl_paper_bound` below, with the SAME
    caveat (that bound's citation was never independently verified either).
    `target_SR_annualised` is the TARGET in-sample, ANNUALISED Sharpe ratio
    -- substituting E[max_Z] itself as the target (rather than an actual,
    independently chosen target SR) is exactly v1's defect (documented in
    v1_semantics: v1's harness effectively conflates "the expected max of
    noise" with "the Sharpe ratio a real strategy targets", producing
    MinBTL values an order of magnitude too small).
    """
    if target_sr_annualised <= 0:
        raise ValueError("target_sr_annualised must be > 0")
    e_max = expected_max_z(N)
    return (e_max / target_sr_annualised) ** 2


def min_btl_paper_bound(N: float, target_sr_annualised: float) -> float:
    """An informational upper bound: MinBTL < 2*ln(N)/SR^2. `[D]`/UNVERIFIED
    -- see `min_btl`'s docstring; the true source of this bound was never
    independently located or checked, same disposition as `min_btl` itself.
    Report-only; never used as a gating input."""
    return 2.0 * math.log(N) / target_sr_annualised**2


@dataclass(frozen=True)
class PboResult:
    pbo: float
    S: int
    n_combinations: int
    logits: list[float]


def pbo_cscv(performance_matrix: np.ndarray, *, S: int | None = None) -> PboResult:
    """PBO via Combinatorial Symmetric Cross-Validation -- Alg 2.3 /
    eq.(2.3)-(2.4) of the PBO paper (Bailey, Borwein, Lopez de Prado & Zhu):

    `performance_matrix` is a (T, N) matrix of N trials' per-period
    performance (e.g. returns) over T periods. S (even) sub-samples are
    formed by splitting the T periods into S contiguous blocks; for each of
    C(S, S/2) ways to select S/2 blocks as the "in-sample" (IS) set (the
    complement is out-of-sample, OOS):
      1. compute each trial's IS performance metric (Sharpe ratio here) and
         find n* = argmax IS metric.
      2. compute the OOS metric for all N trials; find the rank of n*'s OOS
         performance among all N trials, convert to a relative rank
         omega_bar = rank(n*)/(N+1) in (0,1).
      3. logit lambda_c = ln(omega_bar / (1-omega_bar)).
    PBO = P(lambda_c <= 0) across all C(S,S/2) combinations (the fraction of
    combinations where the IS-best trial performs BELOW the OOS median,
    i.e. its IS outperformance failed to generalise).

    NOTE (documented erratum, per the verified research extraction): the
    PBO paper's own worked example states S=16 gives "12,780" combinations;
    the correct value is C(16,8) = 12,870 -- a typo in the source paper.
    This implementation always computes C(S,S/2) directly via
    math.comb, so it is unaffected by that typo.
    """
    from itertools import combinations
    from math import comb

    T, N = performance_matrix.shape
    if S is None:
        S = 16 if T >= 16 else (T - (T % 2))  # default from the paper's worked example; must be even
    if S % 2 != 0 or S < 2:
        raise ValueError("S must be a positive even integer")
    if S > T:
        raise ValueError("S cannot exceed the number of periods T")

    block_bounds = np.array_split(np.arange(T), S)
    n_combos = comb(S, S // 2)
    logits: list[float] = []

    for is_blocks in combinations(range(S), S // 2):
        is_idx = np.concatenate([block_bounds[b] for b in is_blocks])
        oos_blocks = [b for b in range(S) if b not in is_blocks]
        oos_idx = np.concatenate([block_bounds[b] for b in oos_blocks])

        is_perf = performance_matrix[is_idx, :]
        oos_perf = performance_matrix[oos_idx, :]

        is_sharpe = is_perf.mean(axis=0) / (is_perf.std(axis=0, ddof=1) + 1e-300)
        oos_sharpe = oos_perf.mean(axis=0) / (oos_perf.std(axis=0, ddof=1) + 1e-300)

        n_star = int(np.argmax(is_sharpe))
        # rank of n_star's OOS performance among all N trials, 1 = worst
        rank = int(np.sum(oos_sharpe <= oos_sharpe[n_star]))
        omega_bar = rank / (N + 1)
        omega_bar = min(max(omega_bar, 1e-9), 1 - 1e-9)
        logit = math.log(omega_bar / (1 - omega_bar))
        logits.append(logit)

    pbo = float(np.mean([logit <= 0 for logit in logits]))
    return PboResult(pbo=pbo, S=S, n_combinations=n_combos, logits=logits)
