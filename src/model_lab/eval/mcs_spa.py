"""Model Confidence Set / Hansen SPA / Romano-Wolf StepM via `arch` —
MODEL1_foundation_contracts.md §14, §15 (Branch S/F component F3/S7/S8).

Thin, typed wrappers around `arch.bootstrap.{SPA, StepM, MCS}` so callers
never touch the `arch` API directly and so results compose with this
repo's `n`/`n_independent` reporting convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arch.bootstrap import MCS, SPA, StepM


@dataclass(frozen=True)
class SpaResult:
    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    n: int
    benchmark_beats_all_models: bool


def hansen_spa(
    benchmark_losses: np.ndarray, model_losses: np.ndarray, *, alpha: float = 0.10, reps: int = 1000, seed: int = 0
) -> SpaResult:
    """Hansen's (2005) Superior Predictive Ability test. `model_losses` is a
    (n, k) matrix of k candidate models' losses; a *low* p-value means the
    benchmark is significantly beaten by at least one candidate. Returns the
    'consistent' p-value (Hansen's recommended studentized variant)."""
    bench = np.asarray(benchmark_losses, float)
    models = np.asarray(model_losses, float)
    if models.ndim == 1:
        models = models.reshape(-1, 1)
    rng = np.random.RandomState(seed)
    spa = SPA(bench, models, reps=reps, seed=rng)
    spa.compute()
    pvals = spa.pvalues
    return SpaResult(
        p_value_consistent=float(pvals["consistent"]),
        p_value_lower=float(pvals["lower"]),
        p_value_upper=float(pvals["upper"]),
        n=len(bench),
        benchmark_beats_all_models=float(pvals["consistent"]) >= alpha,
    )


@dataclass(frozen=True)
class StepMResult:
    superior_model_indices: list[int]
    n: int
    n_models: int


def romano_wolf_stepm(
    benchmark_losses: np.ndarray, model_losses: np.ndarray, *, alpha: float = 0.10, reps: int = 1000, seed: int = 0
) -> StepMResult:
    """Romano-Wolf (2005) StepM: the set of models significantly superior to
    the benchmark, with familywise error rate control."""
    bench = np.asarray(benchmark_losses, float)
    models = np.asarray(model_losses, float)
    if models.ndim == 1:
        models = models.reshape(-1, 1)
    rng = np.random.RandomState(seed)
    stepm = StepM(bench, models, size=alpha, reps=reps, seed=rng)
    stepm.compute()
    # arch's stubs type these as generic Hashable (pandas Index element type);
    # at runtime they are always the integer column positions of `models`.
    superior = (
        [int(i) for i in stepm.superior_models] if stepm.superior_models is not None else []  # type: ignore[call-overload]
    )
    return StepMResult(superior_model_indices=superior, n=len(bench), n_models=models.shape[1])


@dataclass(frozen=True)
class McsResult:
    included_model_indices: list[int]
    excluded_model_indices: list[int]
    n: int
    n_models: int


def model_confidence_set(losses: np.ndarray, *, alpha: float = 0.10, reps: int = 1000, seed: int = 0) -> McsResult:
    """Hansen, Lunde & Nason (2011) Model Confidence Set on a (n, k) loss
    matrix of k candidate models (NOT including a separate benchmark — the
    MCS itself determines the subset of models statistically
    indistinguishable from the best)."""
    losses = np.asarray(losses, float)
    rng = np.random.RandomState(seed)
    mcs = MCS(losses, size=alpha, reps=reps, seed=rng)
    mcs.compute()
    # Same stub-typing note as romano_wolf_stepm above.
    included = [int(i) for i in mcs.included]  # type: ignore[call-overload]
    excluded = [int(i) for i in mcs.excluded]  # type: ignore[call-overload]
    return McsResult(
        included_model_indices=included, excluded_model_indices=excluded, n=losses.shape[0], n_models=losses.shape[1]
    )
