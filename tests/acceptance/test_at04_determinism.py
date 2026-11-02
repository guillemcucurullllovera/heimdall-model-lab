"""AT-04: Determinism tiers.

BITWISE: identical output, same env, threads=1. NUMERIC: rtol 1e-9 across
thread counts {1,16}. STATISTICAL: >=20 seeds for stochastic methods.

Real multi-thread BLAS variation is not reliably forceable/observable from
inside a single pytest process on this host (numpy/scipy's threading layer
is fixed at process start in most builds; re-importing with a different
`OMP_NUM_THREADS` mid-process does not retroactively change already-loaded
BLAS thread pools). We therefore test BITWISE determinism directly
(same-process, same-env, repeated calls must match exactly), and test the
NUMERIC tier via a SUBPROCESS re-invocation with different
OMP_NUM_THREADS/MKL_NUM_THREADS env vars (a real, if coarse, thread-count
variation), and the STATISTICAL tier via >=20 seeds on a stochastic method
(the SPA bootstrap). This is documented rather than silently assumed
equivalent to varying threads within one process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from model_lab.eval.mcs_spa import hansen_spa
from model_lab.gate_v2.formulas import dsr, expected_max_z
from model_lab.splitter import LabelSpan, build_cpcv

pytestmark = pytest.mark.acceptance


def test_bitwise_determinism_same_process_same_env() -> None:
    """Repeated calls to a pure deterministic computation (splitter +
    formulas) with no RNG involved must produce BIT-IDENTICAL output."""
    groups = []
    idx = 0
    for _g in range(6):
        spans = []
        for _ in range(5):
            spans.append(LabelSpan(index=idx, t_start=float(idx), t_end=float(idx) + 0.9))
            idx += 1
        groups.append(spans)

    run1 = build_cpcv(groups, N=6, k=2)
    run2 = build_cpcv(groups, N=6, k=2)
    assert [s.train_indices for s in run1.splits] == [s.train_indices for s in run2.splits]
    assert [s.test_indices for s in run1.splits] == [s.test_indices for s in run2.splits]

    v1 = expected_max_z(46)
    v2 = expected_max_z(46)
    assert v1 == v2  # exact bitwise equality, not approx

    d1 = dsr(0.158114, 100, 1.0 / 500.0, 1250, -3, 10)
    d2 = dsr(0.158114, 100, 1.0 / 500.0, 1250, -3, 10)
    assert d1 == d2


def test_numeric_determinism_across_thread_counts_via_subprocess() -> None:
    """Re-invoke a pure numeric computation (expected_max_z / dsr — no
    external RNG) in two subprocesses with OMP_NUM_THREADS/MKL_NUM_THREADS
    set to 1 and 16 respectively, and confirm the results agree to
    rtol=1e-9. (These particular functions don't hit BLAS internally, so
    this is a genuine but weak-signal check; documented as such — a truly
    BLAS-heavy computation, e.g. a large matrix decomposition, would be a
    stronger test but isn't part of this codebase's hot path yet.)"""
    script = (
        "import json\n"
        "from model_lab.gate_v2.formulas import expected_max_z, dsr\n"
        "out = {'e_max': expected_max_z(250), 'dsr': dsr(0.158114, 100, 1.0/500.0, 1250, -3, 10)}\n"
        "print(json.dumps(out))\n"
    )
    results = {}
    for threads in (1, 16):
        env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "PATH": "/usr/bin:/bin"}
        import os

        env["HOME"] = os.environ.get("HOME", "")
        env["VIRTUAL_ENV"] = os.environ.get("VIRTUAL_ENV", "")
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, **env}, cwd=Path(__file__).resolve().parents[2]
        )
        assert proc.returncode == 0, proc.stderr
        results[threads] = json.loads(proc.stdout.strip().splitlines()[-1])

    assert results[1]["e_max"] == pytest.approx(results[16]["e_max"], rel=1e-9)
    assert results[1]["dsr"] == pytest.approx(results[16]["dsr"], rel=1e-9)


def test_statistical_determinism_over_20_seeds() -> None:
    """A stochastic method (the SPA bootstrap) run with >=20 different
    seeds should give qualitatively STABLE conclusions (not bitwise-equal
    p-values, but consistent accept/reject decisions) on a case with a
    clear, strong effect."""
    rng = np.random.RandomState(0)
    n = 500
    bench = np.abs(rng.randn(n)) + 1.0
    better = np.abs(rng.randn(n)) * 0.1
    decisions = []
    for seed in range(20):
        res = hansen_spa(bench, better.reshape(-1, 1), reps=200, seed=seed)
        decisions.append(not res.benchmark_not_rejected)
    assert sum(decisions) >= 18  # stable conclusion in >=90% of seeds for a strong effect
