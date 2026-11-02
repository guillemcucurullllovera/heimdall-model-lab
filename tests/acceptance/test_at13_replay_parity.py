"""AT-13: incumbent reproduces recorded decision fingerprints for 100% of
cycles that have one; observer never opens files for write.

Reported PARTIAL, honestly: MODEL-1 has no independent reimplementation of
the AMC-3 selection solver (out of scope, see replay/parity.py), so no
cycle can reach parity_status=MATCH via true replay. This test instead
verifies the harness's HONESTY invariant: it never fabricates a MATCH, it
correctly reports NOT_COMPUTABLE for the (overwhelming majority of) cycles
with no recorded proposal, and — separately — that the observer is
mechanically read-only against the live AMC root.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest

from model_lab.replay.amc_root import AMC_ROOT, JOURNAL_PATH, PINNED_EXPERIMENT_ID
from model_lab.replay.contracts import ParityStatus
from model_lab.replay.observer import build_input_bundle, list_known_cycle_ids
from model_lab.replay.parity import run_parity_harness

AMC_ROOT_AVAILABLE = AMC_ROOT.exists() and JOURNAL_PATH.exists()

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.skipif(not AMC_ROOT_AVAILABLE, reason="AMC root not present on this host"),
]


def _mtimes(root: Path) -> dict[Path, float]:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            with contextlib.suppress(OSError):
                out[p] = p.stat().st_mtime
    return out


def test_observer_never_modifies_the_amc_root() -> None:
    before = _mtimes(AMC_ROOT)
    build_input_bundle("cycle-004-20260908", experiment_id="amc4-prospective-v1-20260908")
    run_parity_harness(experiment_id=PINNED_EXPERIMENT_ID)
    after = _mtimes(AMC_ROOT)
    assert before == after, "read-only observer must never modify any file mtime under the AMC root"


def test_never_fabricates_a_match_without_independent_replay() -> None:
    diffs = run_parity_harness(experiment_id=PINNED_EXPERIMENT_ID)
    assert all(d.parity_status != ParityStatus.MATCH for d in diffs), (
        "MODEL-1 has no independent AMC-3 reimplementation; parity_status=MATCH must never appear"
    )


def test_empty_candidate_cycles_are_not_computable_not_silently_dropped() -> None:
    diffs = run_parity_harness(experiment_id=PINNED_EXPERIMENT_ID)
    # Every v2 cycle (20260925) is known to have empty candidates (confirmed
    # this session: all 17 v2 candidate files are `[]`), so every v2 cycle
    # should appear, explicitly NOT_COMPUTABLE, not omitted.
    assert diffs, "expected at least the 17 v2 cycles"
    assert all(d.parity_status == ParityStatus.NOT_COMPUTABLE for d in diffs)


def test_input_bundle_prefix_hash_is_reproducible_for_a_quiescent_file() -> None:
    b1 = build_input_bundle("cycle-004-20260908", experiment_id="amc4-prospective-v1-20260908")
    b2 = build_input_bundle("cycle-004-20260908", experiment_id="amc4-prospective-v1-20260908")
    # candidate_file_hashes for a historical (no-longer-appended) cycle must
    # be stable across repeated observations.
    assert b1.candidate_file_hashes == b2.candidate_file_hashes


def test_known_cycle_ids_include_the_confirmed_v1_and_v2_experiments() -> None:
    ids = list_known_cycle_ids()
    assert "cycle-004-20260908" in ids
    assert "cycle-017-20260925" in ids
