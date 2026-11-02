"""Pinned regression vectors for Gate v2 formulas, from the verified
research extraction of gate_v2_spec.json / D7_gate_v2_spec.md."""

from __future__ import annotations

import math

import pytest

from model_lab.gate_v2.formulas import dsr, expected_max_z, min_btl, min_trl, se_sr, sr0

E_MAX_TABLE = {
    2: 0.519755,
    10: 1.574598,
    46: 2.244171,
    88: 2.485295,
    100: 2.530603,
    250: 2.837677,
    1000: 3.255122,
    10000: 3.860665,
}


@pytest.mark.parametrize("N,expected", list(E_MAX_TABLE.items()))
def test_expected_max_z_pinned_table(N: int, expected: float) -> None:
    assert expected_max_z(N) == pytest.approx(expected, abs=1e-5)


def test_expected_max_z_domain_error_below_2() -> None:
    with pytest.raises(ValueError):
        expected_max_z(1)


def test_kat1_worked_example() -> None:
    """N=100, V=0.5 (annual), T=1250, skew=-3, kurt=10, 250 obs/yr,
    SR_ann=2.5 -> SR0_daily=0.1132, DSR=0.9004, DSR@N=46=0.9505 (largest N)."""
    N = 100
    V_daily = 0.5 / 250.0
    T = 1250
    skew, kurt = -3.0, 10.0
    sr_hat = 2.5 / math.sqrt(250.0)

    assert sr0(N, V_daily) == pytest.approx(0.1132, abs=2e-4)
    assert dsr(sr_hat, N, V_daily, T, skew, kurt) == pytest.approx(0.9004, abs=2e-4)
    assert dsr(sr_hat, 46, V_daily, T, skew, kurt) == pytest.approx(0.9505, abs=2e-4)

    # N=46 must be the LARGEST N with DSR >= 0.95
    assert dsr(sr_hat, 46, V_daily, T, skew, kurt) >= 0.95
    assert dsr(sr_hat, 47, V_daily, T, skew, kurt) < 0.95


def test_kat1_normal_case_variant() -> None:
    """skew=0, kurt=3 (Normal) -> largest N with DSR>=0.95 is N=88, DSR@88=0.9505."""
    V_daily = 0.5 / 250.0
    T = 1250
    sr_hat = 2.5 / math.sqrt(250.0)
    assert dsr(sr_hat, 88, V_daily, T, 0.0, 3.0) == pytest.approx(0.9505, abs=2e-4)
    assert dsr(sr_hat, 88, V_daily, T, 0.0, 3.0) >= 0.95
    assert dsr(sr_hat, 89, V_daily, T, 0.0, 3.0) < 0.95


def test_se_a_pinned_vector() -> None:
    """SR=0.1, T=1000, skew=-2, kurt=8 -> SE_a = 0.034910."""
    assert se_sr(0.1, 1000, -2.0, 8.0) == pytest.approx(0.034910, abs=1e-6)


def test_min_trl_pinned_vector() -> None:
    """SR=0.1, SR*=0, skew=-1, kurt=6, alpha=0.05 -> T*=301.99."""
    assert min_trl(0.1, 0.0, -1.0, 6.0, 0.05) == pytest.approx(301.99, abs=0.1)


@pytest.mark.parametrize(
    "N,expected",
    [(10, 2.479), (45, 4.998), (100, 6.404), (1000, 10.596), (10000, 14.905)],
)
def test_min_btl_pinned_vectors_at_target_sr_1(N: int, expected: float) -> None:
    assert min_btl(N, 1.0) == pytest.approx(expected, abs=5e-3)


def test_sr0_single_configuration_is_zero() -> None:
    """N_raw_cum == 1 (exactly one pre-registered configuration ever) -> SR0 = 0."""
    assert sr0(1, 0.5) == 0.0


def test_dsr_is_a_probability() -> None:
    val = dsr(0.15, 50, 0.3, 500, -1.0, 5.0)
    assert 0.0 <= val <= 1.0
