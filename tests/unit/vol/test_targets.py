from __future__ import annotations

import numpy as np

from model_lab.vol.targets import forward_realised_variance


def test_forward_realised_variance_known_answer() -> None:
    r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = forward_realised_variance(r, horizon=2)
    # target[i] = r[i+1]**2 + r[i+2]**2
    expected = np.array([2.0**2 + 3.0**2, 3.0**2 + 4.0**2, 4.0**2 + 5.0**2])
    assert len(target) == len(r) - 2
    np.testing.assert_allclose(target, expected)


def test_forward_realised_variance_horizon_one_is_next_squared_return() -> None:
    r = np.array([0.01, -0.02, 0.03, -0.01])
    target = forward_realised_variance(r, horizon=1)
    np.testing.assert_allclose(target, r[1:] ** 2)


def test_forward_realised_variance_rejects_nonpositive_horizon() -> None:
    import pytest

    with pytest.raises(ValueError, match="horizon"):
        forward_realised_variance(np.array([1.0, 2.0]), horizon=0)


def test_forward_realised_variance_empty_when_horizon_exceeds_series() -> None:
    target = forward_realised_variance(np.array([1.0, 2.0]), horizon=5)
    assert len(target) == 0
