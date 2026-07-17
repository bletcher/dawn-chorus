"""ECDF primitives: grid evaluation, quantile crossing, KS / Wasserstein."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dawnchorus.ecdf import (_crossing, _ecdf_on_grid, _ks_stat, _wasserstein_1d,
                             ecdf_quantiles)


def test_ecdf_on_grid_is_monotone_and_reaches_one():
    grid = np.arange(0, 11, 1.0)
    F = _ecdf_on_grid(np.array([2.0, 4.0, 6.0, 8.0]), grid)
    assert np.all(np.diff(F) >= 0)          # non-decreasing
    assert F[-1] == 1.0                      # everything counted at/after last
    assert F[0] == 0.0                       # nothing before the first sample


def test_crossing_interpolates_linearly():
    grid = np.array([0.0, 10.0])
    F = np.array([0.0, 1.0])
    assert abs(_crossing(grid, F, 0.5) - 5.0) < 1e-9
    assert _crossing(grid, F, 0.0) == 0.0


def test_ks_stat_zero_for_identical_one_for_disjoint():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert _ks_stat(a, a) == 0.0
    b = a + 10.0
    assert _ks_stat(a, b) == 1.0             # CDFs never overlap -> max gap 1


def test_wasserstein_equals_the_constant_shift():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert _wasserstein_1d(a, a) == 0.0
    # A pure translation by 10 has W1 distance exactly 10 (minutes).
    assert abs(_wasserstein_1d(a, a + 10.0) - 10.0) < 1e-9


def test_ecdf_quantiles_on_empty_returns_empty_not_keyerror():
    # Sparse input (no morning cleared the detection floor) -> empty ECDF frame.
    out = ecdf_quantiles(pd.DataFrame(), by="month")
    assert out.empty
    assert "month" in out.columns  # shaped, so downstream .to_csv/consumers are safe
