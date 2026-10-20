#!/usr/bin/env python3
"""wft.xasym: the opt-in X neighbour-asymmetry position correction.

  1. scale 0 is the identity, NaN positions stay NaN,
  2. the shift is -K dx in the map's cell, linear in scale,
  3. excluded / unknown builds refuse instead of silently passing through,
  4. the map frame: cell i spans [49.92 i, 49.92 (i+1)) mm.

    ../../.venv/bin/python -m pytest wft/tests/test_xasym.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import xasym  # noqa: E402

X = np.array([10.0, 120.0, 260.0, 390.0, np.nan])
Y = np.array([30.0, 200.0, 310.0, 5.0, 100.0])


@pytest.mark.parametrize('build', ['det2', 'det3 June-15'])
def test_identity_and_nan(build):
    out = xasym.correct_x(X, Y, build, scale=0.0)
    assert np.array_equal(out[:4], X[:4])
    assert np.isnan(out[4])


def test_shift_is_minus_K_dx_and_linear():
    M = xasym.maps()
    G = M['builds']['det2']['dx_um']
    i, j = 2, 6                                   # x 99.84..149.76, y 299.52..349.44
    x, y = 120.0, 310.0
    want = -M['K'] * G[i][j] * 1e-3
    assert xasym.correct_x([x], [y], 'det2')[0] - x == pytest.approx(want)
    assert xasym.correct_x([x], [y], 'det2', scale=2.5)[0] - x == pytest.approx(2.5 * want)


def test_refuses_unmapped_builds():
    for b in ('det4', 'det3 May', 'nonsense'):
        with pytest.raises(KeyError):
            xasym.correct_x(X, Y, b)


def test_cells_and_size():
    d = xasym.dx_mm(X[:4], Y[:4], 'det3 June-15')
    assert np.all(np.abs(d) < 0.5)                # tens of um, never mm
    a = xasym.dx_mm([49.91], [0.0], 'det2')[0]
    b = xasym.dx_mm([49.93], [0.0], 'det2')[0]
    G = xasym.maps()['builds']['det2']['dx_um']
    assert a == pytest.approx((G[0][0] or 0) * 1e-3) and b == pytest.approx((G[1][0] or 0) * 1e-3)
