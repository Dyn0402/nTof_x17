#!/usr/bin/env python3
"""Per-view kernel keys: c2_over_c1_<plane>, sigma_p0_<plane>, c1_asym_<plane>.

Added 2026-10-09 (sps_beam_test_26/analysis/plane_ratio).  The head-on
neighbour pattern differs by view: Y carries a large, delayed +-2 copy, X
almost none, and X's +-1 is one-sided.  What must hold:

  1. absent keys, build_matrix is bit-identical to the global form,
  2. each key acts on its own plane only,
  3. c1_asym = 0 is a no-op and a != 0 moves +-1 charge between sides
     without changing the total,
  4. the ordering gate judges each view separately.

    ../../.venv/bin/python -m pytest wft/tests/test_plane_kernel.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft.calib import check_kernel_ordering, effective_c2  # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

POS = (np.arange(7) - 3) * wm.PITCH


def _M(plane, hyper):
    return wm.build_matrix(plane, POS, 0.1, 0.0, 200.0, hyper)


def _strips(plane, hyper):
    h = dict(hyper, sigma_p0=hyper.get('sigma_p0', 0.01), Dp=0.001)
    q = np.zeros(wm.K)
    q[3] = 1.0
    W = (wm.build_matrix(plane, POS, 0.0, 0.0, 200.0, h) @ q).reshape(7, wm.NSAMP)
    return W.sum(axis=1)                 # area per strip, centre = index 3


@pytest.fixture(params=['delay', 'lp'])
def base(request):
    wm.use_calibration(synth_bundle())
    wm.set_share_mode(request.param)
    return dict(wm.HYPER, c2=0.0, c2_over_c1=0.6, c1=0.2)


def test_absent_keys_identical(base):
    for p in ('x', 'y'):
        a = _M(p, base)
        b = _M(p, dict(base, c2_over_c1_x=0.6, c2_over_c1_y=0.6,
                       sigma_p0_x=base['sigma_p0'], sigma_p0_y=base['sigma_p0'],
                       c1_asym_x=0.0, c1_asym_y=0.0))
        assert np.array_equal(a, b), p


def test_plane_ratio_acts_on_its_plane_only(base):
    h = dict(base, c2_over_c1_y=0.9)
    assert np.array_equal(_M('x', base), _M('x', h))
    ay, by = _strips('y', base), _strips('y', h)
    assert by[5] > ay[5] and by[1] > ay[1]          # +-2 grew on Y
    # +-1 and centre move only by the c2 copy of the tiny direct charge 2 away
    assert np.allclose(ay[[2, 3, 4]], by[[2, 3, 4]], rtol=1e-3)
    hx = dict(base, c2_over_c1_x=0.1)
    ax, bx = _strips('x', base), _strips('x', hx)
    assert bx[5] < ax[5]
    assert np.array_equal(_M('y', base), _M('y', hx))


def test_plane_sigma_p0(base):
    h = dict(base, sigma_p0_x=0.6)
    assert np.array_equal(_M('y', base), _M('y', h))
    assert not np.array_equal(_M('x', base), _M('x', h))


def test_asymmetry(base):
    s0 = _strips('x', base)
    s = _strips('x', dict(base, c1_asym_x=0.3))
    # strip j takes (1+a) from j+1: the -1 neighbour (index 2) gains
    assert s[2] > s0[2] and s[4] < s0[4]
    assert np.isclose(s.sum(), s0.sum(), rtol=1e-9)
    assert np.array_equal(_M('y', base), _M('y', dict(base, c1_asym_x=0.3)))


def test_gate_per_view(monkeypatch):
    monkeypatch.delenv('WFT_ALLOW_INVERTED_KERNEL', raising=False)
    h = dict(c1=0.1, c2=0.0, c2_over_c1=0.5)
    check_kernel_ordering(dict(h, c2_over_c1_x=0.1, c2_over_c1_y=0.95))
    assert np.isclose(effective_c2(dict(h, c2_over_c1_y=0.9), 'y'), 0.09)
    assert np.isclose(effective_c2(dict(h, c2_over_c1_y=0.9), 'x'), 0.05)
    with pytest.raises(ValueError):
        check_kernel_ordering(dict(h, c2_over_c1_y=1.2))
