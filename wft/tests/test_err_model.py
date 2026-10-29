#!/usr/bin/env python3
"""The error-model passes of fit_plane_raw (ERR_K / ERR_F / STRIP_TAU, 2026-10-29).

  1. all off: fit_plane_raw is _fit_plane_raw1 exactly (and the module state is clean afterwards),
  2. on: the fit runs, its weighted chi2/dof falls (the variance grows with the prediction), the geometry stays near the
     noise-only fit, and the pass state (_SIG, _SSCALE) is always cleared,
  3. STRIP_TAU -> tiny: the strip scales stay at 1 (the prior wins).
    ../../.venv/bin/python -m pytest wft/tests/test_err_model.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from wft import model as wm                                    # noqa: E402
from wft.tests.test_lm_fitter import HAVE_DATA, _setup          # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason='no det3 vladder cache/bundle on this machine')


@pytest.fixture
def data():
    cal, C = _setup()
    yield cal, C
    wm.ERR_K = wm.ERR_F = wm.STRIP_TAU = 0.0


def _planes(C, n=6):
    for eid in sorted(C)[:n]:
        for p in 'xy':
            if p in C[eid]:
                yield C[eid][p], p


def test_off_is_identical(data):
    cal, C = data
    for P, p in _planes(C):
        g = wm.init_guess(P, p)
        a = wm.fit_plane_raw(P, p, *g)
        b = wm._fit_plane_raw1(P, p, *g)
        assert a['chi2'] == b['chi2'] and a['p0'] == b['p0'] and a['w'] == b['w']
        assert wm._SIG is None and wm._SSCALE is None


@pytest.mark.parametrize('k,f,tau', [(11.5, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, 0.0, 0.1), (11.5, 0.0, 0.1)])
def test_on_runs_and_clears(data, k, f, tau):
    cal, C = data
    dw = []
    for P, p in _planes(C):
        g = wm.init_guess(P, p)
        b = wm._fit_plane_raw1(P, p, *g)
        wm.ERR_K, wm.ERR_F, wm.STRIP_TAU = k, f, tau
        a = wm.fit_plane_raw(P, p, *g)
        wm.ERR_K = wm.ERR_F = wm.STRIP_TAU = 0.0
        assert wm._SIG is None and wm._SSCALE is None
        assert np.isfinite(a['chi2'])
        if k > 0 or f > 0:
            assert a['chi2'] <= b['chi2'] + 1e-6
        dw.append(abs(a['w'] - b['w']))
    # typical slope change small (single blob planes move; this bundle leaves t0, hence p0, free)
    assert np.median(dw) < 0.5e-3


def test_tiny_tau_keeps_unit_scales(data):
    cal, C = data
    P, p = next(_planes(C))
    wm.STRIP_TAU = 1e-4
    r = wm.fit_plane_raw(P, p, *wm.init_guess(P, p))
    assert np.allclose(r['strip_scale'], 1.0, atol=1e-3)


def test_plain_charge_resolve(data):
    """QSOLVE_PLAIN: same geometry as the capped fit, charge from the uncapped solve (larger on bright planes)."""
    cal, C = data
    P, p = max(_planes(C, 12), key=lambda t: float(np.max(t[0]['W'])))
    g = wm.init_guess(P, p)
    try:
        wm.SAT_CAP = 600.0
        a = wm.fit_plane_raw(P, p, *g)
        wm.QSOLVE_PLAIN = True
        b = wm.fit_plane_raw(P, p, *g)
    finally:
        wm.SAT_CAP, wm.QSOLVE_PLAIN = 0.0, False
    assert b['p0'] == a['p0'] and b['w'] == a['w'] and b['t0'] == a['t0']
    assert np.allclose(b['q_fit'], a['q'])
    assert b['q'].sum() > a['q'].sum()
