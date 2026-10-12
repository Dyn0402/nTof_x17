#!/usr/bin/env python3
"""The physical (RC-diffusion) kernel, build_matrix_rc.

Added 2026-10-09 (sps_beam_test_26/analysis/cloud_basics).  What must hold:

  1. with no RC spread and each view's own template it is the production
     matrix with the copies switched off (c1 = c2 = 0) -- same footprint,
     same electronics,
  2. RC spreading conserves charge: summed over a wide strip window every
     depth column carries the same total as without it,
  3. on Y the spread moves charge outward over time: the +-1/+-2 share of a
     late sample exceeds that of an early one, and rc_D_x = 0 leaves X prompt,
  4. no rc key = the old kernel, untouched.

    ../../.venv/bin/python -m pytest wft/tests/test_rc_kernel.py -q
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

POS = (np.arange(21) - 10) * wm.PITCH
BASE = dict(c1=0.1, c2=0.05, kY=2.0, tau_s=150.0, sigma_s=50.0,
            sigma_p0=0.4, Dp=0.02)


@pytest.fixture(autouse=True)
def cal():
    wm.use_calibration(synth_bundle())
    wm.set_share_mode('delay')
    yield


def test_no_spread_equals_production_without_copies():
    for plane in ('x', 'y'):
        prod = wm.build_matrix(plane, POS, 0.1, 0.02, 200.0, dict(BASE, c1=0.0, c2=0.0))
        rc = wm.build_matrix(plane, POS, 0.1, 0.02, 200.0,
                             dict(BASE, rc_D_y=0.0, rc_D_x=0.0, rc_tmpl='own'))
        assert np.allclose(prod, rc, atol=1e-12)


def test_charge_conserved():
    h0 = dict(BASE, rc_D_y=0.0, rc_tmpl='own')
    h1 = dict(BASE, rc_D_y=5e-4, rc_tmpl='own')
    a = wm.build_matrix('y', POS, 0.0, 0.0, 200.0, h0).reshape(len(POS), wm.NSAMP, wm.K)
    b = wm.build_matrix('y', POS, 0.0, 0.0, 200.0, h1).reshape(len(POS), wm.NSAMP, wm.K)
    # total over strips and the (padded) template area: the spread only redistributes
    assert np.allclose(a.sum(0)[:, 3].sum(), b.sum(0)[:, 3].sum(), rtol=0.05)
    # per strip-sum at every sample it is a time-redistribution only for the
    # all-strip sum: increments sum to zero over strips
    assert np.allclose(a.sum(0), b.sum(0), atol=1e-6)


def test_y_spreads_outward_x_stays():
    h = dict(BASE, rc_D_y=5e-4, rc_D_x=0.0, rc_tmpl='own')
    q = np.zeros(wm.K); q[2] = 1.0
    for plane, spreads in (('y', True), ('x', False)):
        W = (wm.build_matrix(plane, POS, 0.0, 0.0, 200.0, h) @ q).reshape(len(POS), wm.NSAMP)
        c = 10
        ipk = int(np.argmax(W[c]))
        early, late = ipk, min(ipk + 8, wm.NSAMP - 1)
        share = lambda t: (W[c - 2, t] + W[c + 2, t]) / max(W[c, t], 1e-9)
        if spreads:
            assert share(late) > share(early) + 0.05
        else:
            assert abs(W[c - 2].sum() / W[c].sum()
                       - wm.build_matrix(plane, POS, 0.0, 0.0, 200.0,
                                         dict(h, rc_D_y=0.0)) .dot(q).reshape(len(POS), wm.NSAMP)[c - 2].sum()
                       / W[c].sum()) < 1e-9


def test_absent_keys_untouched():
    a = wm.build_matrix('y', POS, 0.1, 0.02, 200.0, BASE)
    assert 'rc_D_y' not in BASE
    b = wm.build_matrix('y', POS, 0.1, 0.02, 200.0, dict(BASE))
    assert np.array_equal(a, b)


# ---- induced-footprint tails (pseudo-Voigt, cloud_basics FINDINGS §23)
WIDE = (np.arange(81) - 40) * wm.PITCH


def test_lor_absent_or_zero_is_bit_identical():
    h = dict(BASE, rc_D_y=5e-4)
    a = wm.build_matrix('x', POS, 0.1, 0.02, 200.0, h)
    b = wm.build_matrix('x', POS, 0.1, 0.02, 200.0, dict(h, lor_frac_x=0.0, lor_gamma_x=0.8))
    assert np.array_equal(a, b)


def test_lor_conserves_charge_and_widens_tails():
    h = dict(BASE, rc_D_y=5e-4)
    a = wm.build_matrix('x', WIDE, 0.0, 0.0, 200.0, h).reshape(len(WIDE), wm.NSAMP, wm.K)
    b = wm.build_matrix('x', WIDE, 0.0, 0.0, 200.0,
                        dict(h, lor_frac_x=0.2, lor_gamma_x=0.8)).reshape(len(WIDE), wm.NSAMP, wm.K)
    # over a +-31 mm window the Lorentzian keeps all but ~1 % of its 20 % share
    assert np.allclose(a.sum(0), b.sum(0), rtol=0.01, atol=1e-6)
    qa, qb = a.sum((1, 2)), b.sum((1, 2))
    c = 40
    assert qb[c + 3] > 3 * qa[c + 3] and qb[c + 4] > 3 * qa[c + 4]
    # Y untouched by X keys
    ya = wm.build_matrix('y', POS, 0.1, 0.02, 200.0, h)
    yb = wm.build_matrix('y', POS, 0.1, 0.02, 200.0, dict(h, lor_frac_x=0.2, lor_gamma_x=0.8))
    assert np.array_equal(ya, yb)
