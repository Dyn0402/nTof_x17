#!/usr/bin/env python3
"""The c2_over_c1 slave: c2 = r * c1, and OFF by default.

Added 2026-08-18 with the hyper it tests.  The shipped bundles all carry
c2 > c1 -- the +-2 strip receiving more than the +-1 strip -- because the
ref-pinned cosmic chi2 is flat in that direction.  The head-on beam measures
the ratio directly at 0.45 +- 0.02 and near-vertical bench cosmics at
0.63 +- 0.09, so slaving it removes a free hyper and makes the ordering
structural.  What must hold:

  1. absent the key, nothing changes (no existing bundle carries it),
  2. with the key, the +-2 content scales as r * c1 and ignores hyper['c2'],
  3. the slave is applied BEFORE the per-plane kY scaling, so the ratio is
     the same on both planes,
  4. it works in both share modes.

    ../../.venv/bin/python -m pytest wft/tests/test_c2_ratio.py -q
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import model as wm                          # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402


def _pm2(plane, hyper):
    """Peak amplitude on the +2 strip from a single depth bin, geometry off."""
    h = dict(hyper)
    h['sigma_p0'], h['Dp'] = 0.01, 0.001
    pos = (np.arange(7) - 3) * wm.PITCH
    M = wm.build_matrix(plane, pos, 0.0, 0.0, 200.0, h)
    q = np.zeros(wm.K)
    q[3] = 1.0
    W = (M @ q).reshape(7, wm.NSAMP)
    return float(W[5].max()), float(W[4].max())      # (+2, +1)


def test_absent_key_is_a_no_op():
    wm.use_calibration(synth_bundle())
    base = dict(wm.HYPER)
    for mode in ('delay', 'lp'):
        wm.set_share_mode(mode)
        a = _pm2('x', base)
        b = _pm2('x', dict(base, c2_over_c1=None))
        assert np.allclose(a, b), mode


def test_ratio_replaces_c2():
    wm.use_calibration(synth_bundle())
    base = dict(wm.HYPER)
    for mode in ('delay', 'lp'):
        wm.set_share_mode(mode)
        # the +2 amplitude must follow r, and must NOT depend on hyper['c2']
        p_ref, _ = _pm2('x', dict(base, c2=0.10))
        for r in (0.3, 0.6):
            got = {}
            for c2_junk in (0.0, 0.5, 1.3):
                p2, _p1 = _pm2('x', dict(base, c2=c2_junk, c2_over_c1=r))
                got[c2_junk] = p2
            vals = list(got.values())
            assert np.allclose(vals, vals[0], rtol=1e-9), (mode, r, got)
            # amplitude is linear in c2, so the ratio carries straight through
            assert np.isclose(vals[0] / p_ref, r * base['c1'] / 0.10,
                              rtol=1e-6), (mode, r)


def test_slave_precedes_the_plane_scaling():
    """c2_eff/c1_eff must be r on BOTH planes, whatever kY is."""
    wm.use_calibration(synth_bundle())
    for mode in ('delay', 'lp'):
        wm.set_share_mode(mode)
        h = dict(wm.HYPER, kY=2.5, cX=1.0, c2=0.0, c2_over_c1=0.6,
                 tau_s=1.0, sigma_s=1.0)
        # with a negligible delay the two copies share one shape, so the
        # +2/+1 peak ratio is c2_eff/c1_eff directly
        rx = np.divide(*_pm2('x', h))
        ry = np.divide(*_pm2('y', h))
        assert np.isclose(rx, ry, rtol=2e-2), (mode, rx, ry)
        assert np.isclose(rx, 0.6, rtol=5e-2), (mode, rx)


if __name__ == '__main__':
    test_absent_key_is_a_no_op()
    test_ratio_replaces_c2()
    test_slave_precedes_the_plane_scaling()
    print('ok')
