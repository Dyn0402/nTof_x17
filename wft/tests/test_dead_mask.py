#!/usr/bin/env python3
"""Dead-channel censoring (T1.3): a channel listed in the bundle's ``dead``
mask must contribute NOTHING to the fit — its samples leave the chi2 sum and
the dof, and the one-sided saturation penalty cannot pull on it either.

Self-contained (synthetic bundle, like test_share_modes).

    ../../.venv/bin/python wft/tests/test_dead_mask.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft.calib import CalibrationBundle              # noqa: E402
from wft import model as wm                          # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402


def window(dead_val=0.0):
    """5-strip window: model-generated track + one channel forced to a
    constant (its 'broken-connection' readout)."""
    pos = (np.arange(5) - 2) * wm.PITCH
    M = wm.build_matrix('x', pos, 0.05, 0.004, 200.0, wm.HYPER)
    q = np.zeros(wm.K)
    q[2:8] = 900.0
    W = (M @ q).reshape(5, wm.NSAMP)
    W[3] = dead_val                                  # channel 3 reads baseline
    return dict(W=W, pos=pos, noise=np.full(5, 8.0), ch=np.arange(5))


def chi2(P, dead):
    cal = synth_bundle()
    cal.dead = {'x': dead}
    wm.use_calibration(cal)
    wm.set_nsamp(32)
    W, noise, pos, sat = wm.prep_plane(P, 'x')
    c, _ = wm.chi2_plane('x', W, noise, pos, sat, 0.05, 0.004, 200.0,
                         wm.HYPER, snap_t0=False)
    return c, int((~sat).sum())


def main():
    wm.use_calibration(synth_bundle())
    wm.set_nsamp(32)
    P = window()

    c_masked, dof_masked = chi2(P, dead=[3])
    c_free, dof_free = chi2(P, dead=[])
    print(f'masked: chi2 {c_masked:.2f} dof {dof_masked}   '
          f'free: chi2 {c_free:.2f} dof {dof_free}')
    assert dof_free - dof_masked == wm.NSAMP, \
        'dead channel must leave the dof'
    assert c_masked < c_free, \
        'masking the baseline-reading channel must remove its chi2 pull'

    # the masked chi2 must be INDEPENDENT of what the dead channel reads —
    # baseline, garbage, or apparent saturation (no censor penalty either)
    for val, label in ((0.0, 'baseline'), (500.0, 'garbage'),
                       (wm.SAT + 10, 'saturated')):
        c, dof = chi2(window(dead_val=val), dead=[3])
        print(f'dead reads {label:>9}: chi2 {c:.6f} dof {dof}')
        assert abs(c - c_masked) < 1e-6, \
            f'dead channel reading {label} leaked into chi2'
        assert dof == dof_masked

    # empty-window guard: all channels dead -> inf, not a spurious perfect fit
    c_all, _ = chi2(P, dead=[0, 1, 2, 3, 4])
    assert np.isinf(c_all), 'fully-censored window must return inf'
    print('all dead -> chi2 inf (guard ok)')

    # a bundle round-trips the mask
    import tempfile
    cal = synth_bundle()
    cal.dead = {'x': [3], 'y': []}
    with tempfile.TemporaryDirectory() as td:
        cal.save(os.path.join(td, 'b'))
        back = CalibrationBundle.load(os.path.join(td, 'b'))
    assert back.dead == {'x': [3], 'y': []}, 'dead mask must round-trip'
    print('bundle round-trip ok')
    print('OK')


def test_dead_mask():
    main()


if __name__ == '__main__':
    main()
