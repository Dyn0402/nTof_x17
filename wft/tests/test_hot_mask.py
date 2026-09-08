#!/usr/bin/env python3
"""Hot-channel wildcard (HANDOFF_D_NOISY_CHANNELS.md, 2026-09-08): a channel
listed in the bundle's ``hot`` mask must stay IN the fit and IN the seed
cluster it belongs to (unlike ``dead``, which is censored) but be unable to
either dominate the chi2 or, on its own, qualify a cluster as a seed.

Self-contained (synthetic bundle, like test_dead_mask/test_share_modes).

    ../../.venv/bin/python wft/tests/test_hot_mask.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft.calib import CalibrationBundle              # noqa: E402
from wft import model as wm                          # noqa: E402
from wft import seed as ws                            # noqa: E402
from wft.tests.test_share_modes import synth_bundle  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def window():
    """5-strip window: a clean model-generated track, nothing forced -- a hot
    channel still reads real (if noise-shaped) signal, so unlike the dead-mask
    test there is no 'broken readout' value to substitute."""
    pos = (np.arange(5) - 2) * wm.PITCH
    M = wm.build_matrix('x', pos, 0.05, 0.004, 200.0, wm.HYPER)
    q = np.zeros(wm.K)
    q[2:8] = 900.0
    W = (M @ q).reshape(5, wm.NSAMP)
    return dict(W=W, pos=pos, noise=np.full(5, 8.0), ch=np.arange(5))


def test_hot_channel_stays_in_fit_but_down_weighted():
    print('hot channel: down-weighted, not censored')
    cal = synth_bundle()
    cal.hot = {'x': [3]}
    wm.use_calibration(cal)
    wm.set_nsamp(32)
    P = window()

    W, noise, pos, sat = wm.prep_plane(P, 'x')
    check('hot channel is NOT censored (sat untouched)', not sat[3].any(),
         f'sat[3]={sat[3]}')
    check('hot channel noise is inflated by exactly HOT_NOISE_INFLATION',
         abs(noise[3] / (8.0 / wm.GAIN['x'][3]) - wm.HOT_NOISE_INFLATION) < 1e-9,
         f'noise[3]={noise[3]}')
    check('non-hot channels are untouched',
         all(abs(noise[i] - 8.0 / wm.GAIN['x'][i]) < 1e-9 for i in (0, 1, 2, 4)))

    c_masked, _ = wm.chi2_plane('x', W, noise, pos, sat, 0.05, 0.004, 200.0,
                                wm.HYPER, snap_t0=False)
    cal_free = synth_bundle()
    wm.use_calibration(cal_free)
    W2, noise2, pos2, sat2 = wm.prep_plane(P, 'x')
    dof_masked = int((~sat).sum())
    dof_free = int((~sat2).sum())
    check('hot channel keeps its dof (unlike dead, which loses it)',
         dof_masked == dof_free, f'{dof_masked} vs {dof_free}')


def test_bundle_roundtrip():
    print('bundle round-trip')
    import tempfile
    cal = synth_bundle()
    cal.hot = {'x': [3], 'y': []}
    with tempfile.TemporaryDirectory() as td:
        cal.save(os.path.join(td, 'b'))
        back = CalibrationBundle.load(os.path.join(td, 'b'))
    check('hot mask round-trips', back.hot == {'x': [3], 'y': []},
         f'got {back.hot}')


def test_seed_never_seeds_on_hot_alone():
    print('seeding: never seed on a flagged channel alone')
    # a pure noise column, entirely on hot channels -- must not become a seed
    pos = np.array([10.0, 10.8, 11.6, 12.4])
    ch = np.array([100, 101, 102, 103])
    amp = np.array([50.0, 60.0, 55.0, 45.0])
    out = ws.seed_candidates(pos, ch, amp, min_strips=3, hot=ch)
    check('all-hot cluster is rejected as a seed', out == [], f'got {out}')

    # the SAME footprint with real channel numbers (not in `hot`) is accepted
    out2 = ws.seed_candidates(pos, ch, amp, min_strips=3, hot=[999])
    check('the identical cluster is accepted when nothing in it is hot',
         len(out2) == 1 and out2[0].n_strips == 4)


def test_track_crossing_hot_strip_is_not_split_or_lost():
    print('seeding: a track crossing one hot strip stays one cluster')
    # 5 clean strips with one hot strip in the MIDDLE -- gap-clustering must
    # bridge across it (it is still <= gap_mm away), and the cluster must be
    # accepted because it has >= min_strips CLEAN strips even discounting the
    # hot one, and must still contain all 5 channels (nothing dropped).
    pos = np.array([10.0, 10.8, 11.6, 12.4, 13.2])
    ch = np.array([50, 51, 52, 53, 54])
    amp = np.array([100.0, 900.0, 950.0, 920.0, 110.0])   # ch 52 is the hot spike
    out = ws.seed_candidates(pos, ch, amp, min_strips=4, hot=[52])
    check('cluster survives with all 5 channels (not split, not dropped)',
         len(out) == 1 and out[0].n_strips == 5,
         f'got {[c.n_strips for c in out]}')
    check('the hot channel is still a MEMBER of the returned cluster',
         len(out) == 1 and 52 in out[0].channels)

    # but if there are only 3 clean strips plus the hot one, and min_strips=4,
    # the cluster must now be rejected (3 clean < 4) even though raw count is 4
    pos2 = pos[:4]
    ch2 = ch[:4]
    amp2 = amp[:4]
    out3 = ws.seed_candidates(pos2, ch2, amp2, min_strips=4, hot=[52])
    check('a cluster with too few CLEAN strips is rejected even at raw count 4',
         out3 == [], f'got {out3}')


if __name__ == '__main__':
    test_hot_channel_stays_in_fit_but_down_weighted()
    test_bundle_roundtrip()
    test_seed_never_seeds_on_hot_alone()
    test_track_crossing_hot_strip_is_not_split_or_lost()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    sys.exit(1 if FAILS else 0)
