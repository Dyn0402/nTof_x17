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


def test_wide_hot_band_does_not_weld_a_noise_column_onto_a_track():
    """The reason seeding clusters the CLEAN strips (2026-09-08).

    A real track and a noise column, separated by a hot run WIDER than the gap
    threshold — D-y 44-63 is exactly this, 20 strips / 15.6 mm. Clustering on
    all strips bridges through the band and returns ONE cluster spanning both;
    the fit then gets a window twice as wide as the track, which on D measured
    chi2/dof 62 against 25 for an uncontaminated window. Deleting the hot
    strips reopens the gap and the two separate.
    """
    print('seeding: a WIDE hot band must not weld a noise column onto a track')
    p = ws.PITCH_MM
    # track at strips 24-43, hot band 44-63 (D-y's real one), column 64-71
    ch = np.arange(24, 72)
    pos = ch * p
    amp = np.full(len(ch), 100.0)
    hot = np.arange(44, 64)
    check('the band alone is wider than the gap threshold',
         len(hot) * p > ws.GAP_THRESHOLD_MM,
         f'{len(hot) * p:.1f} mm vs {ws.GAP_THRESHOLD_MM} mm')

    welded = ws.seed_candidates(pos, ch, amp, min_strips=5, hot=None)
    check('without the mask the whole 48-strip run is ONE cluster',
         len(welded) == 1 and welded[0].n_strips == 48,
         f'got {[c.n_strips for c in welded]}')

    out = ws.seed_candidates(pos, ch, amp, min_strips=5, n_candidates=3, hot=hot)
    check('with the mask it separates into 2 candidates',
         len(out) == 2, f'got {[c.n_strips for c in out]}')
    check('the best candidate is the 20-strip track, not the welded 48',
         len(out) == 2 and out[0].n_strips == 20,
         f'got {out[0].n_strips if out else None}')
    check('neither candidate reaches into the band',
         all(not set(c.channels.tolist()) & set(hot.tolist()) for c in out))

    allhot = ws.seed_candidates(hot * p, hot, np.full(len(hot), 100.0),
                                min_strips=5, hot=hot)
    check('an all-hot column still never seeds', allhot == [], f'got {allhot}')


def test_narrow_hot_band_is_crossed_for_free():
    """The other half of the same rule, and the common case: D's hot runs are
    mostly narrower than the gap threshold (x 448-461 is 14 strips / 10.9 mm).
    Deleting those strips leaves a SUB-threshold hole, so a track crossing one
    is neither split nor holed — the ordinary gap constant does the work, with
    no special-casing."""
    print('seeding: a track crossing a NARROW hot band stays one cluster')
    p = ws.PITCH_MM
    ch = np.arange(440, 476)
    pos = ch * p
    amp = np.full(len(ch), 100.0)
    hot = np.arange(448, 462)              # 14 strips ~ 10.9 mm < gap_mm
    check('the band is narrower than the gap threshold',
         len(hot) * p < ws.GAP_THRESHOLD_MM,
         f'{len(hot) * p:.1f} mm vs {ws.GAP_THRESHOLD_MM} mm')

    out = ws.seed_candidates(pos, ch, amp, min_strips=5, n_candidates=3, hot=hot)
    check('the track is not split by the band',
         len(out) == 1, f'got {[c.n_strips for c in out]}')
    check('and its window spans the band, hot strips included as members',
         len(out) == 1 and out[0].n_strips == 36,
         f'got {out[0].n_strips if out else None}')
    check('the hot strips really are members (they reach the fit, down-weighted)',
         len(out) == 1 and set(hot.tolist()) <= set(out[0].channels.tolist()))


def test_empty_hot_mask_is_the_historical_behaviour():
    print('seeding: no hot mask -> unchanged from before the wildcard')
    p = ws.PITCH_MM
    ch = np.arange(40, 64)
    pos = ch * p
    amp = np.full(len(ch), 100.0)
    a = ws.seed_candidates(pos, ch, amp, min_strips=5, n_candidates=3, hot=None)
    for label, h in (('hot=[]', []), ('hot=all-clean', [900, 901])):
        b = ws.seed_candidates(pos, ch, amp, min_strips=5, n_candidates=3, hot=h)
        check(f'{label} gives the same seeds as hot=None',
             len(a) == len(b) and all(x.n_strips == y.n_strips
                                      and (x.channels == y.channels).all()
                                      for x, y in zip(a, b)))


if __name__ == '__main__':
    test_hot_channel_stays_in_fit_but_down_weighted()
    test_bundle_roundtrip()
    test_seed_never_seeds_on_hot_alone()
    test_track_crossing_hot_strip_is_not_split_or_lost()
    test_wide_hot_band_does_not_weld_a_noise_column_onto_a_track()
    test_narrow_hot_band_is_crossed_for_free()
    test_empty_hot_mask_is_the_historical_behaviour()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    sys.exit(1 if FAILS else 0)
