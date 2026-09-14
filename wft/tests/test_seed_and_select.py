#!/usr/bin/env python3
"""
Unit tests for the two pieces of logic that are not the fit itself: cluster
seeding and candidate selection. These are where the reconstruction decides
*which charge* it is looking at, which is what the det3 gate showed matters
most (a wrong cluster puts the track 37 mm away).

    ../../.venv/bin/python wft/tests/test_seed_and_select.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import seed as ws          # noqa: E402
from wft import reco as wr          # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def test_clustering():
    print('clustering')
    pos = np.array([10.0, 10.8, 11.6, 12.4, 60.0, 60.8, 61.6])   # 4 + 3 strips
    ch = np.arange(len(pos))
    amp = np.array([100, 200, 150, 120, 900, 950, 800.0])
    one = ws.seed_plane(pos, ch, amp)
    check('largest cluster has 4 strips', one.n_strips == 4, f'got {one.n_strips}')
    check('n_dropped counts the other cluster', one.n_dropped == 3,
          f'got {one.n_dropped}')

    cands = ws.seed_candidates(pos, ch, amp, n_candidates=3)
    check('two candidates offered', len(cands) == 2, f'got {len(cands)}')
    check('candidates ranked by strip count',
          cands[0].n_strips >= cands[1].n_strips)
    check('the brighter cluster is available as a runner-up',
          any(abs(c.amp_sum - 2650.0) < 1e-6 for c in cands))

    # a cluster below MIN_STRIPS is not a candidate
    few = ws.seed_candidates(np.array([1.0, 1.8]), np.arange(2),
                             np.array([10.0, 10.0]), n_candidates=3)
    check('sub-threshold cluster rejected', few == [])


def test_significance_floor():
    print('significance floor')
    # FEU 7 max = 50 -> floor 5.0 (the 0.5 strip goes)
    # FEU 8 max = 20 -> floor 2.0 (the 6.0 strip stays, and would NOT survive a
    #                              per-event floor of 5.0 — that is the point)
    df = pd.DataFrame(dict(eventId=[1, 1, 1, 1], feu=[7, 7, 8, 8],
                           channel=[1, 2, 3, 4], amplitude=[10, 100, 10, 100.0],
                           significance=[0.5, 50.0, 6.0, 20.0]))
    out = ws.apply_significance_floor(df, rel=0.10)
    check('floor is per plane, not per event', len(out) == 3,
          f'kept {len(out)} of 4; the weaker plane must keep its 6.0 strip')
    check('the surviving weak-plane strip is the 6.0 one',
          6.0 in set(out['significance']))
    check('disabled floor keeps everything',
          len(ws.apply_significance_floor(df, rel=0)) == 4)


def test_local_floor():
    print('local significance floor')
    # a bright track at 0-3 mm, a faint one 200 mm away, and a weak strip
    # 10 mm from the bright track
    pos = np.array([0.0, 0.8, 1.6, 2.4, 3.2, 200.0, 200.8, 201.6, 202.4, 203.2, 10.0])
    sig = np.array([100, 90, 80, 70, 60, 8, 7, 7, 6, 6, 5.0])
    df = pd.DataFrame(dict(eventId=1, feu=7, channel=np.arange(len(pos)),
                           amplitude=10 * sig, significance=sig))
    g = ws.apply_significance_floor(df, rel=0.10, local_mm=0.0)
    check('plane-wide floor drops the faint track', not ({5, 6, 7, 8, 9} & set(g.channel)))
    loc = ws.apply_significance_floor(df, rel=0.10, local_mm=16.0, pos=pos)
    check('local floor keeps the faint track', {5, 6, 7, 8, 9} <= set(loc.channel))
    check('local floor still drops the weak strip beside the bright track',
          10 not in set(loc.channel))
    try:
        ws.apply_significance_floor(df, rel=0.10, local_mm=16.0)
        refused = False
    except ValueError:
        refused = True
    check('a local floor without positions is refused', refused)
    pm = {7: np.full(512, np.nan)}
    pm[7][:len(pos)] = pos
    check('strip_positions maps channels to positions',
          np.allclose(ws.strip_positions(df, pm), pos))


def test_rescue_floor():
    print('local floor, rescue mode')
    pm = np.arange(512) * 0.78

    def seed(lo, hi):
        ch = np.arange(lo, hi)
        return ws.Seed(channels=ch, n_strips=len(ch), n_dropped=0, amp_sum=0.0, n_raw=0)
    base = [seed(0, 10)]
    extra = [seed(0, 13), seed(300, 308)]
    out = ws.rescue_candidates(base, base, extra, pm, n_candidates=5)
    check('existing seeds are kept exactly', out[0] is base[0])
    check('a cluster overlapping an existing seed is not added',
          not any(s is extra[0] for s in out))
    check('a distant cluster is added', len(out) == 2
          and np.array_equal(out[1].channels, extra[1].channels))
    check('the added cluster is flagged rescued, the kept one is not',
          out[1].rescued and not out[0].rescued)
    check('the candidate cap holds', ws.rescue_candidates(base, base, extra, pm, 1) == base)
    vetoed = [seed(100, 120)]
    check('a cluster the veto removed still blocks a rescue',
          ws.rescue_candidates([], vetoed, [seed(105, 118)], pm, 5) == [])
    check('rescue needs a local width', not ws.local_floor_rescues(0.0, 'rescue'))
    check('rescue mode selected', ws.local_floor_rescues(16.0, 'rescue'))
    check('replace mode is not rescue', not ws.local_floor_rescues(16.0, 'replace'))
    try:
        ws.local_floor_rescues(16.0, 'sometimes')
        refused = False
    except ValueError:
        refused = True
    check('an unknown mode is refused', refused)


def test_split_seeds():
    print('split seeding')
    pm = np.arange(512) * 0.78

    def seed(ch):
        ch = np.asarray(ch)
        return ws.Seed(channels=ch, n_strips=len(ch), n_dropped=0, amp_sum=1.0, n_raw=len(ch))
    # 12 + 11 strips with a 7.0 mm hole: one cluster at 8 mm, two at 6 mm
    two = seed(list(range(0, 12)) + list(range(20, 31)))
    out = ws.split_seeds([two], pm, gap_mm=6.0)
    check('a cluster with a 7 mm hole splits at 6 mm', [s.n_strips for s in out] == [12, 11],
          f'got {[s.n_strips for s in out]}')
    check('the parts account for every strip', sorted(np.concatenate([s.channels for s in out]).tolist())
          == sorted(two.channels.tolist()))
    check('it does not split at 8 mm', ws.split_seeds([two], pm, gap_mm=8.0)[0] is two)
    small = seed(list(range(0, 12)) + list(range(20, 23)))
    check('a part below min_strips keeps the cluster whole',
          ws.split_seeds([small], pm, gap_mm=6.0, min_strips=5)[0] is small)
    check('...and splits once the part qualifies',
          len(ws.split_seeds([small], pm, gap_mm=6.0, min_strips=3)) == 2)
    solid = seed(range(40, 60))
    check('an unbroken cluster is the same object', ws.split_seeds([solid], pm, gap_mm=4.0)[0] is solid)


class _Fit:
    """Minimal stand-in for PlaneFit for the selector tests."""
    def __init__(self, t0, q_uend, tan, dchi2, plausible):
        self.t0, self.q_uend, self.tan_theta = t0, q_uend, tan
        self._dchi2, self._plausible = dchi2, plausible


def test_pair_selection():
    print('pair selection (X/Y time coincidence)')

    class Cal:
        dt_xy = {0: -18.8}
    # x: the true track (t0 = 400) and a noise cluster (t0 = 900)
    # y: the true track at t0 = 418.8 (= 400 + 18.8) and a noise cluster
    xs = [_Fit(900.0, 700.0, 0.2, 500.0, True),     # noise, but higher dchi2
          _Fit(400.0, 700.0, 0.2, 100.0, True)]
    ys = [_Fit(418.8, 700.0, 0.2, 100.0, True),     # the true partner
          _Fit(80.0, 700.0, 0.2, 400.0, True)]
    got = wr.select_pair({'x': xs, 'y': ys}, 0, Cal())
    check('time coincidence beats raw chi2 improvement',
          got['x'].t0 == 400.0 and got['y'].t0 == 418.8,
          f"picked t0x={got['x'].t0}, t0y={got['y'].t0}")

    single = wr.select_pair({'x': [xs[0]], 'y': [ys[0]]}, 0, Cal())
    check('single candidates pass through untouched',
          single['x'] is xs[0] and single['y'] is ys[0])


def test_plausibility_bounds():
    print('plausibility window')
    check('a 700 ns column inside a gap crossing is plausible',
          wr.U_MIN_NS <= 700 <= wr.U_MAX_NS)
    check('a 60 ns spike is not', not (wr.U_MIN_NS <= 60 <= wr.U_MAX_NS))
    check('a 2 us column is not', not (wr.U_MIN_NS <= 2000 <= wr.U_MAX_NS))


if __name__ == '__main__':
    test_clustering()
    test_significance_floor()
    test_local_floor()
    test_rescue_floor()
    test_split_seeds()
    test_pair_selection()
    test_plausibility_bounds()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    sys.exit(1 if FAILS else 0)
