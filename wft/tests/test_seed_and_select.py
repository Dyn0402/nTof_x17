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
    test_pair_selection()
    test_plausibility_bounds()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    sys.exit(1 if FAILS else 0)
