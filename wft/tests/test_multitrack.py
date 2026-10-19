#!/usr/bin/env python3
"""
Unit tests for the multi-track generalisation: select_tracks (disjoint
time-coincident pair ranking) and candidate_rows (the ranked-candidate side
table). The contract under test is the one the bench relies on: pair 0 IS
select_pair's choice, so enabling multi-track output cannot move the
single-track answer.

    ../../.venv/bin/python wft/tests/test_multitrack.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import reco as wr           # noqa: E402
from wft.reco import PlaneFit        # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


class Cal:
    dt_xy = {0: -18.8}


def fit(t0, dchi2, plausible=True, p0=100.0, q_sum=2e4):
    """A real PlaneFit (candidate_rows needs asdict) with the selector's
    attributes attached the way fit_plane_candidates attaches them."""
    f = PlaneFit(p0=p0, w=0.005, t0=t0, tan_theta=0.14, theta_deg=8.0,
                 chi2=500.0, dof=400, p0_err=0.4, w_err=1e-3, tan_err=0.03,
                 t0_err=15.0, q_sum=q_sum, q_u50=350.0, q_u90=600.0,
                 q_uend=700.0, n_strips=8, n_seed=6, n_dropped=0,
                 slope_reliable=True, quality_ok=True, n_candidates=2)
    f._plausible, f._dchi2 = plausible, dchi2
    return f


def test_double_track():
    print('two clean tracks -> two gated pairs, disjoint, best first')
    xs = [fit(400.0, 800.0, p0=100.0), fit(900.0, 600.0, p0=250.0)]
    ys = [fit(418.8, 700.0, p0=50.0), fit(918.8, 500.0, p0=300.0)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('two pairs found', len(pairs) == 2, f'got {len(pairs)}')
    check('both gated', all(g for _i, _j, g in pairs))
    check('disjoint', len({p[0] for p in pairs}) == 2
          and len({p[1] for p in pairs}) == 2)
    check('coincident partners matched (no ghosts)',
          (0, 0) in [(i, j) for i, j, _ in pairs]
          and (1, 1) in [(i, j) for i, j, _ in pairs],
          f'got {[(i, j) for i, j, _ in pairs]}')
    check('best-dchi2 pair ranked first', pairs[0][:2] == (0, 0))

    sel = wr.select_pair({'x': xs, 'y': ys}, 0, Cal())
    i, j, _ = pairs[0]
    check('pair 0 is select_pair\'s choice',
          sel['x'] is xs[i] and sel['y'] is ys[j])


def test_double_counting_guard():
    print('split track / noise -> second pair rejected')
    # fragment of the same track: coincident but not plausible
    xs = [fit(400.0, 800.0), fit(405.0, 200.0, plausible=False)]
    ys = [fit(418.8, 700.0), fit(423.0, 150.0, plausible=False)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('fragment pair not gated', sum(g for *_ij, g in pairs) == 1,
          f'gated {sum(g for *_ij, g in pairs)}')
    # noise cluster: plausible-looking but not time-coincident
    xs = [fit(400.0, 800.0), fit(900.0, 600.0)]
    ys = [fit(418.8, 700.0), fit(80.0, 500.0)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('non-coincident second pair not gated',
          sum(g for *_ij, g in pairs) == 1)


def test_time_degenerate_pairing():
    print('time-coincident tracks: pairing follows summed dchi2 rank, not geometry')
    # Without an xy_pairing calibration. The overlay bench
    # (sept26_prelim_analysis/intra_bench.py) measures ~75 % swaps when the two
    # planes rank the tracks differently; test_charge_pairing is the fix.
    xs = [fit(400.0, 800.0, p0=100.0), fit(405.0, 600.0, p0=250.0)]
    ys = [fit(418.8, 500.0, p0=50.0), fit(423.8, 700.0, p0=300.0)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('both pairs gated', len(pairs) == 2 and all(g for *_ij, g in pairs))
    check('planes rank differently -> x0 pairs with y1 (strongest with strongest)',
          [(i, j) for i, j, _ in pairs] == [(0, 1), (1, 0)],
          f'got {[(i, j) for i, j, _ in pairs]}')
    ys = [fit(418.8, 700.0, p0=50.0), fit(423.8, 500.0, p0=300.0)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('planes rank alike -> x0 pairs with y0',
          [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)],
          f'got {[(i, j) for i, j, _ in pairs]}')


PAIRING = {'features': ['lq'], 'median': {'lq': 0.0}, 'rsig': {'lq': 0.3}}


def test_charge_pairing():
    print('xy_pairing: time-degenerate tracks re-paired by x/y charge')
    xs = [fit(400.0, 800.0, q_sum=3e4), fit(405.0, 600.0, q_sum=1e4)]
    ys = [fit(418.8, 500.0, q_sum=3e4), fit(423.8, 700.0, q_sum=1e4)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal(), pairing=PAIRING)
    check('bright x pairs with bright y despite the dchi2 ranks',
          [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)],
          f'got {[(i, j) for i, j, _ in pairs]}')
    check('both still gated', len(pairs) == 2 and all(g for *_ij, g in pairs))
    check('x members and order unchanged', [i for i, _j, _g in pairs] == [0, 1])

    xs = [fit(400.0, 800.0, q_sum=3e4), fit(900.0, 600.0, q_sum=1e4)]
    ys = [fit(418.8, 700.0, q_sum=1e4), fit(918.8, 500.0, q_sum=3e4)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal(), pairing=PAIRING)
    check('time-separated tracks are never re-paired',
          [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1)],
          f'got {[(i, j) for i, j, _ in pairs]}')

    pairs = wr.select_tracks({'x': [fit(400.0, 800.0, q_sum=3e4)],
                              'y': [fit(418.8, 700.0, q_sum=1e4)]}, 0, Cal(),
                             pairing=PAIRING)
    check('a single track is untouched', len(pairs) == 1 and pairs[0] == (0, 0, True))

    from wft.calib import check_xy_pairing
    for bad in ({'features': ['lq'], 'median': {'lq': 0.0}, 'rsig': {'lq': 0.0}},
                {'features': ['nope'], 'median': {'nope': 0.0}, 'rsig': {'nope': 1.0}},
                {'features': [], 'median': {}, 'rsig': {}}):
        try:
            check_xy_pairing(bad)
            refused = False
        except ValueError:
            refused = True
        check(f'unusable calibration refused ({bad["features"]})', refused)


def test_rescued_candidates():
    print('rescued candidates add tracks, never replace one')
    xs = [fit(400.0, 500.0)]
    ys = [fit(418.8, 500.0), fit(420.0, 900.0)]
    check('without the flag the stronger y wins pair 0',
          wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())[0][:2] == (0, 1))
    ys[1]._rescued = True
    check('flagged as rescued it cannot take pair 0',
          wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())[0][:2] == (0, 0))

    xs = [fit(400.0, 800.0), fit(405.0, 600.0)]
    ys = [fit(418.8, 700.0), fit(423.8, 900.0)]
    ys[1]._rescued = True
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('a rescued candidate still forms a further gated track',
          [(i, j, g) for i, j, g in pairs] == [(0, 0, True), (1, 1, True)],
          f'got {pairs}')

    xs = [fit(400.0, 800.0)]
    ys = [fit(80.0, 700.0), fit(418.8, 100.0)]
    ys[1]._rescued = True
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    check('an ungated production pair 0 is kept over a gated rescued one',
          pairs == [(0, 0, False)], f'got {pairs}')
    rows = wr.candidate_rows(1, {'x': xs, 'y': ys}, pairs)
    check('the side table records the flag',
          [r['rescued'] for r in rows if r['plane'] == 'y'] == [False, True])


def test_single_and_empty():
    print('single-track and degenerate events')
    pairs = wr.select_tracks({'x': [fit(400.0, 800.0)],
                              'y': [fit(418.8, 700.0)]}, 0, Cal())
    check('single track -> one gated pair',
          len(pairs) == 1 and pairs[0][2])
    pairs = wr.select_tracks({'x': [fit(400.0, 800.0)],
                              'y': [fit(80.0, 700.0)]}, 0, Cal())
    check('non-coincident winner kept as pair 0, ungated',
          len(pairs) == 1 and not pairs[0][2])
    check('one-plane event -> no pairs',
          wr.select_tracks({'x': [fit(400.0, 800.0)], 'y': []}, 0, Cal()) == [])
    check('empty event -> no pairs',
          wr.select_tracks({'x': [], 'y': []}, None, Cal()) == [])


def test_candidate_rows():
    print('candidate side table')
    xs = [fit(400.0, 800.0), fit(900.0, 600.0)]
    ys = [fit(418.8, 700.0)]
    pairs = wr.select_tracks({'x': xs, 'y': ys}, 0, Cal())
    rows = wr.candidate_rows(7, {'x': xs, 'y': ys}, pairs,
                             ftst={'x': 3, 'y': 5})
    check('one row per candidate', len(rows) == 3, f'got {len(rows)}')
    r0 = [r for r in rows if r['plane'] == 'x' and r['rank'] == 0][0]
    r1 = [r for r in rows if r['plane'] == 'x' and r['rank'] == 1][0]
    ry = [r for r in rows if r['plane'] == 'y'][0]
    check('winners share track_id 0',
          r0['track_id'] == 0 and ry['track_id'] == 0 and r0['track_gated'])
    check('unpaired candidate marked -1', r1['track_id'] == -1)
    check('selector scores carried',
          r0['dchi2'] == 800.0 and r0['plausible'])
    check('fit fields flattened', abs(r0['p0'] - 100.0) < 1e-9
          and r0['n_strips'] == 8)
    check('per-plane ftst carried', r0['ftst'] == 3 and ry['ftst'] == 5)
    check('isochronous flag computed',
          not r0['isochronous'] and np.isfinite(r0['q_uend']))


if __name__ == '__main__':
    test_double_track()
    test_double_counting_guard()
    test_time_degenerate_pairing()
    test_charge_pairing()
    test_rescued_candidates()
    test_single_and_empty()
    test_candidate_rows()
    print('\n' + ('ALL PASS' if not FAILS else f'FAILURES: {FAILS}'))
    sys.exit(1 if FAILS else 0)
