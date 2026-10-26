#!/usr/bin/env python3
"""Dead strips in seeding and windowing (``WFT_DEAD_SEED``, on by default): a dead
strip cannot fire, so (1) the clustering gap does not count the dead strips it
crosses, and (2) the window pad counts live strips, taking the dead ones it
crosses along. Without a dead list both are the historical behaviour.

    ../../.venv/bin/python wft/tests/test_dead_seed.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import seed as ws          # noqa: E402
from wft import io as wio           # noqa: E402

FAILS = []
P = ws.PITCH_MM


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def test_gap_bridge():
    print('seeding: a dead run inside a track')
    # strips 0..39 on a pitch grid; 10..25 dead (16 strips = 12.5 mm > gap): the
    # track fires 4..9 and 26..31, nothing in between because nothing CAN fire
    idx = np.r_[4:10, 26:32]
    pos, ch, amp = idx * P, idx, np.full(len(idx), 500.0)
    dead_pos = np.arange(10, 26) * P
    old = ws.seed_candidates(pos, ch, amp, n_candidates=3)
    new = ws.seed_candidates(pos, ch, amp, n_candidates=3, dead_pos=dead_pos)
    check('historical: the dead run splits the track in two', len(old) == 2, f'{len(old)} seeds')
    check('dead-aware: one cluster across the dead run', len(new) == 1 and new[0].n_strips == 12,
          f'{[s.n_strips for s in new]}')
    # a REAL 13 mm gap with no dead strips in it still splits
    idx2 = np.r_[4:10, 27:33]
    far = ws.seed_candidates(idx2 * P, idx2, amp, n_candidates=3, dead_pos=np.arange(40, 50) * P)
    check('a live gap still splits', len(far) == 2, f'{len(far)} seeds')
    hot = ws.seed_candidates(pos, ch, amp, n_candidates=3, hot=[4], dead_pos=dead_pos)
    check('the hot (clean-strip) path bridges too', len(hot) == 1, f'{len(hot)} seeds')


def test_window_pad():
    print('window: the pad reaches past dead strips')
    pos_map = np.arange(40) * P
    wfm = np.zeros((40, 32)); noise = np.full(40, 8.0)
    seed = np.arange(10, 16)
    old = wio.extract_window(wfm, noise, pos_map, seed, 3)
    check('historical pad: 3 strips each side', list(old.ch) == list(range(7, 19)), f'{list(old.ch)}')
    same = wio.extract_window(wfm, noise, pos_map, seed, 3, dead=[30])
    check('dead strips away from the pad change nothing', list(same.ch) == list(old.ch))
    new = wio.extract_window(wfm, noise, pos_map, seed, 3, dead=[16, 17, 18, 8])
    check('3 LIVE strips each side, dead ones included',
          list(new.ch) == list(range(6, 22)), f'{list(new.ch)}')
    edge = wio.extract_window(wfm, noise, pos_map, np.arange(36, 39), 3, dead=[39])
    check('stops at the plane edge', list(edge.ch) == list(range(33, 40)), f'{list(edge.ch)}')


def test_dead_positions():
    print('dead_positions: the bundle's dead list as seed positions')
    pm = {6: np.arange(512) * P, 8: np.arange(512) * P}
    on = ws.dead_positions({'x': [30, 10], 'y': []}, pm, 6, 8)
    check('on by default', ws.DEAD_SEED)
    check('x positions sorted', on['x'] is not None and np.allclose(on['x'], [10 * P, 30 * P]))
    check('empty plane -> None', on['y'] is None)
    ws.DEAD_SEED = False
    try:
        off = ws.dead_positions({'x': [30, 10]}, pm, 6, 8)
        check('WFT_DEAD_SEED=0 -> no positions', off['x'] is None and off['y'] is None)
    finally:
        ws.DEAD_SEED = True


if __name__ == '__main__':
    test_dead_positions()
    test_gap_bridge()
    test_window_pad()
    print('\nALL PASS' if not FAILS else f'\nFAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
