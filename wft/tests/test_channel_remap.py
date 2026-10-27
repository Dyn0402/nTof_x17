#!/usr/bin/env python3
"""``CalibrationBundle.remap_channels``: dead / hot / gain carried to another cabling by strip position (F12).

The bench det4 mask (FEU 6/8, plugs normal) applied by channel number to the H4 data (FEU 03, connectors 3-6,
plugs inverted) censors the wrong strips; remapped by position it censors the same strips.

    ../../.venv/bin/python wft/tests/test_channel_remap.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]

from wft.calib import CalibrationBundle          # noqa: E402
from det4_sps_map import POSITION_MM, VIEW       # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def bench_pos():
    c = np.arange(512)
    p = 49.92 * (c // 64) + (63 - c % 64) * 0.78       # every June bench row (strip_position_map, checked)
    return {'x': p.copy(), 'y': p.copy()}


def sps_pos():
    return {v: np.where(VIEW == v, POSITION_MM, np.nan) for v in ('x', 'y')}


def bundle(dead, hot=None, gain=None):
    g = {p: np.ones(512) for p in 'xy'}
    if gain:
        for p, d in gain.items():
            for c, v in d.items():
                g[p][c] = v
    return CalibrationBundle(hyper={'c1': 0.05, 'c2': 0.0, 'tau_s': 100.0, 'sigma_s': 10.0, 'sigma_p0': 0.3, 'Dp': 0.02},
                             v_drift=33.0, grid=np.arange(10.0), tmpl={'x': np.ones(10), 'y': np.ones(10)}, gain=g,
                             dead=dead, hot=hot or {})


def test_bench_to_sps():
    print('bench det4 -> SPS H4 FEU 03')
    bp, sp = bench_pos(), sps_pos()
    # X: bench ch 300 (connector 5, local 44) and 10 (connector 1: not cabled at H4); Y: bench ch 317 (connector 5)
    cal = bundle({'x': [10, 300], 'y': [317]}, hot={'x': [200]}, gain={'x': {300: 0.8}})
    rec = cal.remap_channels(bp, sp, target='sps det4')
    want_x = int(np.flatnonzero((VIEW == 'x') & (np.abs(POSITION_MM - bp['x'][300]) < 0.1))[0])
    want_y = int(np.flatnonzero((VIEW == 'y') & (np.abs(POSITION_MM - bp['y'][317]) < 0.1))[0])
    check('X dead lands on the SPS channel at the same strip', cal.dead['x'] == [want_x], f'{cal.dead["x"]} vs {want_x}')
    check('X: same strip = bench channel - 128', want_x == 300 - 128)
    check('Y dead lands at the same strip (= bench channel + 128)', cal.dead['y'] == [want_y] and want_y == 317 + 128,
          f'{cal.dead["y"]}')
    check('uncabled strip dropped and counted', rec['planes']['x']['dead']['dropped'] == 1)
    check('hot carried too', cal.hot['x'] == [200 - 128])
    check('gain carried to the remapped channel', abs(cal.gain['x'][want_x] - 0.8) < 1e-12 and np.sum(cal.gain['x'] != 1) == 1)
    check('by channel number, bench X ch 300 is a Y strip at H4 (and ch 172 an X strip 100 mm away)',
          VIEW[300] == 'y' and abs(POSITION_MM[172] - bp['x'][172]) > 50)
    check('provenance recorded', cal.provenance['channel_remap']['target'] == 'sps det4')


def test_inverted_plugs():
    print('normal -> inverted plugs (n_TOF): the strip moves within its connector')
    bp = bench_pos()
    c = np.arange(512)
    inv = 49.92 * (c // 64) + (c % 64) * 0.78
    cal = bundle({'x': [64 + 5], 'y': []})
    cal.remap_channels(bp, {'x': inv, 'y': inv}, target='ntof')
    check('connector 1 local 5 -> local 58', cal.dead['x'] == [64 + 58], f'{cal.dead["x"]}')


if __name__ == '__main__':
    test_bench_to_sps()
    test_inverted_plugs()
    print('FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed')
    sys.exit(1 if FAILS else 0)
