#!/usr/bin/env python3
"""Head-on seeding (``WFT_HEADON_SIG``) and the cross-view column rescue (``wft.seed.column_rescue``, used by
``WFT_HEADON_RESCUE``), both opt-in: with them off the seeding is the historical one; a 2-strip cluster is admitted
only above the significance cut and only in a plane production left empty; the column finder takes a drift column
that starts at the other view's t0, and not a flat band, a short blip or a column at another time.

    ../../.venv/bin/python wft/tests/test_headon.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from wft import seed as ws          # noqa: E402

FAILS = []
P = ws.PITCH_MM


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def test_headon_seeds():
    print('head-on 2-strip seeds')
    check('off by default', ws.HEADON_SIG == 0.0 or os.environ.get('WFT_HEADON_SIG'), f'{ws.HEADON_SIG}')
    idx = np.array([100, 101])
    pos, ch = idx * P, idx
    hi = ws.headon_seeds(pos, ch, np.array([300.0, 60.0]), 20.0)
    lo = ws.headon_seeds(pos, ch, np.array([300.0, 9.0]), 20.0)
    check('a 2-strip head-on (2nd strip 60 sigma) is seeded', len(hi) == 1 and hi[0].n_strips == 2 and hi[0].rescued)
    check('a 2-strip blip (2nd strip 9 sigma) is not', lo == [])
    three = ws.headon_seeds(np.array([100, 101, 102]) * P, np.array([100, 101, 102]), np.array([300.0, 60, 40]), 20.0)
    check('a 3-strip cluster is not a head-on seed (production owns it)', three == [])
    check('min_sig 0 = off', ws.headon_seeds(pos, ch, np.array([300.0, 60.0]), 0.0) == [])


def test_column_rescue():
    print('column rescue')
    rng = np.random.default_rng(0)
    ns, nch = 32, 64
    pos = np.arange(nch) * P
    noise = np.full(nch, 8.0)
    t = np.arange(ns) * 60.0
    t0 = 300.0
    col = np.where((t >= t0) & (t <= t0 + 840), 120.0, 0.0)        # 15 samples at 15 sigma per sample
    base = rng.normal(0, 8, (nch, ns))
    W = base.copy(); W[30] += col; W[31] += 0.3 * col
    got = ws.column_rescue(W, noise, pos, t0)
    check('a column at t0 is found', got is not None and 30 in got, f'{got}')
    check('noise alone is not', ws.column_rescue(base, noise, pos, t0) is None)
    band = base.copy(); band[10:20] += 150.0                            # flat for the whole window
    check('a flat band is not a column', ws.column_rescue(band, noise, pos, t0) is None)
    blip = base.copy(); blip[30, 8:10] += 600.0                          # 2 samples
    check('a short blip is not a column', ws.column_rescue(blip, noise, pos, t0) is None)
    early = base.copy(); early[30, 0:16] += 120.0                       # charge before the window
    check('a column that started before t0 - 120 ns is not this track', ws.column_rescue(early, noise, pos, t0) is None)
    late = base.copy(); late[30, 22:30] += 2000.0                        # a pulse after the window only
    check('charge after the window is not a column', ws.column_rescue(late, noise, pos, t0) is None)


def test_rescue_wiring():
    print('reco._headon_rescue wiring (the worker swallows exceptions, so a NameError there is silent)')
    from types import SimpleNamespace
    from wft import reco as wr
    rng = np.random.default_rng(1)
    R = dict(W=rng.normal(0, 8, (64, 32)), noise=np.full(64, 8.0), pos=np.arange(64) * P)
    try:
        r = wr._headon_rescue(R, 'x', SimpleNamespace(t0=300.0, q_sum=1000.0))
        check('noise plane: runs and returns None', r is None)
    except Exception as e:      # noqa: BLE001
        check('noise plane: runs and returns None', False, repr(e))


if __name__ == '__main__':
    test_headon_seeds()
    test_column_rescue()
    test_rescue_wiring()
    print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
