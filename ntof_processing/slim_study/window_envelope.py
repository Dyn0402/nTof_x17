#!/usr/bin/env python3
"""How wide must a slim window be if it uses only the GLOBAL time map?

The per-bunch (da_b, dk_b) fit cannot be run before the slim -- it needs the
matched sample -- so the slim has to survive on K, T0 and the per-arm offsets
alone. That leaves the ~1 ppm bunch-to-bunch clock drift in the residual, which
grows with time since flash. This measures the resulting envelope, including
its TAILS, which is what actually sets the window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ntof_dream_merge' / 'match_study' / 'scripts'))

import study_common as sc                                  # noqa: E402
from ntof_dream_merge.calibration import load as load_cal  # noqa: E402

KEY = 1e9
ARMS = ('A', 'B', 'C', 'D')
cal = load_cal()

r_all, t_all, arm_all = [], [], []
for sub in sc.SUBRUNS:
    ev = np.load(sc.DATA / f'events_{sub}.npz')
    cd = np.load(sc.DATA / f'cand_{sub}_wp.npz')
    ck = np.sort(cd['bunch'].astype(np.float64) * KEY + cd['t'])
    order = np.argsort(cd['bunch'].astype(np.float64) * KEY + cd['t'])
    carm = cd['arm'][order]

    t, b = ev['t'].astype(np.float64), ev['bunch'].astype(np.float64)
    # global map only, arm-agnostic (the slim cannot know the arm in advance)
    tp = cal.predict(t, arm=None)
    ek = b * KEY + tp
    j = np.searchsorted(ck, ek)
    j0, j1 = np.clip(j - 1, 0, ck.size - 1), np.clip(j, 0, ck.size - 1)
    d0, d1 = ck[j0] - ek, ck[j1] - ek
    pick = np.where(np.abs(d0) <= np.abs(d1), j0, j1)
    r = np.where(np.abs(d0) <= np.abs(d1), d0, d1)
    r_all.append(r)
    t_all.append(t)
    arm_all.append(carm[pick])

r = np.concatenate(r_all)
t = np.concatenate(t_all)
a = np.concatenate(arm_all)

print(f'{r.size:,} DREAM triggers, nearest wall+plastic candidate\n')
print('per-arm offsets in the calibration: '
      + '  '.join(f'{k} {cal.arm_offset_ns[k]:+6.2f}' for k in ARMS))

core = np.abs(r) < 2000
print(f'\nwithin |r| < 2 us: {core.mean():.4%}\n')

print('arm-corrected residual (subtracting the fitted per-arm offset):')
off = np.array([cal.arm_offset_ns[k] for k in ARMS])
rc = r - off[np.clip(a, 0, 3)]

edges = [0, 1e6, 3e6, 10e6, 20e6, 40e6, 1e9]
lbl = ['<1 ms', '1-3 ms', '3-10 ms', '10-20 ms', '20-40 ms', '>40 ms']
print(f'{"t since flash":<10} {"n":>8} {"med":>8} {"68%hw":>8} {"99%":>8} '
      f'{"99.9%":>9} {"max|r|":>9}  frac inside +-W')
print(f'{"":<10} {"":>8} {"":>8} {"":>8} {"":>8} {"":>9} {"":>9}  '
      + '  '.join(f'{w:>6g}' for w in (50, 100, 200, 500, 1000)))
for i in range(len(lbl)):
    m = core & (t >= edges[i]) & (t < edges[i + 1])
    if m.sum() < 10:
        continue
    x = np.abs(rc[m])
    fr = '  '.join(f'{(x <= w).mean():6.3%}' for w in (50, 100, 200, 500, 1000))
    print(f'{lbl[i]:<10} {m.sum():>8,} {np.median(rc[m]):>8.1f} '
          f'{np.percentile(x, 68):>8.1f} {np.percentile(x, 99):>8.1f} '
          f'{np.percentile(x, 99.9):>9.1f} {x.max():>9.1f}  {fr}')

x = np.abs(rc[core])
fr = '  '.join(f'{(x <= w).mean():6.3%}' for w in (50, 100, 200, 500, 1000))
print(f'{"ALL":<10} {core.sum():>8,} {np.median(rc[core]):>8.1f} '
      f'{np.percentile(x, 68):>8.1f} {np.percentile(x, 99):>8.1f} '
      f'{np.percentile(x, 99.9):>9.1f} {x.max():>9.1f}  {fr}')
