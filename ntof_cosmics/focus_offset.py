#!/usr/bin/env python3
"""focus_offset.py -- the capsule focus with an in-plane offset as well as a scale.

focus_split.py found the sharpest s depends on the sign of tan x: A 0.98 (tan > 0)
/ 1.27 (tan < 0), C 1.26 / 1.00.  A and C have opposite u axes in global x, so
that is ONE global-x displacement between where the tracks point and the assumed
beam axis (capsule off the axis, or both chambers' strip coordinates shifted).
The A-C cosmic truth is blind to it (it cancels in x_A - x_C) and the one-
parameter scan folds it into s.  Here: median |miss| over a grid of (s, dx),
miss = (pu - sgn_u dx - pw s tan) / sqrt(1 + (s tan)^2), with dx a shift of the
source in GLOBAL x (sgn_u = u_hat . x: A +1, C -1), per arm and jointly.

    python -m ntof_cosmics.focus_offset [--src .../stage3_is2_v2] [--label is2_v2]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ntof_cosmics.focus_split import SRC, load

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'focus_split'
S_GRID = np.round(np.arange(0.85, 1.2001, 0.01), 3)
DX_GRID = np.round(np.arange(-12.0, 12.001, 0.5), 2)
SGN_U = {'A': 1.0, 'C': -1.0}
TAN_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.1, 0.5)]


def grid(pu, pw, tx, sg, nmax=300_000, rng=None):
    if len(pu) > nmax:
        i = (rng or np.random.default_rng(0)).choice(len(pu), nmax, replace=False)
        pu, pw, tx, sg = pu[i], pw[i], tx[i], sg[i]
    M = np.empty((len(S_GRID), len(DX_GRID)))
    for a, s in enumerate(S_GRID):
        den = np.hypot(s * tx, 1.0)
        base = pu - pw * s * tx
        for b, dx in enumerate(DX_GRID):
            M[a, b] = np.median(np.abs((base - sg * dx) / den))
    a, b = np.unravel_index(np.argmin(M), M.shape)
    return float(S_GRID[a]), float(DX_GRID[b]), float(M[a, b]), M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--label', default='is2_v2')
    ap.add_argument('--ms-lo', type=float, default=20.0, help='cleanest window (focus_split): 20-80 ms')
    ap.add_argument('--ms-hi', type=float, default=80.0)
    a = ap.parse_args()
    d = load(a.src, 0.6)
    d = d[d.ms.between(a.ms_lo, a.ms_hi) & d.wall]
    d['sg'] = d.arm.map(SGN_U)
    print(f'{a.label}: {len(d):,} tracks, {a.ms_lo:g}-{a.ms_hi:g} ms, wall fired')
    rows = []
    rng = np.random.default_rng(2)
    for who in ('A', 'C', 'A+C'):
        g = d if who == 'A+C' else d[d.arm == who]
        for lo, hi in TAN_BINS:
            m = (g.tx.abs() >= lo) & (g.tx.abs() < hi)
            h = g[m]
            arr = [h[c].to_numpy(float) for c in ('pu', 'pw', 'tx', 'sg')]
            s, dx, med, M = grid(*arr, rng=rng)
            # the 1-D answer at dx = 0, for comparison
            j0 = int(np.argmin(np.abs(DX_GRID)))
            s0 = float(S_GRID[int(np.argmin(M[:, j0]))])
            # sign halves at the joint best (s, dx): do they now agree?
            halves = {}
            for sgn in (1, -1):
                k = np.sign(arr[2]) == sgn
                # 1-D s at the fitted dx
                Mh = np.array([np.median(np.abs((arr[0][k] - arr[3][k] * dx - arr[1][k] * s_ * arr[2][k])
                                                / np.hypot(s_ * arr[2][k], 1.0))) for s_ in S_GRID])
                halves[sgn] = float(S_GRID[int(np.argmin(Mh))])
            rows.append(dict(who=who, tan_lo=lo, tan_hi=hi, n=len(h), s_best=s, dx_best=dx, med_miss=med,
                             s_at_dx0=s0, s_pos_at_dx=halves[1], s_neg_at_dx=halves[-1]))
            r = rows[-1]
            print(f'  {who:4s} |tan| {lo:.1f}-{hi:.1f}: s {s:.2f}  dx {dx:+5.1f} mm  miss {med:5.1f}  '
                  f'(s at dx=0: {s0:.2f}; halves at fitted dx: + {r["s_pos_at_dx"]:.2f} / - {r["s_neg_at_dx"]:.2f})  '
                  f'n {len(h):,}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f'focus_offset_{a.label}.csv', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
