#!/usr/bin/env python3
"""capsule_focus.py -- does a chain's angle scale focus the capsule?  The focus
scan of `pair_vertex_imaging.diagnostics._pointing_one`, run on any stage-3
chain (production, or an in-situ re-pass) with a fine scale grid, per |tan| bin.

Every gated track's x-view line is extrapolated to the beam axis with its tan
multiplied by s; the miss is (pu - pw*s*tan)/sqrt(1 + (s*tan)^2) in the
chamber's transverse frame.  If the tracks came from a thin source on the axis
the miss would be sharpest at the right scale, s = 1.

WHAT IT CAN AND CANNOT SAY (HANDOFF_TRACKING §10, §13, §21).  The capsule is
10 mm across, and at D = 234.6 mm a 15 % scale error at |tan| 0.3 misses by
~10 mm, so the focus is sensitive.  But it assumes the tracks point back to the
axis.  Electrons scatter between their vertex and the chamber, the source has
tails, and the wall edges give an effective source distance of ~330 mm, not
234.6.  Each of those moves the best focus away from the true scale.  The
production k (capsule band) is the same kind of estimator.  So a sharper image
is evidence about the scale only next to a simulation that carries the same
effects; on its own it is a consistency check.  The x view only: the capsule is
80 mm long in y, so y has no focus.

    python -m ntof_cosmics.capsule_focus --src <stage3 dir> --label is2_v2
    python -m ntof_cosmics.capsule_focus --src .../stage3_fullpass --label prod
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results' / 'capsule_focus'
SCALES = np.round(np.arange(0.60, 1.401, 0.025), 3)
TAN_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
EDGES = np.arange(0, 400.5, 0.5)          # |miss| histogram, mm
U_HAT = {'A': np.array([1., 0., 0.]), 'C': np.array([-1., 0., 0.])}
W_HAT = {'A': np.array([0., 0., 1.]), 'C': np.array([0., 0., -1.])}
PRE_ACCESS = ('run_79', 'run_81')
COLS = ['arm', 'gated', 'angle_calibrated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
        'tanx', 'tany', 'dca_axis_mm']


def _one(path: str, tan_max: float, seed: int) -> dict:
    t = pd.read_parquet(path, columns=COLS)
    t = t[t.gated.astype(bool) & t.angle_calibrated.astype('boolean').fillna(False).astype(bool)]
    t = t[(t.tanx.abs() < tan_max) & (t.tany.abs() < tan_max)]
    rng = np.random.default_rng(seed)
    H = {}
    for arm, g in t.groupby('arm'):
        if arm not in U_HAT:
            continue
        P0 = g[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        D = g[['d_x', 'd_y', 'd_z']].to_numpy(float)
        pu, pw = P0 @ U_HAT[arm], P0 @ W_HAT[arm]
        with np.errstate(divide='ignore', invalid='ignore'):
            tx = (D @ U_HAT[arm]) / (D @ W_HAT[arm])
        ok = np.isfinite(tx) & np.isfinite(pu) & np.isfinite(pw)
        pu, pw, tx = pu[ok], pw[ok], tx[ok]
        e1 = (pu - pw * tx) / np.hypot(tx, 1.0)
        # the same check as diagnostics: rebuilt miss == stage-3 dca_axis_mm
        ref = g.dca_axis_mm.to_numpy(float)[ok]
        fin = np.isfinite(ref)
        if fin.any() and np.nanmax(np.abs(np.abs(e1[fin]) - ref[fin])) > 1e-6:
            raise AssertionError(f'{path} {arm}: transverse frame disagrees with dca_axis_mm')
        b = np.digitize(np.abs(tx), TAN_BINS) - 1
        for s in SCALES:
            e = np.abs((pu - pw * s * tx) / np.hypot(s * tx, 1.0))
            for i in range(len(TAN_BINS) - 1):
                m = b == i
                H[(arm, float(s), i)] = H.get((arm, float(s), i), 0) + np.histogram(e[m], EDGES)[0]
        txs = rng.permutation(tx)                        # the no-source null at s = 1
        en = np.abs((pu - pw * txs) / np.hypot(txs, 1.0))
        for i in range(len(TAN_BINS) - 1):
            m = b == i
            H[(arm, 'null', i)] = np.histogram(en[m], EDGES)[0]
    return H


def _median(h):
    c = np.cumsum(h)
    if c[-1] == 0:
        return np.nan
    j = np.searchsorted(c, c[-1] / 2)
    return 0.5 * (EDGES[j] + EDGES[j + 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--tan-max', type=float, default=0.6, help='own true |tan| cut, both views')
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--include-pre-access', action='store_true')
    a = ap.parse_args()
    files = sorted(p for p in a.src.glob('tracks_run_*_stat090_*.parquet')
                   if a.include_pre_access or re.match(r'tracks_(run_\d+)_', p.name).group(1) not in PRE_ACCESS)
    print(f'{a.label}: {len(files)} sub-runs from {a.src}')
    H = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, h in enumerate(ex.map(_one, map(str, files), [a.tan_max] * len(files), range(len(files)))):
            for k, v in h.items():
                H[k] = H.get(k, 0) + v
    rows = []
    for (arm, s, i), h in H.items():
        rows.append(dict(chain=a.label, arm=arm, scale=s if s != 'null' else np.nan, null=s == 'null',
                         tan_lo=TAN_BINS[i], tan_hi=TAN_BINS[i + 1], n=int(h.sum()),
                         med_miss=_median(h), f10=float(h[EDGES[1:] <= 10].sum() / max(h.sum(), 1)),
                         f20=float(h[EDGES[1:] <= 20].sum() / max(h.sum(), 1))))
    # all |tan| pooled
    for arm in ('A', 'C'):
        for s in [float(x) for x in SCALES] + ['null']:
            h = sum(H[(arm, s, i)] for i in range(len(TAN_BINS) - 1) if (arm, s, i) in H)
            if isinstance(h, int):
                continue
            rows.append(dict(chain=a.label, arm=arm, scale=s if s != 'null' else np.nan, null=s == 'null',
                             tan_lo=0.0, tan_hi=a.tan_max, n=int(h.sum()), med_miss=_median(h),
                             f10=float(h[EDGES[1:] <= 10].sum() / h.sum()),
                             f20=float(h[EDGES[1:] <= 20].sum() / h.sum())))
    R = pd.DataFrame(rows).sort_values(['arm', 'tan_lo', 'tan_hi', 'null', 'scale'])
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / f'focus_{a.label}.csv', index=False)
    pooled = R[(R.tan_lo == 0) & (R.tan_hi == a.tan_max) & ~R.null]
    print(pooled.pivot_table(index='scale', columns='arm', values=['med_miss', 'f10']).round(3).to_string())
    for arm in ('A', 'C'):
        for (lo, hi), g in R[(R.arm == arm) & ~R.null].groupby(['tan_lo', 'tan_hi']):
            j = g.med_miss.idxmin()
            print(f'  {arm} |tan| {lo:.1f}-{hi:.1f}: sharpest at s = {g.scale[j]:.3f} '
                  f'(median miss {g.med_miss[j]:.1f} mm, n {int(g.n.iloc[0]):,})')
    print(f'-> {OUT / f"focus_{a.label}.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
