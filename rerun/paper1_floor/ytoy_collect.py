#!/usr/bin/env python3
"""ytoy_collect.py <dir of ytoy npy> [--out results/ytoy.json] -- the FD toy fitted as the Y view (FD_VIEW=y, condor
ytoy.dag): per chamber the floor (L0, exact model), the production fit (L1) and every measured imperfection (L8,
cumulative; no X copy in Y), at 3 / 9 / 16 deg, each weighted to the Y data's own |theta_ref| mix (< 6, 6-12.5, > 12.5 deg,
the X budget's convention, pass1_basis.ang_weights).  toy_all for Y = the L8 rung (the X budget's toy_all is the
quadrature sum of the alone rungs; on X the two agree within 0.02 deg except det4's withdrawn gain rung, not in L8).
The generation is the X one (footprint core + X tails, pulse, CM): Y has no measured tails of its own (K-list), so the
Y toy is approximate in the footprint; its chi2/dof is higher than X's (the Y bundle model against X-like pulses).
"""
import argparse
import glob
import json
import os

import numpy as np

import floorlib as C

EDGES = (6.0, 12.5)
ANG = (3.0, 9.0, 16.0)


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 20 else np.nan


def weights(det):
    g = C.sample(C.load(C.CALIB[det]), 'y')
    th = np.abs(np.degrees(np.arctan(g.tan_ref)))
    w = np.array([(th < EDGES[0]).mean(), ((th >= EDGES[0]) & (th < EDGES[1])).mean(), (th >= EDGES[1]).mean()])
    return w / w.sum()


def wsum(rows, w):
    """the angle-mix weighted s68: draw each angle's residuals in proportion to the weights, then s68."""
    rs = np.random.default_rng(4)
    parts = []
    n = min(len(rows[a]) for a in ANG)
    for a, wi in zip(ANG, w):
        d = rows[a][:, 1]
        parts.append(rs.choice(d, int(round(wi * n * 3)), replace=True))
    return s68(np.concatenate(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'ytoy.json'))
    a = ap.parse_args()
    out = {}
    for det in C.CALIB:
        w = weights(det)
        r = dict(weights=w.tolist(), rungs={})
        for rung in ('L0', 'L1', 'L8'):
            rows = {}
            for ang in ANG:
                fs = glob.glob(os.path.join(a.inp, f'{det}_{rung}_a{int(ang)}_y.npy'))
                if fs:
                    rows[ang] = np.load(fs[0])
            if len(rows) < 3:
                continue
            per = {str(int(k)): dict(n=int(len(v)), s68=s68(v[:, 1]), chi2dof=float(np.nanmedian(v[:, 2])))
                   for k, v in rows.items()}
            boot = []
            rs = np.random.default_rng(9)
            for _ in range(100):
                bb = {k: v[rs.integers(0, len(v), len(v))] for k, v in rows.items()}
                boot.append(wsum(bb, w))
            r['rungs'][rung] = dict(per_angle=per, all_w=wsum(rows, w), all_w_err=float(np.nanstd(boot)))
        if 'L8' in r['rungs']:
            r['toy_all'] = r['rungs']['L8']['all_w']
            r['toy_all_err'] = r['rungs']['L8']['all_w_err']
        out[det] = r
        print(det, 'w', np.round(w, 2), {k: (round(v['all_w'], 3), {a_: (round(p['s68'], 3), round(p['chi2dof'], 1))
                                                                    for a_, p in v['per_angle'].items()})
                                         for k, v in r['rungs'].items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
