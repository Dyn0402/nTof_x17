#!/usr/bin/env python3
"""toy_chi2_select.py <x toy dir> <y toy dir> [--out results/toy_chi2_select.json] -- the control for the misfit share.

Selecting the best chi2/dof quarter improves the angle in data (misfit_retest.py).  Part of that would happen even with
a perfect model: an event whose noise realisation fits well may also measure its slope well.  In the toy (ladder rows:
angle, residual, chi2/dof, ...) the same selection -- the best quarter in chi2/dof per angle point (the toy events share
one amplitude scale and one geometry per point, so this is the toy's version of the geometry-matched rank) -- gives the
selection's share without any mis-modelling.  The data's misfit share minus the toy's is what the misfit itself costs.
"""
import argparse
import glob
import json
import os

import numpy as np

import floorlib as C


def s68(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 20 else np.nan


def one(files):
    out = {}
    for f in files:
        r = np.load(f)
        r = r[np.isfinite(r[:, 1]) & np.isfinite(r[:, 2])]
        c = r[:, 2]
        best = c <= np.percentile(c, 25)
        a, b = s68(r[:, 1]), s68(r[:, 1][best])
        out[os.path.basename(f)] = dict(n=int(len(r)), s68=a, s68_best_quarter=b, delta2=float(a * a - b * b),
                                        share=float((a * a - b * b) / max(a * a, 1e-9)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xdir')
    ap.add_argument('ydir')
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'toy_chi2_select.json'))
    a = ap.parse_args()
    res = {}
    for det in C.CALIB:
        for view, d, suf in (('x', a.xdir, ''), ('y', a.ydir, '_y')):
            for rung in ('L0', 'L8'):
                fs = sorted(f for f in glob.glob(os.path.join(d, f'{det}_{rung}_a*{suf}.npy')) if '_alone' not in f
                            and (suf or not f.endswith('_y.npy')))
                if fs:
                    o = one(fs)
                    sh = float(np.mean([v['share'] for v in o.values()]))
                    res[f'{det}|{view}|{rung}'] = dict(per_angle=o, mean_share=sh)
                    print(det, view, rung, 'best-chi2-quarter share of the toy variance', f'{sh:.2f}',
                          {k.split('_a')[-1]: (round(v['s68'], 3), round(v['s68_best_quarter'], 3)) for k, v in o.items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
