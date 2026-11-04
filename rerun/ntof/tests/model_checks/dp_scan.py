#!/usr/bin/env python3
"""dp_scan.py -- model check 3, second estimator: chi2 of unsaturated head-on X planes against the bundle Dp.

    dp_scan.py --cache C.pkl --bundle B --source ntof|bench --dps 0.016,0.019,0.0225 --out J.json [--tan 0.05]

For each Dp the bundle is copied with only hyper['Dp'] changed and every unsaturated head-on X plane of the cache is
refitted (template_resid.one, 'prod' fit); the score is the median chi2/dof and the median centre-strip residual at
the peak.  The window cut that makes the stack fit degenerate on C and D (a 950-1000 ns column in a 1200 ns window)
does not hurt this one: the fit models the whole window.  Known bias (DECISION_2026-10-29 K2, TEMPLATE_STUDY T9): the
static footprint under-fills the centre strip on every chamber, so chi2 alone leans to a narrower footprint; read the
n_TOF minimum against the same scan on the chamber's bench head-on cache (--source bench), not in absolute terms.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, HERE]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--source', choices=('ntof', 'bench'), default='ntof')
    ap.add_argument('--dps', required=True)
    ap.add_argument('--tan', type=float, default=0.05)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import pickle
    from wft.calib import CalibrationBundle
    import template_resid as T
    ids = sorted(pickle.load(open(a.cache, 'rb')))
    res = []
    for dp in [float(x) for x in a.dps.split(',')]:
        cal = CalibrationBundle.load(a.bundle)
        cal.hyper = dict(cal.hyper, Dp=dp)
        with tempfile.TemporaryDirectory() as td:
            cal.save(td, note=f'dp_scan Dp {dp}')
            T._init(a.cache, td, dict(source=a.source, tan=a.tan, pin=False, planes='x'))
            rows = [r for e in ids for r in T.one(e)]
        cd = np.array([r['prod']['chi2dof'] for r in rows])
        k0 = T.K_OFF.tolist().index(0)
        pk = np.array([r['prod']['res'][2][k0] for r in rows], float)
        pm = np.array([0.5 * (r['prod']['res'][1][k0] + r['prod']['res'][3][k0]) for r in rows], float)
        res.append(dict(Dp=dp, n=len(rows), chi2dof_med=float(np.median(cd)), chi2dof_mean=float(cd.mean()),
                        centre_peak=float(np.nanmedian(pk)), pm1_peak=float(np.nanmedian(pm)),
                        chi2_sum=float(sum(r['prod']['chi2dof'] for r in rows))))
        print(f'Dp {dp:.4f}: n {len(rows)} chi2/dof med {np.median(cd):.3f} mean {cd.mean():.3f}  centre@peak '
              f'{np.nanmedian(pk):+.3f}  +-1@peak {np.nanmedian(pm):+.3f}', flush=True)
    # parabola through the summed chi2/dof (same planes at every Dp) -> preferred Dp
    D = np.array([r['Dp'] for r in res]); Y = np.array([r['chi2_sum'] for r in res])
    best = None
    if len(D) >= 3:
        c = np.polyfit(D, Y, 2)
        if c[0] > 0:
            best = float(-c[1] / (2 * c[0]))
    out = dict(cache=a.cache, bundle=a.bundle, source=a.source, tan=a.tan, scan=res, Dp_chi2_min=best)
    print('chi2-preferred Dp', best)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
