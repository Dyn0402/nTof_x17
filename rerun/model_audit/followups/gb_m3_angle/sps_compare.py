#!/usr/bin/env python3
"""sps_compare.py -- GB part 3: det4's angle resolution at H4, where the reference is the beam itself.

At H4 every track has the same true angle to 0.53 mrad = 0.030 deg (beam divergence measured from the uRWELL z-scan,
sps_beam_test_26/analysis/spatial_resolution/results.json) and 120 GeV pions scatter by ~0.01 mrad in det4's own
material: the per-event spread of the reconstructed angle around its median IS det4's angle resolution (plus any
position dependence across the beam spot).  Data: FE's SPS gate (run_63, mount 25.64 deg, Ar/CF4/iso, ZS, h4 mask,
1000 events per plateau; ~/x17/cosmic_bench/model_audit/fe_unify2/sps/*.npz): view y is inclined (25.64 deg), view x
head-on (0 deg).  theta = atan(w / v); v = v_span (FE convention), and a second version rescaled so the inclined
median sits on the mount angle (the angle scale reads 8 % low at H4, ws_spsv).  s68 in the FD convention (|theta -
median| <= 5 clip, 68th percentile) and the robust sigma; also the far fraction.

Bench comparison: det4 (g_det4) vs M3, FD measured basis, per |theta_ref| bin (fd_floor_realism/results/
measured_chi2.json) and w-space.

    sps_compare.py   -> results/sps_compare.json
"""
import glob
import json
import os
import re

import numpy as np

from common import RES, AUDIT, s68

HERE = os.path.dirname(os.path.abspath(__file__))
VS = {'d675': (235, 14.0), 'd575': (200, 12.4), 'd475': (165, 11.6)}
VBOX = {'d675': 13.5, 'd575': 11.2, 'd475': 10.0}       # ws_spsv box estimator, 28.8 mm gap (d475 ~165 V/cm: 9.6-10.4)
MOUNT = 25.64


def rs(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(AUDIT, 'fe_unify2', 'sps', 'run63_rot25_operating_00_*_h4.npz'))):
        pl = re.search(r'_(d\d+)_h4', f).group(1)
        E, v = VS[pl]
        P = np.load(f, allow_pickle=True)['per']
        res = {'field_V_cm': E, 'v_span': v, 'v_box': VBOX[pl]}
        ty = np.median(np.abs(P[P[:, 1] == 1][:, 3])) / v
        k = np.tan(np.radians(MOUNT)) / ty
        res['tan_scale_to_mount'] = float(k)
        for vi, view in enumerate('xy'):
            R = P[P[:, 1] == vi]
            w = R[:, 3]
            th = np.degrees(np.arctan(w / v))
            thk = np.degrees(np.arctan(k * w / v))
            th0 = np.median(th)
            far = np.abs(th - th0) > 10
            res[view] = dict(n=int(len(R)), theta_med=float(th0), s68=s68(th - th0), rsig=rs(th),
                             s68_rescaled=s68(thk - np.median(thk)), rsig_rescaled=rs(thk),
                             far10=float(far.mean()), w_rsig=rs(w), w_s68_tan=s68(np.degrees(np.arctan(w / v)) - th0))
        out[pl] = res
        print(pl, json.dumps(res))
    fd = json.load(open(os.path.join(HERE, '..', 'fd_floor_realism', 'results', 'measured_chi2.json')))
    b = json.load(open(os.path.join(AUDIT, 'det4', 'calib_bundle', 'bundle.json')))
    out['bench_det4'] = {v: dict(all=fd[f'det4_{v}']['all_s68'],
                                 bins=[(r['lo'], r['hi'], r['n'], r['s68']) for r in fd[f'det4_{v}']['bins']])
                         for v in 'xy'}
    out['bench_det4']['v_drift'] = float(b['v_drift'])
    print('bench det4', out['bench_det4'])
    json.dump(out, open(os.path.join(RES, 'sps_compare.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
