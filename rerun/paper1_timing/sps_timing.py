#!/usr/bin/env python3
"""sps_timing.py <sps_edge npz> ... --bundle <g_det4 calib_bundle> --out OUT.json
-- the bench column-start estimators (estimators.py) on det4 at SPS (Paper I §7).

det4's X and Y sit on one FEU at H4, so they share ftst: it cancels in t_X - t_Y, and the absolute time is tested with
both signs of the ftst correction (+10 ns x ftst is the bench's).  Selection: both views clustered, q_sum >= Q_MIN.
The SPS pass reads det4 with the bench g_det4 bundle, so its template is the one used here.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimators as E  # noqa: E402
from timing_row import s68  # noqa: E402

Q_MIN = 150.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='+'); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    b = np.load(os.path.join(a.bundle, 'arrays.npz'))
    kern = E.kernel(b['grid'], b['tmpl_x'])
    res = {}
    for f in a.npz:
        z = np.load(f); key = str(z['key'])
        ok = z['x_ok'] & z['y_ok'] & (np.nan_to_num(z['x_q']) >= Q_MIN) & (np.nan_to_num(z['y_q']) >= Q_MIN)
        if ok.sum() < 100:
            res[key] = dict(n=int(ok.sum()), skip='fewer than 100 events'); continue
        ns = z['x_S'].shape[1]
        ftst = z['ftst'][ok].astype(float)
        r = dict(n=int(ok.sum()), n_samples=int(ns))
        for est in ('edge25', 'cf20', 'cf50'):
            t = {}
            for p in 'xy':
                S = z[f'{p}_S'][ok].astype(float)
                if est.startswith('edge'):
                    E.EDGE_FRAC = int(est[4:]) / 100
                    ed = E.Edge(kern, ns)
                    t[p] = np.array([ed.fit(s) for s in S])
                else:
                    t[p] = np.array([E.cf(s, int(est[2:]) / 100) for s in S])
            D = t['x'] - t['y']; M = (t['x'] + t['y']) / 2
            r[est] = dict(sigma_xy=s68(D) / np.sqrt(2), abs_raw=s68(M), abs_plus=s68(M + 10 * ftst), abs_minus=s68(M - 10 * ftst),
                          ftst_medians={int(c): float(np.nanmedian(M[ftst == c])) for c in np.unique(ftst)},
                          median=float(np.nanmedian(M)))
        r['q_median'] = [float(np.median(z['x_q'][ok])), float(np.median(z['y_q'][ok]))]
        res[key] = r
        e = r['edge25']
        print(f'{key:28s} n {r["n"]:5d}  edge25 X-Y/sqrt2 {e["sigma_xy"]:6.1f}  abs {e["abs_raw"]:6.1f} '
              f'(+ftst {e["abs_plus"]:6.1f}, -ftst {e["abs_minus"]:6.1f})   cf20 X-Y {r["cf20"]["sigma_xy"]:6.1f}  '
              f'q {r["q_median"][0]:.0f}/{r["q_median"][1]:.0f}', flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
