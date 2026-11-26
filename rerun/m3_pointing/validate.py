#!/usr/bin/env python3
"""validate.py -- does the per-track width describe every row, in every bin of chi2, angle, anomaly and position?

For each row: the chamber term (d0, d1, bias, outlier fraction and width) is fitted with the M3 term held fixed at
the pair fit (fit.json).  The three rows with one chamber (A Run 1, A Run 2, E 6-24) took no part in the M3 fit, so
for them this is a prediction.  The reference is the Paper I model: one pointing per slot (a0 = 1) and a chamber term
without angle dependence, fitted the same way.

Closure test: the pull r_i / sigma_i in bins of each variable.  Its core width (Gaussian fitted in the fixed window
+- 2 s68) should be 1 in every bin if the width is right for each track.  Writes $WORK/validate/{chamber.json,
closure.parquet} and the per-ray sigma table $WORK/validate/sigma_<tag>.parquet (every interior_good ray, fitted or
not, for the efficiency).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import load, WORK, core_sigma, s68  # noqa: E402
from rows import ROWS, PUBLIC  # noqa: E402
from fit import add_m3  # noqa: E402
from model import fit_single, sigma_track  # noqa: E402

OUT = os.path.join(WORK, 'validate')
VARS = {  # binning variable -> (column template, edges)
    'chi2': ('chi2_m{v}', [0, 0.03, 0.08, 0.15, 0.3, 0.5, 0.75, 1.0]),
    'k': ('k_v{v}', [0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]),
    'tan': ('t_v{v}', [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35]),
    'pos_along': ('pos_{v}', [-200, -100, -50, 0, 50, 100, 200]),
    'pos_across': ('posx_{v}', [-200, -100, -50, 0, 50, 100, 200]),
}


def main():
    par = json.load(open(os.path.join(WORK, 'decompose.json')))
    fr = json.load(open(os.path.join(WORK, 'fit.json')))
    pmf = os.path.join(WORK, 'posmap.json')
    pm = json.load(open(pmf)) if os.path.exists(pmf) and not os.environ.get('M3P_NO_POSMAP') else None
    print('position map:', 'on' if pm else 'off')
    os.makedirs(OUT, exist_ok=True)
    Dall = add_m3(load(sample='efficiency'), par)
    Dall['res_ok_ig'] = Dall.res_ok
    for v, c in (('x', 'Y'), ('y', 'X')):
        Dall[f'chi2_m{v}'] = Dall[f'chi2_{c}']
        Dall[f'pos_{v}'] = Dall[f'm3_{c}_at_{v}']                  # along the measured coordinate
        Dall[f'posx_{v}'] = Dall[f"m3_{'X' if c == 'Y' else 'Y'}_at_{v}"]
    chamber, closure = {}, []
    for r in ROWS:
        tag = r['tag']
        A = Dall[Dall.tag == tag].copy()
        S = A[A.res_ok]
        chamber[tag] = {}
        for v in 'xy':
            g = fr[v]['glob']
            q = fit_single(S, g, v, pm=pm)
            q0 = fit_single(S, g, v, const=True)
            chamber[tag][v] = dict(model=q, const=q0, in_m3_fit=bool(r['pair']))
            A[f'sig_{v}'], A[f'vm3_{v}'] = sigma_track(A, g, q, v, pm=pm)
            A[f'sig0_{v}'], _ = sigma_track(A, g, q0, v, const=True)
            A[f'bias_{v}'] = q['m']; A[f'bias0_{v}'] = q0['m']
            S = A[A.res_ok]
            for name, (col, edges) in VARS.items():
                x = S[col.format(v=v)]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    s = S[(x >= lo) & (x < hi)]
                    if len(s) < 80:
                        continue
                    p1 = (s[f'res_{v}'] - q['m']) / s[f'sig_{v}']
                    p0 = (s[f'res_{v}'] - q0['m']) / s[f'sig0_{v}']
                    closure.append(dict(tag=tag, view=v, var=name, lo=lo, hi=hi, n=len(s),
                                        x=float(s[col.format(v=v)].mean()),
                                        core1=core_sigma(p1)[0], core0=core_sigma(p0)[0],
                                        f3_1=float(np.mean(np.abs(p1) < 3)), f3_0=float(np.mean(np.abs(p0) < 3)),
                                        res_core=core_sigma(s[f'res_{v}'])[0], sig_mean=float(s[f'sig_{v}'].mean())))
            print(f"{tag:7s} {v}  model: sigma_D(0) {np.sqrt(max(q['d0'], 0)) * 1e3:4.0f} um d1 {q['d1']:5.2f} "
                  f"eps {q['eps']:.3f} nll {q['nll']:.1f} | const: sigma_D {np.sqrt(max(q0['d0'], 0)) * 1e3:4.0f} um "
                  f"eps {q0['eps']:.3f} nll {q0['nll']:.1f}  -> dNLL {q0['nll'] - q['nll']:.1f}", flush=True)
        keep = ['event_id', 'tag', 'det', 'region', 'res_ok', 'fit', 'x_ok', 'y_ok', 'res_x', 'res_y', 'mesh_x',
                'mesh_y', 'x_p0', 'y_p0', 'tan_x', 'tan_y', 'chi2_X', 'chi2_Y', 'k_vx', 'k_vy', 'mu_vx', 'mu_vy',
                'near_res_x', 'near_res_y', 'x_n_dropped', 'y_n_dropped', 'cat_xy_5', 'det_any_5', 'spark_veto',
                'mult', 'in_table', 'not_measured'] + [f'{a}_{v}' for a in ('sig', 'sig0', 'vm3', 'bias', 'bias0')
                                                         for v in 'xy']
        if pm:
            A[[k for k in keep if k in A.columns]].to_parquet(os.path.join(OUT, f'sigma_{tag}.parquet'))
    C = pd.DataFrame(closure)
    if pm:
        json.dump(chamber, open(os.path.join(OUT, 'chamber.json'), 'w'), indent=1)
        C.to_parquet(os.path.join(OUT, 'closure.parquet'))
    # summary: spread of the pull core width over the bins of each variable (1 = perfect)
    print('\nclosure: pull core width across bins, mean and spread (std); per-track model vs constant pointing')
    summ = {}
    for (var, held), s in C.assign(held=C.tag.map(lambda t: 'held-out' if t in ('A_run1', 'A_run2', 'E_624')
                                                    else 'in fit')).groupby(['var', 'held']):
        summ[f'{var}|{held}'] = dict(mean1=float(s.core1.mean()), std1=float(s.core1.std()),
                                     mean0=float(s.core0.mean()), std0=float(s.core0.std()), bins=len(s))
        print(f'  {var:10s} {held:8s} model {s.core1.mean():.3f} ± {s.core1.std():.3f}   '
              f'const {s.core0.mean():.3f} ± {s.core0.std():.3f}   (bins {len(s)})')
    name = 'closure_summary.json' if pm else 'closure_summary_noposmap.json'
    json.dump(summ, open(os.path.join(OUT, name), 'w'), indent=1)
    if not pm:     # the no-map pass is only for the comparison: keep the products of the map pass
        return


if __name__ == '__main__':
    main()
