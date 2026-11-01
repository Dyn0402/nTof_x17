#!/usr/bin/env python3
"""boot_summary.py [results/bootstrap.json] -- the tables of BOOTSTRAP_2026-11-01.md from boot_analyze.py's output:
the calibrated share of the clean-event remainder with a propagated error (L8, T, D bootstrap errors), the data-truth
ratios, and each structure's own share for data and toy residuals."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'bootstrap.json')))
rs = np.random.default_rng(1)
out = {}
for k, r in d.items():
    t = r['toy']
    row = dict(clean=r['data_clean_s68'], U_cond=r['U_cond'])
    if all(a in t for a in ('L8', 'T', 'D')):
        L8, T, D = (t[a]['s68'] for a in ('L8', 'T', 'D'))
        eL, eT, eD = (t[a]['err'] for a in ('L8', 'T', 'D'))
        sh = [(dd * dd - tt * tt) * ll * ll / (tt * tt * r['U_cond'] ** 2)
              for ll, tt, dd in zip(L8 + eL * rs.normal(size=4000), T + eT * rs.normal(size=4000),
                                    D + eD * rs.normal(size=4000))]
        row.update(L8=L8, T=T, N=t['N']['s68'], D=D, D_over_T=D / T, eps=r['eps'], U_boot=r['U_boot'],
                   U_boot_cal=r['U_boot_cal'], share_cal=r['share_cal'], share_cal_err=float(np.std(sh)),
                   chi_data=r['data_chi2dof'], chi_D=t.get('D_chi'), chi_T=t.get('T_chi'))
    if 'dD' in r['data_truth']:
        row.update(dD=r['data_truth']['dD']['s68'], dN=r['data_truth']['dN']['s68'],
                   dD_over_meas=r['data_truth_over_measured'],
                   dD_eps_over_meas=(r['data_truth']['dD']['s68'] / r['eps'] / r['data_clean_s68'])
                   if r.get('eps') else None, data_truth_share=r.get('data_truth_share'))
    row['own_share'] = r.get('own_share')
    out[k] = row
    print(k, ' '.join(f'{a}={v:.3f}' if isinstance(v, float) else f'{a}={v}' for a, v in row.items()
                      if not isinstance(v, dict)))
    for a in ('own_share', 'data_truth_share'):
        if row.get(a):
            print('   ', a, {s: (round(v, 2) if isinstance(v, float) else
                                {kk: (round(vv, 2) if vv is not None else None) for kk, vv in v.items()})
                             for s, v in row[a].items()})
json.dump(out, open(os.path.join(HERE, 'results', 'bootstrap_summary.json'), 'w'), indent=1)
