#!/usr/bin/env python3
"""show.py <row.json> ... -- print a row result's estimator table (development / quick look)."""
import json
import sys

for f in sys.argv[1:]:
    j = json.load(open(f))
    print(f, j.get('key', ''), 'n_clean', j.get('n_clean'))
    if 'estimators' not in j:
        print('  ', j.get('skip')); continue
    for k, v in j['estimators'].items():
        e = v['err']
        print(f'  {k:7s} ' + '  '.join(f'{a} {v[a]:6.1f}±{e.get(a, 0):.1f}' for a in
                                        ('sigma_xy', 'sigma_x', 'sigma_y', 'sigma_evt', 'sigma_common', 'bias_xy')))
    for k in ('trig_cov_ns2', 'trig_cov_err_ns2', 'startstart_cov_ns2', 'v_timing_um_ns', 'mm',
              'walk_median_evt_by_amp_quartile', 'walk_sigma_evt_by_amp_quartile'):
        print('  ', k, j.get(k))
