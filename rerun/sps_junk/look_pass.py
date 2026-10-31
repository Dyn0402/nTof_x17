#!/usr/bin/env python3
"""look_pass.py <key> ...: in the pass-1 SPS table, the Y fits at 154-160.5 mm against the rest: charge, footprint,
peak time, X view, the own centroid.  Read-only."""
import sys
import numpy as np
import pandas as pd

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1/sps'
for key in sys.argv[1:]:
    t = pd.read_parquet(f'{A}/{key}/events.parquet')
    ok = t.y_ok.astype(bool)
    y = t.y_p0.where(ok)
    j = (y > 154.0) & (y < 160.5)
    c = (t.y_c_pos > 154.0) & (t.y_c_pos < 160.5)
    print(f'== {key}: n {len(t)}, y_ok {ok.mean():.3f}, junk-zone fits {j.mean():.4f}, junk-zone centroid {c.mean():.4f}')
    # beam profile of the centroid in 1.5 mm bins around the zone
    h, e = np.histogram(t.y_c_pos.dropna(), bins=np.arange(130, 186, 1.56))
    print('  y_c_pos profile:', ' '.join(f'{e[i]:.0f}:{h[i]}' for i in range(len(h))))
    cols = ['y_c_q_lead', 'y_c_q_sum', 'y_c_n_strips', 'y_c_t_peak_ns', 'y_chi2', 'x_c_q_lead', 'x_c_pos']
    cols = [k for k in cols if k in t]
    for name, m in (('zone', c), ('rest', ~c & t.y_c_pos.notna())):
        print(f'  {name:5s} n={int(m.sum()):5d} ' + ' '.join(f'{k}={np.nanmedian(t.loc[m, k]):.1f}' for k in cols))
    if 'y_c_t_peak_ns' in t:
        for name, m in (('zone', c), ('rest', ~c & t.y_c_pos.notna())):
            q = np.nanpercentile(t.loc[m, 'y_c_t_peak_ns'], [5, 25, 50, 75, 95])
            print(f'  t_peak {name}: ' + ' '.join(f'{v:.0f}' for v in q))
