#!/usr/bin/env python3
"""dev_dtsplit.py <row.npz> <rays.parquet> -- residual width at small / large |tan| for events with small / large |dt|
(the chamber's own column mid-time offset).  A trigger jitter widens only the large-|tan|, large-|dt| cell."""
import sys

import numpy as np
import pandas as pd

z = np.load(sys.argv[1]); R = pd.read_parquet(sys.argv[2])
T = pd.DataFrame(dict(eid=z['eid'], tm=(z['tx'] + z['ty'] + z['ex'] + z['ey']) / 4))
T = T[~T.eid.duplicated(keep=False)]
m = R.merge(T, left_on='event_id', right_on='eid')
m['dt'] = m.tm - m.tm.median()


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(np.percentile(np.abs(a - np.median(a)), 68.27)) if len(a) > 30 else np.nan


q = np.nanpercentile(np.abs(m.dt), 50)
print('median |dt|', round(q, 1), 'ns')
for v in 'xy':
    s = np.isfinite(m[f'res_{v}']) & (m[f'res_{v}'].abs() < 15) & m[f'{v}_ok'].astype(bool) & np.isfinite(m[f'tan_{v}'])
    mm = m[s]
    for tl, th in ((0, 0.08), (0.15, 0.45)):
        row = []
        for nm, sel in (('|dt| small', mm.dt.abs() < q), ('|dt| large', mm.dt.abs() >= q)):
            c = sel & (mm[f'tan_{v}'].abs() >= tl) & (mm[f'tan_{v}'].abs() < th)
            row.append(f'{nm}: n {c.sum():4d} s68 {s68(mm[f"res_{v}"][c]):.3f}')
        print(v, f'|tan| {tl}-{th}', ' | '.join(row))
