#!/usr/bin/env python3
"""q7_pointing.py -- sideband-subtracted capsule pointing per configuration on the real run_145 tracks.

Pointing excess = N(dca < 15 mm) - N(15 <= dca < 30 mm): the background floor of line-to-axis distances is ~flat at
small dca (through-goers, wall electrons, flash tail), so the 15-30 mm band estimates the chance-pointing tracks inside
15 mm.  Reads $Q7_DATA/real/<stem>.tracks.parquet (q7_real_score.py beam); writes pointing.json.
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402
REAL = os.path.join(L.DATA, 'real')
TB = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.5)]
out = {}
for stem in sys.argv[1:] or ['r145full_A', 'r145full_C']:
    T = pd.read_parquet(os.path.join(REAL, f'{stem}.tracks.parquet'))
    have = T.groupby('tag').cfg.nunique()
    T = T[T.tag.isin(have[have == T.cfg.nunique()].index) & T.gated]
    T['tmax'] = np.maximum(T.tanx.abs(), T.tany.abs())
    res = {}
    for cf, g in T.groupby('cfg'):
        r = {}
        for acc in (0.6, 1.0):
            h = g[g.tmax < acc]
            p, s = int((h.dca < 15).sum()), int(((h.dca >= 15) & (h.dca < 30)).sum())
            r[f'acc{acc:g}'] = dict(n=int(len(h)), p15=p, sb=s, excess=p - s, err=float(np.sqrt(p + s)))
        r['bins'] = []
        for lo, hi in TB:
            h = g[(g.tmax >= lo) & (g.tmax < hi)]
            p, s = int((h.dca < 15).sum()), int(((h.dca >= 15) & (h.dca < 30)).sum())
            r['bins'].append(dict(lo=lo, hi=hi, n=int(len(h)), p15=p, sb=s, excess=p - s))
        res[cf] = r
    out[stem] = res
    print('==', stem)
    for cf, r in sorted(res.items()):
        a6, a1 = r['acc0.6'], r['acc1']
        print(f"{cf:20s} acc0.6 n {a6['n']:5d} excess {a6['excess']:5d}+-{a6['err']:.0f} | acc1 n {a1['n']:5d} excess {a1['excess']:5d}+-{a1['err']:.0f} | "
              + ' '.join(f"{b['lo']}-{b['hi']}:{b['excess']}/{b['n']}" for b in r['bins']))
json.dump(out, open(os.path.join(REAL, 'pointing.json'), 'w'), indent=1)
