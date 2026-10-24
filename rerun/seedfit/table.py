#!/usr/bin/env python3
"""table.py <det> [v_timing] -- ladder / s68 / far / wrong-basin per bundle variant (nomask, dead, deadgain) and fitter,
from ~/x17/cosmic_bench/vladder/s[fm]_<det>_*.json."""
import glob, json, os, sys
import numpy as np
L = os.path.expanduser('~/x17/cosmic_bench/vladder/')
det = sys.argv[1]
vt = float(sys.argv[2]) if len(sys.argv) > 2 else None
files = {'nomask': L + f'sf_{det}_400.json', 'dead': L + f'sm_{det}_bundle_dead.json', 'deadgain': L + f'sm_{det}_bundle_deadgain.json'}
for f in sorted(glob.glob(L + f'sx_{det}_*.json')):
    files[os.path.basename(f)[len(f'sx_{det}_'):-5]] = f


def ladder(tr, w):
    m = np.isfinite(w)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.isfinite(w) & (np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27))
    rng = np.random.default_rng(0); idx = np.flatnonzero(m)
    e = np.std([np.polyfit(tr[j], w[j], 1)[0] for j in (rng.choice(idx, len(idx)) for _ in range(100))])
    return s, c, e


for p in ('x', 'y'):
    print(f'--- {det} {p}' + (f'  (timing {vt})' if vt else ''))
    for var, f in files.items():
        if not os.path.exists(f):
            continue
        E = json.load(open(f))['events']
        for n in ('prod', 'seeded', 'pivot', 'pivot2', 'pivot_fast', 'reco', 'reco_shear', 'reco_2s'):
            rows = [e[p] for e in E.values() if p in e and n in e[p] and 'chi2' in e[p][n]]
            if not rows:
                continue
            tr = np.array([r['tr'] for r in rows]); w = np.array([r[n]['w'] for r in rows])
            s, c, e = ladder(tr, w)
            th = np.degrees(np.arctan((w - c) / s)) - np.degrees(np.arctan(tr))
            far = np.abs(th) > 5
            s68 = np.percentile(np.abs(th[~far] - np.median(th[~far])), 68.27)
            print(f'  {var:9s} {n:10s} ladder {s:6.2f} ± {e:.2f}' + (f' ({100 * (s / vt - 1):+5.1f}%)' if vt else '') +
                  f'  s68 {s68:.3f}  far {far.mean() * 100:4.1f}%  chi2 med {np.median([r[n]["chi2"] for r in rows]):8.0f}  '
                  f'{np.mean([r[n]["sec"] for r in rows]):.2f}s')
