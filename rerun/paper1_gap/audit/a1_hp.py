"""a1_hp.py -- high-pass (cell minus mean of its 4 neighbours) cross-run covariance: how much real structure is at the 1-2 cell scale"""
import json, pickle
import numpy as np
from a1_lib import *  # noqa
maps = pickle.load(open(f'{OUT}/maps.pkl', 'rb'))['maps']
out = {}
rng = np.random.default_rng(3)
def hp(R, e):
    n = R.shape[0]
    d = np.full(R.shape, np.nan); v = np.full(R.shape, np.nan)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            nb = [R[i - 1, j], R[i + 1, j], R[i, j - 1], R[i, j + 1]]
            if np.isfinite(R[i, j]) and np.all(np.isfinite(nb)):
                d[i, j] = R[i, j] - np.mean(nb)
                v[i, j] = e[i, j] ** 2 + (e[i - 1, j] ** 2 + e[i + 1, j] ** 2 + e[i, j - 1] ** 2 + e[i, j + 1] ** 2) / 16
    return d, v
for n in (6, 8, 10, 12):
    for p in 'xy':
        m1, m2 = maps[n]['run1'][p], maps[n]['run2'][p]
        R1 = residual(m1, 'tau', 'etau', 'plane')[0]; R2 = residual(m2, 'tau', 'etau', 'plane')[0]
        d1, v1 = hp(R1, m1['etau']); d2, v2 = hp(R2, m2['etau'])
        ok = np.isfinite(d1) & np.isfinite(d2)
        a, b = d1[ok], d2[ok]
        cov = np.mean(a * b); bs = [np.mean(a[k] * b[k]) for k in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
        out[f'{n}_{p}'] = dict(ncell=int(ok.sum()), cov=float(cov), cov_err=float(np.std(bs)), sig=float(np.sqrt(max(cov, 0))),
                               var1=float(np.mean(a ** 2)), var2=float(np.mean(b ** 2)), noise1=float(np.mean(v1[ok])), noise2=float(np.mean(v2[ok])),
                               r=float(np.corrcoef(a, b)[0, 1]))
        o = out[f'{n}_{p}']
        print(n, p, o['ncell'], f"cov {o['cov']*1e6:.2f}+-{o['cov_err']*1e6:.2f} e-6  sig {o['sig']*1e3:.2f}e-3  var1 {o['var1']*1e6:.1f} noise1 {o['noise1']*1e6:.1f} var2 {o['var2']*1e6:.1f} noise2 {o['noise2']*1e6:.1f}  r {o['r']:.2f}")
json.dump(out, open(f'{OUT}/hp.json', 'w'), indent=1)
