"""e_05_headon.py -- cross-check: head-on tracks only (|tan| < 0.06), z_k per slice from the full regression, delta_k = mean(cen - ref - tan z_k)
-> $WORK/audit/e/headon.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import e_lib as L, e_03_main as M
GM = L.GM
D = M.load('')
out = {}
for v, axis in (('x', 'U'), ('y', 'W'), ('x', 'W'), ('y', 'U')):
    d = M.base_filter(D[v])
    e = GM.GRIDS['x5'][0] if axis == 'U' else GM.GRIDS['y5'][1]
    col = 'xm' if axis == 'U' else 'ym'
    res = []
    for i in range(5):
        s = d[(d[col] >= e[i]) & (d[col] < e[i + 1])]
        full = M.region_fit(s, boot=0)
        h = s[s.tan.abs() < 0.06]
        nb = len(M.EDGES) - 1
        tb = np.digitize(h.t.values, M.EDGES) - 1
        ok = (tb >= 0) & (tb < nb); h = h[ok]; tb = tb[ok]
        z = full['z']
        y = (h.cen - h.ref).values - h.tan.values * z[tb]
        # robust: clip 4 sigma
        keep = np.ones(len(y), bool)
        for _ in range(3):
            for b in range(nb):
                m = tb == b
                if m.sum() > 10:
                    mu = np.median(y[m & keep]); sg = 1.4826 * np.median(np.abs(y[m & keep] - mu))
                    keep[m] = np.abs(y[m] - mu) < 4 * max(sg, 1e-6)
        def prof(wt):
            n = np.bincount(tb, weights=wt * keep, minlength=nb)[:nb]
            return np.bincount(tb, weights=wt * keep * y, minlength=nb)[:nb] / n, n
        dl, n = prof(np.ones(len(y)))
        sel = full['sel'] & (n > 30)
        err = np.array([y[(tb == b) & keep].std() / np.sqrt(max(n[b], 1)) for b in range(nb)])
        w = 1 / np.maximum(err, 1e-3) ** 2
        par = M.fitq(z[sel], dl[sel], w[sel])
        ue, inv = np.unique(h.eid.values, return_inverse=True)
        rng = np.random.default_rng(7 + i)
        pb = []
        for _ in range(M.B):
            cnt = np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(float)
            d_, _n = prof(cnt[inv])
            pb.append(M.fitq(z[sel], d_[sel], w[sel]))
        res.append(dict(par=par, parb=np.array(pb), zr=(float(z[sel].min()), float(z[sel].max())), coord=float(s[axis].mean()),
                        nev=int(h.eid.nunique()), delta=dl.tolist(), z=z.tolist(), sl=float(M.fitlin(z[sel], dl[sel], w[sel])[1]),
                        slb=np.array([M.fitlin(z[sel], prof(c)[0][sel], w[sel])[1] for c in [np.bincount(rng.integers(0, len(ue), len(ue)), minlength=len(ue)).astype(float)[inv] for _ in range(60)]])))
    # gradient needs slb of length B for 'sl': skip 'sl' here
    r = {}
    for key in ('b', 'D', 's_mean'):
        g = M.gradient(res, key)
        r[key] = dict(grad=g['grad'], err=g['err'], y=g['y'], chi2=g['chi2'])
    r['nev'] = [x['nev'] for x in res]
    out[f'{v}|{axis}'] = r
    print(v, axis, {k: (round(r[k]['grad'] * 1e4, 1), round(r[k]['err'] * 1e4, 1)) for k in ('b', 'D', 's_mean')}, r['nev'], flush=True)
json.dump(out, open(f'{L.OUT}/headon.json', 'w'), indent=1, default=float)
