import sys, numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, boot, surf_terms, SURF
def fit(T, p, sel, nboot=40):
    m = view_mask(T, p) & sel
    t = T[f'{p}_t'].to_numpy()[m]; r = T[f'{p}_r'].to_numpy()[m]
    U = (T.u.to_numpy()[m]-200)/100; W = (T.w.to_numpy()[m]-200)/100
    X = np.c_[np.ones_like(U), U, W, t[:,None]*surf_terms(U, W)]
    b = huber(X, r, 0.5); bs = boot(lambda i: huber(X[i], r[i], 0.5), len(r), nboot)
    n = ['a','bu','bw']+SURF; d = {k: float(v) for k, v in zip(n, b)}; d.update({k+'_err': float(e) for k, e in zip(n, bs.std(0))}); return d
print("det3 rows: kuu (curvature along x) and kww (along y) from the SAME view, own axis restricted to |own-200|<110 mm (strip-plane edges out), other axis full")
for name in ['det3', 'det3_p2', 'det3_day625', 'det2', 'det4', 'det6', 'det7']:
    T = load(name); u = T.u.to_numpy(); w = T.w.to_numpy()
    out = []
    diffs = []
    for p in 'xy':
        own = u if p == 'x' else w
        f = fit(T, p, np.abs(own-200) < 110)
        d = f['kww'] - f['kuu']; e = np.hypot(f['kww_err'], f['kuu_err'])
        diffs.append((d, e))
        out.append(f"[{p}] kuu {f['kuu']:+.2f}±{f['kuu_err']:.2f} kww {f['kww']:+.2f}±{f['kww_err']:.2f} (y-x {d:+.2f}±{e:.2f}, n={len(T[f'{p}_t'][view_mask(T,p)&(np.abs(own-200)<110)])})")
    wmean = sum(d/e**2 for d, e in diffs)/sum(1/e**2 for d, e in diffs); werr = (sum(1/e**2 for d, e in diffs))**-0.5
    print(f"{name:12s} " + ' | '.join(out) + f"  => y-x both views {wmean:+.2f}±{werr:.2f}")
