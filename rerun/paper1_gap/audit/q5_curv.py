"""q5_curv.py -- item 5: the curvature of chamber A's gap (and cathode) along x.  Event-level robust fit of
ln q_u90 = quad(U, W) + |tan| terms (as R4), golden mounting, per run, for several x ranges; guu converted to mm per
(100 mm)^2 with beta 0.68 (G0 / (1 + beta)).  Cathode curvature = gap curvature + mesh curvature (bow block h, kuu).
Errors: bootstrap (100)."""
import numpy as np
from r3_lib import load_mounting, G0, BETA0

D, info, bundle = load_mounting('A628')
FAC = G0 / (1 + BETA0)
kuu_mesh = bundle['h']['kuu']
rng = np.random.default_rng(5)


def fit(m):
    ok = m & np.isfinite(D.lnq) & np.isfinite(D.tpx) & np.isfinite(D.tpy)
    X = np.c_[np.ones(D.n), D.U, D.W, D.U ** 2 / 2, D.W ** 2 / 2, D.U * D.W, np.abs(D.tpx), np.abs(D.tpy)]
    keep = ok.copy()
    for _ in range(3):
        c = np.linalg.lstsq(X[keep], D.lnq[keep], rcond=None)[0]
        r = D.lnq - X @ c
        keep = ok & (np.abs(r) < 3.5 * 1.4826 * np.median(np.abs(r[ok])))
    return c


print(f'mesh (bow block h) kuu = {kuu_mesh:+.2f} mm per (100 mm)^2')
for lab, m in (('all x', np.ones(D.n, bool)), ('x > 80 mm', D.U > -1.2), ('x < 320 mm', D.U < 1.2),
               ('80 < x < 320', (D.U > -1.2) & (D.U < 1.2)), ('x < 200', D.U < 0), ('x > 200', D.U > 0)):
    for r in (0, 1):
        mm = m & (D.row == r)
        c = fit(mm)
        bs = []
        for _ in range(100):
            idx = rng.integers(0, D.n, D.n)
            w = np.bincount(idx, minlength=D.n) > 0
            bs.append(fit(mm & w)[3])
        g, e = FAC * c[3], FAC * np.std(bs)
        print(f'{lab:14s} run {r + 1}: gap kuu {g:+.2f} +- {e:.2f}  ->  cathode kuu {g + kuu_mesh:+.2f} mm per (100 mm)^2')
