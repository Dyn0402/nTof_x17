"""The per-track expected residual width: M3 pointing (fit.json globals) + the chamber term of the row.

sigma_track(D, glob, chamber) -> (sigma_x, sigma_y, V_m3_x, V_m3_y) per ray, in mm.  D must carry the add_m3 columns.
The slot (top/bottom) is decided by the alignment z.  The chamber term of a row comes from fit.json (pair rows) or
from fit_single() (rows with one chamber), with the M3 globals held fixed.
"""
import numpy as np
from scipy.optimize import minimize

from fit import MATCH, npdf, hess_cov  # noqa: F401


def v_m3(D, g, v, pm=None):
    """M3 variance of each ray; pm = posmap.json (the position factor of the M3 term), None = no position term"""
    from posmap import factor
    fl = g['a0'] + g['a1'] * (D[f'mu_v{v}'].to_numpy() - 1)
    top = D[f'z_{v}'].to_numpy() > 500
    kap = np.where(top, g['kt'], g['kb'])
    gam = np.where(top, g.get('gt', 0.0), g.get('gb', 0.0))
    V = D[f'P2_v{v}'].to_numpy() * fl + kap * D[f'k_v{v}'].to_numpy() + gam * D[f'g_v{v}'].to_numpy()
    return V * factor(D, v, pm)


def v_m3_const(D, v, a0=1.0):
    """the Paper I assumption: one pointing per slot, from the per-plane sigma (a0 = 1)"""
    return a0 * D[f'P2_v{v}'].to_numpy()


def v_chamber(D, q, v):
    return q['d0'] + q['d1'] * D[f't_v{v}'].to_numpy() ** 2


def fit_single(D, g, v, const=False, pm=None):
    """univariate mixture fit of one row's residuals of view v with the M3 term fixed: the chamber term (d0, d1),
    bias m, outlier fraction eps and width w.  const=True fits the same with the Paper I constant pointing and a
    chamber term without angle dependence (d1 = 0): the reference the per-track model is compared to."""
    r = D[f'res_{v}'].to_numpy()
    V = v_m3_const(D, v) if const else v_m3(D, g, v, pm)
    t2 = D[f't_v{v}'].to_numpy() ** 2

    def unpack(p):
        return dict(d0=p[0], d1=0.0 if const else p[1], m=p[2], eps=1 / (1 + np.exp(-p[3])), w=np.exp(p[4]))

    def nll(p):
        q = unpack(p)
        s2 = q['d0'] + q['d1'] * t2 + V
        if np.any(s2 <= 1e-4):
            return 1e12
        x = r - q['m']
        L = (1 - q['eps']) * npdf(x, s2) + q['eps'] * npdf(x, q['w'] ** 2)
        return -np.sum(np.log(L + 1e-300))

    p0 = np.array([0.05, 0.5, 0.0, np.log(0.05 / 0.95), np.log(3.0)])
    res = minimize(nll, p0, method='Powell', options=dict(maxiter=20000, xtol=1e-6, ftol=1e-10))
    res = minimize(nll, res.x, method='L-BFGS-B')
    q = unpack(res.x)
    q = {k: float(x) for k, x in q.items()}
    q['nll'] = float(res.fun); q['n'] = int(len(r))
    return q


def sigma_track(D, g, q, v, const=False, pm=None):
    V = v_m3_const(D, v) if const else v_m3(D, g, v, pm)
    return np.sqrt(np.maximum(v_chamber(D, q, v) + V, 1e-6)), V
