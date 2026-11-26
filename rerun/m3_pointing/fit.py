#!/usr/bin/env python3
"""fit.py -- the per-track M3 pointing model, fitted on the rows with two chambers in the telescope at once.

For a track i and a chamber view (x or y; view x is read against M3 Y, view y against M3 X, the chambers are rotated
by ~90 deg), the residual r = chamber - M3 at the alignment z has variance

    Var_i = sigma_D^2(t)  +  V_M3(z; mu_i, k_i)

  sigma_D^2(t) = d0 + d1 t^2           the chamber (t = the track's tan in that view); per chamber row
  V_M3 = P_s^2 [a0 + a1 (mu_i/mu_0 - 1)]  +  kappa_s k_i                                       (M3; slot s)

  mu_i  the ordinary chi2 mean at the track's slope (decompose.json): it follows the plane resolution, so the floor
        grows with it.  P_s^2 is the Gaussian pointing of the slot from the measured per-plane sigma
        (m3_self_resolution), and a0 = 1 is the Paper I assumption.
  k_i   the posterior mean of the anomalous part of the chi2 of the matched coordinate (decompose.posterior).

For two chambers on the same track, the M3 error is common, so their residuals are correlated:
    Cov_i = P_tb [a0 + a1 (mu_i/mu_0 - 1)] + kappa_tb k_i,     P_tb = sum_k g_k(z_t) g_k(z_b) sigma_k^2
The floor's top : bottom : cross ratios are fixed by the fit geometry (plane errors), which makes the floor and the
chambers separable: a0 and a1 are pinned by the covariance, which no chamber contributes to.  kappa_top, kappa_bottom
and kappa_cross are free: the slot response to anomalies depends on where they happen.

Each chamber's residual also has a broad outlier component (a Gaussian of width w, fraction eps, independent between
the chambers).  Likelihood: the bivariate mixture over every pair event, per view.  Writes $WORK/fit.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m3common import load, WORK, Z_LAYERS, SIG_PLANE, hat, pointing_gauss  # noqa: E402
from decompose import posterior  # noqa: E402

PAIRS = ('p622', 'p623', 'p626')
MATCH = {'x': 'Y', 'y': 'X'}            # M3 coordinate read by each chamber view
OUT = os.path.join(WORK, 'fit.json')


def add_m3(D, par):
    for c in 'XY':
        D[f'k_{c}'], D[f'mu_{c}'] = posterior(par[c], D[f'chi2_{c}'].to_numpy(), D[f'm3_s{c.lower()}'].to_numpy())
    for v, c in MATCH.items():
        D[f'k_v{v}'] = D[f'k_{c}']; D[f'mu_v{v}'] = D[f'mu_{c}'] / par[c]['mu'][0]
        D[f'g_v{v}'] = np.maximum(D[f'chi2_{c}'] - D[f'k_{c}'], 0.0)       # posterior ordinary part E[g | chi2]
        D[f'P2_v{v}'] = [pointing_gauss(z, c) ** 2 for z in D[f'z_{v}']]
        D[f't_v{v}'] = D[f'tan_{v}'].abs()
    return D


def pair_table(D):
    P = []
    for p in PAIRS:
        S = D[D.pair == p]
        tags = list(S.tag.unique())
        top = [t for t in tags if S[S.tag == t].z_x.iloc[0] > 500][0]
        bot = [t for t in tags if t != top][0]
        P.append(S[S.tag == top].merge(S[S.tag == bot], on='event_id', suffixes=('_t', '_b')))
    return pd.concat(P, ignore_index=True)


def cross_P(zt, zb, c):
    return float(np.sum(hat(Z_LAYERS, zt) * hat(Z_LAYERS, zb) * SIG_PLANE[c] ** 2))


def npdf(x, s2):
    return np.exp(-0.5 * x * x / s2) / np.sqrt(2 * np.pi * s2)


class PairModel:
    """the bivariate fit of one view over the pairs"""
    GLOBAL = ['a0', 'a1', 'kt', 'kb', 'ktb', 'gt', 'gb', 'gtb']
    PER = ['d0', 'd1', 'm', 'leps', 'lw']

    def __init__(self, T, v):
        self.v = v
        self.rows = sorted(set(T.tag_t) | set(T.tag_b))
        self.T = T
        c = MATCH[v]
        self.rt, self.rb = T[f'res_{v}_t'].to_numpy(), T[f'res_{v}_b'].to_numpy()
        self.mu = T[f'mu_v{v}_t'].to_numpy(); self.k = T[f'k_v{v}_t'].to_numpy(); self.g = T[f'g_v{v}_t'].to_numpy()
        self.Pt, self.Pb = T[f'P2_v{v}_t'].to_numpy(), T[f'P2_v{v}_b'].to_numpy()
        self.Ptb = np.array([cross_P(a, b, c) for a, b in zip(T[f'z_{v}_t'], T[f'z_{v}_b'])])
        self.tt, self.tb = T[f't_v{v}_t'].to_numpy(), T[f't_v{v}_b'].to_numpy()
        self.it = np.array([self.rows.index(x) for x in T.tag_t]); self.ib = np.array([self.rows.index(x) for x in T.tag_b])

    def unpack(self, p):
        ng = len(self.GLOBAL)
        g = dict(zip(self.GLOBAL, p[:ng]))
        q = p[ng:].reshape(len(self.rows), len(self.PER))
        return g, q

    def moments(self, p):
        g, q = self.unpack(p)
        fl = g['a0'] + g['a1'] * (self.mu - 1)
        Vt = q[self.it, 0] + q[self.it, 1] * self.tt ** 2 + self.Pt * fl + g['kt'] * self.k + g['gt'] * self.g
        Vb = q[self.ib, 0] + q[self.ib, 1] * self.tb ** 2 + self.Pb * fl + g['kb'] * self.k + g['gb'] * self.g
        C = self.Ptb * fl + g['ktb'] * self.k + g['gtb'] * self.g
        return Vt, Vb, C, q

    def nll(self, p):
        Vt, Vb, C, q = self.moments(p)
        if np.any(Vt <= 1e-4) or np.any(Vb <= 1e-4):
            return 1e12
        det = Vt * Vb - C * C
        if np.any(det <= 1e-8):
            return 1e12
        xt, xb = self.rt - q[self.it, 2], self.rb - q[self.ib, 2]
        quad = (Vb * xt * xt - 2 * C * xt * xb + Vt * xb * xb) / det
        n2 = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
        et = 1 / (1 + np.exp(-q[self.it, 3])); eb = 1 / (1 + np.exp(-q[self.ib, 3]))
        wt, wb = np.exp(q[self.it, 4]) ** 2, np.exp(q[self.ib, 4]) ** 2
        L = ((1 - et) * (1 - eb) * n2 + et * (1 - eb) * npdf(xt, wt) * npdf(xb, Vb)
             + (1 - et) * eb * npdf(xt, Vt) * npdf(xb, wb) + et * eb * npdf(xt, wt) * npdf(xb, wb))
        return -np.sum(np.log(L + 1e-300))

    def p0(self):
        g = [1.0, 0.5, 1.0, 0.7, 0.6, 0.5, 0.3, 0.3]
        per = [0.1, 1.0, 0.0, np.log(0.05 / 0.95), np.log(2.5)] * len(self.rows)
        return np.array(g + per, float)

    def fit(self):
        r = minimize(self.nll, self.p0(), method='Powell', options=dict(maxiter=60000, xtol=1e-5, ftol=1e-9))
        r = minimize(self.nll, r.x, method='L-BFGS-B')
        # errors from the numerical Hessian of the globals (others profiled at their best values)
        cov = hess_cov(self.nll, r.x, idx=range(len(self.GLOBAL)))
        g, q = self.unpack(r.x)
        per = {row: dict(zip(self.PER, q[i])) for i, row in enumerate(self.rows)}
        for row in per.values():
            row['eps'] = float(1 / (1 + np.exp(-row.pop('leps')))); row['w'] = float(np.exp(row.pop('lw')))
            for kk in ('d0', 'd1', 'm'):
                row[kk] = float(row[kk])
        err = dict(zip(self.GLOBAL, np.sqrt(np.clip(np.diag(cov), 0, None))))
        return dict(view=self.v, nll=float(r.fun), n=len(self.rt), glob={k: float(v) for k, v in g.items()},
                    err={k: float(v) for k, v in err.items()}, rows=per, x=list(map(float, r.x)))


def hess_cov(f, x, idx, h=1e-3):
    idx = list(idx); n = len(idx); H = np.zeros((n, n)); x = np.asarray(x, float)
    f0 = f(x)
    for a in range(n):
        for b in range(a, n):
            def e(i, s):
                y = x.copy(); y[idx[i]] += s; return y
            if a == b:
                H[a, a] = (f(e(a, h)) - 2 * f0 + f(e(a, -h))) / h ** 2
            else:
                y = x.copy(); y[idx[a]] += h; y[idx[b]] += h; fpp = f(y)
                y = x.copy(); y[idx[a]] += h; y[idx[b]] -= h; fpm = f(y)
                y = x.copy(); y[idx[a]] -= h; y[idx[b]] += h; fmp = f(y)
                y = x.copy(); y[idx[a]] -= h; y[idx[b]] -= h; fmm = f(y)
                H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full((n, n), np.nan)


def main():
    par = json.load(open(os.path.join(WORK, 'decompose.json')))
    D = add_m3(load(), par)
    T = pair_table(D)
    out = {}
    for v in 'xy':
        res = PairModel(T, v).fit()
        out[v] = res
        g, e = res['glob'], res['err']
        print(f"view {v} (M3 {MATCH[v]}): n {res['n']}  " + '  '.join(f'{k} {g[k]:.3f}±{e[k]:.3f}' for k in g))
        for row, q in res['rows'].items():
            print(f"   {row:6s} sigma_D(t=0) {np.sqrt(max(q['d0'], 0)) * 1000:5.0f} um  d1 {q['d1']:6.2f}  "
                  f"bias {q['m'] * 1000:5.0f} um  eps {q['eps']:.3f}  w {q['w']:.2f} mm")
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
