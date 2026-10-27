#!/usr/bin/env python3
"""gap_route.py [names...] -- the drift gap across the chamber from the column duration, separated from v(x, y).

In each 5 x 5 cell (70 mm, contained tracks: the M3 track is inside 15..385 mm at the mesh AND at the cathode):
  T(cell)  column duration, two estimators: q_uend (last depth bin above 5 % of the profile peak, after the fitted t0)
           and (q_u90 - q_u50)/0.4 (t0-free: the 50 % and 90 % charge-arrival times of a uniform column are T/2 and 0.9 T)
  b(cell)  ladder slope dw/dtan against M3 = the LOCAL drift velocity (times the local angle scale); the slope of the
           micro-TPC line does not depend on how deep the gap is.
  gap(cell) = T * b  (mm), normalised to 30 mm at the median cell.  A v(x, y) map (gas, field) moves T and b in
  opposite directions and cancels; a gap change moves T only.
Charge-matched variant: only tracks with q_sum inside the chamber's 30-70 % range (the column end depends on amplitude,
GAP_STUDY_2026-07-30 caveat 3).  A quadratic surface is fitted to each map (weights from cell bootstrap errors) and its
curvatures compared with the position-route mesh height h(U, W) from xcheck: a bow of the mesh by h with the cathode
following a fraction f gives gap = const - (1 - f) h.
Writes results/gap_<name>.json.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import DATASETS, RES, load, view_mask, huber, surf_terms, GAP

E = np.linspace(25, 375, 6)
C = 0.5 * (E[:-1] + E[1:])


def cellstat(x, rng, nb=30):
    lo, hi = np.percentile(x, [5, 95])
    f = lambda y: np.mean(np.clip(y, lo, hi))
    return f(x), np.std([f(x[q]) for q in (rng.integers(0, len(x), len(x)) for _ in range(nb))])


def quadfit(M, Me):
    """weighted quadratic surface over the 5x5 cell centres; returns coefficients [1,U,W,U2/2,W2/2,UW] and errors."""
    U, W = np.meshgrid((C - 200) / 100, (C - 200) / 100, indexing='ij')
    g = np.isfinite(M) & np.isfinite(Me) & (Me > 0)
    X = surf_terms(U[g], W[g]); wt = 1 / Me[g]
    A = X * wt[:, None]
    beta, *_ = np.linalg.lstsq(A, M[g] * wt, rcond=None)
    chi2 = float(np.sum((A @ beta - M[g] * wt) ** 2)); dof = max(int(g.sum()) - 6, 1)
    cov = np.linalg.inv(A.T @ A) * max(chi2 / dof, 1.0)
    return beta, np.sqrt(np.diag(cov)), chi2 / dof


def run(name):
    T = load(name)
    rng = np.random.default_rng(5)
    lo, hi = 15.0, 385.0
    cont = ((T.u_mesh > lo) & (T.u_mesh < hi) & (T.w_mesh > lo) & (T.w_mesh < hi) &
            (T.u_cath > lo) & (T.u_cath < hi) & (T.w_cath > lo) & (T.w_cath < hi)).to_numpy()
    u, w = T.u.to_numpy(), T.w.to_numpy()
    X = json.load(open(f'{RES}/xcheck_{name}.json'))
    out = dict(name=name, det=T.attrs['det'], slot=T.attrs['slot'])
    for p in 'xy':
        t = T[f'{p}_t'].to_numpy(); wv = T[f'{p}_wv'].to_numpy(); v = T.attrs['v']
        base = view_mask(T, p) & cont & (np.abs(t) < 0.35)
        q = T[f'{p}_q_sum'].to_numpy()
        qlo, qhi = np.nanpercentile(q[base], [30, 70])
        ests = {'uend': T[f'{p}_q_uend'].to_numpy(), 'u90m50': (T[f'{p}_q_u90'] - T[f'{p}_q_u50']).to_numpy() / 0.4}
        # ladder slope per cell (all contained tracks, not only the charge-matched ones)
        Bm = np.full((5, 5), np.nan); Be = Bm.copy()
        mb = view_mask(T, p) & np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
        for i in range(5):
            for j in range(5):
                k = mb & (u >= E[i]) & (u < E[i + 1]) & (w >= E[j]) & (w < E[j + 1])
                if k.sum() < 60:
                    continue
                Xd = np.c_[np.ones(k.sum()), t[k]]
                Bm[i, j] = huber(Xd, wv[k], 0.03 * v)[1]
                Be[i, j] = np.std([huber(Xd[s], wv[k][s], 0.03 * v)[1] for s in (rng.integers(0, k.sum(), k.sum()) for _ in range(20))])
        res = dict(b=Bm.tolist(), b_err=Be.tolist())
        for en, d in ests.items():
            for qm in (False, True):
                m = base & np.isfinite(d) & ((q >= qlo) & (q <= qhi) if qm else True)
                Tm = np.full((5, 5), np.nan); Te = Tm.copy()
                for i in range(5):
                    for j in range(5):
                        k = m & (u >= E[i]) & (u < E[i + 1]) & (w >= E[j]) & (w < E[j + 1])
                        if k.sum() < 30:
                            continue
                        Tm[i, j], Te[i, j] = cellstat(d[k], rng)
                Gm = Tm * Bm; Ge = Gm * np.hypot(Te / Tm, Be / Bm)
                norm = GAP / np.nanmedian(Gm)
                Gm, Ge = Gm * norm, Ge * norm
                Tn = GAP * Tm / np.nanmedian(Tm); Tne = GAP * Te / np.nanmedian(Tm)
                bg, eg, c2g = quadfit(Gm, Ge)
                bt, et, c2t = quadfit(Tn, Tne)
                key = f'{en}{"_q" if qm else ""}'
                res[key] = dict(T=Tm.tolist(), T_err=Te.tolist(), gap=Gm.tolist(), gap_err=Ge.tolist(),
                                gap_fit=bg.tolist(), gap_fit_err=eg.tolist(), gap_chi2dof=c2g,
                                Tonly_fit=bt.tolist(), Tonly_fit_err=et.tolist(), n=int(m.sum()),
                                gap_range=float(np.nanmax(Gm) - np.nanmin(Gm)))
        hx = X[f'pos_{p}']
        res['h_kuu'], res['h_kww'] = hx['kuu'], hx['kww']
        for key in ('uend', 'uend_q', 'u90m50', 'u90m50_q'):
            g = res[key]['gap_fit']
            res[key]['one_minus_f_uu'] = float(-g[3] / hx['kuu']) if abs(hx['kuu']) > 0.05 else np.nan
            res[key]['one_minus_f_ww'] = float(-g[4] / hx['kww']) if abs(hx['kww']) > 0.05 else np.nan
        out[p] = res
        r = res['uend_q']
        print(f'{name:14s} {p}: gap kuu {r["gap_fit"][3]:+.2f}±{r["gap_fit_err"][3]:.2f} kww {r["gap_fit"][4]:+.2f}±'
              f'{r["gap_fit_err"][4]:.2f} (h: {hx["kuu"]:+.2f} {hx["kww"]:+.2f}) -> 1-f {r["one_minus_f_uu"]:.2f} '
              f'{r["one_minus_f_ww"]:.2f} | u90m50_q 1-f {res["u90m50_q"]["one_minus_f_uu"]:.2f} '
              f'{res["u90m50_q"]["one_minus_f_ww"]:.2f} | T-only kuu {r["Tonly_fit"][3]:+.2f} kww {r["Tonly_fit"][4]:+.2f}',
              flush=True)
    json.dump(out, open(f'{RES}/gap_{name}.json', 'w'), indent=1)


if __name__ == '__main__':
    for n in (sys.argv[1:] or [k for k in DATASETS if os.path.exists(f"{RES}/xcheck_{k}.json") and not k.endswith(("d500", "d700"))]):
        run(n)
