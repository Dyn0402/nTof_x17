#!/usr/bin/env python3
"""Pivot-anchored vs t0-anchored mesh-height estimator.
t0-anchored (the FB route): r = p0 - ref_mesh, h = dr/dtan.
pivot-anchored: r_piv = r + (w - v*tan) * tau,  tau = q_u50 (median charge-arrival time after t0, ns):
  if the fitted line is right at the charge centroid and wrong at t0 (a slope-scale bias), r_piv removes that bias;
  if the line is right at t0 and the local slope differs from v*tan for real reasons (true v(x,y)), r_piv adds a bias.
A geometric h is the same surface for X and Y strips, so the estimator with the smaller X-Y cell disagreement and the
smoother (inner-consistent) dish is the less contaminated one.
"""
import sys, os
import numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, boot, surf_terms, SURF, DETS

EDGES5 = np.linspace(25, 375, 6)


def resid(T, p, v, mode):
    r = T[f'{p}_r'].to_numpy().copy()
    if mode == 'piv':
        wv = T[f'{p}_wv'].to_numpy(); t = T[f'{p}_t'].to_numpy(); tau = T[f'{p}_q_u50'].to_numpy()
        r = r + (wv - v * t) * tau * 1e-3
    return r


def fit2d(T, p, v, mode, sel=None, nboot=20):
    m = view_mask(T, p)
    r_all = resid(T, p, v, mode)
    m &= np.isfinite(r_all)
    if sel is not None:
        m &= sel
    t = T[f'{p}_t'].to_numpy()[m]; r = r_all[m]
    U = (T.u.to_numpy()[m] - 200) / 100; W = (T.w.to_numpy()[m] - 200) / 100
    X = np.c_[np.ones_like(U), U, W, t[:, None] * surf_terms(U, W)]
    beta = huber(X, r, 0.5)
    bs = boot(lambda i: huber(X[i], r[i], 0.5), len(r), nboot)
    names = ['a', 'bu', 'bw'] + SURF
    out = {k: float(x) for k, x in zip(names, beta)}
    out.update({k + '_err': float(e) for k, e in zip(names, bs.std(0))})
    out['n'] = int(len(r))
    return out


def cellmap(T, p, v, mode):
    m = view_mask(T, p)
    r = resid(T, p, v, mode); m &= np.isfinite(r)
    t = T[f'{p}_t'].to_numpy(); u = T.u.to_numpy(); w = T.w.to_numpy()
    H = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            k = m & (u >= EDGES5[i]) & (u < EDGES5[i + 1]) & (w >= EDGES5[j]) & (w < EDGES5[j + 1])
            if k.sum() < 60 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10:
                continue
            H[i, j] = huber(np.c_[np.ones(k.sum()), t[k]], r[k], 0.5)[1]
    return H


def xy_stats(Hx, Hy):
    g = np.isfinite(Hx) & np.isfinite(Hy)
    d = (Hx[g] - Hx[g].mean()) - (Hy[g] - Hy[g].mean())
    return float(np.corrcoef(Hx[g], Hy[g])[0, 1]), float(np.std(d)), float(np.std(Hx[g])), float(np.std(Hy[g]))


def fmt(d, k):
    return f"{d[k]:+.2f}±{d[k+'_err']:.2f}"


def main(names):
    for name in names:
        T = load(name); v = T.attrs['v']
        U = (T.u.to_numpy() - 200) / 100; W = (T.w.to_numpy() - 200) / 100
        inner = (np.abs(U) < 1.25) & (np.abs(W) < 1.25)
        print(f'\n===== {name} v={v:.1f}')
        for mode in ('t0', 'piv'):
            for p in 'xy':
                own, oth = ('kuu', 'kww') if p == 'x' else ('kww', 'kuu')
                f = fit2d(T, p, v, mode); fi = fit2d(T, p, v, mode, sel=inner)
                print(f'  {mode:3s} [{p}] full: own {fmt(f, own)} other {fmt(f, oth)} kuw {fmt(f, "kuw")} | inner<125: own {fmt(fi, own)} other {fmt(fi, oth)}')
            Hx = cellmap(T, 'x', v, mode); Hy = cellmap(T, 'y', v, mode)
            c, rms, sx, sy = xy_stats(Hx, Hy)
            print(f'  {mode:3s} cells: corr(hX,hY) {c:+.2f}  rms(hX-hY) {rms:.2f} mm   spread hX {sx:.2f} hY {sy:.2f} mm')
            # also the same-surface terms across views
        # kuu from X vs Y, kww from X vs Y for both modes
        for mode in ('t0', 'piv'):
            fx = fit2d(T, 'x', v, mode); fy = fit2d(T, 'y', v, mode)
            print(f'  {mode:3s} same-surface: kuu X {fmt(fx,"kuu")} Y {fmt(fy,"kuu")} | kww X {fmt(fx,"kww")} Y {fmt(fy,"kww")} | kuw X {fmt(fx,"kuw")} Y {fmt(fy,"kuw")}')


if __name__ == '__main__':
    main(sys.argv[1:] or DETS)
