#!/usr/bin/env python3
"""Skeptical re-checks of the FB bow conclusion on the golden rows (both views).
 A  position route with extra no-tan quadratic terms (U^2, W^2, UW): does the curvature survive?
 B  inner region only (|U|,|W| < 1.0 and < 1.25): is the dish smooth or an edge hook?
 C  tan subsets (|tan| < 0.2 vs 0.2-0.45): geometry is tan-independent, a reco bias need not be
 D  cell maps: h (X view) vs h (Y view) agreement; h vs ladder slope b and column duration T correlations
 E  fitted t0 (minus ftst-class median) vs position: any position-dependent delay?
 F  dish vertex per view from the quadratic (U_c = -gu/kuu etc.)
 G  1-D profiles along the own axis, 9 bins: parabola fit vs inner-5-bin parabola
"""
import sys, os, json
import numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/model_audit/followups/fb_bow_maps')
from fbcommon import load, view_mask, huber, boot, surf_terms, SURF, DETS

EDGES5 = np.linspace(25, 375, 6)


def posfit(T, p, sel=None, extra=False, nboot=20):
    m = view_mask(T, p)
    if sel is not None:
        m &= sel
    t = T[f'{p}_t'].to_numpy()[m]; r = T[f'{p}_r'].to_numpy()[m]
    U = (T.u.to_numpy()[m] - 200) / 100; W = (T.w.to_numpy()[m] - 200) / 100
    S = surf_terms(U, W)
    cols = [np.ones_like(U), U, W]
    names = ['a', 'bu', 'bw']
    if extra:
        cols += [U * U, W * W, U * W]; names += ['qu', 'qw', 'quw']
    X = np.c_[tuple(cols) + (t[:, None] * S,)]
    names += SURF
    beta = huber(X, r, 0.5)
    bs = boot(lambda i: huber(X[i], r[i], 0.5), len(r), nboot)
    out = {k: float(v) for k, v in zip(names, beta)}
    out.update({k + '_err': float(e) for k, e in zip(names, bs.std(0))})
    out['n'] = int(len(r))
    return out


def fmt(d, k):
    return f"{d[k]:+.2f}±{d[k+'_err']:.2f}"


def cells(T, p, v):
    m = view_mask(T, p)
    t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
    u = T.u.to_numpy(); w = T.w.to_numpy()
    t0 = T[f'{p}_t0'].to_numpy() if f'{p}_t0' in T else np.full(len(T), np.nan)
    fc = T[f'{p}_ftst'].to_numpy() if f'{p}_ftst' in T else np.zeros(len(T))
    tp = np.full_like(t0, np.nan)
    for c in np.unique(fc[np.isfinite(fc)]):
        k = fc == c; tp[k] = np.nanmedian(t0[k & m])
    dt0 = t0 - tp
    dur = (T[f'{p}_q_u90'] - T[f'{p}_q_u50']).to_numpy() if f'{p}_q_u90' in T else np.full(len(T), np.nan)
    H = np.full((5, 5), np.nan); B = H.copy(); D = H.copy(); T0 = H.copy(); N = np.zeros((5, 5), int)
    for i in range(5):
        for j in range(5):
            k = m & (u >= EDGES5[i]) & (u < EDGES5[i + 1]) & (w >= EDGES5[j]) & (w < EDGES5[j + 1])
            N[i, j] = k.sum()
            if k.sum() < 60 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10:
                continue
            H[i, j] = huber(np.c_[np.ones(k.sum()), t[k]], r[k], 0.5)[1]
            kk = k & np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
            B[i, j] = huber(np.c_[np.ones(kk.sum()), t[kk]], wv[kk], 0.03 * v)[1]
            kd = k & np.isfinite(dur) & (np.abs(t) < 0.35)
            if kd.sum() > 30:
                x = dur[kd]; D[i, j] = np.mean(np.clip(x, *np.percentile(x, [5, 95])))
            kt = k & np.isfinite(dt0)
            if kt.sum() > 30:
                T0[i, j] = np.median(dt0[kt])
    return H, B, D, T0, N


def corr(a, b):
    g = np.isfinite(a) & np.isfinite(b)
    if g.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[g], b[g])[0, 1])


def main(names):
    for name in names:
        T = load(name); v = T.attrs['v']
        print(f'\n=================== {name}  v={v:.1f} kw={T.attrs["kw"]} w0={T.attrs["w0"]}  n={len(T)}')
        U = (T.u.to_numpy() - 200) / 100; W = (T.w.to_numpy() - 200) / 100
        inner10 = (np.abs(U) < 1.0) & (np.abs(W) < 1.0)
        inner125 = (np.abs(U) < 1.25) & (np.abs(W) < 1.25)
        for p in 'xy':
            own, oth = ('kuu', 'kww') if p == 'x' else ('kww', 'kuu')
            base = posfit(T, p)
            ext = posfit(T, p, extra=True)
            i10 = posfit(T, p, sel=inner10)
            i125 = posfit(T, p, sel=inner125)
            tt = np.abs(T[f'{p}_t'].to_numpy())
            lo = posfit(T, p, sel=tt < 0.2)
            hi = posfit(T, p, sel=(tt >= 0.2))
            print(f' [{p}] A base      own {fmt(base, own)} other {fmt(base, oth)} kuw {fmt(base, "kuw")}   n={base["n"]}')
            print(f' [{p}]   +U2W2UW   own {fmt(ext, own)} other {fmt(ext, oth)} kuw {fmt(ext, "kuw")}   (qu {fmt(ext, "qu")} qw {fmt(ext, "qw")})')
            print(f' [{p}] B inner<100 own {fmt(i10, own)} other {fmt(i10, oth)}   n={i10["n"]}')
            print(f' [{p}]   inner<125 own {fmt(i125, own)} other {fmt(i125, oth)}   n={i125["n"]}')
            print(f' [{p}] C |tan|<0.2 own {fmt(lo, own)} other {fmt(lo, oth)}   n={lo["n"]}')
            print(f' [{p}]   |tan|>0.2 own {fmt(hi, own)} other {fmt(hi, oth)}   n={hi["n"]}')
            uc = -base['gu'] / base['kuu'] * 100 + 200 if abs(base['kuu']) > 1e-6 else np.nan
            wc = -base['gw'] / base['kww'] * 100 + 200 if abs(base['kww']) > 1e-6 else np.nan
            print(f' [{p}] F vertex   u_c={uc:.0f} mm  w_c={wc:.0f} mm   (gu {base["gu"]:+.2f} gw {base["gw"]:+.2f})')
        Hx, Bx, Dx, T0x, Nx = cells(T, 'x', v)
        Hy, By, Dy, T0y, Ny = cells(T, 'y', v)
        # same-surface test: Hx vs Hy cell by cell (remove each view's mean)
        d = (Hx - np.nanmean(Hx)) - (Hy - np.nanmean(Hy))
        print(f' D  h(X) vs h(Y) cells: corr {corr(Hx, Hy):+.2f}; rms diff {np.nanstd(d):.2f} mm; '
              f'range h(X) {np.nanmin(Hx):+.2f}..{np.nanmax(Hx):+.2f}, h(Y) {np.nanmin(Hy):+.2f}..{np.nanmax(Hy):+.2f}')
        print(f' D  corr(h,b): X {corr(Hx, Bx):+.2f} Y {corr(Hy, By):+.2f}; corr(h,T): X {corr(Hx, Dx):+.2f} Y {corr(Hy, Dy):+.2f}; '
              f'b range X {np.nanmin(Bx):.1f}..{np.nanmax(Bx):.1f} Y {np.nanmin(By):.1f}..{np.nanmax(By):.1f} um/ns; '
              f'T(u90-u50) range X {np.nanmin(Dx):.0f}..{np.nanmax(Dx):.0f} Y {np.nanmin(Dy):.0f}..{np.nanmax(Dy):.0f} ns')
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        print(' h(X) cells [rows u, cols w]:\n', Hx)
        print(' h(Y) cells:\n', Hy)
        print(' b(X) cells:\n', Bx)
        print(' T(X) cells (u90-u50, ns):\n', Dx)
        print(f' E  t0 - class median, cells (ns): X range {np.nanmin(T0x):+.1f}..{np.nanmax(T0x):+.1f}, Y {np.nanmin(T0y):+.1f}..{np.nanmax(T0y):+.1f}')
        print(' t0(X) cells:\n', T0x)
        print(' t0(Y) cells:\n', T0y)


if __name__ == '__main__':
    main(sys.argv[1:] or DETS)
