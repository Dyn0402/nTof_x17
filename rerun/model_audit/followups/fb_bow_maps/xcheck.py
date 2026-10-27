#!/usr/bin/env python3
"""xcheck.py [names...] -- the chamber-bow cross-checks for one or more datasets (FB, 2026-10-27).

For every dataset (common.DATASETS) and both views, four routes to the same surface:

 P  position route  r = p0 - ref_mesh = a + bu U + bw W + tan * h(U, W)    (event-level Huber fit, h quadratic)
    -- also with p0 moved along the track to a PINNED t0 (median t0 of the event's ftst class), which removes any
       position-dependent t0 slide (a t0 error dt moves p0 by w*dt = v*tan*dt and would read as a height v*dt).
    -- bu, bw are the residual scale terms: an M3 plane-scale error would show here (eps = bu / 100 mm).
 A  angle route     w = A(U, W) + B tan (1 + su U + sw W),  A quadratic.  Bow prediction: A_x = v r_f dh/du,
    A_y = v r_f dh/dw, i.e. cu(X) = v r_f kuu / 100, cw(Y) = v r_f kww / 100, cw(X) = cu(Y) = v r_f kuw / 100.
 W  width route     the tan at which the seed cluster (n_seed: strips above threshold, amplitudes only, no times) is
    narrowest, in bins of the view's own coordinate: a drift tilt alpha makes the column vertical at tan = -alpha.
 T  column route    column duration (u90 - u50: t0-free; uend; u90) in 5 x 5 cells of contained tracks, and the
    local ladder slope b = dw/dtan (a pure v measurement: the slope does not depend on the gap) in the same cells
    -> gap(cell) proportional to T * b.  This is how the gap is separated from v(x, y): the duration carries gap/v,
    the ladder carries v alone.
Writes results/xcheck_<name>.json.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import (DATASETS, RES, load, view_mask, huber, boot, surf_terms, SURF, GAP)

EDGES5 = np.linspace(25, 375, 6)


def pos_route(T, p, pin=False, nboot=30):
    m = view_mask(T, p)
    t = T[f'{p}_t'].to_numpy()[m]; r = T[f'{p}_r'].to_numpy()[m]
    if pin:
        t0 = T[f'{p}_t0'].to_numpy()[m]; fc = T[f'{p}_ftst'].to_numpy()[m]; wv = T[f'{p}_wv'].to_numpy()[m]
        tp = np.zeros_like(t0)
        for c in np.unique(fc[np.isfinite(fc)]):
            k = fc == c
            tp[k] = np.nanmedian(t0[k])
        r = r + wv * 1e-3 * (tp - t0)
        g = np.isfinite(r)
        t, r = t[g], r[g]; m2 = np.where(m)[0][g]
    else:
        m2 = np.where(m)[0]
    U = (T.u.to_numpy()[m2] - 200) / 100; W = (T.w.to_numpy()[m2] - 200) / 100
    S = surf_terms(U, W)
    X = np.c_[np.ones_like(U), U, W, t[:, None] * S]
    beta = huber(X, r, 0.5)
    bs = boot(lambda i: huber(X[i], r[i], 0.5), len(r), nboot)
    names = ['a', 'bu', 'bw'] + SURF
    return dict(n=int(len(r)), **{k: float(v) for k, v in zip(names, beta)},
                **{k + '_err': float(e) for k, e in zip(names, bs.std(0))})


def angle_route(T, p, v, nboot=30):
    m = view_mask(T, p)
    t = T[f'{p}_t'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
    m &= np.isfinite(wv)
    m &= np.abs(wv - v * t) < 0.15 * v
    t, wv = t[m], wv[m]
    U = (T.u.to_numpy()[m] - 200) / 100; W = (T.w.to_numpy()[m] - 200) / 100
    X = np.c_[surf_terms(U, W), t, t * U, t * W]
    beta = huber(X, wv, 0.03 * v)
    bs = boot(lambda i: huber(X[i], wv[i], 0.03 * v), len(t), nboot)
    names = ['c0', 'cu', 'cw', 'auu', 'aww', 'auw', 'B', 'Bu', 'Bw']
    return dict(n=int(len(t)), **{k: float(x) for k, x in zip(names, beta)},
                **{k + '_err': float(e) for k, e in zip(names, bs.std(0))})


def cell_maps(T, p, v):
    """h (position route) and w0, b (angle route) in 5 x 5 cells."""
    m = view_mask(T, p)
    t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
    u = T.u.to_numpy(); w = T.w.to_numpy()
    H = np.full((5, 5), np.nan); He = H.copy(); A = H.copy(); Ae = H.copy(); Bm = H.copy(); Be = H.copy(); N = np.zeros((5, 5), int)
    rng = np.random.default_rng(7)
    for i in range(5):
        for j in range(5):
            k = m & (u >= EDGES5[i]) & (u < EDGES5[i + 1]) & (w >= EDGES5[j]) & (w < EDGES5[j + 1])
            N[i, j] = k.sum()
            if k.sum() < 60 or (t[k] > 0.05).sum() < 10 or (t[k] < -0.05).sum() < 10:
                continue
            X = np.c_[np.ones(k.sum()), t[k]]
            H[i, j] = huber(X, r[k], 0.5)[1]
            He[i, j] = np.std([huber(X[q], r[k][q], 0.5)[1] for q in (rng.integers(0, k.sum(), k.sum()) for _ in range(25))])
            kk = k & np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
            X2 = np.c_[np.ones(kk.sum()), t[kk]]
            bb = huber(X2, wv[kk], 0.03 * v); A[i, j], Bm[i, j] = bb
            bs = np.array([huber(X2[q], wv[kk][q], 0.03 * v) for q in (rng.integers(0, kk.sum(), kk.sum()) for _ in range(25))])
            Ae[i, j], Be[i, j] = bs.std(0)
    return dict(h=H.tolist(), h_err=He.tolist(), w0=A.tolist(), w0_err=Ae.tolist(), b=Bm.tolist(), b_err=Be.tolist(),
                n=N.tolist())


def width_route(T, p):
    """tan of the narrowest seed cluster in 6 bins of the view's own coordinate (hyperbolic V fit, grid in t*)."""
    if f'{p}_n_seed' not in T:
        return None
    m = view_mask(T, p, tmax=0.25)
    t = T[f'{p}_t'].to_numpy(); n = T[f'{p}_n_seed'].to_numpy().astype(float)
    m &= np.isfinite(n) & (n > 0)
    if f'{p}_q_sum' in T:     # bright tracks only: wider than threshold noise, same for all positions
        q = T[f'{p}_q_sum'].to_numpy(); m &= q > np.nanpercentile(q[m], 30)
    pos = (T.u if p == 'x' else T.w).to_numpy()
    edges = np.linspace(40, 360, 7)
    grid = np.arange(-0.08, 0.0801, 0.002)
    rows = []
    rng = np.random.default_rng(3)

    def vertex(tt, nn):
        best = (np.inf, np.nan)
        for ts in grid:
            X = np.c_[np.ones_like(tt), np.sqrt((tt - ts) ** 2 + 0.02 ** 2)]
            b, res, *_ = np.linalg.lstsq(X, nn, rcond=None)
            s = float(np.sum((X @ b - nn) ** 2))
            if s < best[0]:
                best = (s, ts)
        return best[1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = m & (pos >= lo) & (pos < hi)
        if k.sum() < 150:
            rows.append((0.5 * (lo + hi), np.nan, np.nan, int(k.sum()))); continue
        tt, nn = t[k], n[k]
        ts = vertex(tt, nn)
        bs = [vertex(tt[q], nn[q]) for q in (rng.integers(0, len(tt), len(tt)) for _ in range(20))]
        rows.append((0.5 * (lo + hi), float(ts), float(np.std(bs)), int(k.sum())))
    return rows


def column_maps(T, v):
    """Column duration and ladder slope in 5 x 5 cells of contained tracks; both views."""
    out = {}
    lo, hi = 15.0, 385.0
    cont = ((T.u_mesh > lo) & (T.u_mesh < hi) & (T.w_mesh > lo) & (T.w_mesh < hi) &
            (T.u_cath > lo) & (T.u_cath < hi) & (T.w_cath > lo) & (T.w_cath < hi)).to_numpy()
    u = T.u.to_numpy(); w = T.w.to_numpy()
    rng = np.random.default_rng(11)
    for p in 'xy':
        m = view_mask(T, p) & cont
        if f'{p}_q_u90' not in T:
            continue
        d1 = (T[f'{p}_q_u90'] - T[f'{p}_q_u50']).to_numpy()
        d2 = T[f'{p}_q_uend'].to_numpy(); d3 = T[f'{p}_q_u90'].to_numpy()
        res = {}
        for nm, d in (('u90m50', d1), ('uend', d2), ('u90', d3)):
            M = np.full((5, 5), np.nan); E = M.copy()
            mm = m & np.isfinite(d) & (np.abs(T[f'{p}_t'].to_numpy()) < 0.35)
            for i in range(5):
                for j in range(5):
                    k = mm & (u >= EDGES5[i]) & (u < EDGES5[i + 1]) & (w >= EDGES5[j]) & (w < EDGES5[j + 1])
                    if k.sum() < 40:
                        continue
                    x = d[k]
                    M[i, j] = np.mean(np.clip(x, *np.percentile(x, [5, 95])))
                    E[i, j] = np.std([np.mean(np.clip(x[q], *np.percentile(x, [5, 95])))
                                      for q in (rng.integers(0, len(x), len(x)) for _ in range(30))])
            res[nm] = M.tolist(); res[nm + '_err'] = E.tolist()
        out[p] = res
    return out


def run(name):
    T = load(name)
    v = T.attrs['v']
    res = dict(name=name, det=T.attrs['det'], slot=T.attrs['slot'], note=T.attrs['note'], v=v, n_events=int(len(T)))
    for p in 'xy':
        res[f'pos_{p}'] = pos_route(T, p)
        try:
            res[f'pospin_{p}'] = pos_route(T, p, pin=True)
        except Exception as e:                                   # noqa: BLE001
            res[f'pospin_{p}'] = dict(error=str(e))
        res[f'ang_{p}'] = angle_route(T, p, v)
        res[f'cells_{p}'] = cell_maps(T, p, v)
        res[f'width_{p}'] = width_route(T, p)
    res['column'] = column_maps(T, v)
    # bow-vs-angle closure: r_f from each of the four predicted relations
    px, py, ax, ay = res['pos_x'], res['pos_y'], res['ang_x'], res['ang_y']
    rel = {}
    for lab, num, den in (('X: dw0/du vs kuu(X)', ax['cu'], px['kuu']), ('Y: dw0/dw vs kww(Y)', ay['cw'], py['kww']),
                          ('X: dw0/dw vs kuw(X)', ax['cw'], px['kuw']), ('Y: dw0/du vs kuw(Y)', ay['cu'], py['kuw'])):
        rel[lab] = float(num * 100 / (v * den)) if abs(den) > 1e-9 else np.nan
    res['r_f'] = rel
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(f'{RES}/xcheck_{name}.json', 'w'), indent=1)
    f = lambda d, k: f'{d[k]:+.2f}±{d[k + "_err"]:.2f}'
    print(f'{name:14s} [{res["slot"]}] X: kuu {f(px, "kuu")} kww {f(px, "kww")} kuw {f(px, "kuw")} bu {f(px, "bu")} | '
          f'Y: kuu {f(py, "kuu")} kww {f(py, "kww")} kuw {f(py, "kuw")} bw {f(py, "bw")} | '
          f'A: cu(X) {f(ax, "cu")} cw(Y) {f(ay, "cw")} | r_f ' + ' '.join(f'{x:.2f}' for x in rel.values()), flush=True)
    return res


if __name__ == '__main__':
    for n in (sys.argv[1:] or list(DATASETS)):
        run(n)
