#!/usr/bin/env python3
"""cellobs.py <work> -- every per-cell observable of chamber A's two surfaces, on one grid, both runs.  -> <work>/cellobs.json

The surfaces: the mesh at height h(U, W) and the cathode at height c(U, W) (mm, M3 z up, U, W in 100 mm from (200, 200)).
Observables per cell and run (track at mid-gap, >= 30 mm inside the active area):
  hp_x, hp_y   position route: slope of (p0 - M3 at the nominal plane) against the M3 slope = the local mesh height h
               (the track point at the nominal plane is p0 - tan h, wft.reco.apply_bow); X and Y independently
  w0_x, w0_y   ladder intercept (um/ns) of w*1e3 = w0 + v tan_ref: the drift tilt v (dh/ds + dc/ds) / 2 (+ a constant)
  v_x, v_y     ladder slope (um/ns)
  T            column duration, X and Y strip sums averaged, amplitude walk removed (ns)
  G, GY        geometric gap, X / Y strips, +/- tan averaged (from gapmap.json)
  dX, dY       drift tilt from the +/- tan asymmetry of the geometric gap: (G+ - G-) / (2 G <1/|tan|>), timing-free
Errors: line-fit covariance (hp, w0, v), bootstrap (T, G), propagated (d).
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gapmap as GM  # noqa: E402

GRID = '4x4'


def line(t, y, clip=3.5):
    """robust y = a + b t: (a, b, err_a, err_b, n)."""
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 40:
        return (np.nan,) * 4 + (0,)
    keep = np.ones(len(t), bool)
    for _ in range(3):
        A = np.vstack([np.ones(keep.sum()), t[keep]]).T
        c = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = y - (c[0] + c[1] * t)
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < clip * rs
    A = np.vstack([np.ones(keep.sum()), t[keep]]).T
    c = np.linalg.lstsq(A, y[keep], rcond=None)[0]
    s2 = np.sum((y[keep] - A @ c) ** 2) / max(keep.sum() - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(c[0]), float(c[1]), float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1])), int(keep.sum())


def inv_tan(incl, ex, ey):
    """per cell and view: <1/|tan|> of the inclined tracks the geometric gap used (|tan| 0.2-0.8 in that view)."""
    out = {}
    ev = list(incl.values())
    xm = np.array([e['ref_mesh_x'] + GM.MID * e['tan_x'] for e in ev])
    ym = np.array([e['ref_mesh_y'] + GM.MID * e['tan_y'] for e in ev])
    i, j = GM.cell_of(xm, ym, ex, ey)
    for p in 'xy':
        t = np.abs(np.array([e[f'tan_{p}'] for e in ev]))
        s = (t >= 0.2) & (t < 0.8)
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                m = s & (i == a) & (j == b)
                out[(p, a, b)] = float(np.mean(1 / t[m])) if m.sum() > 10 else np.nan
    return out


def main():
    work = sys.argv[1]
    gm = json.load(open(f'{work}/gapmap.json'))
    ex, ey = GM.GRIDS[GRID]
    out = {}
    for run, R in GM.RUNS.items():
        tab = GM.load_table(work, R['rel'])
        m, _ = GM.colsum_T(work, run, tab)
        incl = pickle.load(open(f'{work}/build_{R["key"]}/out/{R["key"]}/incl.pkl', 'rb'))
        it_ = inv_tan(incl, ex, ey)
        cells = gm['runs'][run]['cells']
        it, jt = GM.cell_of(tab.xm.values, tab.ym.values, ex, ey)
        im, jm = GM.cell_of(m.xm.values, m.ym.values, ex, ey)
        rows = []
        for a in range(len(ex) - 1):
            for b in range(len(ey) - 1):
                d = tab[(it == a) & (jt == b)]
                r = dict(i=a, j=b, U=(0.5 * (ex[a] + ex[a + 1]) - 200) / 100, W=(0.5 * (ey[b] + ey[b + 1]) - 200) / 100)
                for p in 'xy':
                    tr = d[f'tan_ref_{p}'].values
                    # position route: residual at the nominal plane against the reference slope
                    _, h, _, he, _ = line(tr, d[f'{p}_p0'].values - d[f'ref_mesh_{p}'].values)
                    r[f'hp_{p}'], r[f'hp_{p}_err'] = h, he
                    s = np.abs(tr) <= GM.TAN_LADDER
                    w0, v, w0e, ve, n = line(tr[s], d[f'{p}_w'].values[s] * 1e3)
                    r[f'w0_{p}'], r[f'w0_{p}_err'], r[f'v_{p}'], r[f'v_{p}_err'], r[f'n_{p}'] = w0, w0e, v, ve, n
                mc = m[(im == a) & (jm == b)]
                r['T'] = GM.tmean(mc['T_amp']) if len(mc) > 20 else np.nan
                r['T_err'] = GM.boot(GM.tmean, mc['T_amp'].values) if len(mc) > 20 else np.nan
                c = cells[f'{GRID}:{a},{b}']
                for k, src in (('G', 'G'), ('GY', 'GY')):
                    r[k], r[f'{k}_err'] = c[f'{src}_sym'], c[f'{src}_sym_err']
                    gp, gn = c[f'{src}_plus'], c[f'{src}_minus']
                    p = 'x' if k == 'G' else 'y'
                    q = it_[(p, a, b)]
                    g = c[f'{src}_sym']
                    dl = (gp - gn) / (2 * g * q) if np.isfinite(gp) and np.isfinite(gn) and np.isfinite(q) else np.nan
                    r[f'd{p.upper()}'] = float(dl)
                    # error: the sym error scales the half difference by ~2 (the halves are independent)
                    r[f'd{p.upper()}_err'] = float(2 * c[f'{src}_sym_err'] / (2 * g * q)) if np.isfinite(dl) else np.nan
                rows.append(r)
        out[run] = rows
        print(run, 'cells', len(rows), flush=True)
    json.dump(out, open(f'{work}/cellobs.json', 'w'), indent=1, default=float)
    print('wrote', f'{work}/cellobs.json')


if __name__ == '__main__':
    main()
