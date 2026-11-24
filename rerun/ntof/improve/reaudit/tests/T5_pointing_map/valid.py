#!/usr/bin/env python3
"""T5 task 5: apply pointing-derived corrections to the clean cosmic lines (A-C for A, C; B-D for D) and compare with
the cosmic map trained on the other half (run parity for A, C; event parity for D, as bd_map.py).

Methods, all on the UNMAPPED global slope of the test lines, sigma = 1.4826 MAD of (corrected - j):
  none        no correction
  cosmic      the cosmic map (angle_map.fit_map / bd_map 100 mm) trained on the other half, bilinear
  cosmic_glob only the global a0, s0 of that training half
  P0          slope - c(cell)                          (pointing alone; c = pointing residual centre, piecewise const)
  P1          (slope - (c - s_cos e_cell)) / (1 + s_cos)  s_cos = the training-half cosmic map's s at the line (bilinear)
  P1g         same with the training-half global s0 only (pointing local information + one global cosmic scale)
  P0g         slope - c(cell) then the training-half global (a0, s0) scale: (slope - c - a0 + a0... ) see code
  cos+P       cosmic map, then minus (c - pred_cos)(cell): what pointing adds beyond the cosmic map
Coverage: the pointing map covers only some cells (reach / near-normal / dead bands), so every method is evaluated on
the lines in valid pointing cells ("covered") and, for the whole chamber, with no pointing correction outside them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pm

RNG = np.random.default_rng(17)


def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def cellval(Z, edges, x, y):
    ix, iy, ok = pm.cell_index(x, y, edges)
    out = np.full(len(x), np.nan)
    out[ok] = Z[ix[ok], iy[ok]]
    return out


def lines(arm):
    from rerun.ntof.improve import angle_map as AM
    if arm in 'AC':
        G = AM.clean_lines(pd.read_parquet(pm.LINES), arm.lower(), 'x')
        half = (G.run.str.replace('run_', '').astype(int) % 2).to_numpy()
        return G, half, AM.EDGES, AM.N_MIN
    from rerun.ntof.improve import bd_map as BD
    G = BD.clean(pd.read_parquet(pm.BDLINES), 'z')
    return G, (G.ev.astype(int) % 2).to_numpy(), BD.EDGES, 60


def fit_cos(G, edges, nmin):
    from rerun.ntof.improve import angle_map as AM
    old = AM.EDGES, AM.CENTRES, AM.N_MIN
    AM.EDGES, AM.CENTRES, AM.N_MIN = edges, 0.5 * (edges[1:] + edges[:-1]), nmin
    try:
        return AM.fit_map(G)
    finally:
        AM.EDGES, AM.CENTRES, AM.N_MIN = old


def corrections(arm, cellsize):
    o = dict(np.load(pm.HERE / f'hmap_{arm}_{cellsize}.npz'))
    edges = o['edges']
    v = o['valid']
    C = np.where(v, o['c'], np.nan)
    Dp = np.where(v, o['c'] - o['pred'], np.nan)           # pointing minus the FULL cosmic map's prediction
    E = o['e']
    G, half, cedges, nmin = lines(arm)
    x, y, j = G.x.to_numpy(float), G.y.to_numpy(float), G.j.to_numpy(float)
    slope = G.r.to_numpy(float) + j
    c, e_c, dp = cellval(C, edges, x, y), cellval(E, edges, x, y), cellval(Dp, edges, x, y)
    cov = np.isfinite(c)
    res = {k: np.full(len(G), np.nan) for k in ('none', 'cosmic', 'cosmic_glob', 'P0', 'P1', 'P1g', 'cos+P', 'cos+Pd')}
    for tr in (0, 1):
        te = half != tr
        mp = fit_cos(G[half == tr], cedges, nmin)
        a, s = pm.interp(mp, x[te], y[te])
        sl = slope[te]
        res['none'][te] = sl
        res['cosmic'][te] = (sl - a) / (1 + s)
        res['cosmic_glob'][te] = (sl - mp['a0']) / (1 + mp['s0'])
        cc = c[te]
        res['P0'][te] = np.where(cov[te], sl - cc, sl)
        ap = cc - s * e_c[te]
        res['P1'][te] = np.where(cov[te], (sl - ap) / (1 + s), sl)
        apg = cc - mp['s0'] * e_c[te]
        res['P1g'][te] = np.where(cov[te], (sl - apg) / (1 + mp['s0']), sl)
        # cosmic + pointing excess: the pointing-minus-cosmic difference of the cell, with the held-out cosmic map
        dph = cc - (a + s * e_c[te])          # pointing minus the TRAINING-half cosmic prediction at t_p
        res['cos+P'][te] = np.where(cov[te], (sl - a - dph) / (1 + s), (sl - a) / (1 + s))
        # only the LOCAL part of the pointing-minus-cosmic difference: remove its arm-wide alpha + gamma e first
        okc = cov[te]
        X = np.vstack([np.ones(okc.sum()), e_c[te][okc]]).T
        dd = np.zeros(te.sum())
        if okc.sum() > 10:
            # fit alpha, gamma on the cells (one value per cell), not per line
            cc_cells = np.where(v, o['c'], np.nan)
            ii, jj = np.where(v)
            pa, ps = pm.interp(mp, 0.5 * (edges[ii] + edges[ii + 1]), 0.5 * (edges[jj] + edges[jj + 1]))
            dcell = cc_cells[ii, jj] - (pa + ps * E[ii, jj])
            Xc = np.vstack([np.ones(len(ii)), E[ii, jj]]).T
            ag = np.linalg.lstsq(Xc, dcell, rcond=None)[0]
            dd[okc] = dph[okc] - X @ ag
        res['cos+Pd'][te] = (sl - a - dd) / (1 + s)
    rows = []
    for reg, m in (('covered', cov), ('all', np.ones(len(G), bool)), ('not_covered', ~cov)):
        if m.sum() < 100:
            continue
        r0 = {k: v_[m] - j[m] for k, v_ in res.items()}
        boot = {k: [] for k in r0}
        n = m.sum()
        for _ in range(200):
            ii = RNG.integers(0, n, n)
            for k in r0:
                boot[k].append(mad(r0[k][ii]))
        b = {k: np.array(vv) for k, vv in boot.items()}
        row = dict(arm=arm, cell=cellsize, region=reg, n_lines=int(n))
        for k in r0:
            row[f'sig_{k}'] = mad(r0[k])
            row[f'err_{k}'] = float(np.std(b[k]))
            row[f'dsig_{k}_vs_none'] = float(np.mean(b[k] - b['none']))
            row[f'err_d_{k}_vs_none'] = float(np.std(b[k] - b['none']))
            row[f'dsig_{k}_vs_cosmic'] = float(np.mean(b[k] - b['cosmic']))
            row[f'err_d_{k}_vs_cosmic'] = float(np.std(b[k] - b['cosmic']))
        rows.append(row)
    # angle dependence on covered lines: |j - e_cell| small vs large (P0 is exact only at j = t_p)
    dj = np.abs(j - e_c)
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 1.0)):
        m = cov & (dj >= lo) & (dj < hi)
        if m.sum() < 100:
            continue
        rows.append(dict(arm=arm, cell=cellsize, region=f'cov |j-tp| {lo}-{hi}', n_lines=int(m.sum()),
                         **{f'sig_{k}': mad(vv[m] - j[m]) for k, vv in res.items()}))
    return rows


def main():
    rows = []
    for arm in 'ACD':
        for cs in ('25', '50'):
            rows += corrections(arm, cs)
    T = pd.DataFrame(rows)
    T.to_csv(pm.HERE / 'validation_cosmic_lines.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 100)
    cols = ['arm', 'cell', 'region', 'n_lines'] + [f'sig_{k}' for k in ('none', 'cosmic', 'cosmic_glob', 'P0', 'P1',
                                                                          'P1g', 'cos+P', 'cos+Pd')]
    print(T[cols].round(4).to_string())
    print(T[['arm', 'cell', 'region'] + [c for c in T.columns if c.startswith('dsig_') or c.startswith('err_d_')]]
          .dropna().round(4).T.to_string())


if __name__ == '__main__':
    main()
