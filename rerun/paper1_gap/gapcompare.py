#!/usr/bin/env python3
"""gapcompare.py <work> -- compare the gap maps of gapmap.py: run against run, method against method, the field
signature of the timing maps, the board bow; quadratic surfaces; figures.  -> results/compare.json, figures/*.

Deviations dg (mm) are taken against each method's own chamber-wide value, so a method's absolute offset (the colsum
box T differs from the stack T by ~3 %, the Y peak estimator reads high) does not enter.
  G   X view, window integral, +/- tan averaged (the S4 gap estimator)       dg = G_cell - G_all
  GY  Y view, peak sample, +/- tan averaged (relative only)                  dg = GY_cell - GY_all
  VT  angle-scale velocity x column duration                                 dg = 28.19 (v_rel T_rel - 1)
  T   column duration alone, read as a gap change (field included)           dg = 28.19 ln T_rel / (1 + beta)
  V   angle-scale velocity alone, read as a gap change                       dg = -28.19 ln v_rel / beta
  bow flat cathode on the measured dish of the board: the CURVATURE terms of the pass-1 bow block (cross-view 'h':
      h = gu U + gw W + kuu U^2/2 + kww W^2/2 + kuw U W, mesh height, z up, cathode above), dg = -(h_quad - <h_quad>).
      The linear terms are the rigid tilt of the whole chamber in the reference frame, which the cathode shares.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))

#: pass-1 bow block of both det3 golden rows (identical), cross-view mean surface 'h' (calib_bundle/bundle.json)
BOW = {'kuu': 0.8818883015769696, 'kww': 1.4923582013350711, 'kuw': -0.06296886468393921}
METHODS = ('G', 'GY', 'VT', 'T', 'V', 'bow')
LABEL = {'G': 'geometric, X strips', 'GY': 'geometric, Y strips', 'VT': r'$v_\mathrm{ref}\times T$',
         'T': 'column duration only', 'V': 'angle-scale velocity only', 'bow': 'bow, flat cathode'}


def dev(c, all_, m):
    if m == 'G':
        return c['G_sym'] - all_['G_sym'], c['G_sym_err']
    if m == 'GY':
        return c['GY_sym'] - all_['GY_sym'], c['GY_sym_err']
    if m == 'VT':
        return c['dg_VT'], c['dg_VT_err']
    if m == 'T':
        return c['dg_T'], c['dg_T_err']
    if m == 'V':
        return c['dg_V'], c['dg_V_err']
    if m == 'bow':
        U, W = (c['xc'] - 200) / 100, (c['yc'] - 200) / 100
        return -(BOW['kuu'] * U * U / 2 + BOW['kww'] * W * W / 2 + BOW['kuw'] * U * W), 0.0


def table(R, run, grid, m):
    C = R['runs'][run]['cells']
    rows = []
    for k, c in C.items():
        if k.startswith(grid + ':'):
            d, e = dev(c, C['all'], m)
            ok = np.isfinite(d) and np.isfinite(e) and (e > 0 or m == 'bow')
            rows.append((c['xc'], c['yc'], d if ok else np.nan, e if ok else np.nan))
    return np.array(rows)


def combine(a, b):
    """inverse-variance mean of the two runs per cell, and the run-difference chi2."""
    out = a.copy()
    for i in range(len(a)):
        d = [(a[i, 2], a[i, 3]), (b[i, 2], b[i, 3])]
        d = [(x, e) for x, e in d if np.isfinite(x) and np.isfinite(e) and e > 0]
        if not d:
            out[i, 2:] = np.nan if not (np.isfinite(a[i, 2]) and a[i, 3] == 0) else (0.5 * (a[i, 2] + b[i, 2]), 0.0)
            continue
        w = np.array([1 / e ** 2 for _, e in d])
        out[i, 2] = float(np.sum([x for x, _ in d] * w) / w.sum()); out[i, 3] = float(1 / np.sqrt(w.sum()))
    m = np.isfinite(a[:, 2]) & np.isfinite(b[:, 2]) & (a[:, 3] > 0)
    chi2 = float(np.sum((a[m, 2] - b[m, 2]) ** 2 / (a[m, 3] ** 2 + b[m, 3] ** 2))) if m.any() else np.nan
    return out, chi2, int(m.sum())


def const_chi2(t):
    m = np.isfinite(t[:, 2]) & (t[:, 3] > 0)
    w = 1 / t[m, 3] ** 2
    mu = np.sum(t[m, 2] * w) / w.sum()
    return float(np.sum((t[m, 2] - mu) ** 2 * w)), int(m.sum() - 1), float(np.sqrt(np.average((t[m, 2] - mu) ** 2, weights=w)))


def surface(t):
    """weighted quadratic dg = c0 + a U + b W + uu U^2 + ww W^2 + uw U W (U, W in 100 mm from (200, 200))."""
    m = np.isfinite(t[:, 2]) & (t[:, 3] > 0)
    U, W = (t[m, 0] - 200) / 100, (t[m, 1] - 200) / 100
    A = np.vstack([np.ones_like(U), U, W, U * U, W * W, U * W]).T
    w = 1 / t[m, 3]
    c, *_ = np.linalg.lstsq(A * w[:, None], t[m, 2] * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    chi2 = float(np.sum(((A @ c - t[m, 2]) * w) ** 2)); ndf = int(m.sum() - 6)
    s = np.sqrt(max(chi2 / ndf, 1.0)) if ndf > 0 else 1.0
    gu, gw = np.meshgrid(np.linspace(-1.4, 1.4, 29), np.linspace(-1.2, 1.2, 25))
    S = c[0] + c[1] * gu + c[2] * gw + c[3] * gu ** 2 + c[4] * gw ** 2 + c[5] * gu * gw
    return dict(c=c.tolist(), err=(np.sqrt(np.diag(cov)) * s).tolist(), chi2=chi2, ndf=ndf,
                span=float(S.max() - S.min()), names=['c0', 'U', 'W', 'UU', 'WW', 'UW'])


def wcorr(a, b):
    m = np.isfinite(a[:, 2]) & np.isfinite(b[:, 2])
    if m.sum() < 4:
        return np.nan, 0
    return float(np.corrcoef(a[m, 2], b[m, 2])[0, 1]), int(m.sum())


def slope(a, b):
    """weighted fit a = s b through the origin (errors of both, effective variance)."""
    m = np.isfinite(a[:, 2]) & np.isfinite(b[:, 2]) & (a[:, 3] > 0)
    x, y, ex, ey = b[m, 2], a[m, 2], b[m, 3], a[m, 3]
    s = 1.0
    for _ in range(20):
        w = 1 / (ey ** 2 + s ** 2 * ex ** 2)
        s = float(np.sum(w * x * y) / np.sum(w * x * x))
    err = float(1 / np.sqrt(np.sum(w * x * x)))
    chi2 = float(np.sum(w * (y - s * x) ** 2))
    return s, err, chi2, int(m.sum() - 1)


def main():
    work = sys.argv[1]
    R = json.load(open(f'{work}/gapmap.json'))
    out = dict(chamber={}, grids={})
    for run in ('run1', 'run2'):
        a = R['runs'][run]['cells']['all']
        out['chamber'][run] = dict(G=a['G_sym'], G_err=a['G_sym_err'], G_pooled=a['G_pooled'], G_plus=a['G_plus'],
                                   G_minus=a['G_minus'], GY=a['GY_sym'], GY_err=a['GY_sym_err'], n_G=a['G_n'],
                                   vX=a['vX'], vY=a['vY'], T=a['T'], n_T=a['n_T'], T_info=R['runs'][run]['T_info'])
    tabs = {}
    for grid in ('4x4', '3x3', 'x5', 'y5'):
        G = out['grids'][grid] = {}
        for m in METHODS:
            t1, t2 = table(R, 'run1', grid, m), table(R, 'run2', grid, m)
            if m == 'bow':
                ref = table(R, 'run1', '4x4', 'bow')[:, 2].mean()
                t1[:, 2] -= ref; t2[:, 2] -= ref
            tc, chi2, n = combine(t1, t2)
            tabs[(grid, m)] = (t1, t2, tc)
            r = dict(run1=t1.tolist(), run2=t2.tolist(), comb=tc.tolist(), run_diff_chi2=chi2, run_diff_n=n)
            if m != 'bow':
                r['flat_chi2'], r['flat_ndf'], r['rms'] = const_chi2(tc)
                r['flat_chi2_run1'], r['flat_ndf_run1'], _ = const_chi2(t1)
                r['flat_chi2_run2'], r['flat_ndf_run2'], _ = const_chi2(t2)
                r['rms_run1'], r['rms_run2'] = const_chi2(t1)[2], const_chi2(t2)[2]
            if grid == '4x4' and m != 'bow':
                r['surface'] = surface(tc)
            if grid == '4x4' and m == 'bow':
                r['surface'] = dict(names=['c0', 'U', 'W', 'UU', 'WW', 'UW'],
                                    c=[0, 0, 0, -BOW['kuu'] / 2, -BOW['kww'] / 2, -BOW['kuw']])
            G[m] = r
        # method against method on the combined map
        G['corr'] = {f'{a}~{b}': wcorr(tabs[(grid, a)][2], tabs[(grid, b)][2])
                     for i, a in enumerate(METHODS) for b in METHODS[i + 1:]}
        G['slope_vs_G'] = {m: slope(tabs[(grid, m)][2], tabs[(grid, 'G')][2]) for m in ('GY', 'VT', 'T', 'V')}
        # field signature: dlnv against dlnT (pure gap: -beta/(1+beta); pure gas: -1)
        C1 = [R['runs'][r]['cells'] for r in ('run1', 'run2')]
        lt, lv, et, ev = [], [], [], []
        for C in C1:
            for k, c in C.items():
                if k.startswith(grid + ':') and np.isfinite(c['T_rel']) and np.isfinite(c['v_rel']):
                    lt.append(np.log(c['T_rel'])); lv.append(np.log(c['v_rel']))
                    et.append(c['T_err']); ev.append(c['v_err'])
        lt, lv, et, ev = map(np.array, (lt, lv, et, ev))
        k = -0.5
        for _ in range(30):
            w = 1 / (ev ** 2 + k ** 2 * et ** 2)
            k = float(np.sum(w * lt * lv) / np.sum(w * lt * lt))
        G['k'] = dict(k=k, err=float(1 / np.sqrt(np.sum(w * lt * lt))), chi2=float(np.sum(w * (lv - k * lt) ** 2)),
                      ndf=int(len(lt) - 1), pure_gap=-R['beta'] / (1 + R['beta']), pure_gas=-1.0)
    os.makedirs(f'{HERE}/results', exist_ok=True)
    json.dump(out, open(f'{HERE}/results/compare.json', 'w'), indent=1, default=float)
    # print a summary
    for run, c in out['chamber'].items():
        print(run, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != 'T_info'})
    for grid, G in out['grids'].items():
        print('=== grid', grid, 'k', {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in G['k'].items()})
        for m in METHODS:
            r = G[m]
            print(f"  {m:4s} rundiff chi2 {r['run_diff_chi2']:6.1f}/{r['run_diff_n']}"
                  + (f"  flat chi2 {r['flat_chi2']:7.1f}/{r['flat_ndf']} (run1 {r['flat_chi2_run1']:.1f}/{r['flat_ndf_run1']}, run2 {r['flat_chi2_run2']:.1f}/{r['flat_ndf_run2']})  rms {r['rms']:.2f} mm" if m != 'bow' else '')
                  + (f"  surface span {r['surface']['span']:.2f} chi2 {r['surface']['chi2']:.1f}/{r['surface']['ndf']}" if 'span' in r.get('surface', {}) else ''))
        print('  corr', {k: (round(v[0], 2), v[1]) for k, v in G['corr'].items()})
        print('  slope vs G', {k: tuple(round(x, 2) for x in v) for k, v in G['slope_vs_G'].items()})


if __name__ == '__main__':
    main()
