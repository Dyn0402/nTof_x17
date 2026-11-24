#!/usr/bin/env python3
"""T4: drift-time (column duration) maps of the n_TOF chambers from late capsule tracks.

Per-track estimator (see report): T90 = (q_u90 + DT/2) / 0.9 per view, DT = 60 ns.  For a uniform column of duration T
starting at t0, _profile_summary's interpolated 90 % point is 0.9 T - DT/2 (the cumulative sum is read at bin centres),
so T90 = T for a uniform column; it is a scale estimator (map = ratios), not an absolute duration.  'xy' = mean of the
two views' T90.

Model, per arm and series (x, y, xy), on ln T:
    ln T = cell(x_local, y_local; 25 mm) + f_tanx + f_tany + f_t0 + f_q + f_table   (+ ln v_t/<v_t> removed first)
fitted by backfitting with binned medians (step functions; bins below).  The cell term, minus the chamber median, is the
map.  Errors per cell: 1.2533 * 1.4826 MAD(residual) / sqrt(n).

    python tmaps.py            -> tmaps.npz, results.json, figures/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path.insert(0, str(REPO))

DT = 60.0
E25 = np.arange(-200.0, 200.1, 25.0)
E50 = np.arange(-200.0, 200.1, 50.0)
E100 = np.arange(-200.0, 200.1, 100.0)
NMIN = 30
TAN_BINS = np.arange(-0.5, 0.501, 0.05)
T0_BINS = np.r_[-1e9, np.arange(-150.0, 150.1, 20.0), 1e9]
B_T0 = (-50.0, 250.0)          # B: no coincidence timing; keep tracks whose column fits the window
#: kappa = v_angle / v_t per arm and view (docs/PARAMETERS.md, vt_score/vt_kappa.json); D has no line (1)
KAPPA = {'A': {'x': 0.968, 'y': 0.973}, 'C': {'x': 0.978, 'y': 0.979}, 'D': {'x': 1.0, 'y': 1.0}}


def series(d, s):
    if s in ('x', 'y'):
        return ((d[f'{s}_q_u90'] + DT / 2) / 0.9).to_numpy(float)
    return 0.5 * (series(d, 'x') + series(d, 'y'))


def codes(v, bins):
    return np.clip(np.digitize(v, bins) - 1, 0, len(bins) - 2)


def cell_codes(d, edges):
    n = len(edges) - 1
    ix = np.floor((d.x_local.to_numpy() - edges[0]) / (edges[1] - edges[0])).astype(int)
    iy = np.floor((d.y_local.to_numpy() - edges[0]) / (edges[1] - edges[0])).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    return np.where(ok, ix * n + iy, -1), n


def gmed(code, r, n=None):
    s = pd.Series(r).groupby(code).median()
    if n is None:
        return s
    out = np.full(n, np.nan)
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def factors(d, arm, which=('tanx', 'tany', 't0', 'q', 'table')):
    F = {}
    if arm != 'B':
        if 'tanx' in which:
            F['tanx'] = codes(d.tanx.to_numpy(float), TAN_BINS)
        if 'tany' in which:
            F['tany'] = codes(d.tany.to_numpy(float), TAN_BINS)
    if 't0' in which:
        F['t0'] = codes(0.5 * (d.x_t0 + d.y_t0).to_numpy(float), T0_BINS)
    if 'q' in which:
        q = np.log((d.x_q_sum + d.y_q_sum).to_numpy(float))
        F['q'] = codes(q, np.nanpercentile(q, np.linspace(0, 100, 21)))
    if 'table' in which:
        F['table'] = pd.factorize(d.run.astype(str) + '/' + d.subrun.astype(str))[0]
    return F


def backfit(lnT, cell, ncell, F, n_iter=5):
    ok = np.isfinite(lnT) & (cell >= 0)
    lnT, cell = lnT[ok], cell[ok]
    F = {k: v[ok] for k, v in F.items()}
    f = {k: np.zeros(len(lnT)) for k in F}
    prof = {}
    cm = np.zeros(len(lnT))
    for _ in range(n_iter):
        r = lnT - sum(f.values()) if f else lnT.copy()
        cmap = gmed(cell, r, ncell)
        cm = cmap[cell]
        for k, c in F.items():
            r = lnT - cm - sum(v for kk, v in f.items() if kk != k)
            g = gmed(c, r)
            fk = g.reindex(c).to_numpy()
            off = np.median(fk)
            f[k] = fk - off
            prof[k] = (g - off)
    tot = sum(f.values()) if f else 0.0
    corr = lnT - tot
    cmap = gmed(cell, corr, ncell)
    res = corr - cmap[cell]
    n = np.bincount(cell, minlength=ncell)
    mad = gmed(cell, np.abs(res), ncell)
    err = 1.2533 * 1.4826 * mad / np.sqrt(np.maximum(n, 1))
    med = np.median(corr)
    m = cmap - med
    m[n < NMIN] = np.nan
    err[n < NMIN] = np.nan
    return dict(lnmap=m, err=err, n=n, prof=prof, med=float(med), sig=float(1.4826 * np.median(np.abs(res))))


def prep(D, arm):
    d = D[D.arm == arm].copy()
    if arm == 'B':
        t0 = 0.5 * (d.x_t0 + d.y_t0)
        d = d[(t0 > B_T0[0]) & (t0 < B_T0[1])]
    else:
        d = d[d.gated.astype(bool)]
    return d


def lnT_of(d, s, arm):
    T = series(d, s)
    with np.errstate(invalid='ignore', divide='ignore'):
        lt = np.log(T)
    if arm != 'B':
        vt = d.v_t.to_numpy(float)
        lt = lt + np.log(vt / np.nanmedian(vt))     # T * v_t is the gas-independent part
    return lt


def build(d, arm, s, edges=E25, which=('tanx', 'tany', 't0', 'q', 'table')):
    cell, n = cell_codes(d, edges)
    out = backfit(lnT_of(d, s, arm), cell, n * n, factors(d, arm, which))
    for k in ('lnmap', 'err', 'n'):
        out[k] = out[k].reshape(n, n)
    return out


def quad_fit(m, e, edges, rmax=150.0):
    c = 0.5 * (edges[1:] + edges[:-1]) / 100.0
    X, Y = np.meshgrid(c, c, indexing='ij')
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0) & (np.abs(X) < rmax / 100) & (np.abs(Y) < rmax / 100)
    m = m - np.median(m[ok])
    A = np.c_[np.ones(ok.sum()), X[ok], Y[ok], X[ok] ** 2, Y[ok] ** 2, X[ok] * Y[ok]]
    w = 1.0 / e[ok]
    p, *_ = np.linalg.lstsq(A * w[:, None], m[ok] * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    res = m[ok] - A @ p
    chi2 = float(np.sum((res / e[ok]) ** 2))
    dof = int(ok.sum() - 6)
    sc = max(np.sqrt(chi2 / max(dof, 1)), 1.0)
    pe = np.sqrt(np.diag(cov)) * sc
    return dict(p=p.tolist(), pe=pe.tolist(), chi2=chi2, dof=dof, n_cells=int(ok.sum()),
                tilt_x_pct_per_100mm=100 * p[1], tilt_x_err=100 * pe[1],
                tilt_y_pct_per_100mm=100 * p[2], tilt_y_err=100 * pe[2],
                quad_x_pct_at_150=100 * p[3] * 2.25, quad_x_err=100 * pe[3] * 2.25,
                quad_y_pct_at_150=100 * p[4] * 2.25, quad_y_err=100 * pe[4] * 2.25,
                cross_pct=100 * p[5], cross_err=100 * pe[5],
                resid_rms_pct=float(100 * np.std(res)), med_err_pct=float(100 * np.median(e[ok])),
                range_pct=[float(100 * np.nanpercentile(m[ok], 2)), float(100 * np.nanpercentile(m[ok], 98))])


def diff_stats(m1, e1, m2, e2, nmin=None, n1=None, n2=None):
    ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(e1) & np.isfinite(e2)
    if n1 is not None:
        ok &= (n1 >= nmin) & (n2 >= nmin)
    a, b = m1[ok], m2[ok]
    dd = (a - np.median(a)) - (b - np.median(b))       # each normalised to its own median already; remove offset
    chi2 = float(np.sum(dd ** 2 / (e1[ok] ** 2 + e2[ok] ** 2)))
    return dict(n_cells=int(ok.sum()), corr=float(np.corrcoef(a, b)[0, 1]), chi2=chi2,
                rms_diff_pct=float(100 * np.std(dd)), med_err_pct=float(100 * np.median(np.hypot(e1[ok], e2[ok]))))


if __name__ == '__main__':
    pass
