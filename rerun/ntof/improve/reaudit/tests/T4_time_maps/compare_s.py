#!/usr/bin/env python3
"""T4 part 2: compare the T maps with the scale part s(x, y) of the cosmic angle map (angle_map_r2.json).

1 + s = v_true / v_assumed per cell (r = slope - j = a + s j).  Per-cell errors on s from a bootstrap of the clean
lines (A, C: lines.parquet + angle_map.clean_lines, 50 mm, N_MIN 80; D: bd_lines.parquet + bd_map.clean, 100 mm,
N_MIN 60), 60 replicas.  Fit ln(1+s) = a + b ln(T/T_med) per cell, weighted by the effective variance.
Writes results_s.json, s_boot.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path[:0] = [str(HERE), str(REPO)]
from rerun.ntof.improve import angle_map as AM  # noqa: E402
from rerun.ntof.improve import bd_map as BD  # noqa: E402

TQ = Path('/media/ucla/work/ntof_pass1/ana_r2/track_quality')
MAP = json.load(open(REPO / 'rerun/ntof/stage3/angle_map/angle_map_r2.json'))['maps']
NB = 60
rng = np.random.default_rng(17)


def boot(G, edges, nmin):
    AM.EDGES, AM.CENTRES, AM.N_MIN = edges, 0.5 * (edges[1:] + edges[:-1]), nmin
    m0 = AM.fit_map(G)
    S = []
    for _ in range(NB):
        S.append(np.asarray(AM.fit_map(G.iloc[rng.integers(0, len(G), len(G))])['s']))
    return np.asarray(m0['s']), np.std(S, axis=0), np.asarray(m0['n'])


def wfit(x, y, ex, ey):
    b = 0.0
    for _ in range(10):
        w = 1.0 / (ey ** 2 + (b * ex) ** 2)
        A = np.c_[np.ones_like(x), x]
        cov = np.linalg.inv((A * w[:, None]).T @ A)
        p = cov @ (A * w[:, None]).T @ y
        b = p[1]
    chi2 = float(np.sum(w * (y - A @ p) ** 2))
    dof = len(x) - 2
    sc = max(np.sqrt(chi2 / dof), 1.0)
    return dict(a=float(p[0]), b=float(p[1]), b_err=float(np.sqrt(cov[1, 1]) * sc), b_err_stat=float(np.sqrt(cov[1, 1])),
                chi2=chi2, dof=dof, r=float(np.corrcoef(x, y)[0, 1]), n=len(x))


def grad(m, e, edges):
    c = 0.5 * (edges[1:] + edges[:-1]) / 100.0
    X, Y = np.meshgrid(c, c, indexing='ij')
    ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
    A = np.c_[np.ones(ok.sum()), X[ok], Y[ok]]
    w = 1 / e[ok]
    p, *_ = np.linalg.lstsq(A * w[:, None], m[ok] * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    chi2 = np.sum(((m[ok] - A @ p) * w) ** 2)
    sc = max(np.sqrt(chi2 / (ok.sum() - 3)), 1)
    return dict(gx=float(p[1]), gx_err=float(np.sqrt(cov[1, 1]) * sc), gy=float(p[2]), gy_err=float(np.sqrt(cov[2, 2]) * sc))


def main():
    T = {}
    for f in HERE.glob('tmaps_*.npz'):
        T.update(dict(np.load(f)))
    L = pd.read_parquet(TQ / 'lines.parquet')
    BL = pd.read_parquet(TQ / 'bd_lines.parquet')
    out, sb = {}, {}
    for arm, view in [('A', 'x'), ('A', 'y'), ('C', 'x'), ('C', 'y'), ('D', 'x'), ('D', 'y')]:
        key = f'{arm}_{view}'
        if arm in 'AC':
            G = AM.clean_lines(L, arm.lower(), view)
            edges, nmin, tag = AM.np.arange(-200.0, 201.0, 50.0), 80, '50'
        else:
            G = BD.clean(BL, {'x': 'z', 'y': 'y'}[view])
            edges, nmin, tag = np.arange(-200.0, 201.0, 100.0), 60, '100'
        s0, se, ns = boot(G, edges, nmin)
        diff = np.nanmax(np.abs(s0 - np.asarray(MAP[key]['s'])))
        sb[f'{key}_s'], sb[f'{key}_s_err'], sb[f'{key}_n'] = s0, se, ns
        y = np.log1p(s0)
        ey = se / (1 + s0)
        res = dict(n_lines=len(G), max_diff_vs_shipped_map=float(diff))
        for ts in (view, 'xy'):
            m = np.log(T[f'{arm}_{ts}_T{tag}'])
            e = T[f'{arm}_{ts}_err{tag}']
            nt = T[f'{arm}_{ts}_n{tag}']
            ok = (ns >= nmin) & np.isfinite(m) & (nt >= 100) & np.isfinite(e)
            fit = wfit(m[ok], y[ok], e[ok], ey[ok])
            ok2 = ok & (np.abs(0.5 * (edges[1:] + edges[:-1]))[:, None] < 150) & \
                (np.abs(0.5 * (edges[1:] + edges[:-1]))[None, :] < 150)
            fit_in = wfit(m[ok2], y[ok2], e[ok2], ey[ok2]) if ok2.sum() > 4 else None
            res[f'T_{ts}'] = dict(all=fit, interior=fit_in, T_range_pct=[float(100 * m[ok].min()), float(100 * m[ok].max())],
                                  s_range=[float(s0[ok].min()), float(s0[ok].max())],
                                  grad_s=grad(y, np.where(ok, ey, np.nan), edges),
                                  grad_lnT=grad(np.where(ok, m, np.nan), np.where(ok, e, np.nan), edges))
        out[key] = res
        print(key, json.dumps(res, default=float)[:1500], flush=True)
    np.savez(HERE / 's_boot.npz', **sb)
    json.dump(out, open(HERE / 'results_s.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
