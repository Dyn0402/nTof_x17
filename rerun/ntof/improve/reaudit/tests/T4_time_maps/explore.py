#!/usr/bin/env python3
"""Estimator study: per-track column-duration estimators, their widths, x/y agreement, and dependence on t0, tan,
charge, time since flash at fixed position (centre region |x|,|y| < 75 mm, binned within 25 mm cells)."""
import numpy as np
import pandas as pd

D = pd.read_parquet('cache.parquet')
DT = 60.0


def est(d, p):
    u50, u90, ue = d[f'{p}_q_u50'], d[f'{p}_q_u90'], d[f'{p}_q_uend']
    return {'T50': 2 * (u50 + DT / 2), 'T90': (u90 + DT / 2) / 0.9, 'Td': (u90 - u50) / 0.4, 'Tend': ue}


def rw(v):
    v = v[np.isfinite(v)]
    m = np.median(v)
    return m, 1.4826 * np.median(np.abs(v - m)) / m


def within_cell(d, val, by, bins):
    """median of val/cell-median in bins of `by`, cells of 25 mm, cells weighted by counts."""
    ix = np.floor(d.x_local / 25).astype(int)
    iy = np.floor(d.y_local / 25).astype(int)
    cell = ix * 100 + iy
    rel = np.log(val) - pd.Series(np.log(val)).groupby(cell.values).transform('median').values
    b = np.digitize(by, bins)
    out = []
    for k in range(1, len(bins)):
        m = (b == k) & np.isfinite(rel)
        out.append((0.5 * (bins[k - 1] + bins[k]), m.sum(), np.median(rel[m]) * 100 if m.sum() > 100 else np.nan))
    return out


for arm in 'ACDB':
    d = D[D.arm == arm]
    c = d[(d.x_local.abs() < 75) & (d.y_local.abs() < 75)]
    print(f'==== {arm}: n {len(d)}, centre {len(c)}')
    for p in 'xy':
        E = est(c, p)
        print(' ', p, ' '.join(f'{k}: med {rw(v.to_numpy())[0]:.0f} rw {rw(v.to_numpy())[1]:.3f}' for k, v in E.items()))
    Ex, Ey = est(c, 'x'), est(c, 'y')
    for k in Ex:
        r = np.log(Ex[k] / Ey[k]).to_numpy()
        r = r[np.isfinite(r)]
        cc = np.corrcoef(*[np.log(v.to_numpy()) for v in (Ex[k], Ey[k])][:2])[0, 1] if False else np.nan
        ok = np.isfinite(Ex[k]) & np.isfinite(Ey[k]) & (Ex[k] > 0) & (Ey[k] > 0)
        cc = np.corrcoef(np.log(Ex[k][ok]), np.log(Ey[k][ok]))[0, 1]
        print(f'   {k}: x/y median ratio {np.exp(np.median(r)):.3f}, rw of ln(x/y) {1.4826*np.median(np.abs(r-np.median(r))):.3f}, corr {cc:.2f}')
    # combined estimator: mean of the two views
    for k in ('T50', 'T90', 'Td'):
        Tc = 0.5 * (Ex[k] + Ey[k])
        print(f'   {k} xy-mean: rw {rw(Tc.to_numpy())[1]:.3f}')
    for k in ('T90', 'Td', 'T50'):
        T = 0.5 * (Ex[k] + Ey[k]).to_numpy()
        t0 = 0.5 * (c.x_t0 + c.y_t0).to_numpy()
        print(f'  {k} vs t0 [ns]:', [(int(b), n, round(v, 1)) for b, n, v in within_cell(c, T, t0, np.arange(-200, 801, 50)) if n > 100])
        if arm != 'B':
            for tc in ('tanx', 'tany'):
                print(f'  {k} vs {tc}:', [(round(b, 2), n, round(v, 1)) for b, n, v in within_cell(c, T, c[tc].to_numpy(), np.arange(-0.5, 0.51, 0.1))])
        q = np.log((c.x_q_sum + c.y_q_sum).to_numpy())
        qb = np.nanpercentile(q, np.arange(0, 101, 10))
        print(f'  {k} vs charge decile:', [(n, round(v, 1)) for b, n, v in within_cell(c, T, q, qb)])
        ts = c.t_since_flash_ns.to_numpy() / 1e6
        print(f'  {k} vs t_since_flash [ms]:', [(int(b), n, round(v, 1)) for b, n, v in within_cell(c, T, ts, np.array([20, 25, 30, 40, 50, 60, 80, 100, 200]))])
