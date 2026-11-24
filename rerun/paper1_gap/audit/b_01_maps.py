#!/usr/bin/env python3
"""b_01_maps.py: per-run, per-n cell maps (T, v, hp, w0, charge-time proxies) for audit test B.  -> $WORK/audit/b/maps.pkl"""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gapmap as GM, cellobs as CO  # noqa
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/b'
XE, YE = (30.0, 370.0), (49.0, 351.0)      # U in [-1.7, 1.7], W in [-1.51, 1.51]
PROX = ['u50', 'u90', 'uend']


def tm_err(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan, np.nan
    return GM.tmean(a), GM.boot(GM.tmean, a, nb=100)


def maps_for(tab, m, n):
    ex, ey = np.linspace(*XE, n + 1), np.linspace(*YE, n + 1)
    it, jt = GM.cell_of(tab.xm.values, tab.ym.values, ex, ey)
    im, jm = GM.cell_of(m.xm.values, m.ym.values, ex, ey)
    keys = ['T', 'T_err', 'nT', 'vx', 'vx_err', 'vy', 'vy_err', 'hpx', 'hpx_err', 'hpy', 'hpy_err',
            'w0x', 'w0x_err', 'w0y', 'w0y_err', 'nv']
    keys += [f'{v}{k}' for v in 'xy' for k in PROX] + [f'{v}{k}_err' for v in 'xy' for k in PROX]
    # tan-range variants of the ladder slope
    keys += [f'v{p}_{r}' for p in 'xy' for r in ('t03', 't016')] + [f'v{p}_{r}_err' for p in 'xy' for r in ('t03', 't016')]
    M = {k: np.full((n, n), np.nan) for k in keys}
    U = (0.5 * (ex[:-1] + ex[1:]) - 200) / 100
    W = (0.5 * (ey[:-1] + ey[1:]) - 200) / 100
    for a in range(n):
        for b in range(n):
            d = tab[(it == a) & (jt == b)]
            mc = m[(im == a) & (jm == b)]
            M['nv'][a, b] = len(d); M['nT'][a, b] = len(mc)
            M['T'][a, b], M['T_err'][a, b] = tm_err(mc['T_amp'].values)
            for p in 'xy':
                tr = d[f'tan_ref_{p}'].values; w = d[f'{p}_w'].values * 1e3
                _, h, _, he, _ = CO.line(tr, d[f'{p}_p0'].values - d[f'ref_mesh_{p}'].values)
                M[f'hp{p}'][a, b], M[f'hp{p}_err'][a, b] = h, he
                s = np.abs(tr) <= GM.TAN_LADDER
                w0, v, w0e, ve, _ = CO.line(tr[s], w[s])
                M[f'w0{p}'][a, b], M[f'w0{p}_err'][a, b], M[f'v{p}'][a, b], M[f'v{p}_err'][a, b] = w0, w0e, v, ve
                for lab, sel in (('t03', np.abs(tr) <= 0.3), ('t016', (np.abs(tr) >= 0.1) & (np.abs(tr) <= 0.6))):
                    _, v2, _, ve2, _ = CO.line(tr[sel], w[sel])
                    M[f'v{p}_{lab}'][a, b], M[f'v{p}_{lab}_err'][a, b] = v2, ve2
                for k in PROX:
                    M[f'{p}{k}'][a, b], M[f'{p}{k}_err'][a, b] = tm_err(d[f'{p}_q_{k}'].values)
    return dict(U=U, W=W, **M)


def main():
    res = {}
    for run, R in GM.RUNS.items():
        tab = GM.load_table(WORK, R['rel'])
        m, info = GM.colsum_T(WORK, run, tab)
        res[run] = {}
        for n in (4, 6, 8, 10):
            res[run][n] = maps_for(tab, m, n)
        # chamber-wide ladder per tan range
        glob = {}
        for p in 'xy':
            tr = tab[f'tan_ref_{p}'].values; w = tab[f'{p}_w'].values * 1e3
            for lab, sel in (('t06', np.abs(tr) <= 0.6), ('t03', np.abs(tr) <= 0.3),
                             ('t016', (np.abs(tr) >= 0.1) & (np.abs(tr) <= 0.6))):
                a0, v, _, ve, nn = CO.line(tr[sel], w[sel]); glob[f'{p}_{lab}'] = (v, ve, a0, nn)
        res[run]['glob'] = glob
        print(run, len(tab), len(m), {k: np.round(v[:2], 3) for k, v in glob.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(res, open(f'{OUT}/maps.pkl', 'wb'))


if __name__ == '__main__':
    main()
