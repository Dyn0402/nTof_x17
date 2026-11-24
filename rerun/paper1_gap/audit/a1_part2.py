"""a1_part2.py -- Part 2: curl test (plane level, fine structure by finite differences, Helmholtz decomposition) with an
event bootstrap.  Needs maps.pkl from a1_part1.py.  -> part2.json, part2_boot.pkl, phi.pkl"""
import json
import pickle
import sys
import numpy as np
from multiprocessing import Pool
from a1_lib import *  # noqa

NCURL = (4, 6, 8, 10, 12)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 120
RUNS = ('run1', 'run2')


def resid_maps(mp_run, kind='plane'):
    """plane-subtracted (kind 'plane') or raw (kind 'none') tau maps of one run: (Rx, ex, Ry, ey)"""
    out = []
    for p in 'xy':
        m = mp_run[p]
        if kind == 'none':
            R = m['tau'].copy()
        else:
            R = residual(m, 'tau', 'etau', kind)[0]
        e = np.where(np.isfinite(R), m['etau'], np.nan)
        out += [R, e]
    return out


def helm_stats(n, kind, mp):
    """mp[run][view] map dicts.  Gradient-only description of the two views, runs combined and alone."""
    out = {}
    sols = {}
    for tag, runs in (('both', RUNS), ('run1', ('run1',)), ('run2', ('run2',))):
        A, y, ids = helmholtz_system(n, [resid_maps(mp[r], kind) for r in runs])
        phi, res, H, rank, cov = helmholtz_solve(A, y)
        chi2 = float(np.sum(res ** 2)); dof = len(y) - rank
        e = np.array([i[4] for i in ids])
        out[tag] = dict(chi2=chi2, dof=int(dof), chi2_dof=chi2 / dof, ndata=len(y), rank=rank,
                        exp_noise_rms=float(np.sqrt(np.mean(e ** 2 * np.clip(1 - H, 0, None)))),
                        obs_rms=float(np.sqrt(np.mean((res * e) ** 2))))
        sols[tag] = (ids, res * e, phi, cov, A)
    # noise-free cross-covariance of the solenoidal residuals of run1 alone and run2 alone
    d1 = {(c, i, j): r for (mi, c, i, j, e_), r in zip(sols['run1'][0], sols['run1'][1])}
    d2 = {(c, i, j): r for (mi, c, i, j, e_), r in zip(sols['run2'][0], sols['run2'][1])}
    keys = [k for k in d1 if k in d2]
    a = np.array([d1[k] for k in keys]); b = np.array([d2[k] for k in keys])
    # total (non-gradient + gradient) cross-covariance of the same components
    R = {r: resid_maps(mp[r], kind) for r in RUNS}
    tot1 = np.array([R['run1'][0 if k[0] == 'x' else 2][k[1], k[2]] for k in keys])
    tot2 = np.array([R['run2'][0 if k[0] == 'x' else 2][k[1], k[2]] for k in keys])
    out['sol_cov'] = float(np.mean(a * b)); out['sol_n'] = len(keys)
    out['tot_cov'] = float(np.mean(tot1 * tot2))
    out['sol_sig'] = float(np.sqrt(max(out['sol_cov'], 0))); out['tot_sig'] = float(np.sqrt(max(out['tot_cov'], 0)))
    out['sol_frac_var'] = float(out['sol_cov'] / out['tot_cov']) if out['tot_cov'] > 0 else np.nan
    return out, sols


def one(mp):
    """all Part 2 numbers for a set mp[n][run][view]"""
    res = {}
    for n in NCURL:
        Rx = {r: residual(mp[n][r]['x'], 'tau', 'etau', 'plane')[0] for r in RUNS}
        Ry = {r: residual(mp[n][r]['y'], 'tau', 'etau', 'plane')[0] for r in RUNS}
        ex = {r: np.where(np.isfinite(Rx[r]), mp[n][r]['x']['etau'], np.nan) for r in RUNS}
        ey = {r: np.where(np.isfinite(Ry[r]), mp[n][r]['y']['etau'], np.nan) for r in RUNS}
        d = dict(curl=curl_stats(Rx, ex, Ry, ey, n))
        for kind in ('plane', 'none'):
            d[f'helm_{kind}'] = helm_stats(n, kind, mp[n])[0]
        # plane level: c_x = dtau_x/dW and b_y = dtau_y/dU from the cell planes
        pl = {}
        for r in RUNS:
            cx = residual(mp[n][r]['x'], 'tau', 'etau', 'plane'); cy = residual(mp[n][r]['y'], 'tau', 'etau', 'plane')
            pl[r] = dict(cx=float(cx[1][2]), cx_err=float(np.sqrt(cx[2][2, 2])), by=float(cy[1][1]),
                         by_err=float(np.sqrt(cy[2][1, 1])), bx=float(cx[1][1]), cy=float(cy[1][2]))
        d['plane'] = pl
        res[n] = d
    return res


def build(args):
    seed = args
    rng = np.random.default_rng(max(seed, 0) + 1000)
    T = tables()
    mp = {n: {r: {} for r in RUNS} for n in NCURL}
    for r in RUNS:
        d = T[r]
        k = rng.integers(0, len(d), len(d)) if seed >= 0 else np.arange(len(d))
        dd = d.iloc[k]
        U = (dd.xm.values - 200) / 100; W = (dd.ym.values - 200) / 100
        for p in 'xy':
            t = dd[f'tan_ref_{p}'].values; w = dd[f'{p}_w'].values * 1e3
            s = np.abs(t) <= GM.TAN_LADDER
            vrun = rline(t[s], w[s], nmin=40)[1]
            for n in NCURL:
                mp[n][r][p] = to_tau(cellmap(U[s], W[s], t[s], w[s], n), vrun)
    return one(mp)


if __name__ == '__main__':
    base = build(-1)
    json.dump(base, open(f'{OUT}/part2.json', 'w'), indent=1, default=float)
    print('baseline done', flush=True)
    with Pool(4) as pool:
        boots = pool.map(build, range(NB))
    pickle.dump(boots, open(f'{OUT}/part2_boot.pkl', 'wb'))
    print('boot done', len(boots))
