"""a1_part1.py -- Part 1: cell maps of tau = w0/v_run and v_rel for n in NS, both runs and views; residual statistics and
run1 x run2 reproducibility.  -> $WORK/audit/a1/maps.pkl, part1.json"""
import json
import pickle
import numpy as np
from a1_lib import *  # noqa

maps = {}   # maps[n][run][view] -> dict
vrun = {}
for run in GM.RUNS:
    for p in 'xy':
        U, W, t, w = view_arrays(run, p)
        _, v, _, ev, nk = rline(t, w, nmin=40)
        c0 = CO.line(t, w)
        assert abs(c0[1] - v) < 1e-9, (c0, v)
        vrun[(run, p)] = (v, ev, len(t))
        print(run, p, 'v_run', round(v, 3), '+-', round(ev, 3), 'N', len(t), flush=True)
for n in NS:
    maps[n] = {}
    for run in GM.RUNS:
        maps[n][run] = {}
        for p in 'xy':
            U, W, t, w = view_arrays(run, p)
            cm = cellmap(U, W, t, w, n)
            maps[n][run][p] = to_tau(cm, vrun[(run, p)][0])
    print('built', n, flush=True)
pickle.dump(dict(maps=maps, vrun=vrun), open(f'{OUT}/maps.pkl', 'wb'))

res = {}
for n in NS:
    res[n] = {}
    for p in 'xy':
        m1, m2 = maps[n]['run1'][p], maps[n]['run2'][p]
        # tau minus weighted plane
        R1, c1, cv1, ch1, d1 = residual(m1, 'tau', 'etau', 'plane')
        R2, c2, cv2, ch2, d2 = residual(m2, 'tau', 'etau', 'plane')
        s = cross_stats(R1, m1['etau'], R2, m2['etau'])
        s['plane'] = {k: dict(coef=c.tolist(), err=np.sqrt(np.diag(cv)).tolist(), chi2=ch, dof=d)
                      for k, (c, cv, ch, d) in dict(run1=(c1, cv1, ch1, d1), run2=(c2, cv2, ch2, d2)).items()}
        s['sig_deg'] = float(np.degrees(np.arctan(s['sig'])))
        s['sig_deg_range'] = [float(np.degrees(np.arctan(s['sig_lo']))), float(np.degrees(np.arctan(s['sig_hi'])))]
        s['rms_deg'] = [float(np.degrees(np.arctan(s['rms1']))), float(np.degrees(np.arctan(s['rms2'])))]
        s['ncell_valid'] = [int(np.isfinite(m1['tau']).sum()), int(np.isfinite(m2['tau']).sum())]
        res[n][f'tau_{p}'] = s
        # v_rel: mean-removed and quadratic-removed
        for kind in ('mean', 'quad'):
            Q1 = residual(m1, 'vrel', 'evrel', kind)[0]; Q2 = residual(m2, 'vrel', 'evrel', kind)[0]
            res[n][f'vrel_{p}_{kind}'] = cross_stats(Q1, m1['evrel'], Q2, m2['evrel'])
    print(n, {k: (round(v['rms1'], 4), round(v['rms2'], 4), round(v['r'], 2), round(v['sig'], 4)) for k, v in res[n].items()
              if 'rms1' in v}, flush=True)
json.dump(res, open(f'{OUT}/part1.json', 'w'), indent=1, default=float)

# event-level plane of tau over the chamber (shared v): w = a + b U + c W + v t
ev_plane = {}
for run in GM.RUNS:
    for p in 'xy':
        U, W, t, w = view_arrays(run, p)
        X = np.vstack([np.ones_like(t), U, W, t]).T
        keep = np.ones(len(t), bool)
        for _ in range(3):
            c = np.linalg.lstsq(X[keep], w[keep], rcond=None)[0]
            r = w - X @ c; rs = 1.4826 * np.median(np.abs(r[keep])); keep = np.abs(r) < 3.5 * rs
        Xk = X[keep]; c = np.linalg.lstsq(Xk, w[keep], rcond=None)[0]
        s2 = np.sum((w[keep] - Xk @ c) ** 2) / (keep.sum() - 4)
        cov = s2 * np.linalg.inv(Xk.T @ Xk)
        v = c[3]
        ev_plane[f'{run}_{p}'] = dict(a=c[0] / v, b=c[1] / v, cc=c[2] / v, v=v, ea=np.sqrt(cov[0, 0]) / v,
                                      eb=np.sqrt(cov[1, 1]) / v, ec=np.sqrt(cov[2, 2]) / v, N=int(keep.sum()),
                                      resid_rms=float(np.sqrt(s2)))
        print(run, p, {k: round(float(x), 4) for k, x in ev_plane[f'{run}_{p}'].items()})
json.dump(ev_plane, open(f'{OUT}/part1_eventplane.json', 'w'), indent=1, default=float)
