#!/usr/bin/env python3
"""xw_tan.py -- X width against the SIGNED reference angle: is the drift field tilted?

A non-flat cathode (det3's charge column ends at 25.7-29.2 mm across the chamber,
mx_june_wft/GAP_STUDY_2026-07-30.md) tilts the drift field, which walks the charge sideways in
proportion to depth: in the X view an extra angle delta that the M3 reference does not see.  A head-on
stack of tracks at reference angle tan then grows as if the track had tan + delta, so the fitted width
is smallest at tan = -delta.  The FINDINGS §10 split was in |tan| and could not see the sign.

Per dataset, unsaturated events (window max < SAT) in signed tan_x bins: x_width.fit_x on the stack
(bootstrap errors), and the same bins on xw_sim toy replicas (vertical field, the dataset's truth), which
give the curve an untilted chamber makes.  delta from a fit of D_T^2 = a + b (tan + delta)^2 to the data,
with b free and with b fixed to the toy's.

    xw_tan.py [--sets det3L,det4,det7R,det7H] [--nrep 10] [--nboot 20]   -> results/xw_tan.json
    xw_tan.py --split ref_mesh_x|ref_mesh_y [--sets det3L]                  delta per position tercile
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x_width as xw                                          # noqa: E402
import xw_sim as X                                            # noqa: E402

EDGES = np.array([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
SAT = 3500.0
TANMAX = 1.0                                                  # stack's own cut off: the bins select


def fit_bin(args):
    h, ev, T0, seed = args
    if seed is not None:
        rng = np.random.default_rng(seed)
        ks = list(ev)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    n, M = xw.stack(ev, 'x', tan_max=TANMAX)
    try:
        return xw.fit_x(h, M, T0)
    except Exception:
        return None


def bins_of(ev):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        out.append({k: e for k, e in ev.items() if lo <= e['tan_x'] < hi and e['x']['W'].max() < SAT})
    return out


def measure_bins(pool, h, ev, T0, nboot):
    rows = []
    for (lo, hi), sub in zip(zip(EDGES[:-1], EDGES[1:]), bins_of(ev)):
        if len(sub) < 40:
            rows.append(dict(lo=float(lo), hi=float(hi), n=len(sub)))
            continue
        rs = list(pool.map(fit_bin, [(h, sub, T0, None)] + [(h, sub, T0, 5000 + b) for b in range(nboot)]))
        c, bs = rs[0], [b for b in rs[1:] if b is not None]
        rows.append(dict(c, lo=float(lo), hi=float(hi), n=len(sub),
                         tan_mean=float(np.mean([e['tan_x'] for e in sub.values()])),
                         **{k + '_err': float(np.std([b[k] for b in bs])) for k in ('DT', 'v', 'sig0')}))
    return rows


def toy_rep(args):
    S, seed = args
    X.TAN = 1.0                                               # generate every layout, not just |tan| < 0.03
    ev = X.Gen(S, 'base', seed).sample()
    out = []
    for sub in bins_of(ev):
        f = fit_bin((S['h_fit'], sub, S['T0'], None)) if len(sub) >= 40 else None
        out.append(None if f is None else f['DT'])
    return out


def parab(tm, D, De, b_fix=None):
    """D_T^2 = a + b (tan + delta)^2 ->  (delta, err, a, b)."""
    ok = np.isfinite(D) & np.isfinite(De) & (De > 0)
    t, y, ye = tm[ok], D[ok] ** 2, 2 * D[ok] * De[ok]

    def res(q):
        a, d = q[0], q[1]
        b = q[2] if b_fix is None else b_fix
        return (a + b * (t + d) ** 2 - y) / ye
    q0 = [y.min(), 0.0] + ([1e7] if b_fix is None else [])
    r = least_squares(res, q0)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * max((r.fun ** 2).sum() / max(len(y) - len(q0), 1), 1.0)
    b = r.x[2] if b_fix is None else b_fix
    return dict(delta=float(r.x[1]), delta_err=float(np.sqrt(cov[1, 1])), a=float(r.x[0]), b=float(b),
                chi2=float((r.fun ** 2).sum()), ndf=int(len(y) - len(q0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='det3L,det4,det7R,det7H')
    ap.add_argument('--nrep', type=int, default=10)
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--split', choices=['ref_mesh_x', 'ref_mesh_y'])
    a = ap.parse_args()
    inp = pickle.load(open(X.INPUTS, 'rb'))
    out_path = f'{HERE}/results/xw_tan.json'
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    with ProcessPoolExecutor(a.workers) as pool:
        for name in a.sets.split(','):
            S = inp[name]
            ev = pickle.load(open(f'{xw.CACHES}/headon_{S["key"]}.pkl', 'rb'))
            ev = {k: e for k, e in ev.items() if 'x' in e}
            mids = 0.5 * (EDGES[:-1] + EDGES[1:])
            if a.split:
                r = out.setdefault(name, {})
                b_toy = r.get('fit_free', {}).get('b')
                b_fix = r.get('toy_fit', {}).get('b')
                vals = np.array([e[a.split] for e in ev.values()])
                q = np.quantile(vals, [0, 1 / 3, 2 / 3, 1]); q[-1] += 1e-6
                rows = []
                for lo, hi in zip(q[:-1], q[1:]):
                    sub = {k: e for k, e in ev.items() if lo <= e[a.split] < hi}
                    br = measure_bins(pool, S['h_fit'], sub, S['T0'], a.nboot)
                    D = np.array([x.get('DT', np.nan) for x in br]); De = np.array([x.get('DT_err', np.nan) for x in br])
                    tm = np.array([x.get('tan_mean', m) for x, m in zip(br, mids)])
                    f = parab(tm, D, De, b_fix)
                    rows.append(dict(lo=float(lo), hi=float(hi), bins=br, fit=f))
                    print(f'{name} {a.split} {lo:6.1f}-{hi:6.1f}: ' + ' '.join(f'{x:4.0f}' for x in D) +
                          f'  delta {f["delta"]:+.4f}±{f["delta_err"]:.4f}', flush=True)
                r[f'split_{a.split}'] = rows
                json.dump(out, open(out_path, 'w'), indent=1)
                continue
            br = measure_bins(pool, S['h_fit'], ev, S['T0'], a.nboot)
            toys = np.array(list(pool.map(toy_rep, [(S, 900000 + k) for k in range(a.nrep)])), dtype=float)
            D = np.array([x.get('DT', np.nan) for x in br]); De = np.array([x.get('DT_err', np.nan) for x in br])
            tm = np.array([x.get('tan_mean', m) for x, m in zip(br, mids)])
            Tm, Ts = np.nanmean(toys, 0), np.nanstd(toys, 0, ddof=1)
            tf = parab(mids, Tm, Ts / np.sqrt(a.nrep))
            r = dict(set=name, truth_DT=S['DT'], bins=br, toy_mean=Tm.tolist(), toy_rms=Ts.tolist(),
                     toy_fit=tf, fit_free=parab(tm, D, De), fit_toyb=parab(tm, D, De, tf['b']))
            out[name] = r
            print(f'{name}: tan bins ' + ' '.join(f'{m:+.2f}' for m in mids))
            print('   data D_T ' + ' '.join(f'{d:4.0f}±{e:2.0f}' for d, e in zip(D, De)) + '   n ' + ' '.join(str(x['n']) for x in br))
            print('   toy  D_T ' + ' '.join(f'{d:4.0f}±{e:2.0f}' for d, e in zip(Tm, Ts)) + f'   (truth {S["DT"]:.0f})')
            for lab in ('toy_fit', 'fit_free', 'fit_toyb'):
                f = r[lab]
                print(f'   {lab:9s} delta {f["delta"]:+.4f}±{f["delta_err"]:.4f}  b {f["b"]:.3g}  a^0.5 {np.sqrt(max(f["a"], 0)):.0f}  '
                      f'chi2 {f["chi2"]:.1f}/{f["ndf"]}', flush=True)
            json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
