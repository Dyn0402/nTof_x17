#!/usr/bin/env python3
"""x_width.py -- X drift-width growth on every bench dataset, and by position on the chamber.

FINDINGS §28: the joint X+Y head-on fit puts the bench drift diffusion Dd at the dry Magboltz value,
while v says 0.4-0.6 % water (whose Magboltz D_T is 25-30 % smaller).  That fit shares one sigma_0
between the views and used one dataset per chamber.  Here X alone (no RC term, so the width growth
with drift time is the whole story), with bench_attach's model:

    W_o(t) = A * sum_u sum_s h(t - dt - u - s) dF_o(sigma(u)),   sigma^2 = sigma_0^2 + 2 Dd u,
    u uniform over T = GAP / v

fitted on the unbiased head-on stack (rc_diffusion.stack), free sigma_0, Dd, A, dt and T (so v comes
from the same stack; 'Tbundle' arm pins T at the bundle v as a check).  Read out as

    D_T,eff = sqrt(2 Dd / v)  [um/sqrt(cm)]   and   sigma_drift(gap) = sqrt(2 Dd T)  [mm],

and compared with Magboltz (gasmodel.GasGrid 'bench', air_hs) at the run's drift field: the water
fraction the fitted v implies, its D_T, and dry.  Errors: bootstrap over events.

Datasets: the head-on caches of headon_cache.py (every reference event with |tan| < 0.06 in a view).
Splits (|tan_x| < TAN): by ref_mesh_x (across the X strips) and ref_mesh_y (along them), and centre
vs edge.

    x_width.py [--keys k1,k2,...] [--nboot 30] [--tan 0.03] [--workers 12]
        -> results/x_width.json
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
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa')]
from rc_diffusion import stack, GRID, F, OFFS, STEP
from bench_attach import predict, template, GAP

CACHES = '/home/dylan/x17/cosmic_bench/cloud_basics/xwidth/caches'
KEYS = {'o22_long_det2': 'det2', 'o22_det2': 'det2',
        'sat_det3': 'det3', 'o22_long_det3': 'det3', 'o22_det3': 'det3',
        'g_det4': 'det4',
        'g_det6_long': 'det6', 'g_det6_longer': 'det6',
        'g_det7_long': 'det7', 'g_det7_longer': 'det7'}
V_BUNDLE = {'det2': 39.94, 'det3': 36.6, 'det4': 34.16, 'det6': 26.7, 'det7': 36.6}
LAM = 1e6
MIN_EV = 120


SCAN = 'sat_det3__'          # headon_cache.py --subrun caches of the det3 6-27 drift / amplification scans
# starting v per drift voltage: driftscan_fit's composition curve (0.95 % water), rescaled to GAP
V_SCAN = {500: 15.9, 700: 25.1, 900: 32.8, 1000: 35.3, 1100: 37.8}


def det_of(key):
    return 'det3' if key.startswith(SCAN) else KEYS[key]


def drift_hv(key):
    if key.startswith(SCAN):
        import re
        return float(re.search(r'drift_(\d+)V', key).group(1))
    from qa_config import get_config
    c = get_config(key)
    rc = json.load(open(f'{c.BASE_PATH}{c.RUN}/run_config.json'))
    det = next(d for d in rc['detectors'] if d['name'] == c.DET_NAME)
    card, ch = det['hv_channels']['drift']
    sub = next(s for s in rc['sub_runs'] if s['sub_run_name'] == c.SUB_RUN)
    return float(sub['hvs'][str(card)][str(ch)])


def predict_x(h, sig0, Dd, T, A, t_shift, n):
    """bench_attach.predict for D_rc = 0, no attenuation, vectorised: with no spread after landing the
    charge from slice u_k sits at F_o(sigma(u_k)) for good, so W_o = A sum_k F_o(sigma_k) h(t - u_k).
    The last slice carries the fractional weight T/STEP - floor, so the model is continuous in T
    (predict rounds T to whole slices, which makes T unfittable)."""
    nf = T / STEP
    nu = int(np.ceil(nf))
    w = np.ones(nu); w[-1] = nf - (nu - 1)
    u = (np.arange(nu) + 0.5) * STEP
    Fo = np.array([F(np.sqrt(sig0 ** 2 + 2 * Dd * u), o) for o in OFFS]) * w      # (9, nu)
    H = np.zeros((nu, n))
    for k in range(nu):
        m = min(len(h), n - k)
        if m > 0:
            H[k, k:k + m] = h[:m]
    P = A * (Fo @ H)
    t = np.arange(n) * STEP
    return np.array([np.interp(t - t_shift, t, p, left=0.0, right=p[-1]) for p in P])


def fit_x(h, M, T0, free_T=True, sig0_fix=None):
    """sig0_fix: pin sigma_0 (and T at T0) -- the split arm that keeps the sigma_0-Dd trade out."""
    sel = np.all(np.isfinite(M), axis=0)
    n = len(GRID)
    if sig0_fix is not None:
        free_T = False

    def unpack(q):
        sig0, Dd, A, dt = q[:4]
        if sig0_fix is not None:
            sig0 = sig0_fix
        return sig0, Dd, A, dt, (q[4] if free_T else T0)

    def res(q):
        sig0, Dd, A, dt, T = unpack(q)
        return (predict_x(h, sig0, Dd, T, A, dt, n)[:, sel] - M[:, sel]).ravel()

    x0, lo, hi, sc = [0.4, 3e-4, 0.02, 0.0], [0.02, 0.0, 1e-4, -600], [1.5, 0.005, 10, 600], [0.1, 1e-4, 0.01, 30]
    if sig0_fix is not None:                         # q[0] is carried but ignored
        x0[0], lo[0], hi[0] = sig0_fix, sig0_fix - 1e-6, sig0_fix + 1e-6
    if free_T:
        x0.append(T0); lo.append(0.6 * T0); hi.append(1.5 * T0); sc.append(30)
    best = None
    for d0 in (100.0, 200.0, 300.0, 400.0):
        x0[3] = d0
        r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, x_scale=sc)
        if best is None or r.cost < best.cost:
            best = r
    sig0, Dd, A, dt, T = unpack(best.x)
    return dict(sig0=float(sig0), Dd=float(Dd), T=float(T), v=float(GAP / T * 1e3),
                DT=float(np.sqrt(2 * Dd / (GAP / T)) * np.sqrt(10) * 1e3), sdrift=float(np.sqrt(2 * Dd * T)),
                rms=float(np.sqrt((best.fun ** 2).mean())), dt=float(dt))


def one(args):
    h, ev, T0, free_T, sig0_fix, seed = args
    if seed is not None:
        rng = np.random.default_rng(seed)
        ks = list(ev)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    n, M = stack(ev, 'x', tan_max=TAN)
    return fit_x(h, M, T0, free_T, sig0_fix)


TAN = 0.03


def measure(pool, h, ev, T0, nboot, free_T=True, sig0_fix=None):
    jobs = [(h, ev, T0, free_T, sig0_fix, s) for s in [None] + [1000 + b for b in range(nboot)]]
    rs = list(pool.map(one, jobs))
    c, bs = rs[0], rs[1:]
    out = dict(c, n=int(sum(abs(e['tan_x']) < TAN for e in ev.values())))
    for k in ('sig0', 'Dd', 'v', 'DT', 'sdrift'):
        out[k + '_err'] = float(np.std([b[k] for b in bs]))
    return out


def magboltz(v, E):
    from gasmodel import GasGrid
    from bench_water_e import invert
    return invert(GasGrid('bench', 'air_hs'), v, E)


def split(ev, coord, edges):
    return [{k: e for k, e in ev.items() if lo <= e[coord] < hi} for lo, hi in zip(edges[:-1], edges[1:])]


def main():
    global TAN
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=','.join(KEYS))
    ap.add_argument('--nboot', type=int, default=30)
    ap.add_argument('--tan', type=float, default=0.03)
    ap.add_argument('--nsplit', type=int, default=3)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    TAN = a.tan
    out_path = f'{HERE}/results/x_width.json'
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    with ProcessPoolExecutor(a.workers) as pool:
        for key in a.keys.split(','):
            det = det_of(key)
            h, _ = template(det)
            ev = pickle.load(open(f'{CACHES}/headon_{key}.pkl', 'rb'))
            T0 = GAP / (V_BUNDLE[det] * 1e-3)
            if key.startswith(SCAN):
                T0 = GAP / (V_SCAN[int(drift_hv(key))] * 1e-3)
            E = drift_hv(key) / 2.88
            r = dict(det=det, E_Vcm=E, v_bundle=V_BUNDLE[det])
            r['all'] = measure(pool, h, ev, T0, a.nboot)
            r['all_Tbundle'] = measure(pool, h, ev, T0, a.nboot, free_T=False)
            r['magboltz'] = magboltz(r['all']['v'], E)
            q = r['all']
            mb = r['magboltz']
            print(f'{key:14s} {det} E {E:4.0f}  n {q["n"]:4d}  v {q["v"]:5.2f}±{q["v_err"]:.2f} (bundle {V_BUNDLE[det]})  '
                  f'sig0 {q["sig0"]:.3f}±{q["sig0_err"]:.3f}  D_T {q["DT"]:4.0f}±{q["DT_err"]:3.0f}  '
                  f'[MB wet {mb["DT_um_rtcm"]:4.0f} @ {mb["water_pct"]:.2f} % / dry {mb["DT_dry"]:4.0f}]  '
                  f'Tbundle D_T {r["all_Tbundle"]["DT"]:4.0f}±{r["all_Tbundle"]["DT_err"]:3.0f}', flush=True)
            # position splits on the X head-on events
            xo = {k: e for k, e in ev.items() if abs(e['tan_x']) < TAN}
            r['splits'] = {}
            for coord in ('ref_mesh_x', 'ref_mesh_y'):
                vals = np.array([e[coord] for e in xo.values()])
                edges = np.quantile(vals, np.linspace(0, 1, a.nsplit + 1)); edges[-1] += 1e-6
                rows = []
                for (lo, hi), sub in zip(zip(edges[:-1], edges[1:]), split(xo, coord, edges)):
                    if len(sub) < MIN_EV:
                        continue
                    m = measure(pool, h, sub, T0, a.nboot)
                    m['fixed'] = measure(pool, h, sub, q['T'], a.nboot, sig0_fix=q['sig0'])
                    m.update(lo=float(lo), hi=float(hi), mid=float(np.median([e[coord] for e in sub.values()])))
                    rows.append(m)
                    f = m['fixed']
                    print(f'    {coord} {lo:5.0f}-{hi:5.0f}: n {m["n"]:4d} v {m["v"]:5.2f}±{m["v_err"]:.2f} '
                          f'sig0 {m["sig0"]:.3f}±{m["sig0_err"]:.3f} D_T {m["DT"]:4.0f}±{m["DT_err"]:3.0f}   '
                          f'| sig0,T fixed: D_T {f["DT"]:4.0f}±{f["DT_err"]:3.0f} sdrift {f["sdrift"]:.3f}±{f["sdrift_err"]:.3f}',
                          flush=True)
                r['splits'][coord] = rows
            out[key] = r
            json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
