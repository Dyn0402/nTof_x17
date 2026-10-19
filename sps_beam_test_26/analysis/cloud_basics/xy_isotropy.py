#!/usr/bin/env python3
"""xy_isotropy.py -- is the X width excess (FINDINGS §29) drift diffusion, or an X-readout effect?

Drift diffusion is isotropic: whatever widens X with depth widens Y by the same amount.  A term in the X
readout (the X strips sit one board layer deeper) would not.  On its own Y cannot tell: it carries the RC
spread along the resistive strips, sigma^2 = sigma_0^2 + 2 Dd u + 2 D_rc s (u = arrival time, s = time
since landing), and in one stack Dd_y and D_rc trade off (§27).  The det3 6-27 drift scan breaks that:
D_rc is resistive-layer physics and does not depend on the drift field, while Dd(E) and T(E) do.  So
D_rc is shared over the fields and profiled on a grid; at each grid point every field is fitted on its
own, X and Y together (bench_attach.predict, X template for both views, uniform arrival over T):

    per field: sigma_0x, sigma_0y, T, A_x, A_y, dt_x, dt_y, and Dd_x, Dd_y   (arm 'free')
               ... with Dd_y = Dd_x                                          (arm 'iso': one gas)
               ... with Dd_y = Magboltz at det3's composition, Dd_x free     (arm 'gasY': the excess is X-only)

Total cost per arm vs D_rc; at the best D_rc of each arm, Dd_y/Dd_x per field with bootstrap errors.
Reads the head-on caches of headon_cache.py (the drift-scan sub-runs and the long run at 1000 V).

    xy_isotropy.py [--nboot 20] [--workers 14] [--tan 0.03]
        -> results/xy_isotropy.json
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
from rc_diffusion import stack, GRID
from bench_attach import predict, template, GAP
import gaps                                                 # noqa: E402

CACHES = '/home/dylan/x17/cosmic_bench/cloud_basics/xwidth/caches'
# drift HV -> cache; 100 V and 300 V are left out: their drift time outlasts the 3.6 us stack (x_width.py)
RUNS = {500: 'sat_det3__drift_scan_resist_490V_drift_500V',
        700: 'sat_det3__drift_scan_resist_490V_drift_700V',
        900: 'sat_det3__drift_scan_resist_490V_drift_900V',
        1000: 'sat_det3__long_run_resist_490V_drift_1000V',
        1100: 'sat_det3__drift_scan_resist_490V_drift_1100V'}
GAP_DET3 = gaps.gap('det3')   # 27.9 (timing) until 2026-10-19; dished cathode (HANDOFF X-width gotchas); fits use GAP = 30, scaled when compared
WATER = 0.95             # det3 six-field composition fit, FINDINGS §25 (no air)
DRC_GRID = [1.0e-4, 1.5e-4, 2.0e-4, 2.8e-4, 3.6e-4, 4.5e-4, 5.5e-4, 7.0e-4]
LAM = 1e6
TAN = 0.03
SAT = 3500.0     # drop events whose X or Y window reaches this ADC (saturation biases v and the widths: unsat_width.py)


def magboltz_Dd(E, v_fit):
    """Magboltz Dd [mm^2/ns] at det3's composition, as the fit's GAP=30 frame sees it.  D_T is a gas
    property; the fit's sigma_drift^2(gap) = 2 Dd T is measured over the real 27.9 mm, so in the fit
    frame Dd = 0.5 D_T^2 v_fit (27.9/30)."""
    from gasmodel import GasGrid
    g = GasGrid('bench', 'air_hs')
    DT = g(WATER, 0.0, E)['DT']                      # um/sqrt(cm)
    return 0.5 * (DT * 1e-3 / np.sqrt(10)) ** 2 * v_fit * 1e-3 * GAP_DET3 / GAP, float(DT)


def fit_field(h, Ms, sels, T0, drc, arm, dd_y_fix=None):
    n = len(GRID)

    def unpack(q):
        s0x, s0y, T, ax, ay, dx, dy, ddx = q[:8]
        ddy = q[8] if arm == 'free' else (ddx if arm == 'iso' else dd_y_fix)
        return s0x, s0y, T, ax, ay, dx, dy, ddx, ddy

    def res(q):
        s0x, s0y, T, ax, ay, dx, dy, ddx, ddy = unpack(q)
        r = []
        for p, s0, A, dt, dd, d_rc in (('x', s0x, ax, dx, ddx, 0.0), ('y', s0y, ay, dy, ddy, drc)):
            P = predict(h, s0, dd, d_rc, LAM, T, A, dt, n)
            r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
        return np.concatenate(r)

    x0 = [0.4, 0.4, T0, 0.02, 0.02, 0.0, 0.0, 3e-4]
    lo = [0.02, 0.02, 0.6 * T0, 1e-4, 1e-4, -600, -600, 0.0]
    hi = [1.5, 1.5, 1.5 * T0, 10, 10, 600, 600, 0.005]
    sc = [0.1, 0.1, 30, 0.01, 0.01, 30, 30, 1e-4]
    if arm == 'free':
        x0.append(3e-4); lo.append(0.0); hi.append(0.005); sc.append(1e-4)
    best = None
    for d0 in (100.0, 250.0, 400.0):
        x0[5] = x0[6] = d0
        r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, x_scale=sc)
        if best is None or r.cost < best.cost:
            best = r
    s0x, s0y, T, ax, ay, dx, dy, ddx, ddy = unpack(best.x)
    nx = int(sels['x'].sum()) * Ms['x'].shape[0]
    return dict(sig0_x=s0x, sig0_y=s0y, T=T, v=GAP / T * 1e3, Dd_x=ddx, Dd_y=float(ddy),
                cost=float(best.cost), rms_x=float(np.sqrt((best.fun[:nx] ** 2).mean())),
                rms_y=float(np.sqrt((best.fun[nx:] ** 2).mean())))


def stacks(ev, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        ks = list(ev)
        ev = {i: ev[ks[j]] for i, j in enumerate(rng.integers(0, len(ks), len(ks)))}
    Ms = {p: stack(ev, p, tan_max=TAN)[1] for p in ('x', 'y')}
    return Ms, {p: np.all(np.isfinite(M), axis=0) for p, M in Ms.items()}


def job(args):
    hv, drc, arm, seed, dd_y_fix = args
    h, _ = template('det3')
    ev = pickle.load(open(f'{CACHES}/headon_{RUNS[hv]}.pkl', 'rb'))
    if SAT:
        ev = {k: e for k, e in ev.items() if max(e['x']['W'].max(), e['y']['W'].max()) < SAT}
    Ms, sels = stacks(ev, seed)
    T0 = GAP / (V0[hv] * 1e-3)
    return (hv, drc, arm, seed), fit_field(h, Ms, sels, T0, drc, arm, dd_y_fix)


# starting v per drift HV (x_width.py's V_SCAN, 0.95 % water composition curve)
V0 = {500: 15.9, 700: 25.1, 900: 32.8, 1000: 35.3, 1100: 37.8}


def main():
    global TAN, SAT
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=20)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--tan', type=float, default=0.03)
    ap.add_argument('--sat', type=float, default=3500.0, help='0 = keep saturated events')
    a = ap.parse_args()
    TAN, SAT = a.tan, a.sat
    out = dict(runs=RUNS, sat_adc=SAT, water_pct=WATER, drc_grid=DRC_GRID, tan_max=TAN, gap_fit=GAP, gap_det3=GAP_DET3)
    with ProcessPoolExecutor(a.workers) as pool:
        # 1. free and iso arms on the D_rc grid (central stacks)
        jobs = [(hv, d, arm, None, None) for hv in RUNS for d in DRC_GRID for arm in ('free', 'iso')]
        res = dict(pool.map(job, jobs))
        # gasY arm: Dd_y pinned at Magboltz, using each field's free-arm v at that D_rc
        mb = {}
        jobs = []
        for hv in RUNS:
            for d in DRC_GRID:
                v = res[(hv, d, 'free', None)]['v']
                mb[(hv, d)] = magboltz_Dd(hv / 2.88, v)
                jobs.append((hv, d, 'gasY', None, mb[(hv, d)][0]))
        res.update(dict(pool.map(job, jobs)))
        prof = {}
        for arm in ('free', 'iso', 'gasY'):
            prof[arm] = [dict(Drc=d, cost=sum(res[(hv, d, arm, None)]['cost'] for hv in RUNS)) for d in DRC_GRID]
            best = min(prof[arm], key=lambda r: r['cost'])
            print(f'{arm:5s} profile: ' + '  '.join(f'{r["Drc"]:.1e}:{r["cost"]:.4e}' for r in prof[arm])
                  + f'   best D_rc {best["Drc"]:.1e}', flush=True)
        out['profile'] = prof
        out['fields'] = {str(hv): {arm: {f'{d:.1e}': res[(hv, d, arm, None)] for d in DRC_GRID}
                                   for arm in ('free', 'iso', 'gasY')} for hv in RUNS}
        out['magboltz'] = {str(hv): {f'{d:.1e}': dict(Dd=mb[(hv, d)][0], DT=mb[(hv, d)][1]) for d in DRC_GRID}
                           for hv in RUNS}
        json.dump(out, open(f'{HERE}/results/xy_isotropy.json', 'w'), indent=1)
        # 2. bootstrap the free arm at its best D_rc and at the neighbours
        dbest = min(prof['free'], key=lambda r: r['cost'])['Drc']
        i = DRC_GRID.index(dbest)
        dset = DRC_GRID[max(i - 1, 0):i + 2]
        jobs = [(hv, d, 'free', 1000 + b, None) for hv in RUNS for d in dset for b in range(a.nboot)]
        bres = dict(pool.map(job, jobs))
    boot = {}
    for d in dset:
        for hv in RUNS:
            bs = [bres[(hv, d, 'free', 1000 + b)] for b in range(a.nboot)]
            c = res[(hv, d, 'free', None)]
            rat = np.array([b['Dd_y'] / b['Dd_x'] if b['Dd_x'] > 0 else np.nan for b in bs])
            row = dict(Dd_x=c['Dd_x'], Dd_y=c['Dd_y'], Dd_x_err=float(np.std([b['Dd_x'] for b in bs])),
                       Dd_y_err=float(np.std([b['Dd_y'] for b in bs])),
                       ratio=c['Dd_y'] / c['Dd_x'] if c['Dd_x'] > 0 else float('nan'),
                       ratio_err=float(np.nanstd(rat)), Dd_mb=mb[(hv, d)][0], v=c['v'],
                       v_err=float(np.std([b['v'] for b in bs])))
            boot.setdefault(f'{d:.1e}', {})[str(hv)] = row
            print(f'D_rc {d:.1e}  {hv:5d} V ({hv / 2.88:4.0f} V/cm): v {row["v"]:5.2f}±{row["v_err"]:.2f}  '
                  f'Dd_x {row["Dd_x"]:.2e}±{row["Dd_x_err"]:.1e}  Dd_y {row["Dd_y"]:.2e}±{row["Dd_y_err"]:.1e}  '
                  f'y/x {row["ratio"]:.2f}±{row["ratio_err"]:.2f}   Magboltz {row["Dd_mb"]:.2e}', flush=True)
    out['bootstrap'] = boot
    out['drc_best_free'] = dbest
    json.dump(out, open(f'{HERE}/results/xy_isotropy.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
