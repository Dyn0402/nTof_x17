#!/usr/bin/env python3
"""driftscan_fit.py -- one bench gas, one chamber, six drift fields: what composition?

Data: bench_driftscan.json (det3, 6-27, drift 100-1100 V = 35-382 V/cm, all-strip
trigger-placed arriving-current stacks, X view).  Everything physical is SHARED across
the fields: the composition (water, air) of the bench gas, the drift-field gradient k
(geometry: a fixed relative profile at every voltage), the gap spread (det3's dished
cathode, nominal gap 27.9 mm), the electronics shaper and the trigger jitter.  Free per
field: amplitude and t0.  A per-time loss, a per-depth loss and a field gradient each
leave a different signature across the six fields.

    driftscan_fit.py [--source air_hs] [--emin 60]
Output: results/driftscan_fit_<source>.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gasmodel as M                                         # noqa: E402
import predict as P                                          # noqa: E402
from bench_fit import jitter                                  # noqa: E402

GAP = 27.9
WIN = (-200.0, 1600.0)
KS = (-0.2, -0.1, 0.0, 0.1, 0.2)
SG = (0.0, 0.75, 1.5, 2.25)
_C = {}


def cur(G, comp, E, k, sg):
    key = (comp, E, k, sg)
    if key not in _C:
        lk = lambda e: G(comp[0], comp[1], e)
        if sg <= 0:
            _C[key] = P.current_field(lk, E, GAP, k)
        else:
            xs = np.linspace(-2, 2, 7); wk = np.exp(-0.5 * xs ** 2); wk /= wk.sum()
            _C[key] = sum(w * P.current_field(lk, E, GAP + x * sg, k) for x, w in zip(xs, wk))
    return _C[key]


def fit_field(grid, y, e, I, par, sj):
    m = (grid >= WIN[0]) & (grid <= WIN[1]) & np.isfinite(y)
    s, off = P.shaped(I, par); s = jitter(s, sj)
    a0 = 1.0 / max(s.max(), 1e-12)
    mod = lambda q, t: q[0] * np.interp(t, P.FINE + off + q[1], s)
    sol = least_squares(lambda q: (mod(q, grid[m]) - y[m]) / e[m], [a0, 0.0], x_scale=[0.1 * a0, 20])
    return float(np.sum(sol.fun ** 2)), int(m.sum()), mod(sol.x, grid), sol.x


def total(G, data, comp, k, sg, par, sj):
    c = 0.0; n = 0
    for E, (grid, y, e) in data.items():
        ci, ni, _, _ = fit_field(grid, y, e, cur(G, comp, E, k, sg), par, sj)
        c += ci; n += ni
    return c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='air')
    ap.add_argument('--emin', type=float, default=60.0)
    ap.add_argument('--view', default='x')
    a = ap.parse_args()
    G = M.GasGrid('bench', a.source)
    D = json.load(open(os.path.join(HERE, 'results', 'bench_driftscan.json')))
    grid = np.array(D['grid'])
    data = {}
    for V, E in zip(D['volts'], D['E_Vcm']):
        if E < a.emin or E < G.E.min() - 1.0:          # 34.7 V/cm sits on the 35 V/cm node
            continue
        r = D[str(V)][a.view]
        data[float(E)] = (grid, np.array(r['stack']), np.maximum(np.nan_to_num(np.array(r['band']), nan=1.0), 0.004))
    print('fields used:', list(data))
    par = np.array([7.6, 49.6, 0.10, 1200.0]); sj = 25.0
    comp = (0.9, 0.0); k = 0.0; sg = 1.5
    ws = [float(x) for x in np.round(np.linspace(G.W.min(), G.W.max(), 21), 4)]
    as_ = [float(x) for x in np.round(np.linspace(0.0, min(0.04, G.A.max()), 9), 4)]
    for it in range(3):
        # shaper + jitter at the current physics
        def fsh(p):
            if p[0] < 0.5 or p[1] < 5 or p[3] < 50 or p[4] < 0:
                return np.full(600, 1e3)
            out = []
            for E, (gr, y, e) in data.items():
                _, _, mc, _ = fit_field(gr, y, e, cur(G, comp, E, k, sg), p[:4], p[4])
                m = (gr >= WIN[0]) & (gr <= WIN[1]) & np.isfinite(y)
                out.append((mc[m] - y[m]) / e[m])
            return np.concatenate(out)
        sol = least_squares(fsh, list(par) + [sj], x_scale=[0.3, 10, 0.01, 100, 10], diff_step=[1e-2] * 5)
        par, sj = sol.x[:4], sol.x[4]
        # geometry at the current composition
        best = min(((total(G, data, comp, kk, ss, par, sj)[0], kk, ss) for kk in KS for ss in SG))
        k, sg = best[1], best[2]
        # composition scan
        chi = np.array([[total(G, data, (w, x), k, sg, par, sj)[0] for x in as_] for w in ws])
        i, j = np.unravel_index(np.argmin(chi), chi.shape)
        comp = (ws[i], as_[j])
        print(f'iteration {it}: shaper n {par[0]:.2f} tau {par[1]:.1f} u {par[2]:.3f} tu {par[3]:.0f} jitter {sj:.0f} | '
              f'k {k:+.1f} gap spread {sg:.2f} | water {comp[0]:.2f} % air {comp[1]:.3f} % | chi2 {chi.min():.0f}')
    ok = chi <= chi.min() + 2.30
    out = dict(source=a.source, view=a.view, fields=list(data), shaper=list(par), jitter=sj, k=k, gap_spread=sg,
               water=comp[0], air=comp[1], o2_ppm=comp[1] * 2095,
               water_68=[min(np.array(ws)[ok.any(1)]), max(np.array(ws)[ok.any(1)])],
               air_68=[min(np.array(as_)[ok.any(0)]), max(np.array(as_)[ok.any(0)])],
               grid_w=ws, grid_a=as_, chi=chi.tolist(), per_field={})
    for E, (gr, y, e) in data.items():
        c, n, mc, q = fit_field(gr, y, e, cur(G, comp, E, k, sg), par, sj)
        c0, _, m0, _ = fit_field(gr, y, e, cur(G, (comp[0], 0.0), E, k, sg), par, sj)
        gp = G(comp[0], comp[1], E)
        out['per_field'][str(E)] = dict(chi2=c, n=n, chi2_noair=c0, v=gp['v'], etav=gp['etav'],
                                        curve=mc.tolist(), curve_noair=m0.tolist())
        print(f'  {E:6.1f} V/cm: v {gp["v"]:5.1f} um/ns  eta*v {gp["etav"] * 1e4:.3f}e-4/ns  chi2 {c:.0f}/{n} (no air {c0:.0f})')
    print(f'best: water {comp[0]:.2f} % {out["water_68"]}, air {comp[1]:.3f} % {out["air_68"]} '
          f'({comp[1] * 2095:.0f} ppm O2), k {k:+.1f}, gap spread {sg:.2f} mm')
    json.dump(out, open(os.path.join(HERE, 'results', f'driftscan_fit_{a.source}_{a.view}.json'), 'w'))


if __name__ == '__main__':
    main()
