#!/usr/bin/env python3
"""beam_comp_fit.py -- the beam gas composition from the full run_71 stacks (+ run_63 ladder v).

Same physics and machinery as the bench drift scan (driftscan_fit.py): uniform ionisation
over a 30 mm gap, Magboltz v, eta, D_L for Ar/CF4/iso 88/10/2 + water + air
(gasmodel 'beam'), a linear drift-field profile k and a gap spread, (x) a parametric
shaper (x) a Gaussian jitter.  SHARED: composition, k, gap spread, shaper, jitter.  Free
per stack: amplitude and t0.  Data: run_71 RAW head-on, X +-2 and Y +-8, at 243 / 150 /
92 V/cm (headon_masked_k12.json), plus the drift velocity measured on the run_63 ladder
at 142 / 108 / 75 V/cm (3 % error) -- the 150 and 92 V/cm stacks have no drift end in the
window, so on their own they hardly constrain v.

    beam_comp_fit.py [--source air_hs]
Output: results/beam_comp_fit_<source>.json
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

GAP = 30.0
WIN = (500.0, 3800.0)
KS = (-0.1, -0.05, 0.0, 0.05, 0.1)
SG = (0.0, 0.5, 1.0)
LADDER_V = {142.0: 7.77, 108.0: 5.83, 75.0: 4.04}
V_ERR = 0.03
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


def fit_stack(t, y, e, I, par, sj):
    m = (t >= WIN[0]) & (t <= WIN[1])
    s, off = P.shaped(I, par); s = jitter(s, sj)
    a0 = 1.0 / max(s.max(), 1e-12)
    mod = lambda q, tt: q[0] * np.interp(tt, P.FINE + off + q[1], s)
    sol = least_squares(lambda q: (mod(q, t[m]) - y[m]) / e[m], [a0, 650.0], x_scale=[0.1 * a0, 20])
    return float(np.sum(sol.fun ** 2)), int(m.sum()), mod(sol.x, t), sol.x


def chi_v(G, comp):
    return sum((G(comp[0], comp[1], E)['v'] - v) ** 2 / (V_ERR * v) ** 2 for E, v in LADDER_V.items())


def total(G, data, comp, k, sg, par, sj):
    c = chi_v(G, comp)
    for (E, view), (t, y, e) in data.items():
        c += fit_stack(t, y, e, cur(G, comp, E, k, sg), par, sj)[0]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='air')
    a = ap.parse_args()
    G = M.GasGrid('beam', a.source)
    H = json.load(open(os.path.join(HERE, 'results', 'headon_masked_k12.json')))
    t = np.array(H['t'])
    data = {}
    for lab, E in (('raw700', 243.0), ('raw450', 150.0), ('raw275', 92.0)):
        for view, h in (('x', '2'), ('y', '8')):
            S = np.array(H[lab][view]['sum'][h]); y = S / S[(t >= 1080) & (t <= 1260)].mean()
            data[(E, view)] = (t, y, np.maximum(np.array(H[lab][view]['band'][h]), 0.005))
    par = np.array([8.886, 40.73, 0.062, 2438.0]); sj = 1.0
    comp = (1.55, 0.07); k = 0.0; sg = 0.0
    ws = [float(x) for x in np.round(np.linspace(max(1.2, G.W.min()), min(1.9, G.W.max()), 15), 4)]
    as_ = [float(x) for x in np.round(np.linspace(0.0, min(0.15, G.A.max()), 16), 4)]
    for it in range(3):
        def fsh(p):
            if p[0] < 0.5 or p[1] < 5 or p[3] < 50:
                return np.full(700, 1e3)
            out = []
            for (E, view), (tt, y, e) in data.items():
                _, _, mc, _ = fit_stack(tt, y, e, cur(G, comp, E, k, sg), p, sj)
                m = (tt >= WIN[0]) & (tt <= WIN[1])
                out.append((mc[m] - y[m]) / e[m])
            return np.concatenate(out)
        sol = least_squares(fsh, par, x_scale=[0.3, 10, 0.01, 100], diff_step=[1e-2] * 4)
        par = sol.x
        best = min(((total(G, data, comp, kk, ss, par, sj), kk, ss) for kk in KS for ss in SG))
        k, sg = best[1], best[2]
        chi = np.array([[total(G, data, (w, x), k, sg, par, sj) for x in as_] for w in ws])
        i, j = np.unravel_index(np.argmin(chi), chi.shape)
        comp = (ws[i], as_[j])
        print(f'iteration {it}: shaper n {par[0]:.2f} tau {par[1]:.1f} u {par[2]:.3f} tu {par[3]:.0f} | '
              f'k {k:+.2f} gap spread {sg:.1f} | water {comp[0]:.2f} % air {comp[1]:.3f} % | chi2 {chi.min():.0f}')
    ok = chi <= chi.min() + 2.30
    out = dict(source=a.source, shaper=list(par), jitter=sj, k=k, gap_spread=sg, water=comp[0], air=comp[1],
               o2_ppm=comp[1] * 2095, water_68=[min(np.array(ws)[ok.any(1)]), max(np.array(ws)[ok.any(1)])],
               air_68=[min(np.array(as_)[ok.any(0)]), max(np.array(as_)[ok.any(0)])],
               grid_w=ws, grid_a=as_, chi=chi.tolist(), ladder_v={str(E): [v, G(*comp, E)['v']] for E, v in LADDER_V.items()},
               per_stack={})
    for (E, view), (tt, y, e) in data.items():
        c, n, mc, q = fit_stack(tt, y, e, cur(G, comp, E, k, sg), par, sj)
        c0, _, m0, _ = fit_stack(tt, y, e, cur(G, (comp[0], 0.0), E, k, sg), par, sj)
        gp = G(comp[0], comp[1], E)
        out['per_stack'][f'{E}_{view}'] = dict(chi2=c, n=n, chi2_noair=c0, v=gp['v'], etav=gp['etav'], t0=q[1],
                                               curve=mc.tolist(), curve_noair=m0.tolist())
        print(f'  {E:5.0f} V/cm {view}: v {gp["v"]:5.2f}  eta*v {gp["etav"] * 1e4:.2f}e-4/ns  chi2 {c:.0f}/{n} (no air {c0:.0f})  t0 {q[1]:.0f}')
    print('ladder v (meas, model):', {E: (v, round(G(*comp, E)['v'], 2)) for E, v in LADDER_V.items()})
    print(f'best: water {comp[0]:.2f} % {out["water_68"]}, air {comp[1]:.3f} % {out["air_68"]} '
          f'({comp[1] * 2095:.0f} ppm O2), k {k:+.2f}, gap spread {sg:.1f} mm')
    json.dump(out, open(os.path.join(HERE, 'results', f'beam_comp_fit_{a.source}.json'), 'w'))


if __name__ == '__main__':
    main()
