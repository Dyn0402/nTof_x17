#!/usr/bin/env python3
"""bench_fit.py -- what water and air does each bench chamber's gas need?

Data: bench_stack.json (all-strip, trigger-placed arriving-current stacks, X view; det4
head-on only, its amplification stripes run across X).
Model (predict.current_field): uniform ionisation over the chamber's measured gap, drift
in a linear field profile E(z) = E0 (1 + k (1/2 - z/G)) with Magboltz v, eta, D_L for the
bench gas (Ar/iso 95/5 + water + air, gasmodel 'bench'), averaged over a Gaussian spread
of gap lengths (non-flat cathode), (x) a shared parametric shaper (x) a Gaussian trigger
jitter.  Per chamber: composition (water, air) scanned; field gradient k and gap spread
profiled; amplitude, t0 and jitter fitted.  Shared over chambers: the shaper.

Fields: 700 V <-> 243 V/cm (600 V 208, 1000 V 347).  Gaps: GAP_STUDY (det2 30.6,
det3 27.9, det7 27.5 mm); det4's own is unusable -> 30.

    bench_fit.py [--source air_hs] [--shaper beam|free]
Output: results/bench_fit_<view>_<shaper>.json
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

CH = {'det2': (347.0, 30.6), 'det3': (347.0, 27.9), 'det4': (208.0, 30.0), 'det7': (243.0, 27.5)}
WIN = (-200.0, 1500.0)
SG = (0.0, 0.75, 1.5, 2.25, 3.0)                 # gap spread [mm]
KS = (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4)   # field gradient
BEAM_SHAPER = (8.886, 40.73, 0.062, 2438.0)
_CUR = {}


def cur(G, comp, E0, gap, k, sg):
    key = (comp, E0, gap, k, sg)
    if key not in _CUR:
        if len(_CUR) > 8000:
            _CUR.clear()
        lk = lambda E: G(comp[0], comp[1], E)
        if sg <= 0:
            _CUR[key] = P.current_field(lk, E0, gap, k)
        else:
            ks = np.linspace(-2, 2, 7); wk = np.exp(-0.5 * ks ** 2); wk /= wk.sum()
            _CUR[key] = sum(w * P.current_field(lk, E0, gap + x * sg, k) for x, w in zip(ks, wk))
    return _CUR[key]


def jitter(s, sj):
    if sj <= 1:
        return s
    kk = np.arange(-4 * sj, 4 * sj + P.DT, P.DT)
    g = np.exp(-0.5 * (kk / sj) ** 2); g /= g.sum()
    return np.convolve(s, g, mode='same')


def fit_curve(grid, y, e, I, par):
    """amp, t0, jitter for a given current; returns (q, chi2, n, model curve)."""
    m = (grid >= WIN[0]) & (grid <= WIN[1]) & np.isfinite(y)
    s, off = P.shaped(I, par)
    a0 = 1.0 / max(s.max(), 1e-12)
    mod = lambda q, t: q[0] * np.interp(t, P.FINE + off + q[1], jitter(s, q[2]))
    sol = least_squares(lambda q: (mod(q, grid[m]) - y[m]) / e[m], [a0, 0.0, 30.0],
                        bounds=([0, -400, 0], [np.inf, 400, 300]), x_scale=[0.1 * a0, 20, 20])
    return sol.x, float(np.sum(sol.fun ** 2)), int(m.sum()), mod(sol.x, grid)


def profile_geom(G, grid, y, e, comp, E0, gap, par):
    best = None
    for k in KS:
        for sg in SG:
            c = fit_curve(grid, y, e, cur(G, comp, E0, gap, k, sg), par)[1]
            if best is None or c < best[0]:
                best = (c, k, sg)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='air')
    ap.add_argument('--view', default='x')
    ap.add_argument('--shaper', default='free', choices=('free', 'beam'))
    a = ap.parse_args()
    G = M.GasGrid('bench', a.source)
    S = json.load(open(os.path.join(HERE, 'results', 'bench_stack.json')))
    grid = np.array(S['grid'])
    data = {d: (np.array(S[d][a.view]['stack']), np.maximum(np.array(S[d][a.view]['band']), 0.004))
            for d in CH if d in S}
    ws = [float(x) for x in np.round(np.linspace(G.W.min(), G.W.max(), 21), 4)]
    as_ = [float(x) for x in np.round(np.linspace(0.0, min(0.04, G.A.max()), 9), 4)]
    par = np.array(BEAM_SHAPER)
    best = {d: dict(w=0.5, a=0.0, k=0.0, sg=0.0) for d in data}
    for it in range(3 if a.shaper == 'free' else 2):
        for d, (y, e) in data.items():
            E, gap = CH[d]
            b = best[d]
            b['k'], b['sg'] = profile_geom(G, grid, y, e, (b['w'], b['a']), E, gap, par)
            chi = np.full((len(ws), len(as_)), np.inf)
            for i, w in enumerate(ws):
                for j, x in enumerate(as_):
                    chi[i, j] = fit_curve(grid, y, e, cur(G, (w, x), E, gap, b['k'], b['sg']), par)[1]
            i, j = np.unravel_index(np.argmin(chi), chi.shape)
            b.update(w=ws[i], a=as_[j], chi=chi)
        if a.shaper == 'free':
            def fsh(p):
                if p[0] < 0.5 or p[1] < 5 or p[3] < 50:
                    return np.full(400, 1e3)
                out = []
                for d, (y, e) in data.items():
                    E, gap = CH[d]; b = best[d]
                    q, _, _, mc = fit_curve(grid, y, e, cur(G, (b['w'], b['a']), E, gap, b['k'], b['sg']), p)
                    m = (grid >= WIN[0]) & (grid <= WIN[1]) & np.isfinite(y)
                    out.append((mc[m] - y[m]) / e[m])
                return np.concatenate(out)
            sol = least_squares(fsh, par, x_scale=[0.3, 10, 0.01, 100], diff_step=[1e-2] * 4)
            par = sol.x
        print(f'iteration {it}: shaper n {par[0]:.2f} tau {par[1]:.1f} u {par[2]:.3f} tu {par[3]:.0f}  ' +
              ' '.join(f'{d}:w{b["w"]:.2f}/a{b["a"]:.3f}/k{b["k"]:+.1f}/sg{b["sg"]:.1f}' for d, b in best.items()))
    out = {'shaper': par.tolist(), 'shaper_mode': a.shaper, 'source': a.source, 'view': a.view,
           'grid_w': ws, 'grid_a': as_}
    for d, (y, e) in data.items():
        E, gap = CH[d]; b = best[d]; chi = b['chi']
        ok = chi <= chi.min() + 2.30
        gp = G(b['w'], b['a'], E)
        q, c, n, mc = fit_curve(grid, y, e, cur(G, (b['w'], b['a']), E, gap, b['k'], b['sg']), par)
        c0 = fit_curve(grid, y, e, cur(G, (b['w'], 0.0), E, gap, b['k'], b['sg']), par)[1]
        cflat = fit_curve(grid, y, e, cur(G, (b['w'], b['a']), E, gap, 0.0, b['sg']), par)[1]
        noloss = fit_curve(grid, y, e, cur(G, (b['w'], 0.0), E, gap, b['k'], b['sg']), par)[3]
        out[d] = dict(E=E, gap=gap, water=b['w'], air=b['a'], o2_ppm=b['a'] * 2095, k=b['k'], gap_spread=b['sg'],
                      water_68=[min(np.array(ws)[ok.any(1)]), max(np.array(ws)[ok.any(1)])],
                      air_68=[min(np.array(as_)[ok.any(0)]), max(np.array(as_)[ok.any(0)])],
                      v=gp['v'], etav=gp['etav'], chi2=c, n=n, chi2_noair=c0, chi2_k0=cflat,
                      amp=q[0], t0=q[1], jitter=q[2], curve=mc.tolist(), curve_noair=noloss.tolist(),
                      chi=chi.tolist())
        print(f'{d}: E {E:.0f} gap {gap}: water {b["w"]:.2f} % {out[d]["water_68"]}, air {b["a"]:.3f} % '
              f'{out[d]["air_68"]} ({b["a"] * 2095:.0f} ppm O2), v(E0) {gp["v"]:.1f}, k {b["k"]:+.1f}, '
              f'gap spread {b["sg"]:.1f} mm, jitter {q[2]:.0f} ns | chi2 {c:.0f}/{n} (no air {c0:.0f}, k=0 {cflat:.0f})')
    json.dump(out, open(os.path.join(HERE, 'results', f'bench_fit_{a.view}_{a.shaper}.json'), 'w'))


if __name__ == '__main__':
    main()
