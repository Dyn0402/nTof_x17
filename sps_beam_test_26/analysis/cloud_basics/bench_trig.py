#!/usr/bin/env python3
"""bench_trig.py -- the bench no-attachment control, without data-driven alignment.

§10/§11 concluded "no attachment on the bench" from threshold-aligned stacks,
which §14 showed distort the early pulse.  Here each event is placed by the
TRIGGER: t = sample * 60 ns - t0_abs[view][ftst] (the bundle's per-ftst-class
absolute t0: scintillator trigger -> DREAM clock phase), and the 9-strip sum of
each view is centred on the M3 prediction, not on the pulse.  A view enters when
it is head-on itself (|tan_view| < 0.03): every depth then lands on the same
strips, so the wide sum is the drift current alone.

The bench drift (~800 ns) is short against the shaping, so there is no flat
stretch to read a slope from.  Instead the stack is fitted forward:
    A * [tmpl_view (x) box(0, T) * exp(-r t)](t - t0)
with A, t0, T, r free (T free: the effective gap is not exactly 30 mm), tmpl the
bundle's impulse template (brightest strips of inclined tracks -- no attachment
assumption).  r converts to O2 with the bench-gas Magboltz (Ar/iso 95/5 +
0.5 % H2O, 0.05 % O2) at the bundle's v.

    bench_trig.py det3=<big_cache.pkl>:<bundle_dir> [det7=...] ...
Output: results/bench_trig.json
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_NS = 60.0
GAP_MM = 30.0
HALF = int(os.environ.get("BENCH_HALF", 4))   # strips either side; Y needs ~7 to hold its RC spread
GRID = np.arange(-300.0, 1800.0, 20.0)


def stack(ev, view, t0a):
    acc = np.zeros(len(GRID)); cnt = np.zeros(len(GRID)); n = 0
    for e in ev.values():
        if abs(e[f'tan_{view}']) > 0.03 or view not in e:
            continue
        d = e[view]
        k = np.abs(np.asarray(d['pos']) - e[f'ref_mesh_{view}']) <= (HALF + 0.5) * 0.78
        if k.sum() < 2 * HALF:              # window clipped by the edge of the cached strips
            continue
        s = np.asarray(d['W'], float)[k].sum(0)
        t0 = t0a.get(str(int(e[f'ftst_{view}'])))
        if t0 is None:
            continue
        t = np.arange(len(s)) * SAMPLE_NS - t0
        yy = np.interp(GRID, t, s, left=np.nan, right=np.nan)
        ok = np.isfinite(yy)
        acc[ok] += yy[ok]; cnt[ok] += 1; n += 1
    return n, acc / np.maximum(cnt, 1)


def main():
    mb = {p['E_Vcm']: p for p in json.load(open(os.path.join(
        HERE, 'results', 'magboltz_bench_w0p5_o0p05.json')))['points']}
    vs = np.array(sorted((p['v_true_um_ns'], p['eta_per_cm'] * p['v_true_um_ns'] * 1e-4)
                         for p in mb.values()))
    out = {'grid': GRID.tolist()}
    for arg in sys.argv[1:]:
        det, rest = arg.split('=', 1)
        cache, bdir = rest.split(':', 1)
        ev = pickle.load(open(cache, 'rb'))
        B = json.load(open(os.path.join(bdir, 'bundle.json')))
        v = float(B['v_drift']); T = GAP_MM * 1e3 / v
        rate500 = float(np.interp(v, vs[:, 0], vs[:, 1]))         # /ns at 0.05 % O2
        out[det] = dict(v=v, T_drift_ns=T, rate_per_ns_500ppm=rate500)
        if 't0_abs' not in B:
            print(f'{det}: bundle {os.path.basename(bdir.rstrip("/"))} has no t0_abs -- skipped')
            out.pop(det); continue
        A = np.load(os.path.join(bdir, 'arrays.npz'))
        for view in ('x', 'y'):
            n, y = stack(ev, view, B['t0_abs'][view])
            y = y - np.median(y[GRID < -100])
            y = y / y.max()
            tg, tm = A['grid'], A['tmpl_x']      # X template for both views (Y's carries the RC undershoot)
            dt = tg[1] - tg[0]

            def model(p, t=GRID):
                amp, t0, T, r = p
                u = np.arange(0.0, T, dt)
                src = np.exp(-r * u)
                src[-1] *= (T - u[-1]) / dt          # partial last bin: keeps the model continuous in T
                resp = np.convolve(src, tm)[:len(u) + len(tm) - 1] * dt
                tt = tg[0] + np.arange(len(resp)) * dt + t0
                return amp * np.interp(t, tt, resp, left=0.0, right=0.0)

            m = (GRID > -200) & (GRID < T + 700)
            f = lambda p: model(p)[m] - y[m]
            fits = {}
            for tag, x0, lb, ub in (('free', [1 / T, 0.0, T, 0.0], [0, -300, 0.5 * T, -3e-3], [1, 300, 1.5 * T, 3e-3]),
                                    ('r=0', [1 / T, 0.0, T], [0, -300, 0.5 * T], [1, 300, 1.5 * T])):
                g = (lambda q: f(list(q) + [0.0])) if tag == 'r=0' else f
                sol = least_squares(g, x0, bounds=(lb, ub), x_scale=[1 / T, 50, 50, 1e-4][:len(x0)])
                fits[tag] = sol
            sol = fits['free']
            J = sol.jac; res = sol.fun
            cov = np.linalg.pinv(J.T @ J) * (res @ res) / max(len(res) - 4, 1)
            r, er = sol.x[3], np.sqrt(cov[3, 3]) * np.sqrt(1)
            out[det][view] = dict(n=n, stack=y.tolist(), amp=float(sol.x[0]), t0=float(sol.x[1]),
                                  T=float(sol.x[2]), rate_per_ns=float(r), rate_err=float(er),
                                  o2_ppm=float(500 * r / rate500), o2_err=float(500 * er / rate500),
                                  rms_free=float(np.sqrt(np.mean(res ** 2))),
                                  rms_r0=float(np.sqrt(np.mean(fits['r=0'].fun ** 2))))
            print(f'{det} {view}: n={n:4d}  v={v:.1f}  T_fit={sol.x[2]:4.0f} ns (30mm/v {T:.0f})  '
                  f'r = {r * 1e4:+.2f} +- {er * 1e4:.2f} e-4/ns -> {500 * r / rate500:+5.0f} +- {500 * er / rate500:.0f} ppm O2   '
                  f'rms free/r=0 {out[det][view]["rms_free"]:.4f}/{out[det][view]["rms_r0"]:.4f}')
    json.dump(out, open(os.path.join(HERE, 'results', f'bench_trig{"" if HALF == 4 else f"_h{HALF}"}.json'), 'w'))


if __name__ == '__main__':
    main()
