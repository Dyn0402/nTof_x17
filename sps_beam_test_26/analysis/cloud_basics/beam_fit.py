#!/usr/bin/env python3
"""beam_fit.py -- bench_attach's physics on det4 run_71 head-on (unbiased stacks).

h = det4's bench X template (same DREAM shaping, 180 ns), T = 30 mm / v with v
from the run_62/63 span ladders (14.4 / 12.5 / 11.6 um/ns at 243 / 150 / 92 V/cm,
the lower two window-floored), footprint sigma_0^2 + 2 Dd u (+ 2 D_rc s on Y).
Fitted: sigma_0, Dd, D_rc, per-view amplitude and offset; lambda (charge loss)
pinned at infinity or free.  Magboltz wet (1.7 % H2O) Dd for comparison."""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack, GRID
from bench_attach import predict, template

V = {'sps700': 14.4, 'sps450': 12.5, 'sps275': 11.6}
h, _ = template('det4')
out = {}
for name, v in V.items():
    T = 30.0 / (v * 1e-3)
    ev = pickle.load(open(SOURCES[name], 'rb'))
    Ms = {p: stack(ev, p)[1] for p in ('x', 'y')}
    sels = {p: np.all(np.isfinite(M), axis=0) for p, M in Ms.items()}
    n = len(GRID)
    out[name] = {}
    for lab in ('uniform', 'loss_free'):
        def res(q):
            sig0, Dd, Drc, lam, ax, ay, dx, dy = q
            r = []
            for p, A, dt in (('x', ax, dx), ('y', ay, dy)):
                P = predict(h, sig0, Dd, Drc if p == 'y' else 0.0, lam, T, A, dt, n)
                r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
            return np.concatenate(r)
        lo = [0.02, 0, 0, 1e6 - 2 if lab == 'uniform' else 100, 1e-5, 1e-5, -600, -600]
        hi = [1.5, 0.003, 0.01, 1e6, 10, 10, 600, 600]
        best = None
        for d0 in (100.0, 250.0, 400.0):
            x0 = [0.4, 5e-5, 2e-4, 1e6 - 1 if lab == 'uniform' else 2000, 0.01, 0.01, d0, d0]
            r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3,
                              x_scale=[0.1, 2e-5, 1e-4, 500, 0.005, 0.005, 30, 30])
            if best is None or r.cost < best.cost:
                best = r
        r = best
        J = r.jac; cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - 8, 1)
        e = np.sqrt(np.diag(cov))
        sig0, Dd, Drc, lam = r.x[:4]
        DT = np.sqrt(2 * Dd / (v * 1e-3)) * np.sqrt(10) * 1e3
        out[name][lab] = dict(sig0=sig0, sig0_err=e[0], Dd=Dd, Dd_err=e[1], DT_um_rtcm=DT, Drc=Drc,
                              lam_ns=lam, rms=float(np.sqrt((r.fun ** 2).mean())))
        print(f'{name} {lab:9s}: sig0 {sig0:.3f}±{e[0]:.3f}  Dd {Dd:.2e}±{e[1]:.1e} (D_T {DT:.0f} um/rtcm)  '
              f'Drc {Drc:.2e}  lambda {lam:.0f} ns ({lam * v * 1e-3:.0f} mm)  rms {out[name][lab]["rms"]:.4f}', flush=True)
json.dump(out, open(f'{HERE}/results/beam_fit.json', 'w'), indent=1, default=float)
