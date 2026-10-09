#!/usr/bin/env python3
"""bench_joint.py -- head-on bench stacks with drift diffusion as its own term.

Charge drifting for time u (depth v u) lands with sigma^2 = sigma_0^2 +
2 Dd u (drift diffusion; Magboltz: 2 Dd = D_T^2 v) and, on Y only, spreads on
the resistive strip as 2 D_rc s (s = time since landing).  A head-on bench
cosmic deposits uniformly over the gap (the bench pulse is flat-topped: no
attachment), so the arrival current is a box of length T.  With the all-strip
sum S = h (x) box_T, the step response is H(t) = S(t) + H(t - T) and the
impulse h = dH/dt -- the electronics come out of the data, per view.

    W_o(t) = sum_j h(t - t_j) [Q_o(t_j) - Q_o(t_{j-1})],
    Q_o(t) = (1/T) int_0^min(t,T) F_o( sqrt(sigma_0^2 + 2 Dd u + 2 D_rc (t-u)) ) du

Fitted jointly over X and Y of one chamber: sigma_0 (shared), Dd (shared, one
gas), D_rc (Y), T (shared), plus a per-view time offset.  The beam fits
(rc_diffusion.py) have Dd ~ 0 by construction (front-loaded charge).

    bench_joint.py
"""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack, F, GRID, OFFS, STEP


def step_impulse(S, T):
    n = int(round(T / STEP))
    H = S.copy()
    for i in range(n, len(H)):
        H[i] += H[i - n]
    return np.diff(np.r_[0.0, H])


def predict(S, sig0, Dd, Drc, T, t_shift):
    n = len(S)
    h = step_impulse(S, T)
    t = np.arange(n) * STEP
    nu = max(int(round(T / STEP)), 1)
    u = (np.arange(nu) + 0.5) * STEP
    out = []
    # time of the first arrival inside the grid: the box starts where S rises
    i0 = int(np.argmax(S > 0.02 * S.max()))
    for o in OFFS:
        Q = np.zeros(n)
        for j in range(n):
            tj = (j - i0) * STEP
            if tj <= 0:
                continue
            uu = u[u < tj]
            if len(uu) == 0:
                continue
            sig = np.sqrt(sig0 ** 2 + 2 * Dd * uu + 2 * Drc * (tj - uu))
            Q[j] = F(sig, o).sum()          # sum of dQ over the box = nu = T/STEP, so h (x) dQ = S
        w = np.convolve(h, np.diff(np.r_[0.0, Q]))[:n]
        # continuous shift of the charge-arrival origin (interpolated, so the fit has a gradient)
        out.append(np.interp(t - t_shift, t, w, left=0.0, right=w[-1]))
    return np.array(out)


def main():
    out = {}
    for name in ('det2', 'det3', 'det4', 'det6', 'det7'):
        ev = pickle.load(open(SOURCES[name], 'rb'))
        Ms = {p: stack(ev, p)[1] for p in ('x', 'y')}
        Ss, sels = {}, {}
        for p, M in Ms.items():
            ok = np.all(np.isfinite(M), axis=0)
            S = np.nan_to_num(M).sum(0); S[~ok] = 0
            Ss[p], sels[p] = S, ok

        def res(q):
            sig0, Dd, Drc, T, dx, dy = q
            r = []
            for p, dt in (('x', dx), ('y', dy)):
                P = predict(Ss[p], sig0, Dd, Drc if p == 'y' else 0.0, T, dt)
                r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
            return np.concatenate(r)

        best = None
        for T0 in (600.0, 750.0, 900.0):
            r = least_squares(res, [0.35, 3e-4, 3e-4, T0, 0.0, 0.0],
                              bounds=([0.02, 0, 0, 300, -150, -150], [1.5, 0.01, 0.01, 1500, 150, 150]),
                              x_scale=[0.1, 1e-4, 1e-4, 100, 30, 30], diff_step=1e-3)
            if best is None or r.cost < best.cost:
                best = r
        r = best
        J = r.jac
        cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - len(r.x), 1)
        e = np.sqrt(np.diag(cov))
        names = ('sig0', 'Dd', 'Drc', 'T', 'dt_x', 'dt_y')
        out[name] = dict(zip(names, r.x.tolist()), err=dict(zip(names, e.tolist())),
                         rms=float(np.sqrt((r.fun ** 2).mean())))
        q = out[name]
        print(f'{name}: sig0 {q["sig0"]:.3f}±{e[0]:.3f} mm | Dd {q["Dd"]:.2e} (drift sigma at T: '
              f'{np.sqrt(2 * q["Dd"] * q["T"]):.2f} mm) | Drc {q["Drc"]:.2e} | T {q["T"]:.0f} ns | '
              f'rms {q["rms"]:.4f}', flush=True)
    json.dump(out, open(f'{HERE}/results/bench_joint.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
