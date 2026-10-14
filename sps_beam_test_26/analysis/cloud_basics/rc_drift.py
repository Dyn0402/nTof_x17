#!/usr/bin/env python3
"""rc_drift.py -- the RC-spread fit of rc_diffusion.py with drift diffusion written in.

rc_diffusion.fit gives every arriving electron the same prompt width sigma_0 and lets the
resistive spread grow with the time since arrival s.  Charge from depth z also arrives wider by
drift diffusion, sigma_d^2 = D_T^2 z, and that is not in the fit, so it leaks into sigma_0 and
D_rc by an amount that depends on how much deep charge there is -- i.e. on the gas (D_T, v) AND
on attachment (attach_transfer.py, FINDINGS §27).  Here the stack's source S at time index m is
taken to have arrived at u = max(t_m - t_a, 0) after the first (mesh) charge, so

    sigma^2(m, s) = sigma_0^2 + D_T^2 v u / 10 + 2 D_rc s          (D_T mm/sqrt(cm), v mm/ns)

with D_T and v fixed from Magboltz / the measured drift velocity, and t_a (where the arrival
starts on the stack's threshold-aligned clock) fitted.  Everything else is as rc_diffusion.

    rc_drift.py                       -> results/rc_drift.json   (beam run_71 + bench chambers)
"""
import json
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rc_diffusion as rc                                  # noqa: E402
from width_vs_time import SOURCES                          # noqa: E402
from gasmodel import GasGrid                               # noqa: E402

GRID, STEP, OFFS, PITCH, U = rc.GRID, rc.STEP, rc.OFFS, 0.78, rc.U


def _F(sig, o):
    """rc.F for an array of widths of any shape: fraction above strip o, averaged over U."""
    z = 1.0 / (np.sqrt(2) * np.maximum(sig, 1e-3))[..., None]
    c = o * PITCH - U
    return (0.5 * (erf((c + PITCH / 2) * z) - erf((c - PITCH / 2) * z))).mean(-1)


def predict(p, S, dd2, view):
    """p = (sigma_0, D_rc, t_a); dd2 = D_T^2 v / 10 [mm^2/ns]."""
    sig0, D, ta = p
    n = len(S)
    s = np.arange(n) * STEP
    u = np.maximum(GRID - ta, 0.0)                         # arrival time of the source sample m
    var = sig0 ** 2 + dd2 * u[:, None] + (2 * D * s)[None, :]
    sig = np.sqrt(var)                                     # (m, k)
    out = np.zeros((len(OFFS), n))
    for i, o in enumerate(OFFS):
        Fo = _F(sig, o)                                    # (m, k)
        dF = np.diff(np.c_[np.zeros(n), Fo], axis=1)       # increment at lag k for source m
        # W_o[m + k] += S[m] dF[m, k]
        for m in np.flatnonzero(S != 0):
            out[i, m:] += S[m] * dF[m, :n - m]
    return out


def fit(M, view, dd2, ta0=-150.0):
    ok_t = np.all(np.isfinite(M), axis=0)
    S = np.nan_to_num(M).sum(0)
    S[~ok_t] = 0

    def res(q):
        p = (q[0], q[1] if view == 'y' else 0.0, q[-1])
        return (predict(p, S, dd2, view)[:, ok_t] - M[:, ok_t]).ravel()

    x0 = [0.4, 3e-4, ta0] if view == 'y' else [0.4, ta0]
    lo = [0.02, 0.0, -600.0] if view == 'y' else [0.02, -600.0]
    hi = [1.5, 0.05, 600.0] if view == 'y' else [1.5, 600.0]
    r = least_squares(res, x0, bounds=(lo, hi), x_scale=[0.05, 1e-4, 50.0][:len(x0)] if view == 'y' else [0.05, 50.0])
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - len(r.x), 1)
    e = np.sqrt(np.diag(cov))
    out = dict(sig0=float(r.x[0]), sig0_err=float(e[0]), ta=float(r.x[-1]),
               rms=float(np.sqrt((r.fun ** 2).mean())))
    if view == 'y':
        out.update(D=float(r.x[1]), D_err=float(e[1]))
    return out


# gas per dataset: (grid, water %, air %, field V/cm); bench v from the bundles (bench_pinned.json)
BEAM = {'sps700': 243.0, 'sps450': 150.0, 'sps275': 92.0}


def main():
    gb = GasGrid('beam', 'air_hs')
    gn = GasGrid('bench', 'air_hs')
    pin = json.load(open(os.path.join(HERE, 'results', 'bench_pinned.json')))
    out = {}
    for name, path in SOURCES.items():
        ev = pickle.load(open(path, 'rb'))
        if name in BEAM:
            g = gb(1.51, 0.078, BEAM[name])
            arms = {'magboltz': (g['DT'] * 1e-3) ** 2 * g['v'] * 1e-3 / 10}
        else:
            v = pin[name]['v'] * 1e-3
            # the bench D_T is not settled (FINDINGS §27): v(E) says ~1 % water (D_T ~ 270), the
            # head-on width growth says dry (~480); fit both
            arms = {'wet_DT275': 0.275 ** 2 * v / 10, 'dry_DT480': 0.480 ** 2 * v / 10}
        arms['none'] = 0.0
        out[name] = {}
        for view in ('x', 'y'):
            n, M = rc.stack(ev, view)
            out[name][view] = {'n': n}
            for lab, dd2 in arms.items():
                r = fit(M, view, dd2)
                r['dd2'] = dd2
                out[name][view][lab] = r
                print(f'{name:7s} {view} {lab:10s} sig0 {r["sig0"]:.3f}±{r["sig0_err"]:.3f} '
                      + (f'D_rc {r["D"]:.2e}±{r["D_err"]:.1e} ' if view == 'y' else '')
                      + f't_a {r["ta"]:+.0f} rms {r["rms"]:.4f}', flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'rc_drift.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
