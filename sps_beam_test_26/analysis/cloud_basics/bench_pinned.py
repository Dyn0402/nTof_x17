#!/usr/bin/env python3
"""bench_pinned.py -- bench_joint with drift diffusion pinned from Magboltz and
the box length pinned to gap / v, so sigma_0 is the only free footprint term.

Water fraction per chamber: the one at which Magboltz (Ar/iso 95/5 + H2O, 250
V/cm, Saclay pressure) reproduces the bundle's measured v; D_T interpolated at
that fraction.  2 Dd = D_T^2 v.  Also run with Dd = dry and Dd = 0 to show the
sensitivity of sigma_0 to the diffusion assumption.

    bench_pinned.py [--gap 30]
"""
import argparse, json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack
from bench_joint import predict

V_BUNDLE = {'det2': 39.94, 'det3': 36.6, 'det4': 34.16, 'det6': 26.7, 'det7': 36.6}
E_BENCH = 250.0


def magboltz_curve():
    rows = []
    for tag, w in (('bench_dry', 0), ('bench_w0p25', .25), ('bench_w0p5', .5), ('bench_w1', 1), ('bench_w2', 2)):
        f = f'{HERE}/results/magboltz_{tag}.json'
        if not os.path.exists(f):
            continue
        P = json.load(open(f))['points']
        E = np.array([p['E_Vcm'] for p in P])
        v = np.interp(E_BENCH, E, [p['v_um_ns'] * 10 for p in P])     # file stores cm/ns*1e3
        dt = np.interp(E_BENCH, E, [p['DT_um_rtcm'] for p in P])
        rows.append((w, v, dt))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gap', type=float, default=30.0)
    a = ap.parse_args()
    mb = magboltz_curve()
    print('Magboltz bench @250 V/cm: water %, v, D_T:', [tuple(np.round(r, 1)) for r in mb])
    out = {}
    for name, v in V_BUNDLE.items():
        o = np.argsort(mb[:, 1])
        water = float(np.interp(v, mb[o, 1], mb[o, 0]))
        DT = float(np.interp(water, mb[:, 0], mb[:, 2]))                  # um/sqrt(cm)
        DTmm = DT * 1e-3 / np.sqrt(10.0)                                    # mm/sqrt(mm)
        Dd_mb = 0.5 * DTmm ** 2 * v * 1e-3                                  # mm^2/ns
        Dd_dry = 0.5 * (mb[0, 2] * 1e-3 / np.sqrt(10)) ** 2 * v * 1e-3
        T = a.gap / (v * 1e-3)
        ev = pickle.load(open(SOURCES[name], 'rb'))
        Ms = {p: stack(ev, p)[1] for p in ('x', 'y')}
        Ss, sels = {}, {}
        for p, M in Ms.items():
            ok = np.all(np.isfinite(M), axis=0)
            S = np.nan_to_num(M).sum(0); S[~ok] = 0
            Ss[p], sels[p] = S, ok
        out[name] = dict(v=v, water_pct=water, DT_um_rtcm=DT, T=T)
        for lab, Dd in (('magboltz', Dd_mb), ('dry', Dd_dry), ('none', 0.0)):
            def res(q):
                sig0, Drc, dx, dy = q
                r = []
                for p, dt in (('x', dx), ('y', dy)):
                    P = predict(Ss[p], sig0, Dd, Drc if p == 'y' else 0.0, T, dt)
                    r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
                return np.concatenate(r)
            r = least_squares(res, [0.35, 4e-4, -60, -60], bounds=([0.02, 0, -250, -250], [1.5, 0.01, 250, 250]),
                              x_scale=[0.1, 1e-4, 30, 30], diff_step=1e-3)
            J = r.jac
            cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - 4, 1)
            e = np.sqrt(np.diag(cov))
            out[name][lab] = dict(Dd=Dd, sig0=r.x[0], sig0_err=e[0], Drc=r.x[1], Drc_err=e[1],
                                  dt=r.x[2:].tolist(), rms=float(np.sqrt((r.fun ** 2).mean())))
            print(f'{name} water~{water:.2f}% D_T {DT:.0f} | Dd={lab:8s} {Dd:.2e}: sig0 {r.x[0]:.3f}±{e[0]:.3f} '
                  f'Drc {r.x[1]:.2e} dt {r.x[2]:+.0f}/{r.x[3]:+.0f} rms {out[name][lab]["rms"]:.4f}', flush=True)
    json.dump(out, open(f'{HERE}/results/bench_pinned.json', 'w'), indent=1)


main()
