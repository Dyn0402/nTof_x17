#!/usr/bin/env python3
"""air_fit.py -- how much water and how much air does the beam gas need?

Two measured observables per drift field, both from waveforms:
  v(E)   243 V/cm: the drift end of the run_71 head-on stack (make_figures fit,
         T over a 30 mm gap); 142/108/75 V/cm: the run_63 25.64 deg ladder slope
         (median peak time against strip position, v = 1 / (|dt/du| tan theta))
  r(E)   run_71 RAW head-on loss rate per ns (make_figures fit) at 243/150/92 V/cm
Model: Magboltz on a (water, air) grid for Ar/CF4/iso 88/10/2 at 720.8 Torr,
air = N2/O2/Ar 78.08/20.95/0.93 replacing argon (condor 4410646,
magboltz_beam_w*_a*.json).  Bilinear interpolation in (water, air) and linear in
E; chi2 over both observables; the grid minimum and its 1-sigma contour.

Systematics folded in as errors: v 3 % (the 30 mm gap and ladder angle), r 10 %
(template shape: the fit's chi2/ndf is 2.5-6).  Caveat that no error covers:
Magboltz's three-body O2 attachment (O2 + e + M) with H2O or isobutane as M.

    air_fit.py [--gas beam]
Output: results/air_fit_<gas>.json
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FITS = '/home/dylan/x17/cosmic_bench/cloud_basics/attachment/fits.json'
TILT = 25.64
GAP = 30.0
V_SYS, R_SYS = 0.03, 0.10


def measured():
    F = json.load(open(FITS))
    L = json.load(open(os.path.join(RES, 'ladder_profile.json')))
    tz = 1 / np.tan(np.radians(TILT))
    v = {243: (GAP * 1e3 / F['raw700']['T'], None)}
    for arm, E in (('rot_d425', 142), ('rot_d325', 108), ('rot_d225', 75)):
        f = L[arm]['fit']
        rows = [r for r in L[arm]['rows'] if f['u_range'][0] - 1e-6 <= r['u'] <= f['u_range'][1] + 1e-6
                and np.isfinite(r['t_med'])]
        u = np.array([r['u'] for r in rows]); t = np.array([r['t_med'] for r in rows])
        (b, _a), cov = np.polyfit(u, t, 1, cov=True)
        vv = 1e3 * tz / abs(b)
        v[E] = (vv, vv * np.sqrt(cov[0, 0]) / abs(b))
    r = {E: (F[l]['r'], F[l]['r_err']) for l, E in (('raw700', 243), ('raw450', 150), ('raw275', 92))}
    return v, r


def load_grid(gas):
    G = {}
    for p in glob.glob(os.path.join(RES, 'air', f'magboltz_{gas}_w*_a*.json')):
        m = re.search(rf'{gas}_w([0-9p]+)_a([0-9p]+)\.json', p)
        w, a = (float(x.replace('p', '.')) for x in m.groups())
        d = json.load(open(p))
        G[(w, a)] = {pt['E_Vcm']: (pt['v_true_um_ns'], pt['eta_per_cm']) for pt in d['points']}
    return G


def interp(G, W, A, w, a, E):
    """bilinear in (water, air), linear in E; returns (v um/ns, eta*v per ns)."""
    iw = np.clip(np.searchsorted(W, w) - 1, 0, len(W) - 2); ia = np.clip(np.searchsorted(A, a) - 1, 0, len(A) - 2)
    out = []
    for q in (0, 1):
        vals = []
        for ww in W[iw:iw + 2]:
            row = []
            for aa in A[ia:ia + 2]:
                pts = G[(ww, aa)]; Es = np.array(sorted(pts))
                vv = np.interp(E, Es, [pts[e][0] for e in Es])
                et = np.interp(E, Es, [pts[e][1] for e in Es])
                row.append(vv if q == 0 else et * vv * 1e-4)
            vals.append(row)
        vals = np.array(vals)
        fw = (w - W[iw]) / (W[iw + 1] - W[iw]); fa = (a - A[ia]) / (A[ia + 1] - A[ia])
        out.append((1 - fw) * ((1 - fa) * vals[0, 0] + fa * vals[0, 1]) + fw * ((1 - fa) * vals[1, 0] + fa * vals[1, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gas', default='beam')
    a = ap.parse_args()
    v, r = measured()
    G = load_grid(a.gas)
    W = np.array(sorted({k[0] for k in G})); A = np.array(sorted({k[1] for k in G}))
    missing = [(w, x) for w in W for x in A if (w, x) not in G]
    print(f'grid: water {W.tolist()}  air {A.tolist()}  missing {missing}')
    print('measured v:', {E: f'{x[0]:.2f}' for E, x in v.items()}, ' r [e-4/ns]:', {E: f'{x[0] * 1e4:.2f}' for E, x in r.items()})
    ws = np.linspace(W.min(), W.max(), 121); as_ = np.linspace(A.min(), min(A.max(), 0.5), 201)
    chi = np.full((len(ws), len(as_)), np.inf)
    for i, w in enumerate(ws):
        for j, x in enumerate(as_):
            c = 0.0
            for E, (vm, ve) in v.items():
                vs = interp(G, W, A, w, x, E)[0]
                c += (vs - vm) ** 2 / ((ve or 0) ** 2 + (V_SYS * vm) ** 2)
            for E, (rm, re_) in r.items():
                rs = interp(G, W, A, w, x, E)[1]
                c += (rs - rm) ** 2 / (re_ ** 2 + (R_SYS * rm) ** 2)
            chi[i, j] = c
    i, j = np.unravel_index(np.argmin(chi), chi.shape)
    ok = chi <= chi.min() + 2.30                      # 68 % for two parameters
    wr = [float(ws[ok.any(1)].min()), float(ws[ok.any(1)].max())]
    ar = [float(as_[ok.any(0)].min()), float(as_[ok.any(0)].max())]
    best = dict(water_pct=float(ws[i]), air_pct=float(as_[j]), chi2=float(chi.min()), ndf=len(v) + len(r) - 2,
                water_68=wr, air_68=ar, o2_ppm=float(as_[j] * 0.2095 * 1e4), o2_ppm_68=[x * 0.2095 * 1e4 for x in ar])
    print(f'best: water {best["water_pct"]:.2f} % [{wr[0]:.2f}, {wr[1]:.2f}],  air {best["air_pct"]:.3f} % '
          f'[{ar[0]:.3f}, {ar[1]:.3f}] = O2 {best["o2_ppm"]:.0f} ppm;  chi2 {best["chi2"]:.1f} / {best["ndf"]}')
    pred = {E: interp(G, W, A, best['water_pct'], best['air_pct'], E) for E in sorted(set(v) | set(r))}
    for E in sorted(pred):
        print(f'  {E:3d} V/cm: v {pred[E][0]:5.2f} (meas {v[E][0]:.2f})' if E in v else f'  {E:3d} V/cm:',
              f' r {pred[E][1] * 1e4:.2f} (meas {r[E][0] * 1e4:.2f})' if E in r else '')
    json.dump(dict(best=best, measured_v={str(k): x for k, x in v.items()},
                   measured_r={str(k): x for k, x in r.items()},
                   pred={str(E): p for E, p in pred.items()},
                   grid_water=ws.tolist(), grid_air=as_.tolist(), chi2=chi.tolist()),
              open(os.path.join(RES, f'air_fit_{a.gas}.json'), 'w'))


if __name__ == '__main__':
    main()
