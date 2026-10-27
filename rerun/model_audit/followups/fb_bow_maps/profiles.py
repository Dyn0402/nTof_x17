#!/usr/bin/env python3
"""profiles.py [names...] -- 1-D profiles across each chamber, along each view's own axis (9 bins, 30-370 mm):
  h(p)    mesh height from positions: slope of the residual p0 - ref_mesh against tan_ref (Huber)          [mm]
  w0(p)   transverse-speed offset of a vertical track: intercept of w against tan_ref (Huber)              [um/ns]
  b(p)    ladder slope dw/dtan (local v x angle scale)                                                       [um/ns]
  w0_pred(p) = b * r_f * dh/dp from a parabola fitted to h(p), for r_f = 0.5 (flat cathode) and 1 (parallel bow)
Errors: bootstrap (30).  Writes results/profiles_<name>.json.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import DATASETS, RES, load, view_mask, huber

EDGES = np.linspace(30, 370, 10)


def run(name):
    T = load(name)
    v = T.attrs['v']
    rng = np.random.default_rng(2)
    out = dict(name=name, det=T.attrs['det'], slot=T.attrs['slot'], note=T.attrs['note'])
    for p in 'xy':
        m = view_mask(T, p)
        t = T[f'{p}_t'].to_numpy(); r = T[f'{p}_r'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
        pos = (T.u if p == 'x' else T.w).to_numpy()
        ma = m & np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            k = m & (pos >= lo) & (pos < hi)
            ka = ma & (pos >= lo) & (pos < hi)
            if k.sum() < 50 or ka.sum() < 50:
                rows.append([0.5 * (lo + hi)] + [np.nan] * 6 + [int(k.sum())]); continue
            X = np.c_[np.ones(k.sum()), t[k]]; Xa = np.c_[np.ones(ka.sum()), t[ka]]
            h = huber(X, r[k], 0.5)[1]
            hb = [huber(X[q], r[k][q], 0.5)[1] for q in (rng.integers(0, k.sum(), k.sum()) for _ in range(30))]
            a = huber(Xa, wv[ka], 0.03 * v)
            ab = np.array([huber(Xa[q], wv[ka][q], 0.03 * v) for q in (rng.integers(0, ka.sum(), ka.sum()) for _ in range(30))])
            rows.append([0.5 * (lo + hi), float(h), float(np.std(hb)), float(a[0]), float(ab[:, 0].std()), float(a[1]),
                         float(ab[:, 1].std()), int(k.sum())])
        R = np.array(rows, float)
        g = np.isfinite(R[:, 1])
        if g.sum() < 4 or np.isfinite(R[:, 3]).sum() < 4:
            out[p] = None; print(name, p, 'too few bins'); continue
        c2 = np.polyfit(R[g, 0] - 200, R[g, 1], 2, w=1 / np.maximum(R[g, 2], 0.05))
        bmed = float(np.nanmedian(R[:, 5]))
        dhdp = 2 * c2[0] * (R[:, 0] - 200) + c2[1]                   # mm/mm
        pred = {f'{rf}': (bmed * rf * dhdp).tolist() for rf in (0.5, 1.0)}
        # measured w0 slope vs predicted: fit lines
        ga = np.isfinite(R[:, 3])
        sw = np.polyfit(R[ga, 0] - 200, R[ga, 3], 1, w=1 / np.maximum(R[ga, 4], 0.02))[0]
        out[p] = dict(pos=R[:, 0].tolist(), h=R[:, 1].tolist(), h_err=R[:, 2].tolist(), w0=R[:, 3].tolist(),
                      w0_err=R[:, 4].tolist(), b=R[:, 5].tolist(), b_err=R[:, 6].tolist(), n=R[:, 7].tolist(),
                      h_curv=float(2 * c2[0]), h_par=c2.tolist(), b_med=bmed, w0_pred=pred, w0_slope=float(sw),
                      r_f=float(sw / (bmed * 2 * c2[0])) if c2[0] != 0 else np.nan)
        print(f'{name:14s} {p}: h\'\' {2 * c2[0] * 1e6:+6.0f} /km  dw0/dp {sw * 1e3:+.2f} um/ns per m  r_f {out[p]["r_f"]:.2f}', flush=True)
    json.dump(out, open(f'{RES}/profiles_{name}.json', 'w'), indent=1)


if __name__ == '__main__':
    for n in (sys.argv[1:] or list(DATASETS)):
        run(n)
