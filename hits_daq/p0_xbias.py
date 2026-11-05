#!/usr/bin/env python3
"""p0_xbias.py <boot pickle> <bundle> [--out results/p0_<key>.json] -- where does the X hits bias come from?

The production model has no slow spreading across the resistive strips (rc_D_x = 0), yet hits on X compress the
drift-time ladder as much as on Y (pass 1: H0 implied v 46 vs 36 um/ns on det3).  This asks whether the production
model's same-time mixing (drift diffusion Dp, prompt footprint sigma_p0, the electronics template) is enough to make it.

Input: the residual-bootstrap extraction of one row (rerun/paper1_floor/boot_extract.py): per clean plane the data
window W, the pass model M on it (the event's own fitted p0, w, t0 and depth profile q) and the noise.  The same hit
emulator (V0-like: 5 sigma peak, 30 % leading-edge time, as WaveformAnalyzer.cpp) runs on

  data    W
  model   M as stored, and M regenerated from the bundle (closure: must equal the stored M)
  ablations of the model, each regenerated through wft.model.model_waveforms at the same (p0, w, t0, q):
          Dp=0, sigma_p0=0.07 (induction floor), both, rc_D=0 (Y), 'metallic' (sigma_p0=0.07, rc_D=0, Dp kept) and
          'point' (Dp=0, sigma_p0=0.07, rc_D=0)

Per plane the hit ladder t(x) is fitted (amplitude-weighted least squares); u_h = dx/dt.  The truth is the fit's own w
(the lateral drift per ns of drift time, mm/ns), so ratio = u_h / w (> 1 = compressed ladder = too steep).  Reported as
medians in |w| bins and as the binned-median slope of u_h against w.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

from wft import model as wm  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402

SNS = 60.0


def hits_v0(W, noise, nsig=5.0, frac=0.3):
    """Per strip: (t_le, t_peak, amp) or NaN.  Peak sample >= nsig x noise; time = leading edge at frac x the parabolic
    peak (linear interpolation on the rising side); peak time parabolic."""
    n, ns = W.shape
    out = np.full((n, 3), np.nan)
    for i in range(n):
        y = W[i]
        k = int(np.argmax(y))
        a = y[k]
        if not (a >= nsig * noise[i]) or k == 0 or k == ns - 1:
            continue
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        d = float(np.clip(d, -0.5, 0.5))
        A = y1 - 0.25 * (y0 - y2) * d
        tp = (k + d) * SNS
        thr = frac * A
        j = k
        while j > 0 and y[j - 1] >= thr:
            j -= 1
        if j == 0:
            tle = 0.0
        else:
            tle = (j - 1 + (thr - y[j - 1]) / (y[j] - y[j - 1])) * SNS
        out[i] = tle, tp, A
    return out


def ladder_u(pos, t, a, min_hits=3):
    """Amplitude-weighted LSQ of t against x; returns dx/dt [mm/ns] (NaN if degenerate)."""
    k = np.isfinite(t) & np.isfinite(a)
    if k.sum() < min_hits:
        return np.nan
    x, y, wt = pos[k], t[k], a[k]
    xm = np.average(x, weights=wt)
    ym = np.average(y, weights=wt)
    sxx = np.sum(wt * (x - xm) ** 2)
    if sxx <= 0:
        return np.nan
    s = np.sum(wt * (x - xm) * (y - ym)) / sxx          # ns/mm
    return 1.0 / s if abs(s) > 1e-9 else np.nan


def variants(hyper, view):
    rc = f'rc_D_{view}'
    base = dict(hyper)
    V = {'model': base}
    V['Dp0'] = {**base, 'Dp': 0.0}
    V['s0floor'] = {**base, 'sigma_p0': 0.07, f'sigma_p0_{view}': 0.07}
    V['Dp0_s0floor'] = {**V['Dp0'], 'sigma_p0': 0.07, f'sigma_p0_{view}': 0.07}
    if float(base.get(rc, 0.0)) > 0:
        V['rc0'] = {**base, rc: 1e-12}      # tiny but > 0 keeps the RC kernel path (rc_D_y > 0 enables it)
    # metallic readout: prompt footprint at the induction floor, no resistive spreading, drift diffusion kept (gas physics)
    V['metallic'] = {**V['s0floor'], rc: 1e-12} if float(base.get(rc, 0.0)) > 0 else V['s0floor']
    V['point'] = {**V['Dp0_s0floor'], rc: 1e-12} if float(base.get(rc, 0.0)) > 0 else V['Dp0_s0floor']
    return V


def summarise(w, u, bins):
    r = u / w
    ok = np.isfinite(r) & (np.abs(w) > 0)
    rows = []
    aw = np.abs(w)
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = ok & (aw >= lo) & (aw < hi)
        rows.append(dict(lo=lo, hi=hi, n=int(k.sum()), ratio_med=float(np.median(r[k])) if k.sum() > 10 else None))
    # binned-median slope of u against w over the inclined range (the ladder compression as one number)
    k = ok & (aw >= bins[1])
    q = np.quantile(w[k], np.linspace(0, 1, 21)) if k.sum() > 50 else None
    slope = None
    if q is not None:
        xm, ym = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            kk = k & (w >= lo) & (w < hi)
            if kk.sum() > 5:
                xm.append(np.median(w[kk]))
                ym.append(np.median(u[kk]))
        slope = float(np.polyfit(xm, ym, 1)[0])
    return dict(bins=rows, slope=slope, n=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickle')
    ap.add_argument('bundle')
    ap.add_argument('--out')
    ap.add_argument('--nmax', type=int, default=0)
    a = ap.parse_args()
    D = pickle.load(open(a.pickle, 'rb'))
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    hyper = dict(cal.hyper)
    v = D['meta']['v_drift'] / 1000.0                      # mm/ns
    P = D['planes'][:a.nmax] if a.nmax else D['planes']
    # |w| bins: w = v tan  ->  tan 0.05, 0.1, 0.2, 0.3, 0.45, 0.7
    bins = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.7]) * v
    res = dict(meta=D['meta'], hyper={k: hyper[k] for k in ('sigma_p0', 'Dp', 'rc_D_y', 'rc_D_x') if k in hyper},
               views={})
    for view in ('x', 'y'):
        L = [p for p in P if p['view'] == view]
        V = variants(hyper, view)
        w = np.array([p['w'] for p in L])
        U = {k: np.full(len(L), np.nan) for k in ['data_le', 'data_pk', 'stored_le'] + [f'{k}_le' for k in V]
             + [f'{k}_pk' for k in V]}
        clos = []
        for i, p in enumerate(L):
            pos, nz = p['pos'].astype(float), p['noise'].astype(float)
            h = hits_v0(p['W'].astype(float), nz)
            U['data_le'][i] = ladder_u(pos, h[:, 0], h[:, 2])
            U['data_pk'][i] = ladder_u(pos, h[:, 1], h[:, 2])
            h = hits_v0(p['M'].astype(float), nz)
            U['stored_le'][i] = ladder_u(pos, h[:, 0], h[:, 2])
            q = p['q'].astype(float)
            for k, hy in V.items():
                M = wm.model_waveforms(view, pos, p['p0'], p['w'], p['t0'], q, hy)
                if k == 'model':
                    clos.append(float(np.max(np.abs(M - p['M'])) / max(1.0, float(np.max(np.abs(p['M']))))))
                h = hits_v0(M, nz)
                U[f'{k}_le'][i] = ladder_u(pos, h[:, 0], h[:, 2])
                U[f'{k}_pk'][i] = ladder_u(pos, h[:, 1], h[:, 2])
        res['views'][view] = dict(n=len(L), closure_max_rel=float(np.max(clos)) if clos else None,
                                  closure_med_rel=float(np.median(clos)) if clos else None,
                                  arms={k: summarise(w, u, bins) for k, u in U.items()})
        print(f'[{view}] n={len(L)} closure max rel {res["views"][view]["closure_max_rel"]:.2e}')
        for k, s in res['views'][view]['arms'].items():
            meds = ' '.join(f'{b["ratio_med"]:.2f}' if b['ratio_med'] else '  - ' for b in s['bins'])
            print(f'   {k:16s} slope {s["slope"] if s["slope"] is None else round(s["slope"], 3)!s:>6}  ratio/bin {meds}')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
