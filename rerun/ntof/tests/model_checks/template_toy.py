#!/usr/bin/env python3
"""template_toy.py -- what a template mismatch does to angles and positions (model check 1, systematic translation).

    template_toy.py --bundle B --out J.json [--tail-scale 0.1 --tail-from 300] [--stretch 1.05] [--n 60] [--workers 3]

Truth: the pass model (the bundle) with a MODIFIED X template -- the tail beyond ``tail_from`` ns after the template
peak scaled by (1 + tail_scale), and/or the time axis stretched by ``stretch`` -- driving uniform full-gap columns
(T = gap / v) at |tan| = 0, 0.1, 0.2, 0.4, 0.6, p0 mid-plate, amplitude ~1000 ADC peak, Gaussian noise 10 ADC, t0 = 0.
Fit: wft.model.fit_plane_raw with the UNMODIFIED bundle, started at the truth (the question is the bias at the
minimum, not the search).  Reported per |tan|: median and 68 % of (fitted - true) tan and p0 (at mid-column), with
the same toy on the unmodified template as the null (estimator floor).  --truth-dp D: the truth carries Dp = D instead
(the fit keeps the bundle's): what a wrong Dp does.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
TANS = (0.0, 0.1, 0.2, 0.4, 0.6)
GAPS = {'mx17_3': 28.08, 'mx17_2': 31.12, 'mx17_6': 28.0, 'mx17_7': 29.44}
_C = {}


def _init(bundle, tail_scale, tail_from, stretch, ns, truth_dp=None):
    from wft.calib import CalibrationBundle
    from wft import model as wm
    cal = CalibrationBundle.load(bundle)
    g, h = np.asarray(cal.grid, float), {p: np.asarray(cal.tmpl[p], float) for p in 'xy'}
    mod = {}
    for p in 'xy':
        t = h[p].copy()
        if stretch != 1.0:
            t = np.interp(g / stretch, g, h[p]) / stretch * 1.0
            t *= h[p].max() / t.max()
        tp = g[np.argmax(t)]
        f = np.where(g > tp + tail_from, 1.0 + tail_scale, 1.0)
        mod[p] = t * f
    _C.update(cal=cal, g=g, h=h, mod=mod, ns=ns, dp_fit=float(cal.hyper['Dp']), dp_truth=truth_dp)


def _model(plane, pos, p0, w, t0, q, tmpl, truth=False):
    from wft import model as wm
    cal = _C['cal']
    cal.tmpl = dict(cal.tmpl, **{plane: tmpl})
    if truth and _C.get('dp_truth'):
        cal.hyper = dict(cal.hyper, Dp=_C['dp_truth'])
    wm.use_calibration(cal)
    cal.hyper = dict(cal.hyper, Dp=_C['dp_fit'])
    wm.set_nsamp(_C['ns'])
    M = wm.model_waveforms(plane, pos, p0, w, t0, q, wm.HYPER)
    return M


def one(args):
    from wft import model as wm
    tan, seed, which = args
    rng = np.random.default_rng(seed)
    cal = _C['cal']
    plane = 'x'
    v = cal.v_drift
    gap = GAPS.get(cal.detector, 28.0)
    T = gap / (v * 1e-3)
    w = tan * v * 1e-3
    p0 = 200.0 + rng.uniform(-0.39, 0.39)
    ext = abs(w) * T
    pos = np.arange(200.0 - 5 - (ext if w < 0 else 0), 200.0 + 5 + (ext if w > 0 else 0), 0.78)
    K = int(cal.n_depth_bins)
    nb = int(round(T / 60.0))
    q = np.zeros(K)
    q[:nb] = 1.0
    q[nb - 1] = T / 60.0 - (nb - 1) if T / 60.0 > nb - 1 else 1.0
    tm = _C['mod'] if which == 'mod' else _C['h']
    M = _model(plane, pos, p0, w, 0.0, q, tm[plane], truth=(which == 'mod'))
    M *= 1000.0 / M.max()
    noise = 10.0
    W = M + rng.normal(0, noise, M.shape)
    P = dict(W=W.astype(np.float32), pos=pos.astype(np.float32), noise=np.full(len(pos), noise, np.float32),
             ch=np.arange(len(pos)) % 512)
    cal.tmpl = dict(cal.tmpl, x=_C['h']['x'], y=_C['h']['y'])
    wm.use_calibration(cal)
    wm.set_nsamp(_C['ns'])
    try:
        r = wm.fit_plane_raw(P, plane, p0, w, 0.0)
    except Exception:
        return None
    if r is None or not np.isfinite(r.get('chi2', np.inf)):
        return None
    mid = 0.5 * T
    return dict(tan=tan, which=which, dtan=(r['w'] - w) / (v * 1e-3), dp0=(r['p0'] + r['w'] * mid) - (p0 + w * mid),
                dt0=float(r['t0']), chi2=float(r['chi2']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tail-scale', type=float, default=0.0)
    ap.add_argument('--tail-from', type=float, default=300.0)
    ap.add_argument('--stretch', type=float, default=1.0)
    ap.add_argument('--ns', type=int, default=20)
    ap.add_argument('--truth-dp', type=float, default=None, help="truth Dp (the 'mod' arm), fit with the bundle's")
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    jobs = [(t, 1000 * i + s, wh) for i, t in enumerate(TANS) for s in range(a.n) for wh in ('null', 'mod')]
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(a.bundle, a.tail_scale, a.tail_from, a.stretch, a.ns, a.truth_dp)) as pool:
        rows = [r for r in pool.map(one, jobs, chunksize=4) if r]
    out = dict(bundle=a.bundle, truth_dp=a.truth_dp, tail_scale=a.tail_scale, tail_from=a.tail_from, stretch=a.stretch, ns=a.ns, bins=[])
    for t in TANS:
        b = dict(tan=t)
        for wh in ('null', 'mod'):
            R = [r for r in rows if r['tan'] == t and r['which'] == wh]
            dt = np.array([r['dtan'] for r in R]); dp = np.array([r['dp0'] for r in R])
            b[wh] = dict(n=len(R), dtan_med=float(np.median(dt)), dtan_68=float(np.percentile(np.abs(dt - np.median(dt)), 68)),
                         dp0_med=float(np.median(dp)), dp0_68=float(np.percentile(np.abs(dp - np.median(dp)), 68)),
                         dtan_med_err=float(1.2533 * np.std(dt) / np.sqrt(max(len(dt), 1))),
                         dp0_med_err=float(1.2533 * np.std(dp) / np.sqrt(max(len(dp), 1))))
        b['bias_tan'] = b['mod']['dtan_med'] - b['null']['dtan_med']
        b['bias_p0'] = b['mod']['dp0_med'] - b['null']['dp0_med']
        b['bias_deg'] = float(np.degrees(np.arctan(t + b['bias_tan']) - np.arctan(t)))
        out['bins'].append(b)
        print(f"tan {t:.1f}: bias tan {b['bias_tan']:+.4f} ({b['bias_deg']:+.3f} deg, rel "
              f"{b['bias_tan'] / t if t else float('nan'):+.3%})  p0 {b['bias_p0']:+.3f} mm   "
              f"[null dtan {b['null']['dtan_med']:+.4f} dp0 {b['null']['dp0_med']:+.3f}]", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
