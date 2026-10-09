#!/usr/bin/env python3
"""width_vs_time.py -- model-free cluster width against arrival time, head-on.

For a track normal to a view every depth lands at the same strip position, so
the charge arriving at time t came from depth z = v (t - t_mesh) and its
lateral spread is sigma^2(z) = sigma_0^2 + D_T^2 z (prompt cloud + drift
diffusion), plus whatever the resistive layer adds later.  Per time sample:

    m2(t) = sum_events sum_strips W(s,t) (x_s - c)^2  /  sum_events sum_strips W(s,t)

with c the event's time-integrated centroid.  The sums are signed, so
zero-mean noise cancels in the stack; strip discretisation adds pitch^2/12
(Sheppard), subtracted.  Events are aligned on the 50 % rise of the summed
waveform and stacked on a 20 ns grid.  No model, no fit of the kernel.

    width_vs_time.py --out results/width_vs_time.json
"""
import argparse
import json
import os
import pickle

import numpy as np

A = '/media/dylan/data/x17/cosmic_bench/Analysis'
SPS = '/media/dylan/data/x17/sps_run53_det4_check/plane_ratio'
SOURCES = {
    'det2': f'{A}/mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/wft/plane_ratio/big_cache_3000.pkl',
    'det3': f'{A}/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/plane_ratio/big_cache_3000.pkl',
    'det4': f'{A}/mx17_det4_day_6-24-26/long_run/mx17_4/wft/plane_ratio/big_cache_3000.pkl',
    'det6': f'{A}/mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/wft/plane_ratio/big_cache_3000.pkl',
    'det7': f'{A}/mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/wft/plane_ratio/big_cache_3000.pkl',
    'sps700': f'{SPS}/sps_run71_raw700.pkl',
    'sps450': f'{SPS}/sps_run71_raw450.pkl',
    'sps275': f'{SPS}/sps_run71_raw275.pkl',
}
SAMPLE_NS = 60.0
PITCH = 0.78
GRID = np.arange(-400.0, 2000.0, 20.0)
HALF = 4            # strips either side of the centroid strip used in the moments


def stack(events, plane, tan_max):
    num = np.zeros(len(GRID))
    den = np.zeros(len(GRID))
    n = 0
    for ev in events.values():
        if plane not in ev or abs(ev[f'tan_{plane}']) > tan_max:
            continue
        P = ev[plane]
        W = np.asarray(P['W'], float)
        pos = np.asarray(P['pos'], float)
        q = W.sum(axis=1)
        ic = int(np.argmax(q))
        lo, hi = ic - HALF, ic + HALF + 1
        if lo < 0 or hi > len(pos):
            continue
        Ww, xw = W[lo:hi], pos[lo:hi]
        Q = Ww.sum()
        if Q <= 0:
            continue
        c = float((Ww.sum(axis=1) * xw).sum() / Q)
        s = Ww.sum(axis=0)
        ipk = int(np.argmax(s))
        half = 0.5 * s[ipk]
        k = next((k for k in range(ipk, 0, -1) if s[k - 1] < half <= s[k]), None)
        if k is None:
            continue
        t50 = (k - 1 + (half - s[k - 1]) / (s[k] - s[k - 1])) * SAMPLE_NS
        t = np.arange(W.shape[1]) * SAMPLE_NS - t50
        m2 = (Ww * (xw[:, None] - c) ** 2).sum(axis=0)
        num += np.interp(GRID, t, m2, left=0, right=0)
        den += np.interp(GRID, t, s, left=0, right=0)
        n += 1
    return n, num, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tan-max', type=float, default=0.03)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'results', 'width_vs_time.json'))
    a = ap.parse_args()
    out = {}
    for name, path in SOURCES.items():
        with open(path, 'rb') as f:
            ev = pickle.load(f)
        out[name] = {}
        for plane in ('x', 'y'):
            n, num, den = stack(ev, plane, a.tan_max)
            frac = den / den.max()
            ok = frac > 0.05
            var = np.where(ok, num / np.where(ok, den, 1) - PITCH ** 2 / 12, np.nan)
            out[name][plane] = dict(n=n, t=GRID.tolist(), charge=frac.tolist(),
                                    var=[None if not np.isfinite(v) else float(v) for v in var])
            sel = (GRID >= -150) & (GRID <= 600) & ok
            print(f'{name:7s} {plane} n={n:4d}  sigma(t) [mm] at t50-100/0/+200/+400/+600: ' +
                  ' '.join(f'{np.sqrt(max(np.interp(tt, GRID[ok], var[ok]), 0)):.3f}'
                           for tt in (-100, 0, 200, 400, 600)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
