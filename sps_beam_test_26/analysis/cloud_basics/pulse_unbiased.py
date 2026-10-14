#!/usr/bin/env python3
"""pulse_unbiased.py -- head-on X pulse without the max-normalisation bias.

Per event: sum of the 5 strips around the centre, NOT normalised; aligned on the
first sample crossing a fixed absolute threshold (THR ADC, interpolated); plain
mean over events.  Compared with box(T = 30 mm / v) (x) the bundle X template,
aligned the same way.  If the earlier sag was the alignment/normalisation bias,
this stack is flat-topped."""
import os, pickle, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES, SAMPLE_NS
from bench_attach import BUNDLES, BUNDLE
V = {'det2': 39.94, 'det3': 36.6, 'det4': 34.16, 'det6': 26.7, 'det7': 36.6, 'sps700': 14.0}
GRID = np.arange(-300.0, 3300.0, 30.0)
THR = 60.0
out = {}
for name in ('det2', 'det3', 'det4', 'det6', 'det7', 'sps700', 'sps275'):
    ev = pickle.load(open(SOURCES[name], 'rb'))
    acc = np.zeros(len(GRID)); cnt = np.zeros(len(GRID)); n = 0
    for e in ev.values():
        if abs(e['tan_x']) > 0.03:
            continue
        W = np.asarray(e['x']['W'], float)
        ic = int(np.argmax(W.sum(1)))
        if ic < 2 or ic > len(W) - 3:
            continue
        s = W[ic - 2:ic + 3].sum(0)
        k = next((k for k in range(1, len(s)) if s[k - 1] < THR <= s[k]), None)
        if k is None or k < 3:
            continue
        t = (np.arange(len(s)) - (k - 1 + (THR - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
        y = np.interp(GRID, t, s, left=np.nan, right=np.nan)
        ok = np.isfinite(y); acc[ok] += y[ok]; cnt[ok] += 1; n += 1
    m = np.where(cnt > 0.6 * n, acc / np.maximum(cnt, 1), np.nan)
    pk = np.nanmax(m)
    line = f'{name:7s} n={n:4d} data: ' + ' '.join(f'{x:.2f}' for x in (m / pk)[(GRID % 150 == 0) & (GRID >= 0) & (GRID <= 1500)])
    print(line)
    if name.startswith('det'):
        z = np.load(f'{BUNDLES}/{BUNDLE[name]}/arrays.npz'); g, h = z['grid'], z['tmpl_x']
        dt = g[1] - g[0]; T = 30000 / V[name]
        p = np.convolve(h, np.ones(int(T / dt)))
        tt = g[0] + np.arange(len(p)) * dt
        p = p / p.max(); k = np.argmax(p >= THR / pk)   # same absolute threshold relative to the peak
        tt -= tt[k]
        print(f'{"":7s}        box*h: ' + ' '.join(f'{np.interp(x, tt, p):.2f}' for x in GRID[(GRID % 150 == 0) & (GRID >= 0) & (GRID <= 1500)]))
    out[name] = dict(n=n, t=GRID.tolist(), mean=np.nan_to_num(m).tolist())
print('t columns:', GRID[(GRID % 150 == 0) & (GRID >= 0) & (GRID <= 1500)])
json.dump(out, open(f'{HERE}/results/pulse_unbiased.json', 'w'))
