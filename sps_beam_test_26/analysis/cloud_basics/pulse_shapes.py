#!/usr/bin/env python3
"""pulse_shapes.py -- head-on summed waveform (+-2 strips around the centre),
20 %-trimmed mean over events, aligned on the 50 % rise, bench vs beam, with
the bundle templates for comparison.  figures/pulse_shapes.png"""
import os, pickle, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from width_vs_time import SOURCES, SAMPLE_NS
GRID = np.arange(-400.0, 3600.0, 30.0)
HERE = os.path.dirname(os.path.abspath(__file__))


def mean_pulse(ev, plane, tan_max=0.03, strips=2):
    rows = []
    for e in ev.values():
        if plane not in e or abs(e[f'tan_{plane}']) > tan_max:
            continue
        W = np.asarray(e[plane]['W'], float)
        ic = int(np.argmax(W.sum(1)))
        if ic < strips or ic > len(W) - strips - 1:
            continue
        s = W[ic - strips:ic + strips + 1].sum(0)
        c = W[ic]
        ipk = int(np.argmax(s)); half = 0.5 * s[ipk]
        k = next((k for k in range(ipk, 0, -1) if s[k - 1] < half <= s[k]), None)
        if k is None or s[ipk] <= 0:
            continue
        t = (np.arange(len(s)) - (k - 1 + (half - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
        rows.append((np.interp(GRID, t, s / s[ipk], left=np.nan, right=np.nan),
                     np.interp(GRID, t, c / s[ipk], left=np.nan, right=np.nan)))
    R = np.array(rows)
    out = []
    for i in range(2):
        m = np.full(len(GRID), np.nan)
        for j in range(len(GRID)):
            col = R[:, i, j]; col = col[np.isfinite(col)]
            if len(col) > 0.5 * len(R):
                m[j] = trim_mean(col, 0.2)
        out.append(m)
    return len(R), out


def main():
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, plane in zip(axs, 'xy'):
        for name in ('det3', 'det4', 'det7', 'sps700', 'sps275'):
            ev = pickle.load(open(SOURCES[name], 'rb'))
            n, (s, c) = mean_pulse(ev, plane)
            res[f'{name}_{plane}'] = dict(n=n, t=GRID.tolist(), sum=np.nan_to_num(s).tolist())
            ax.plot(GRID, s, lw=1.2, ls='-' if name.startswith('det') else '--', label=f'{name} (n={n})')
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_title(f'{plane.upper()} view, head-on, sum of 5 strips')
        ax.set_xlabel('t − t50 [ns]')
    axs[0].set_ylabel('normalised'); axs[0].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(f'{HERE}/figures', exist_ok=True)
    fig.savefig(f'{HERE}/figures/pulse_shapes.png', dpi=130)
    json.dump(res, open(f'{HERE}/results/pulse_shapes.json', 'w'))
    for k, v in res.items():
        s = np.array(v['sum']); t = GRID
        print(k, ' '.join(f'{tt:.0f}:{np.interp(tt, t, s):+.2f}' for tt in (0, 200, 400, 600, 800, 1000, 1400, 1800, 2400, 3000)))


main()
