#!/usr/bin/env python3
"""
rc_line_step3.py — measure the +-1 neighbour transfer function directly.

On near-vertical tracks the charge sits on 1-2 strips, so the +-1 neighbour
of a bright strip carries (mostly) the shared copy. Average those neighbour
waveforms aligned to the source strip's t50 and compare with the model's
assumption: c1*kY x (template delayed tau_s, smeared sigma_s). A shape
mismatch here mis-times shared charge on every fit and is the leading
suspect for the residual Y slope scale (kw = 0.967).

    ../../.venv/bin/python mx_june_wft/bench/rc_line_step3.py sat_det3
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

TAN_MAX = 0.06
AMP_MIN = 600.0
AMP_MAX = 3400.0
GRID = np.arange(-360, 1400, 10.0)


def t50(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    import pickle
    cfg = get_config(args.run_key)
    W = os.path.join(cfg.OUT_BASE, 'wft')
    cal = CalibrationBundle.load(os.path.join(W, 'calib_bundle'))
    with open(os.path.join(W, 'bench_cache.pkl'), 'rb') as f:
        events = pickle.load(f)['events']

    acc = {'x': {'src': [], 'nb': []}, 'y': {'src': [], 'nb': []}}
    for ev in events.values():
        t = ev['truth']
        for plane in ('x', 'y'):
            wins = ev['wins'].get(plane)
            if not wins or not np.isfinite(t[f'tan_{plane}']) \
                    or abs(t[f'tan_{plane}']) > TAN_MAX:
                continue
            P = wins[0]
            Wf = np.asarray(P['W'], np.float32)
            ns = Wf.shape[1]
            amax = Wf.max(axis=1)
            i = int(np.argmax(amax))
            a = amax[i]
            if not (AMP_MIN <= a <= AMP_MAX):
                continue
            if not (0 < i < len(Wf) - 1):
                continue
            ipk = int(np.argmax(Wf[i]))
            if ipk < 6 or ipk > ns - 12:
                continue
            c = t50(Wf[i])
            if not np.isfinite(c):
                continue
            tt = (np.arange(ns) - c) * 60.0
            # the dimmer of the two neighbours carries the smaller direct
            # share (the track sits toward the brighter side)
            j = i - 1 if amax[i - 1] <= amax[i + 1] else i + 1
            if amax[j] > 0.75 * a:
                continue
            acc[plane]['src'].append(np.interp(GRID, tt, Wf[i] / a,
                                               left=np.nan, right=np.nan))
            acc[plane]['nb'].append(np.interp(GRID, tt, Wf[j] / a,
                                              left=np.nan, right=np.nan))

    from scipy.ndimage import gaussian_filter1d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for k, plane in enumerate(('x', 'y')):
        src = np.array(acc[plane]['src'])
        nb = np.array(acc[plane]['nb'])
        if len(src) < 20:
            print(f'{plane}: too few pairs ({len(src)})')
            continue
        ms = np.nanmedian(src, axis=0)
        mn = np.nanmedian(nb, axis=0)
        ms -= np.nanmedian(ms[GRID < -250])
        mn -= np.nanmedian(mn[GRID < -250])
        h = cal.hyper
        kY = h.get('kY', 1.0) if plane == 'y' else 1.0
        c1 = h['c1'] * kY
        tmpl = np.asarray(cal.tmpl[plane], float)
        tmpl = tmpl / tmpl.max()
        sm = gaussian_filter1d(tmpl, max(h['sigma_s'], 1.0) / 10.0)
        model = c1 * np.interp(GRID - h['tau_s'], GRID, sm, left=0, right=0)
        pk_d = GRID[int(np.nanargmax(mn))]
        pk_m = GRID[int(np.nanargmax(model))]
        r_amp = float(np.nanmax(mn))
        print(f'{plane}: n={len(src):4d}  data +-1: peak {r_amp:.3f} @ {pk_d:.0f} ns, '
              f'undershoot {np.nanmin(mn):+.3f} | model: peak {model.max():.3f} '
              f'@ {pk_m:.0f} ns, undershoot {model.min():+.3f}')
        ax = axs[k]
        ax.plot(GRID, ms, 'k', lw=1, label='source strip (median)')
        ax.plot(GRID, mn, 'b', lw=2, label='neighbour, measured')
        ax.plot(GRID, model, 'r--', lw=1.5,
                label=f'model copy (c1={c1:.2f}, tau={h["tau_s"]:.0f})')
        ax.set_xlim(-300, 1400)
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f'{plane}: +-1 transfer, near-vertical tracks (n={len(src)})')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(W, 'rc_line_step3.png')
    fig.savefig(png, dpi=110)
    print('wrote', png)


if __name__ == '__main__':
    main()
