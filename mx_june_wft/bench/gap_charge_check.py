#!/usr/bin/env python3
"""
gap_charge_check.py — does the collected charge follow the gap map?

An independent, timing-free test of the drift-gap topography. A muon crossing a
drift gap of length L deposits primary ionisation proportional to L, so a region
where the cathode really sits 2 mm closer must also collect ~7 % LESS total
charge. A reconstruction artefact that merely truncates the arrival-time column
does not touch the total charge.

For each dataset with a gap study we build, on the same sliding kernel:
  * the endpoint map  gap(x, y)          [mm]
  * the charge map    median qsum(x, y)  [fit units], near-vertical tracks only
and fit  ln(q/q_med) = alpha * ln(gap/gap_med).  alpha ~ 1 means the charge
follows the geometry; alpha ~ 0 means it does not.

    ../../.venv/bin/python mx_june_wft/bench/gap_charge_check.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

sys.path.insert(0, HERE)
from gap_compare import DATASETS, load, grid_map, fit_T   # noqa: E402

TAN_MAX = 0.20          # near-vertical: path length ~ the gap itself
KERNEL_R = 45.0
MIN_EVENTS = 60


def charge_map(d, xs, ys, sel):
    x, y, q = d['x'][sel], d['y'][sel], d['qsum'][sel]
    M = np.full((len(ys), len(xs)), np.nan)
    for j, yc in enumerate(ys):
        dy2 = (y - yc) ** 2
        for i, xc in enumerate(xs):
            s = dy2 + (x - xc) ** 2 < KERNEL_R ** 2
            if s.sum() >= MIN_EVENTS:
                M[j, i] = np.median(q[s])
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'mx_june_wft'))
    args = ap.parse_args()

    rows, panels = [], []
    for label, key, det, note in DATASETS:
        d = load(key, 'x')
        if d is None:
            continue
        d.update(label=label, det=det)
        xs, ys, G, _E, _N = grid_map(d)
        sel = d['tan'] < TAN_MAX
        if sel.sum() < 500:
            print(f'-- {label}: only {sel.sum()} near-vertical tracks, skipped')
            continue
        Q = charge_map(d, xs, ys, sel)
        m = np.isfinite(G) & np.isfinite(Q)
        if m.sum() < 20:
            continue
        lg = np.log(G[m] / np.nanmedian(G[m]))
        lq = np.log(Q[m] / np.nanmedian(Q[m]))
        alpha, b = np.polyfit(lg, lq, 1)
        r = float(np.corrcoef(lg, lq)[0, 1])
        rows.append(dict(dataset=label, det=det, n_grid=int(m.sum()),
                         n_tracks=int(sel.sum()),
                         gap_spread_pct=round(100 * np.std(lg), 1),
                         charge_spread_pct=round(100 * np.std(lq), 1),
                         alpha=round(float(alpha), 2), corr=round(r, 2)))
        panels.append((label, lg, lq, alpha, r))

    tab = pd.DataFrame(rows)
    print('\n== charge vs gap topography (near-vertical tracks, |tan| < '
          f'{TAN_MAX}) ==')
    print('alpha = d ln(charge) / d ln(gap): 1 = charge follows the geometry, '
          '0 = it does not')
    print(tab.to_string(index=False))
    os.makedirs(args.out, exist_ok=True)
    tab.to_csv(os.path.join(args.out, 'gap_charge_check.csv'), index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(panels)
    if not n:
        return
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.6 * nrow),
                            layout='constrained', squeeze=False)
    for k, (label, lg, lq, alpha, r) in enumerate(panels):
        ax = axs[k // ncol][k % ncol]
        ax.plot(100 * lg, 100 * lq, '.', ms=4, alpha=0.5)
        xx = np.linspace(lg.min(), lg.max(), 10)
        ax.plot(100 * xx, 100 * (alpha * xx), 'r-', lw=1.5,
                label=f'slope {alpha:.2f}')
        ax.plot(100 * xx, 100 * xx, 'k--', lw=1, label='charge ~ path length')
        ax.set_xlabel('gap deviation [%]', fontsize=8)
        ax.set_ylabel('charge deviation [%]', fontsize=8)
        ax.set_title(f'{label}  (r = {r:.2f})', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].axis('off')
    out = os.path.join(args.out, 'gap_charge_check.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
