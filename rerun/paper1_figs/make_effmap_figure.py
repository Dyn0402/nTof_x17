#!/usr/bin/env python3
"""make_effmap_figure.py --root ROOT [--out papers/paper1/figures/efficiency_map.pdf] -- Paper I §5 efficiency maps on
pass 1: the five chambers' plateau pools (rerun/effbudget/pass1 fine_maps.py: same denominator as the loss budget, every
measured M3 ray in the active area; numerator 'detected'; 8 mm disk, 2 mm grid; the pools are in
rerun/effbudget/pass1/results/map_pools.json), with the active-area box and the interior >= 30 mm (x) contour.
ROOT holds budget/<run>/<subrun>/<det>/rays_<det>.parquet (the paper1_eff outputs)."""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
EB = os.path.join(REPO, 'rerun', 'effbudget', 'pass1')
sys.path[:0] = [EB, REPO]
import fine_maps as fm  # noqa: E402
from common.mx17_active_area import TRUE_ACTIVE_BY_DET  # noqa: E402

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=os.path.join(REPO, 'papers', 'paper1', 'figures', 'efficiency_map.pdf'))
    ap.add_argument('--r', type=float, default=8.0)
    a = ap.parse_args()
    T = pd.read_csv(os.path.join(EB, 'results', 'rows.csv'))
    pools = json.load(open(os.path.join(EB, 'results', 'map_pools.json')))
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.9))
    summ = {}
    for ax, d in zip(axs, DETS):
        rows = T[T.key.isin(pools[d]['pool'])]
        R = fm.load(a.root, rows)
        e, m, n = fm.emap(R, a.r)
        im = ax.imshow(100 * m.T, origin='lower', extent=[e[0], e[-1], e[0], e[-1]], vmin=0, vmax=100, cmap='viridis')
        b = TRUE_ACTIVE_BY_DET['mx17_' + d[-1]]
        ax.add_patch(Rectangle((b['x'][0], b['y'][0]), b['x'][1] - b['x'][0], b['y'][1] - b['y'][0], fill=False, ec='w', lw=0.8))
        ax.add_patch(Rectangle((b['x'][0] + 30, b['y'][0] + 30), b['x'][1] - b['x'][0] - 60, b['y'][1] - b['y'][0] - 60,
                               fill=False, ec='w', lw=0.8, ls='--'))
        if d == 'det7':
            ax.axvline(199.3, color='w', ls=':', lw=1); ax.text(100, 8, 'X ch 0-255\n(not quoted)', color='w', ha='center', fontsize=7)
        ax.set_title(f'{d}: {len(rows)} rows, {len(R):,} M3 rays', fontsize=9)
        ax.set_xlabel('x at mid-gap [mm]'); ax.set_ylabel('y at mid-gap [mm]'); ax.set_xlim(-10, 410); ax.set_ylim(-10, 410)
        summ[d] = dict(rows=len(rows), rays=int(len(R)))
    fig.subplots_adjust(wspace=0.35)
    fig.colorbar(im, ax=axs, label='detection efficiency [%]', fraction=0.015, pad=0.01)
    fig.suptitle(f'Detection efficiency, pass 1, {a.r:g} mm disk (solid: active area; dashed: the >= 30 mm interior)', fontsize=10)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches='tight'); fig.savefig(os.path.join(HERE, 'figures', 'efficiency_map.png'), dpi=110, bbox_inches='tight')
    json.dump(summ, open(os.path.join(HERE, 'results', 'efficiency_map.json'), 'w'), indent=1)
    print(summ)


if __name__ == '__main__':
    main()
