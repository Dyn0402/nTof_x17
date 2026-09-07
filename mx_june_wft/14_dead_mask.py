#!/usr/bin/env python3
"""
14_dead_mask.py — per-channel dead mask from signal RATE (T1.3).

A broken connection downstream of the preamp still shows a normal pedestal,
so dead channels must be found by rate, not pedestal (F1; same logic as
`find_dead_strips()` in beam_track_finding.py). Here the rate is measured on
the bench cache's candidate windows: for every channel, the fraction of
window rows in which its per-strip peak clears 5x its noise. A channel that
keeps appearing inside track windows but essentially never fires is dead —
its neighbours triggered the window, it read baseline.

Writes the mask JSON (ready for the bundle's ``dead`` field) and, when strips
are found, patches them into a bundle copy is left to the caller — the bench
can inject the mask via a variant's ``model_globals`` first.

    ../.venv/bin/python mx_june_wft/14_dead_mask.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/dead_mask.json (+ .png)
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

FIRE_NSIG = 5.0         # per-strip peak > this x noise counts as a signal
MIN_APPEAR = 100        # need this many window rows to judge a channel
DEAD_FRACTION = 0.05    # dead if rate < this x median active rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = get_config(args.run_key)
    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft',
                                       'bench_cache_ftst.pkl')
    out_dir = args.out or cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    with open(cache, 'rb') as f:
        events = pickle.load(f)['events']

    appear = {'x': defaultdict(int), 'y': defaultdict(int)}
    fired = {'x': defaultdict(int), 'y': defaultdict(int)}
    for ev in events.values():
        for plane in ('x', 'y'):
            for P in (ev['wins'].get(plane) or []):
                W = np.asarray(P['W'], dtype=float)
                ch = np.asarray(P['ch'], dtype=int)
                noise = np.maximum(np.asarray(P['noise'], dtype=float), 3.0)
                hot = W.max(axis=1) > FIRE_NSIG * noise
                for c, h in zip(ch, hot):
                    appear[plane][int(c)] += 1
                    fired[plane][int(c)] += int(h)

    res = dict(fire_nsig=FIRE_NSIG, min_appear=MIN_APPEAR,
               dead_fraction=DEAD_FRACTION, n_events=len(events),
               dead={}, rates={})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, plane in zip(axes, ('x', 'y')):
        chs = np.array(sorted(appear[plane]))
        n = np.array([appear[plane][c] for c in chs], float)
        r = np.array([fired[plane][c] for c in chs]) / n
        judged = n >= MIN_APPEAR
        med = float(np.median(r[judged & (r > 0)])) if judged.any() else np.nan
        dead = chs[judged & (r < DEAD_FRACTION * med)]
        res['dead'][plane] = [int(c) for c in dead]
        res['rates'][plane] = {int(c): dict(rate=float(q), n=int(m))
                               for c, q, m in zip(chs, r, n)}
        print(f'{plane}: {len(chs)} channels seen, {int(judged.sum())} judged '
              f'(>= {MIN_APPEAR} rows), median fire rate {med:.2f}, '
              f'dead: {sorted(int(c) for c in dead) or "none"}')
        ax.scatter(chs[judged], r[judged], s=10, label='judged')
        ax.scatter(chs[~judged], r[~judged], s=10, color='0.7',
                   label=f'< {MIN_APPEAR} rows')
        if np.isfinite(med):
            ax.axhline(DEAD_FRACTION * med, color='crimson', ls='--',
                       label=f'dead threshold ({DEAD_FRACTION:.2f} x median)')
        for c in dead:
            ax.axvline(c, color='crimson', alpha=0.4)
        ax.set_ylabel(f'{plane.upper()} fire rate / window row')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_xlabel('channel')
    fig.suptitle('T1.3 — per-channel signal rate inside candidate windows')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dead_mask.png'), dpi=130)
    with open(os.path.join(out_dir, 'dead_mask.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'wrote {out_dir}/dead_mask.json + .png')


if __name__ == '__main__':
    main()
