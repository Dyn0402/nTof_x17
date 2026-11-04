#!/usr/bin/env python3
"""beam_common.py -- capsule pointing of the SAME beam tracks under two configs (run_145): angle quality, not yield.

    beam_common.py BASE CFG [--dir .../t0pin/beam/merged]

Tracks are matched by (event, x and y position within 2 mm at the mesh). For matched tracks the axis DCA is compared:
the 68 % / median DCA among tracks that point within 25 mm in either config, and the paired difference (bootstrap).
"""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
import capsule_point
RC = '/media/ucla/work/ntof_pass1/runs/run_145/run_config.json'
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('cfg')
ap.add_argument('--dir', default='/media/ucla/work/ntof_pass1/t0pin/beam/merged'); a = ap.parse_args()
rng = np.random.default_rng(2)
for arm in 'AC':
    T = {}
    for lab in (a.base, a.cfg):
        _, t = capsule_point.score(os.path.join(a.dir, f'r145_{arm}_{lab}.parquet'), arm, RC, lab)
        T[lab] = t[['event_id', 'x_p0', 'y_p0', 'dca', 'tanx', 'tany']] if 'x_p0' in t else t
    b, c = T[a.base], T[a.cfg]
    px = [k for k in b.columns if k in ('x_p0', 'p0_x_local', 'x_pos')]
    m = b.merge(c, on='event_id', suffixes=('_b', '_c'))
    # same particle: local positions agree (columns from build_tracks.pair_rows)
    pos = [k for k in ('x_p0', 'y_p0') if f'{k}_b' in m]
    keep = np.ones(len(m), bool)
    for k in pos:
        keep &= np.abs(m[f'{k}_b'] - m[f'{k}_c']) < 2.0
    m = m[keep].drop_duplicates('event_id', keep=False)
    near = (m.dca_b < 25) | (m.dca_c < 25)
    mm = m[near]
    d = (mm.dca_c - mm.dca_b).to_numpy()
    bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
    print(f'{arm} {a.cfg} vs {a.base}: matched {len(m)} (of {len(b)} / {len(c)}), pointing<25 in either {len(mm)}; '
          f'DCA median {np.median(mm.dca_b):.2f} -> {np.median(mm.dca_c):.2f} mm, 68% {np.percentile(mm.dca_b, 68):.2f} -> '
          f'{np.percentile(mm.dca_c, 68):.2f}; paired median diff {np.median(d):+.3f} +- {np.std(bs):.3f}; '
          f'<10 mm {(mm.dca_b < 10).sum()} -> {(mm.dca_c < 10).sum()}; |dtan| med x {np.median(np.abs(m.tanx_c - m.tanx_b)):.4f} y {np.median(np.abs(m.tany_c - m.tany_b)):.4f}')
