#!/usr/bin/env python3
"""score_h2c.py <label>=<glob> ... [--json OUT] -- score h2c_run.py tables against the reference, with tiers.score.

Per label and view: W, the naive hit ladder converted with the hit v (u_dream / v, as H-naive), and H2c.  Only the
planes common to every label are scored (same event ids), so the rows are directly comparable.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sets', nargs='+')
    ap.add_argument('--family', default='dream')
    ap.add_argument('--json')
    ap.add_argument('--bins', action='store_true', help='also print H2c steep per |tan_ref| bin (tiers.TBINS)')
    a = ap.parse_args()
    T = {}
    for s in a.sets:
        lab, pat = s.split('=', 1)
        T[lab] = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pat))], ignore_index=True)
    keys = None
    for t in T.values():
        k = set(zip(t.event_id, t.view))
        keys = k if keys is None else keys & k
    out = {}
    f = a.family
    for lab, t in T.items():
        t = t[[k in keys for k in zip(t.event_id, t.view)]]
        for v, g in t.groupby('view'):
            tr = g.tan_ref.to_numpy(float)
            row = dict(W=score(g.tan_W.to_numpy(float), tr)['all'],
                       H2c=score(g[f'tan_h2c_{f}'].to_numpy(float), tr)['all'])
            out[f'{lab}/{v}'] = row
            h = row['H2c']
            print(f'{lab:14s} {v}  n {h["n"]:5d}  W s68 {row["W"]["s68"]:.2f}  H2c s68 {h["s68"]:.2f} '
                  f'steep {h["steep"]:+.2f} bias {h["bias"]:+.2f}  in5 {h["frac_in5"]:.3f}  s68_full {h["s68_full"]:.2f}')
            if a.bins:
                full = score(g[f'tan_h2c_{f}'].to_numpy(float), tr)
                row['H2c_bins'] = {k: d for k, d in full.items() if k != 'all'}
                print('    |tan| bin steep/s68: ' + '  '.join(
                    f'{k} {d["steep"]:+.2f}/{d["s68"]:.2f}' for k, d in row['H2c_bins'].items()
                    if d['n'] > 0 and d['steep'] is not None and d['s68'] is not None))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
