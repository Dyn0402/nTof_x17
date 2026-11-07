#!/usr/bin/env python3
"""resim_score.py '<glob of resim_*.parquet>' [--json OUT] -- score the Arm S re-simulation (resim.py) per variant.

Each row of a resim table is one real plane regenerated through model_waveforms with a modified kernel and exact truth
(w_true, tan_true), then read by every hit family (u_<f>, ur_<f>) and refitted by the waveform fit (w_fit, --refit).
All slopes are dx/dt in mm/ns, so the scores need no reference and no drift velocity:
  ratio   median u / w_true over |tan_true| > 0.1: the ladder compression of that family (> 1 = reads too steep);
          the same for W (w_fit / w_true)
  angle   tan = slope / v with v = median(w_true / tan_true) of the table (the truth's own v), scored against tan_true
          with tiers.score (s68 within 5 deg, steep, per-|tan| bins)
Truth is exact here, so this is the resolution/bias of each readout on the model, not on data (Arm D gives those).
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

FAMS = ('dream', 'peak', 'thr', 'tot', 'fd', 'binary', 'ideal')
CODES = dict(dream='V0', peak='V1', thr='V2', tot='V3', fd='V4', binary='V5', ideal='V6')
ORDER = ('point', 'metal', 'dp0', 's0_07', 's0_15', 'base', 's0_45', 'rc0', 'rc025', 'rc05', 'rc2', 'rc5', 'rc10', 'sheet')


def ratio(s, wt, tt):
    k = np.isfinite(s) & np.isfinite(wt) & (np.abs(tt) > 0.1)
    return float(np.median(s[k] / wt[k])) if k.sum() > 30 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tables')
    ap.add_argument('--json')
    a = ap.parse_args()
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(a.tables))], ignore_index=True)
    out = {}
    for view in ('x', 'y'):
        G = T[T.view == view]
        k = np.abs(G.tan_true) > 0.1
        v = float(np.median(G.w_true[k] / G.tan_true[k]))
        out[view] = dict(v_mm_ns=v, variants={})
        print(f'[{view}] v_true {v * 1e3:.1f} um/ns   ratio = slope / true slope (|tan|>0.1); s68 / steep in deg')
        print(f'  {"variant":8s} {"W ratio":>8s} {"W s68":>6s} | ' + ' | '.join(f'{CODES[f]} ratio s68 steep' for f in ('dream', 'peak', 'ideal')))
        for var in [x for x in ORDER if x in set(G.variant)]:
            g = G[G.variant == var]
            wt, tt = g.w_true.to_numpy(float), g.tan_true.to_numpy(float)
            o = dict(n=len(g), W=dict(ratio=ratio(g.w_fit.to_numpy(float), wt, tt), score=score(g.w_fit.to_numpy(float) / v, tt)),
                     fams={})
            for f in FAMS:
                for suf, col in (('', f'u_{f}'), ('r', f'ur_{f}')):
                    s = g[col].to_numpy(float)
                    o['fams'][f'{CODES[f]}{suf}_{f}'] = dict(ratio=ratio(s, wt, tt), score=score(s / v, tt))
            out[view]['variants'][var] = o
            Ws = o['W']['score']['all']

            def cell(n):
                t = o['fams'][n]
                s = t['score']['all']
                return f'{t["ratio"]:.3f} {s["s68"] or float("nan"):4.2f} {s["steep"] if s["steep"] is not None else float("nan"):+5.2f}'
            print(f'  {var:8s} {o["W"]["ratio"]:8.3f} {Ws["s68"] or float("nan"):6.2f} | '
                  + ' | '.join(cell(f'{CODES[f]}_{f}') for f in ('dream', 'peak', 'ideal')))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
