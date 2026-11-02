#!/usr/bin/env python3
"""ac_line.py -- score n_TOF reconstructions against the beam-off A-C cosmic line (the n_TOF truth; P5 / N-G1).

    ac_line.py --a A.parquet --c C.parquet [--label L] [--sep-max 60] [--out summary.json]

A clean A-C through-goer (one paired track in each arm, the two chamber lines within --sep-max mm of each other) is one
straight line through both mesh crossings, 469 mm apart (ntof_cosmics/insitu_calib.py's truth).  Each chamber's own
fitted direction is compared with that line: the residual angle between them, binned in the line's angle to the
chamber normal.  Positions do not depend on v; the line therefore tests the angle scale (v, kw), the bow, and the
angular acceptance (tan_max) of whatever produced the tables.

Inputs are ntof_reco.py outputs (their ``.candidates.parquet`` siblings are read: the events table cannot hold a
second track).  Geometry: build_tracks.local_and_global with k = 1 (the bundle's v is the working v), the arm
transforms from the run's run_config.json, y up (COORDINATES.md).  The direction sign of a chamber line is ambiguous
(drift-into-gap); the residual is taken against +-L, whichever is closer.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

BINS_DEG = (0, 10, 20, 31, 40, 45, 55, 70)


def tracks(path, arm, tr):
    from sept26_prelim_analysis import build_tracks as BT
    cand = pd.read_parquet(path.replace('.parquet', '.candidates.parquet'))
    if 'tag' not in cand:
        cand['tag'] = 'all'
    df = BT.pair_rows(cand)
    df = BT.local_and_global(df, tr, k=1.0)
    df['arm'] = arm
    return df


def dca(p1, d1, p2, d2):
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(np.einsum('ij,ij->i', p2 - p1, n)) / nn


def score(a_path, c_path, run_config, sep_max=60.0, label=''):
    from ntof_tracking.reco import geometry as G
    trs = G.detector_transforms(json.load(open(run_config)))
    A = tracks(a_path, 'A', trs['mx17_A'])
    C = tracks(c_path, 'C', trs['mx17_C'])
    out = dict(label=label, n_tracks_A=int(len(A)), n_tracks_C=int(len(C)),
               n_events_A=int(A.event_id.nunique()), n_events_C=int(C.event_id.nunique()))
    A1 = A[A.gated].groupby('event_id').filter(lambda g: len(g) == 1)
    C1 = C[C.gated].groupby('event_id').filter(lambda g: len(g) == 1)
    M = A1.merge(C1, on='event_id', suffixes=('_a', '_c'))
    P = lambda s: M[[f'p0_x_{s}', f'p0_y_{s}', f'p0_z_{s}']].to_numpy(float)
    D = lambda s: M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)
    pa, pc, da, dc = P('a'), P('c'), D('a'), D('c')
    L = pc - pa
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    M['sep_mm'] = dca(pa, da, pc, dc)
    out['n_ac_events'] = int(len(M))
    keep = M['sep_mm'].to_numpy() < sep_max
    out['n_ac_line'] = int(keep.sum())
    rows = []
    for s, d, nrm in (('a', da, np.array([0., 0., 1.])), ('c', dc, np.array([0., 0., -1.]))):
        cosr = np.abs(np.einsum('ij,ij->i', d, L)).clip(0, 1)
        res = np.degrees(np.arccos(cosr))
        inc = np.degrees(np.arccos(np.abs(L @ nrm).clip(0, 1)))
        for lo, hi in zip(BINS_DEG[:-1], BINS_DEG[1:]):
            m = keep & (inc >= lo) & (inc < hi) & np.isfinite(res)
            r = res[m]
            rows.append(dict(arm=s.upper(), inc_lo=lo, inc_hi=hi, n=int(m.sum()),
                             res_med=float(np.median(r)) if len(r) else np.nan,
                             res_68=float(np.percentile(r, 68)) if len(r) else np.nan,
                             tail_gt5=float((r > 5).mean()) if len(r) else np.nan))
    out['bins'] = rows
    return out, M[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True)
    ap.add_argument('--c', required=True)
    ap.add_argument('--run-config', required=True)
    ap.add_argument('--sep-max', type=float, default=60.0)
    ap.add_argument('--label', default='')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    s, _ = score(a.a, a.c, a.run_config, a.sep_max, a.label)
    print(json.dumps({k: v for k, v in s.items() if k != 'bins'}))
    print(pd.DataFrame(s['bins']).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    if a.out:
        json.dump(s, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
