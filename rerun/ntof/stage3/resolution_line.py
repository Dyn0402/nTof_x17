#!/usr/bin/env python3
"""resolution_line.py -- per-track angle resolution of chambers A and C on n_TOF pass 1, against the A-C cosmic line.

The pass-1 counterpart of ntof_cosmics/angle_response.py's `resolution` (is2-era table
ntof_cosmics/results/tracking/pooled/resolution.csv, same columns), on the analysis view (v(t) angles, k = 1):

  * sample: run_149 beam-off triggers with exactly one gated track in A and one in C (gated = true |tan| < 0.5,
    t0 cuts none), whose two lines pass within SEP_MM (60, the is2-era primary selection) of each other;
  * truth: the joined line's global slopes j = (p0_C - p0_A) / (z_C - z_A), x and y (469 mm lever arm);
  * residual: (track slope d_x/d_z or d_y/d_z  -  j) * sign(j), in true-tan units.  Pass 1 has no response
    correction (the angles are true tans), so the bias is reported, not removed;
  * per |j| bin: n, bias (median), sigma (1.4826 MAD), tail_frac (|res| > 0.15), tan_err_quoted (median x/y_tan_err),
    pull_mad.

    resolution_line.py [--out rerun/ntof/stage3/resolution/resolution_pass1.csv]
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)

from sept26_prelim_analysis import pass1_tracks as P1   # noqa: E402

RES_EDGES = np.array([0.0, 0.04, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60])
SEP_MM = 60.0
TAIL = 0.15
COLS = ['subrun', 'event_id', 'arm', 'gated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'x_tan_err', 'y_tan_err']


def mad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def dca(pa, da, pc, dc):
    n = np.cross(da, dc)
    nn = np.linalg.norm(n, axis=1)
    return np.abs(np.einsum('ij,ij->i', pc - pa, n)) / np.where(nn > 0, nn, np.nan)


def sample(run=149):
    d = P1.read(COLS, arms=('A', 'C'), runs=[run])
    d = d[d.gated.fillna(False).astype(bool)]
    one = d.groupby(['subrun', 'event_id', 'arm']).p0_x.transform('size') == 1
    d = d[one]
    A, C = d[d.arm == 'A'], d[d.arm == 'C']
    M = A.merge(C, on=['subrun', 'event_id'], suffixes=('_a', '_c'))
    P = lambda s: M[[f'p0_x_{s}', f'p0_y_{s}', f'p0_z_{s}']].to_numpy(float)   # noqa: E731
    D = lambda s: M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)      # noqa: E731
    M['sep_mm'] = dca(P('a'), D('a'), P('c'), D('c'))
    jz = M.p0_z_c - M.p0_z_a
    M['jx'], M['jy'] = (M.p0_x_c - M.p0_x_a) / jz, (M.p0_y_c - M.p0_y_a) / jz
    return M


def resolution(M):
    rows = []
    g = M[M.sep_mm < SEP_MM]
    for arm, s in (('A', 'a'), ('C', 'c')):
        for ax in 'xy':
            j = g[f'j{ax}'].to_numpy()
            slope = (g[f'd_{ax}_{s}'] / g[f'd_z_{s}']).to_numpy()
            res = (slope - j) * np.where(j >= 0, 1.0, -1.0)
            err = g[f'{ax}_tan_err_{s}'].to_numpy()
            for lo, hi in zip(RES_EDGES[:-1], RES_EDGES[1:]):
                m = (np.abs(j) >= lo) & (np.abs(j) < hi) & np.isfinite(res)
                if m.sum() < 20:
                    continue
                rows.append(dict(arm=arm, axis=ax, lo=lo, hi=hi, n=int(m.sum()), bias=float(np.median(res[m])),
                                 sigma=mad(res[m]), tail_frac=float((np.abs(res[m]) > TAIL).mean()),
                                 tan_err_quoted=float(np.median(err[m])), pull_mad=mad(res[m] / err[m])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'resolution', 'resolution_pass1.csv'))
    a = ap.parse_args()
    M = sample()
    R = resolution(M)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    R.to_csv(a.out, index=False)
    json.dump(dict(run='run_149', n_ac_pairs=int(len(M)), n_line=int((M.sep_mm < SEP_MM).sum()), sep_mm=SEP_MM,
                   source='pass-1 analysis view (pass1_tracks, v(t) angles)', tail=TAIL),
              open(a.out.replace('.csv', '.json'), 'w'), indent=1)
    pd.set_option('display.width', 200)
    print(f'{len(M)} single-track A-C pairs, {(M.sep_mm < SEP_MM).sum()} within {SEP_MM:.0f} mm')
    print(R.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
