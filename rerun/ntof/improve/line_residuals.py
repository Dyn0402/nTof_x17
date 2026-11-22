#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
line_residuals.py -- per-chamber angle and position residuals against the A-C cosmic line, over every beam-off
sub-run of n_TOF pass 1 (not just run_149), mapped over the chamber and against track quality.

The truth is external to each chamber: a muon that crosses A and C defines the joined line (469 mm lever arm) from
the two positions; each chamber's own angle is compared with it.  This is the resolution_line.py sample (one
gated track in A and one in C, lines within SEP_MM), on all beam-off sub-runs (registry class cosmic; until 2026-11-22
this was "t_since_flash null", which took in 14,742 production events, tracking audit finding 13).  The lines are
built on the UNMAPPED slopes (the `_nomap` twins of the analysis view): the angle map is fitted on them.

    python -m rerun.ntof.improve.line_residuals build     -> <out>/lines.parquet
    python -m rerun.ntof.improve.line_residuals maps      -> <out>/line_bias_*.csv  (bias vs true tan, vs position)

Residual convention: res_x = (d_x/d_z - jx), in true-tan units; signed by the global frame (not by sign(j)), so a
scale error shows up as a slope of res against j and a local field distortion as structure against position.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from sept26_prelim_analysis import pass1_tracks as P1  # noqa: E402
from rerun.ntof.improve.track_quality import out_dir  # noqa: E402

SEP_MM = 60.0
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'gated', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
        'x_local', 'y_local', 'tanx', 'tany', 'x_tan_err', 'y_tan_err', 'chi2dof_x', 'chi2dof_y', 'x_n_strips',
        'y_n_strips', 'x_dchi2', 'y_dchi2', 'x_q_sum', 'y_q_sum', 'x_t0', 'y_t0', 'drift_len_mm', 'drift_railed',
        'x_n_dropped', 'y_n_dropped', 'n_cand_x', 'n_cand_y', 't_since_flash_ns', 'x_p0', 'y_p0',
        'x_w', 'y_w', 'x_tan_theta', 'y_tan_theta', 'tanx_frozen', 'tany_frozen']
KEY = ['run', 'subrun', 'tag', 'event_id']


def dca(pa, da, pc, dc):
    n = np.cross(da, dc)
    nn = np.linalg.norm(n, axis=1)
    return np.abs(np.einsum('ij,ij->i', pc - pa, n)) / np.where(nn > 0, nn, np.nan)


def build() -> pd.DataFrame:
    names = P1.campaign_schema().names
    cols = COLS + [c + '_nomap' for c in COLS if c + '_nomap' in names] + \
        [c for c in ('censored_window',) if c in names]
    d = P1.nomap(P1.read(cols, arms=('A', 'C'), filter=P1.cosmic_filter()))
    d = d[d.gated.fillna(False).astype(bool)]
    one = d.groupby(KEY + ['arm']).p0_x.transform('size') == 1
    d = d[one]
    A, C = d[d.arm == 'A'].drop(columns='arm'), d[d.arm == 'C'].drop(columns='arm')
    M = A.merge(C, on=KEY, suffixes=('_a', '_c'))
    P = lambda s: M[[f'p0_x_{s}', f'p0_y_{s}', f'p0_z_{s}']].to_numpy(float)   # noqa: E731
    D = lambda s: M[[f'd_x_{s}', f'd_y_{s}', f'd_z_{s}']].to_numpy(float)      # noqa: E731
    M['sep_mm'] = dca(P('a'), D('a'), P('c'), D('c'))
    jz = M.p0_z_c - M.p0_z_a
    M['jx'], M['jy'] = (M.p0_x_c - M.p0_x_a) / jz, (M.p0_y_c - M.p0_y_a) / jz
    for s in 'ac':
        M[f'res_x_{s}'] = M[f'd_x_{s}'] / M[f'd_z_{s}'] - M.jx
        M[f'res_y_{s}'] = M[f'd_y_{s}'] / M[f'd_z_{s}'] - M.jy
    M = M[M.sep_mm < SEP_MM]
    M.to_parquet(out_dir() / 'lines.parquet', index=False)
    print(len(M), 'A-C lines from', M.run.nunique(), 'beam-off runs')
    return M


def mad(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) > 20 else np.nan


def stat(g, col):
    r = g[col].to_numpy(float)
    return dict(n=len(r), bias=float(np.nanmedian(r)) if len(r) > 20 else np.nan, sigma=mad(r),
                tail=float(np.mean(np.abs(r) > 0.15)) if len(r) else np.nan)


def maps():
    M = pd.read_parquet(out_dir() / 'lines.parquet')
    out = out_dir()
    rows = []
    jb = np.array([-0.5, -0.4, -0.3, -0.2, -0.12, -0.06, 0, 0.06, 0.12, 0.2, 0.3, 0.4, 0.5])
    for s, arm in (('a', 'A'), ('c', 'C')):
        for v in 'xy':
            j = M[f'j{v}']
            k = np.digitize(j, jb) - 1
            for b in range(len(jb) - 1):
                g = M[k == b]
                rows.append(dict(arm=arm, view=v, lo=jb[b], hi=jb[b + 1], **stat(g, f'res_{v}_{s}')))
    T = pd.DataFrame(rows)
    T.to_csv(out / 'line_bias_vs_tan.csv', index=False)
    # position maps: bias and sigma per (x_local, y_local) 40 mm cell, |j| < 0.3 so every cell has all angles
    pb = np.arange(-200, 201, 50)
    rows = []
    for s, arm in (('a', 'A'), ('c', 'C')):
        for v in 'xy':
            G = M[np.abs(M[f'j{v}']) < 0.4].copy()
            # remove the global scale (bias linear in j) before mapping, so the map shows local structure
            ok = np.isfinite(G[f'res_{v}_{s}'])
            a, b0 = np.polyfit(G.loc[ok, f'j{v}'], G.loc[ok, f'res_{v}_{s}'].clip(-0.15, 0.15), 1)
            G['r'] = G[f'res_{v}_{s}'] - (a * G[f'j{v}'] + b0)
            ix = np.digitize(G[f'x_local_{s}'], pb) - 1
            iy = np.digitize(G[f'y_local_{s}'], pb) - 1
            for i in range(len(pb) - 1):
                for jj in range(len(pb) - 1):
                    g = G[(ix == i) & (iy == jj)]
                    rows.append(dict(arm=arm, view=v, scale_slope=a, offset=b0, x_lo=pb[i], y_lo=pb[jj],
                                     **stat(g, 'r')))
    P = pd.DataFrame(rows)
    P.to_csv(out / 'line_bias_vs_pos.csv', index=False)
    pd.set_option('display.width', 200, 'display.max_rows', 400)
    print(T.round(4).to_string())
    for (arm, v), g in P.groupby(['arm', 'view']):
        print(f'\n{arm} {v}: global res = {g.scale_slope.iloc[0]:+.4f} j {g.offset.iloc[0]:+.4f}; '
              'median residual (x1000) per cell, rows = y_local high->low, cols = x_local low->high')
        piv = g.pivot(index='y_lo', columns='x_lo', values='bias').sort_index(ascending=False) * 1000
        npv = g.pivot(index='y_lo', columns='x_lo', values='n').sort_index(ascending=False)
        print(piv.where(npv >= 100).round(0).to_string())
        print('sigma x1000:')
        print((g.pivot(index='y_lo', columns='x_lo', values='sigma').sort_index(ascending=False) * 1000)
              .where(npv >= 100).round(0).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['build', 'maps'])
    a = ap.parse_args()
    build() if a.step == 'build' else maps()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
