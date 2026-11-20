#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_quality.py -- which n_TOF pass-1 tracks point badly at the capsule, and why.

The capsule is the one known source of the late (>= 20 ms) A/C tracks.  A track whose line passes far from it
is either a real particle from somewhere else (background) or a capsule particle whose angle (or position) the
reconstruction got wrong.  The two leave different marks:

  background   adds a flat miss distribution: the capsule-pointing excess (within R1 of the source, minus a
               shuffled-angle null) drops, but the pointing core of what is left keeps its width
  bad angles   keep the excess within a wide window (R2) but move it out of the narrow one: the core
               widens and the tail fraction (R1..R2) / (0..R2) grows

So, for each track-level variable (chi2, strip counts, two-track rank, candidates, t0, charge, depth, ...), per
quantile bin: share of tracks, excess within R1 and R2, the tail fraction, and the null-subtracted core width
(robust sigma of the signed miss within +-R2).  The x view only (A/C measure global x; the capsule is long in y).

Reads the analysis view through capsule_tof.load (gated = true |tan| < 0.5, t0 <= 300 ns, pre-access runs out).

    python -m rerun.ntof.improve.track_quality scan  [--win late|early]   -> <out>/scan_<win>.csv
    python -m rerun.ntof.improve.track_quality cache                      -> <out>/tracks_AC.parquet

Out: $WORK_DATA/ntof_pass1/ana/track_quality/ (X17_TQ_OUT overrides).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

R1, R2 = 15.0, 45.0
EXTRA = ['track_id', 'x_rank', 'y_rank', 'n_cand_x', 'n_cand_y', 'chi2dof_x', 'chi2dof_y', 'x_n_strips',
         'y_n_strips', 'x_dchi2', 'y_dchi2', 'x_n_dropped', 'y_n_dropped', 'x_n_seed', 'y_n_seed', 'drift_len_mm',
         'drift_railed', 'q_per_len', 'x_q_sum', 'y_q_sum', 'x_q_uend', 'y_q_uend', 'x_tan_err', 'y_tan_err',
         'x_p0_err', 'x_t0_err', 'y_t0_err', 'x_ftst', 'depth_grid_edge_ns', 'x_slope_reliable', 'y_slope_reliable',
         'x_isochronous', 'y_isochronous', 'arms_lit', 'n_coinc_arms', 'event_class', 'x_local', 'y_local',
         'v_t', 'tan_source', 'x_quality_ok', 'y_quality_ok']


def out_dir() -> Path:
    p = os.environ.get('X17_TQ_OUT')
    if p:
        d = Path(p)
    else:
        d = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'ana' / 'track_quality'
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tracks(refresh=False) -> pd.DataFrame:
    """A/C gated tracks with the pointing quantities (pos, disp) and the quality columns; cached."""
    f = out_dir() / 'tracks_AC.parquet'
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    from sept26_prelim_analysis import capsule_tof as CT
    CT.COLS = list(dict.fromkeys(CT.COLS + EXTRA))
    d = CT.load('pass1', ('A', 'C'))
    for c in ('event_class', 'tan_source'):
        if c in d:
            d[c] = d[c].astype(str)
    d.to_parquet(f, index=False)
    return d


def source_x(d: pd.DataFrame) -> dict:
    from sept26_prelim_analysis import capsule_tof as CT
    xs = {}
    for arm, g in d[d.ms >= 20].groupby('arm'):
        xs[arm] = CT.crossing(g.pos.to_numpy(), g.disp.to_numpy())[0]
    return xs


def pointing(miss: np.ndarray, miss0: np.ndarray) -> dict:
    """miss: signed miss of the real lines; miss0: the same lines with shuffled displacements (null)."""
    n = len(miss)
    if n < 200:
        return dict(n=n)
    a1, a2 = np.mean(np.abs(miss) < R1), np.mean(np.abs(miss) < R2)
    b1, b2 = np.mean(np.abs(miss0) < R1), np.mean(np.abs(miss0) < R2)
    e1, e2 = a1 - b1, a2 - b2
    # null-subtracted core width: histogram difference within +-R2, robust sigma from its central 68 %
    edges = np.linspace(-R2, R2, 91)
    h = np.histogram(miss, edges)[0] - np.histogram(miss0, edges)[0]
    h = np.clip(h, 0, None).astype(float)
    if h.sum() > 50:
        c = np.cumsum(h) / h.sum()
        mid = 0.5 * (edges[1:] + edges[:-1])
        q16, q50, q84 = np.interp([0.16, 0.5, 0.84], c, mid)
        w68, med = 0.5 * (q84 - q16), q50
    else:
        w68 = med = np.nan
    return dict(n=n, f1=a1, null1=b1, exc1=e1, exc2=e2, tail=(e2 - e1) / e2 if e2 > 0.01 else np.nan,
                w68=w68, med=med, exc1_err=np.sqrt(a1 * (1 - a1) / n + b1 * (1 - b1) / n))


SCAN = {  # variable -> how to bin it ('q' quantiles, or explicit edges, or 'cat')
    'chi2dof_x': 'q', 'chi2dof_y': 'q', 'x_n_strips': [0, 3, 4, 5, 6, 8, 10, 14, 200],
    'y_n_strips': [0, 3, 4, 5, 6, 8, 10, 14, 200], 'x_rank': 'cat', 'y_rank': 'cat',
    'n_cand_x': [0, 1.5, 2.5, 3.5, 99], 'n_cand_y': [0, 1.5, 2.5, 3.5, 99], 'x_n_dropped': 'cat', 'y_n_dropped': 'cat',
    'x_n_seed': 'q', 'drift_len_mm': 'q', 'drift_railed': 'cat', 'q_per_len': 'q', 'x_q_sum': 'q', 'y_q_sum': 'q',
    'yx_ratio': 'q', 'x_q_uend': 'q', 'x_tan_err': 'q', 'y_tan_err': 'q', 'x_t0': 'q', 'y_t0': 'q', 'dt0': 'q',
    'x_dchi2': 'q', 'abs_tanx': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], 'abs_tany': 'q',
    'x_slope_reliable': 'cat', 'y_slope_reliable': 'cat', 'x_isochronous': 'cat', 'wall': 'cat',
    'n_coinc_arms': 'cat', 'event_class': 'cat', 'n_trk_arm': 'cat', 'x_local': 'q', 'y_local': 'q',
    'x_quality_ok': 'cat', 'y_quality_ok': 'cat', 'tan_source': 'cat',
}


def derive(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d['yx_ratio'] = d.y_q_sum / d.x_q_sum
    d['dt0'] = d.x_t0 - d.y_t0
    d['abs_tanx'] = d.tanx.abs()
    d['abs_tany'] = d.tany.abs()
    d['n_trk_arm'] = d.groupby(['run', 'subrun', 'event_id', 'arm']).arm.transform('size').clip(upper=4)
    return d


def scan(win='late', nq=8) -> pd.DataFrame:
    d = derive(load_tracks())
    xs = source_x(d)
    sel = d.ms >= 20 if win == 'late' else (d.ms >= 4) & (d.ms < 12)
    d = d[sel & (d.tanx.abs() >= 0.1)]          # the band needs leverage; |tan| < 0.1 points everywhere
    rng = np.random.default_rng(7)
    rows = []
    for arm, g in d.groupby('arm'):
        miss = (g.pos + g.disp - xs[arm]).to_numpy()
        # null: shuffle displacements within |tan x| bins so the null keeps the angle distribution
        miss0 = np.empty_like(miss)
        tb = np.digitize(g.tanx.abs().to_numpy(), [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        disp, pos = g.disp.to_numpy(), g.pos.to_numpy()
        for b in np.unique(tb):
            i = np.flatnonzero(tb == b)
            miss0[i] = pos[i] + disp[rng.permutation(i)] - xs[arm]
        base = pointing(miss, miss0)
        rows.append(dict(arm=arm, var='ALL', bin='all', share=1.0, **base))
        for var, how in SCAN.items():
            if var not in g:
                continue
            v = g[var]
            if how == 'cat':
                lab = v.astype(str).to_numpy()
                cats = pd.Series(lab).value_counts()
                cats = cats[cats >= 500].index
                groups = [(c, lab == c) for c in sorted(cats)]
            else:
                x = v.to_numpy(float)
                ok = np.isfinite(x)
                if how == 'q':
                    edges = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, nq + 1)))
                else:
                    edges = np.asarray(how, float)
                k = np.digitize(x, edges[1:-1])
                groups = [(f'[{edges[j]:.4g},{edges[j + 1]:.4g})', ok & (k == j)) for j in range(len(edges) - 1)]
                if (~ok).sum() >= 500:
                    groups.append(('nan', ~ok))
            for lab, m in groups:
                r = pointing(miss[m], miss0[m])
                rows.append(dict(arm=arm, var=var, bin=lab, share=m.mean(), **r))
    R = pd.DataFrame(rows)
    R.to_csv(out_dir() / f'scan_{win}.csv', index=False)
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['cache', 'scan'])
    ap.add_argument('--win', default='late', choices=['late', 'early'])
    a = ap.parse_args()
    if a.step == 'cache':
        d = load_tracks(refresh=True)
        print(len(d), 'tracks cached')
    else:
        R = scan(a.win)
        pd.set_option('display.width', 220, 'display.max_rows', 500)
        print(R[['arm', 'var', 'bin', 'share', 'n', 'exc1', 'exc2', 'tail', 'w68', 'med']].round(3).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
