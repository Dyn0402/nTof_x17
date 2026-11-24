#!/usr/bin/env python3
"""T4 COSMIC cache (beam-off sub-runs by registry class, no time cut): late (>= 20 ms) single-track-per-chamber capsule tracks from the stage3_r2_final production tables.

Selection kept in the cache (the rest is applied in analysis):
  production sub-runs (registry rclass production, not excluded), t_since_flash >= 20 ms, exactly one track of that arm
  in the event (all tracks counted, gated or not), gated, not low_charge, not t0_railed, not x/y split child.
  B: same, with whatever `gated` is (B has no angle).
Writes cache.parquet in this directory.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
S3 = Path('/media/ucla/work/ntof_pass1/stage3_r2_final')
OUT = Path(__file__).resolve().parent

COLS = ['tag', 'run', 'subrun', 'event_id', 'arm', 'event_class', 'ntof_run', 'x_local', 'y_local', 'tanx', 'tany',
        'tanx_nomap', 'tany_nomap', 'tan_raw_x', 'tan_raw_y', 'x_w', 'y_w', 'x_q_u50', 'x_q_u90', 'x_q_uend',
        'y_q_u50', 'y_q_u90', 'y_q_uend', 'drift_t_end_ns', 'drift_railed', 'x_t0', 'y_t0', 'x_t0_err', 'y_t0_err',
        'x_q_sum', 'y_q_sum', 'q_total', 't_since_flash_ns', 'gated', 'low_charge', 't0_railed', 'x_split_child',
        'y_split_child', 'censored_window', 'd_faint', 'v_t', 'v_t_measured', 'v_drift_um_ns', 'x_n_strips',
        'y_n_strips', 'chi2dof_x', 'chi2dof_y', 'x_n_flagged', 'y_n_flagged', 'x_cens_frac', 'y_cens_frac',
        'xy_time_ok', 'xy_balance_ok', 'x_p0', 'y_p0']


sys.path.insert(0, str(REPO))
from sept26_prelim_analysis.pass1_tracks import cosmic_tags  # noqa: E402
TAGS = set(cosmic_tags())


def one(f):
    d = pq.read_table(f, columns=COLS).to_pandas()
    d = d[d.tag.isin(TAGS)]
    n = d.groupby(['event_id', 'arm']).arm.transform('size')
    b = lambda c: d[c].fillna(False).astype(bool)
    keep = ((n == 1) & b('gated') & ~b('low_charge') & ~b('t0_railed')
            & ~b('x_split_child') & ~b('y_split_child'))
    d = d[keep].copy()
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


def main():
    reg = pd.read_csv(REPO / 'rerun/ntof/ntof_registry.csv')
    reg = reg[(reg.rclass == 'cosmic') & (reg.excluded != True)]
    prod = {(f'run_{r}', s) for r, s in zip(reg.run, reg.subrun)}
    fs = []
    for f in sorted(S3.glob('tracks_run_*.parquet')):
        stem = f.stem[len('tracks_'):]
        run = '_'.join(stem.split('_')[:2])
        sub = stem[len(run) + 1:]
        if (run, sub) in prod:
            fs.append(f)
    print(len(fs), 'cosmic tables', flush=True)
    out = []
    with ProcessPoolExecutor(3) as ex:
        for i, d in enumerate(ex.map(one, fs)):
            out.append(d)
            if i % 50 == 0:
                print(i, sum(map(len, out)), flush=True)
    D = pd.concat(out, ignore_index=True)
    D.to_parquet(OUT / 'cache_cosmic.parquet')
    print(D.arm.value_counts())


if __name__ == '__main__':
    main()
