#!/usr/bin/env python3
"""T5 cache: late (>= 25 ms) gated, one-track-per-chamber A/C/D tracks from stage3_r2_final, unmapped + mapped angles.

Selection (pass1_tracks conventions): production rows (t_since_flash_ns not null, so beam-off tags excluded), arm A/C/D,
gated := gated_nomap & ~censored_window (pass1_tracks.nomap rule, the unmapped |tan| < 0.5 acceptance), exactly one
such track in the (sub-run, event, arm), not low_charge, not t0_railed, not x/y split child.  Mapped columns kept too.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path('/media/ucla/work/ntof_pass1/stage3_r2_final')
OUT = Path(__file__).resolve().parent / 'late_tracks_ext06.parquet'
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'gated', 'gated_reco', 'gated_nomap', 'censored_window', 'x_local', 'y_local',
        'tanx', 'tany', 'tanx_nomap', 'tany_nomap', 'd_x', 'd_y', 'd_z', 'd_x_nomap', 'd_y_nomap', 'd_z_nomap',
        'p0_x', 'p0_y', 'p0_z', 'x_q_sum', 'y_q_sum', 'q_total', 't_since_flash_ns', 'low_charge', 't0_railed',
        'x_split_child', 'y_split_child', 'x_t0', 'y_t0', 'angle_calibrated', 'wall_A', 'wall_C', 'wall_D',
        'ntof_run', 'x_p0', 'y_p0', 'x_w', 'y_w', 'dca_source_mm', 'chi2dof_x', 'chi2dof_y']


def one(f):
    names = set(pq.read_schema(f).names)
    cols = [c for c in COLS if c in names]
    t = pq.read_table(f, columns=cols,
                      filters=[('arm', 'in', ['A', 'C', 'D']), ('t_since_flash_ns', '>=', 25e6)]).to_pandas()
    if t.empty:
        return None
    b = lambda c: t[c].astype('boolean').fillna(False).astype(bool).to_numpy()  # noqa: E731
    g = b('gated_reco') & ~b('censored_window') & (t.tanx_nomap.abs() < 0.6).to_numpy() & (t.tany_nomap.abs() < 0.6).to_numpy()
    t = t[g]
    n = t.groupby(['subrun', 'event_id', 'arm']).arm.transform('size')
    t = t[(n == 1).to_numpy()]
    b = lambda c: t[c].astype('boolean').fillna(False).astype(bool).to_numpy()  # noqa: E731
    t = t[~b('low_charge') & ~b('t0_railed') & ~b('x_split_child') & ~b('y_split_child')]
    for c in ('gated', 'gated_reco', 'gated_nomap', 'censored_window', 'low_charge', 't0_railed', 'x_split_child', 'y_split_child',
              'angle_calibrated', 'gated_reco', 'wall_A', 'wall_C', 'wall_D'):
        if c in t:
            t[c] = t[c].astype('boolean').fillna(False).astype(bool)
    for c in ('run', 'subrun', 'tag', 'arm'):
        t[c] = t[c].astype(str)
    return t


def main():
    fs = sorted(SRC.glob('tracks_run_*_*.parquet'))
    if len(sys.argv) > 1:
        fs = fs[:int(sys.argv[1])]
    parts = []
    with ProcessPoolExecutor(3) as ex:
        for i, t in enumerate(ex.map(one, fs)):
            if t is not None:
                parts.append(t)
            if i % 50 == 0:
                print(i, len(fs), sum(len(p) for p in parts), flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(OUT, index=False)
    print(len(d), d.groupby('arm').size().to_dict())


if __name__ == '__main__':
    main()
