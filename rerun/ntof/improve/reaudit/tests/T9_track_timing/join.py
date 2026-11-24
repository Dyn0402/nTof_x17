#!/usr/bin/env python3
"""Join the A-C cosmic lines with the per-track timing columns of stage3_r2_final -> joined.parquet (one row per line).

Keys: run, subrun, tag, event_id, arm.  lines.parquet keeps one gated (nomap, not censored) track per chamber per
event; the cache applies the same selection, and only (event, arm) with exactly one such track are joined.
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path('/media/ucla/work/claude-jobs/eb8200b3/T9_track_timing')
L = pd.read_parquet('/media/ucla/work/ntof_pass1/ana_r2/track_quality/lines.parquet')
C = pd.read_parquet(D / 'cosmic.parquet')
KEY = ['run', 'subrun', 'tag', 'event_id']
C = C[C.n_trk_arm == 1].rename(columns={'d_x': 'dm_x', 'd_y': 'dm_y', 'd_z': 'dm_z', 'tanx': 'tanxm', 'tany': 'tanym'})
assert not C.duplicated(KEY + ['arm']).any()
keep = ['track_id', 'dm_x', 'dm_y', 'dm_z', 'd_x_nomap', 'd_y_nomap', 'd_z_nomap', 'tanxm', 'tanym', 'x_w', 'y_w',
        'x_t0', 'y_t0', 'x_ftst', 'y_ftst', 'x_dt0', 'y_dt0', 'x_q_u50', 'y_q_u50', 'x_q_u90', 'y_q_u90', 'x_q_uend',
        'y_q_uend', 'drift_t_end_ns', 'dtc_ns', 'x_t0_err', 'y_t0_err', 't0_railed', 'x_split_child',
        'y_split_child', 'wall_A', 'wall_C', 'x_rank', 'y_rank', 'v_t', 'xy_time_ok', 'xy_balance_ok', 'amp_norm',
        'x_bow_a', 'y_bow_a', 'x_isochronous', 'y_isochronous', 'p0_x', 'p0_y', 'p0_z', 'x_local', 'y_local',
        'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'chi2dof_x', 'chi2dof_y']
M = L.copy()
for s, arm in (('a', 'A'), ('c', 'C')):
    g = C[C.arm == arm][KEY + keep].rename(columns={c: f'{c}_{s}' for c in keep if f'{c}_{s}' not in L.columns})
    # columns already in lines (p0, local, q_sum...) are re-read as _{s}chk to verify the join
    g = g.rename(columns={c: f'{c}_{s}chk' for c in keep if f'{c}_{s}' in L.columns})
    n0 = len(M)
    M = M.merge(g, on=KEY, how='left', validate='one_to_one')
    assert len(M) == n0
    miss = M[f'track_id_{s}'].isna().sum()
    # the lines' slopes are the nomap ones: they must equal the cache's _nomap direction
    ds = (M[f'd_x_{s}'] / M[f'd_z_{s}'] - M[f'd_x_nomap_{s}'] / M[f'd_z_nomap_{s}']).abs()
    dp = (M[f'p0_x_{s}'] - M[f'p0_x_{s}chk']).abs()
    print(arm, 'unmatched', miss, '| max |slope_nomap diff|', np.nanmax(ds), '| max |p0_x diff|', np.nanmax(dp))
M = M.drop(columns=[c for c in M.columns if c.endswith('chk')])
M = M[M.track_id_a.notna() & M.track_id_c.notna()].reset_index(drop=True)
M.to_parquet(D / 'joined.parquet', index=False)
print(len(L), 'lines,', len(M), 'joined')
