#!/usr/bin/env python3
"""T9 cache: A/C tracks of stage3_r2_final with the timing columns and t0 - pin.

  cosmic.parquet : every A/C row of the beam-off (registry class cosmic) sub-runs, gated_nomap & ~censored_window
  beam.parquet   : A/C gated, angle_calibrated, t_since_flash >= 25 ms, x/y t0 <= 300, production, not run_79/81
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
NT = REPO / 'rerun' / 'ntof'
sys.path[:0] = [str(REPO), str(NT), str(NT / 'stage3')]
WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
VIEW = WD / 'ntof_pass1' / 'stage3_r2_final'
OUT = WD / 'claude-jobs' / 'eb8200b3' / 'T9_track_timing'

COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'gated_nomap', 'censored_window',
        'angle_calibrated', 't_since_flash_ns', 'x_local', 'y_local', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z',
        'd_x_nomap', 'd_y_nomap', 'd_z_nomap', 'tanx', 'tany', 'tanx_nomap', 'tany_nomap', 'x_w', 'y_w', 'x_p0',
        'y_p0', 'x_t0', 'y_t0', 'x_ftst', 'y_ftst', 'x_q_u50', 'y_q_u50', 'x_q_u90', 'y_q_u90', 'x_q_uend',
        'y_q_uend', 'drift_t_end_ns', 'drift_len_mm', 'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'chi2dof_x',
        'chi2dof_y', 'dtc_ns', 'x_t0_err', 'y_t0_err', 'x_tan_err', 'y_tan_err', 't0_railed', 'low_charge',
        'x_split_child', 'y_split_child', 'wall_A', 'wall_C', 'n_cand_x', 'n_cand_y', 'x_rank', 'y_rank', 'v_t',
        'xy_time_ok', 'xy_balance_ok', 'amp_norm', 'x_dchi2', 'y_dchi2', 'x_bow_a', 'y_bow_a', 'x_isochronous',
        'y_isochronous', 'drift_railed']


def one(args):
    f, rows, canon, kind = args
    import quality_join as Q
    have = set(pq.read_schema(f).names)
    t = pd.read_parquet(f, columns=[c for c in COLS if c in have])
    t = t[t.arm.isin(['A', 'C'])]
    if kind == 'cosmic':
        t = t[t.gated_nomap.fillna(False).astype(bool) & ~t.censored_window.fillna(False).astype(bool)]
    else:
        t = t[t.gated.fillna(False).astype(bool) & t.angle_calibrated.fillna(False).astype(bool)
              & (t.t_since_flash_ns >= 25e6) & (t.x_t0 <= 300) & (t.y_t0 <= 300)]
    t = t.copy()
    pins = Q._pins(rows, canon)
    for p in 'xy':
        ft = t[f'{p}_ftst'].fillna(-1).astype(int).to_numpy()
        pin = np.array([pins.get((tg, a), {}).get(p, {}).get(int(v), np.nan) for tg, a, v in zip(t.tag, t.arm, ft)])
        t[f'{p}_dt0'] = t[f'{p}_t0'].to_numpy(float) - pin
    t['n_trk_arm'] = t.groupby(['tag', 'event_id', 'arm']).arm.transform('size')
    for c in ('event_class',):
        if c in t:
            t[c] = t[c].astype(str)
    return t


def main(kind):
    import ntof_ledger as L
    canon = L.load_canon(str(NT / 'ledger_canon.json'))
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    jobs = []
    for (run, sub), rows in groups.items():
        f = VIEW / f'tracks_run_{run}_{sub}.parquet'
        if not f.exists():
            continue
        if kind == 'cosmic' and rows[0]['rclass'] == 'cosmic':
            jobs.append((f, rows, canon, kind))
        if kind == 'beam' and rows[0]['rclass'] == 'production' and f'run_{run}' not in ('run_79', 'run_81'):
            jobs.append((f, rows, canon, kind))
    with ProcessPoolExecutor(3) as ex:
        T = pd.concat(list(ex.map(one, jobs, chunksize=2)), ignore_index=True)
    if kind == 'cosmic':
        T = T[T.tag.isin(set(r['tag'] for r in reg if r['rclass'] == 'cosmic'))]
    T.to_parquet(OUT / f'{kind}.parquet', index=False)
    print(kind, len(jobs), 'sub-runs', len(T), 'rows')


if __name__ == '__main__':
    main(sys.argv[1])
