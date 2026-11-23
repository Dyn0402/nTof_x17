#!/usr/bin/env python3
"""collect.py -- per-track t0 relative to the trigger pin, for every gated A/C/D track of the analysis view.

t0 is each view's fitted start time; the pin is the bundle's t0_abs for the view's ftst (the time a particle that
fired the trigger is expected at).  The reco lets t0 move within pin +- 180 ns (a hard box); t0_railed marks tracks
at that edge.  Writes $WORK_DATA/ntof_pass1/ana/t0_study/tracks_t0.parquet (one row per gated track) for the
pile-up / out-of-time study (README.md).
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
NT = HERE.parent.parent
sys.path[:0] = [str(NT), str(NT / 'stage3'), str(NT.parent.parent)]
import ntof_ledger as L  # noqa: E402
import quality_join as Q  # noqa: E402

WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
VIEW = WD / 'ntof_pass1' / 'stage3_r2_final'
OUT = WD / 'ntof_pass1' / 'ana' / 't0_study'
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 't_since_flash_ns', 'x_t0', 'y_t0', 'x_ftst',
        'y_ftst', 'x_q_u50', 'y_q_u50', 't0_railed', 'dca_source_mm', 'wall_A', 'wall_C', 'wall_D', 'tanx', 'tany',
        'x_local', 'y_local', 'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips', 'event_class', 'dtc_ns', 'amp_norm',
        'n_cand_x', 'n_cand_y', 'x_split_child', 'y_split_child', 'x_q_u90', 'y_q_u90', 'x_q_uend', 'y_q_uend',
        'drift_t_end_ns', 'drift_railed', 'x_plausible', 'y_plausible', 'x_isochronous', 'y_isochronous']


def one(args):
    f, rows, canon = args
    t = pd.read_parquet(f, columns=COLS)
    t = t[t.gated & t.arm.isin(['A', 'C', 'D'])].copy()
    pins = Q._pins(rows, canon)
    for p in 'xy':
        ft = t[f'{p}_ftst'].fillna(-1).astype(int).to_numpy()
        pin = np.array([pins.get((tg, a), {}).get(p, {}).get(int(v), np.nan) for tg, a, v in zip(t.tag, t.arm, ft)])
        t[f'{p}_dt0'] = (t[f'{p}_t0'].to_numpy(float) - pin).astype('float32')
    t['wall'] = np.select([t.arm == a for a in 'ACD'], [t[f'wall_{a}'] for a in 'ACD'], False)
    t['n_gated_arm'] = t.groupby(['tag', 'event_id', 'arm']).track_id.transform('size')
    return t.drop(columns=['wall_A', 'wall_C', 'wall_D'])


def main():
    canon = L.load_canon(str(NT / 'ledger_canon.json'))
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    jobs = []
    for (run, sub), rows in groups.items():
        f = VIEW / f'tracks_run_{run}_{sub}.parquet'
        if f.exists() and rows[0]['rclass'] == 'production':
            jobs.append((f, rows, canon))
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex:
        T = pd.concat(list(ex.map(one, jobs, chunksize=4)), ignore_index=True)
    T.to_parquet(OUT / 'tracks_t0.parquet', index=False)
    print(len(jobs), 'sub-runs,', len(T), 'gated tracks ->', OUT / 'tracks_t0.parquet')


if __name__ == '__main__':
    main()
