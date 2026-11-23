#!/usr/bin/env python3
"""wall_time.py -- every wall hit within +-1000 ns of the trigger in each track's own wall group.

The slim scintillator hits cover dt_ns in [-1000, +1000] ns around the DREAM trigger.  The scint-stack extraction kept
only (-100, +60); here we keep all of it.  A wall hit is the two ends of one group (channels 2g+1, 2g+2) within
END_TOL ns of each other; its time is the mean of the two ends and its amplitude the sum.  Each gated A/C/D track
(tracks_t0.parquet) is given its wall group from the scint-stack geometry (u at the wall = u + L_wall * lambda * tan)
and every wall hit of that group in the event: time, amplitude.

Writes $WORK_DATA/ntof_pass1/ana/t0_study/wall_hits.parquet (one row per track x wall hit; tracks with none get one
row with NaN time).
"""
import glob, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
OUT = WD / 'ntof_pass1' / 'ana' / 't0_study'
SLIM = WD / 'ntof_pass1' / 'context_r2' / 'slim'
STACK = WD / 'ntof_pass1' / 'ana' / 'scint_stack' / 'tracks'
WE = np.array([-225.0, -125.0, -25.0, 75.0, 175.0])
LW = 97.4 * 0.85          # wall lever x the scint-stack scattering lambda (~0.85 on pass 1)
END_TOL = 15.0
ARM_DET = {'A': 0, 'C': 2, 'D': 3}


def wall_hits(run, sub):
    f = SLIM / f'ntof_hits_{run}_{sub}.parquet'
    if not f.exists():
        return None
    h = pd.read_parquet(f, columns=['eventId', 'det', 'detn', 'dt_ns', 'amp', 'is_control', 'satuflag'])
    h = h[(h.is_control == 0) & (h.det <= 3)]
    out = []
    for arm, d in ARM_DET.items():
        a = h[h.det == d]
        for g in range(4):
            e1 = a[a.detn == 2 * g + 1][['eventId', 'dt_ns', 'amp']]
            e2 = a[a.detn == 2 * g + 2][['eventId', 'dt_ns', 'amp']]
            m = e1.merge(e2, on='eventId', suffixes=('1', '2'))
            m = m[np.abs(m.dt_ns1 - m.dt_ns2) <= END_TOL]
            if not len(m):
                continue
            out.append(pd.DataFrame(dict(event_id=m.eventId.to_numpy(), arm=arm, grp=g,
                                         t_wall=0.5 * (m.dt_ns1 + m.dt_ns2).to_numpy(),
                                         a_wall=(m.amp1 + m.amp2).to_numpy())))
    if not out:
        return None
    w = pd.concat(out, ignore_index=True)
    # one hit per (event, arm, group, ~same time): ends can pair twice
    w['tb'] = np.round(w.t_wall / 10)
    return w.sort_values('a_wall', ascending=False).drop_duplicates(['event_id', 'arm', 'grp', 'tb']).drop(columns='tb')


def one(args):
    run, sub, T = args
    w = wall_hits(run, sub)
    if w is None:
        return None
    T = T.copy()
    T['grp'] = np.digitize(T.u_mm + LW * T.tan_raw_x, WE) - 1
    m = T.merge(w, on=['event_id', 'arm', 'grp'], how='left')
    return m


def main():
    T = pd.read_parquet(OUT / 'tracks_t0.parquet')
    S = pd.concat([pd.read_parquet(f, columns=['run', 'subrun', 'event_id', 'arm', 'track_id', 'u_mm', 'tan_raw_x'])
                   for f in sorted(glob.glob(str(STACK / 'stack_run_*.parquet')))], ignore_index=True)
    S['run'] = S.run.astype(str).where(S.run.astype(str).str.startswith('run_'), 'run_' + S.run.astype(str))
    T['run'] = T.run.astype(str).where(T.run.astype(str).str.startswith('run_'), 'run_' + T.run.astype(str))
    T = T.merge(S, on=['run', 'subrun', 'event_id', 'arm', 'track_id'], how='inner')
    print('tracks with wall geometry', len(T))
    jobs = [(r.replace('run_', 'run_'), s, g) for (r, s), g in T.groupby(['run', 'subrun'])]
    with ProcessPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, [(r, s, g) for r, s, g in jobs], chunksize=4) if p is not None]
    W = pd.concat(parts, ignore_index=True)
    W.to_parquet(OUT / 'wall_hits.parquet', index=False)
    print(len(W), 'rows ->', OUT / 'wall_hits.parquet')


if __name__ == '__main__':
    main()
