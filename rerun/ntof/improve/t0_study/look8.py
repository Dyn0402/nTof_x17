# study script: in-time PAIR reference = pairs whose two tracks both have an in-time hit in their own wall group
import numpy as np, pandas as pd, os
W = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/wall_hits.parquet'))
W['t_ms'] = W.t_since_flash_ns / 1e6
W['child'] = W.x_split_child.fillna(False).astype(bool) | W.y_split_child.fillna(False).astype(bool)
W['tc'] = 0.5 * ((W.x_dt0 + W.x_q_u50) + (W.y_dt0 + W.y_q_u50))
for arm in 'AC':
    a = W[(W.arm == arm) & (W.t_ms >= 12) & (W.t_ms < 75) & (W.n_gated_arm == 2)].copy()
    pk = a.t_wall[(a.t_wall > -60) & (a.t_wall < 40)].median()
    a['in'] = np.abs(a.t_wall - pk) < 30
    k = ['run', 'subrun', 'tag', 'event_id', 'track_id']
    tr = a.groupby(k).agg(tc=('tc', 'first'), win=('in', 'any'), grp=('grp', 'first'), child=('child', 'first'), t=('t_ms', 'first')).reset_index()
    tr = tr.sort_values(k)
    g = tr.groupby(k[:-1])
    e = pd.DataFrame(dict(d=(g.tc.first() - g.tc.last()).abs(), both=g.win.all(), none=~g.win.any(), child=g.child.any(),
                          diffgrp=g.grp.nunique() > 1, t=g.t.first()))
    e = e[~e.child & e.diffgrp]
    ref = e[e.both]
    for lo, hi in ((12, 25), (25, 75)):
        x = e[(e.t >= lo) & (e.t < hi)]; r = x[x.both]; n0 = x[x.none]
        print(f'{arm} {lo}-{hi} ms, different wall groups: pairs {len(x)}, both wall in time {len(r)} ({len(r)/len(x):.2f}): '
              f'|dtc|>300 {np.mean(r.d>300):.3f} | all {np.mean(x.d>300):.3f} | neither in time {np.mean(n0.d>300):.3f} (n {len(n0)})')
