# study script: pair time consistency with the column centre; in-time expectation from mixing singles; vs ToF
import numpy as np, pandas as pd, os
T = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/tracks_t0.parquet'))
T['t_ms'] = T.t_since_flash_ns / 1e6
T['child'] = T.x_split_child.fillna(False).astype(bool) | T.y_split_child.fillna(False).astype(bool)
T['tc'] = 0.5 * ((T.x_dt0 + T.x_q_u50) + (T.y_dt0 + T.y_q_u50))
rng = np.random.default_rng(5)
tb = [12, 16, 20, 25, 35, 50, 75]
for arm in 'AC':
    a = T[(T.arm == arm) & (T.t_ms >= 12) & (T.t_ms < 75)]
    s = a[(a.n_gated_arm == 1) & a.wall & ~a.child].tc.dropna().to_numpy()
    mix = np.abs(rng.choice(s, 400000) - rng.choice(s, 400000))
    p = a[a.n_gated_arm == 2].sort_values(['run', 'subrun', 'tag', 'event_id', 'track_id'])
    g = p.groupby(['run', 'subrun', 'tag', 'event_id'])
    e = pd.DataFrame(dict(d=(g.tc.first() - g.tc.last()).abs(), child=g.child.any(), t=g.t_ms.first(),
                          wall=g.wall.first()))
    e = e[~e.child]
    for cut in (200, 250, 300, 400):
        print(f'{arm} |dtc| > {cut}: non-child pairs {np.mean(e.d > cut):.3f}   in-time mixed singles {np.mean(mix > cut):.3f}')
    ns = a[(a.n_gated_arm == 1)].groupby(pd.cut(a[(a.n_gated_arm == 1)].t_ms, tb), observed=True).size()
    r = e.groupby(pd.cut(e.t, tb), observed=True).agg(pairs=('d', 'size'), beyond300=('d', lambda v: np.mean(v > 300)))
    r['pairs_per_1000_singles'] = 1000 * r.pairs / ns
    r['excess300'] = r.beyond300 - np.mean(mix > 300)
    r['acc_est'] = (r.excess300 / 0.5).clip(lower=0)        # ~half of accidentals land beyond 300 ns (flat +-1 us)
    print(r.round(4).to_string())
