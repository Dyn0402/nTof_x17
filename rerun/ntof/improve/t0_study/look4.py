# study script: chamber t0 response to the particle's arrival time, measured with the wall time of the track's own group
import numpy as np, pandas as pd, os
W = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/wall_hits.parquet'),
                    columns=['run', 'subrun', 'event_id', 'arm', 'track_id', 'n_gated_arm', 't_since_flash_ns',
                             'x_split_child', 'y_split_child', 'x_dt0', 'y_dt0', 't0_railed', 't_wall', 'a_wall',
                             'tanx', 'dca_source_mm'])
W['t_ms'] = W.t_since_flash_ns / 1e6
W['child'] = W.x_split_child.fillna(False).astype(bool) | W.y_split_child.fillna(False).astype(bool)
for arm in 'ACD':
    s = W[(W.arm == arm) & (W.t_ms >= 12) & ~W.child & (W.n_gated_arm == 1)].copy()
    tw = s.t_wall.dropna()
    pk = tw[(tw > -60) & (tw < 40)].median()
    s['dtw'] = s.t_wall - pk
    print(f'== {arm}  singles >=12 ms; wall hits in own group per 100 ns (peak {pk:.1f}):',
          np.histogram(s.dtw.dropna(), np.arange(-1000, 1001, 100))[0].tolist())
    s['tid'] = s.groupby(['run', 'subrun', 'event_id', 'track_id'], sort=False).ngroup()
    s['in'] = np.abs(s.dtw) < 30
    s['off'] = (np.abs(s.dtw) >= 60) & (np.abs(s.dtw) < 1000)
    agg = s.groupby('tid').agg(n_in=('in', 'sum'), n_off=('off', 'sum'))
    good = agg.index[(agg.n_in == 0) & (agg.n_off == 1)]
    o = s[s.tid.isin(good) & s.off]
    print(f'   singles {len(agg)}: in-time own-group hit {np.mean(agg.n_in > 0):.3f}; none in time + one off-time hit '
          f'{len(o)} ({len(o)/len(agg):.4f})')
    bins = np.array([-1000, -700, -500, -400, -300, -240, -180, -120, -60, 60, 120, 180, 240, 300, 400, 500, 700, 1000])
    o = o.assign(b=pd.cut(o.dtw, bins))
    r = o.groupby('b', observed=True).agg(n=('x_dt0', 'size'), x_med=('x_dt0', 'median'), y_med=('y_dt0', 'median'),
                                         x_p16=('x_dt0', lambda v: v.quantile(.16)), x_p84=('x_dt0', lambda v: v.quantile(.84)),
                                         rail=('t0_railed', 'mean'), point15=('dca_source_mm', lambda v: np.mean(v < 15)))
    ref = s[s['in']]
    print(f'   reference (in-time hit): x_dt0 median {ref.x_dt0.median():.1f}, railed {ref.t0_railed.mean():.4f}, '
          f'dca<15 {np.mean(ref.dca_source_mm < 15):.3f}')
    print(r.round(3).to_string())
