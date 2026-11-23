# study script: the column centre tc = (t0 - pin) + u50 as a per-track arrival time; response to the wall time
import numpy as np, pandas as pd, os
W = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/wall_hits.parquet'))
W['t_ms'] = W.t_since_flash_ns / 1e6
W['child'] = W.x_split_child.fillna(False).astype(bool) | W.y_split_child.fillna(False).astype(bool)
for arm in 'ACD':
    s = W[(W.arm == arm) & (W.t_ms >= 12) & ~W.child & (W.n_gated_arm == 1)].copy()
    tw = s.t_wall.dropna(); pk = tw[(tw > -60) & (tw < 40)].median(); s['dtw'] = s.t_wall - pk
    s['tid'] = s.groupby(['run', 'subrun', 'event_id', 'track_id'], sort=False).ngroup()
    s['in'] = np.abs(s.dtw) < 30; s['off'] = (np.abs(s.dtw) >= 60) & (np.abs(s.dtw) < 1000)
    agg = s.groupby('tid').agg(n_in=('in', 'sum'), n_off=('off', 'sum'))
    o = s[s.tid.isin(agg.index[(agg.n_in == 0) & (agg.n_off == 1)]) & s.off].copy()
    ref = s[s['in']].drop_duplicates('tid').copy()
    for d in (o, ref):
        d['tc'] = 0.5 * ((d.x_dt0 + d.x_q_u50) + (d.y_dt0 + d.y_q_u50))
    m0 = ref.tc.median(); q = ref.tc.quantile([.01, .02, .16, .84, .98, .99]).to_numpy()
    print(f'== {arm}: in-time tc median {m0:.0f}, 16-84 % {q[2]-m0:.0f}/{q[3]-m0:+.0f}, 2-98 % {q[1]-m0:.0f}/{q[4]-m0:+.0f}')
    bins = np.array([-1000, -700, -500, -400, -300, -180, -120, -60, 60, 120, 180, 300, 400, 500, 700, 1000])
    o['b'] = pd.cut(o.dtw, bins)
    r = o.groupby('b', observed=True).agg(n=('tc', 'size'), dtc_med=('tc', lambda v: v.median() - m0),
                                         outside_2_98=('tc', lambda v: np.mean((v < q[1]) | (v > q[4]))),
                                         point15=('dca_source_mm', lambda v: np.mean(v < 15)))
    print(r.round(3).to_string())
