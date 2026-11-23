# study script: does the column end (t0 + u90 / uend, and drift_t_end) reveal a late-arriving track?
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
    ref = s[s['in']].drop_duplicates('tid')
    for d in (o, ref):
        d['xe'] = d.x_dt0 + d.x_q_u90; d['ye'] = d.y_dt0 + d.y_q_u90
        d['xend'] = d.x_dt0 + d.x_q_uend
    print(f'== {arm}: in-time reference n {len(ref)}: (t0-pin)+u90 x median {ref.xe.median():.0f} p95 {ref.xe.quantile(.95):.0f} '
          f'p99 {ref.xe.quantile(.99):.0f} | +uend median {ref.xend.median():.0f} p95 {ref.xend.quantile(.95):.0f} | '
          f'drift_t_end median {ref.drift_t_end_ns.median():.0f} p95 {ref.drift_t_end_ns.quantile(.95):.0f}; drift_railed {ref.drift_railed.mean():.3f}')
    thr = ref.xe.quantile(.98)
    bins = np.array([-700, -400, -180, -60, 60, 180, 300, 400, 500, 700, 1000])
    o['b'] = pd.cut(o.dtw, bins)
    r = o.groupby('b', observed=True).agg(n=('xe', 'size'), xe_med=('xe', 'median'), ye_med=('ye', 'median'),
                                         xend_med=('xend', 'median'), dte=('drift_t_end_ns', 'median'),
                                         beyond_p98=('xe', lambda v: np.mean(v > thr)), drail=('drift_railed', 'mean'),
                                         plaus=('x_plausible', 'mean'))
    print(f'   x (t0-pin)+u90 p98 of in-time: {thr:.0f}; in-time beyond: 0.020')
    print(r.round(3).to_string())
