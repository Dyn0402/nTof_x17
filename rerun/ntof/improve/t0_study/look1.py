# study script: t0 - pin of singles and pair members; dt0 between the two tracks of same-chamber pairs
import numpy as np, pandas as pd, os
T = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/tracks_t0.parquet'))
T['t_ms'] = T.t_since_flash_ns / 1e6
T['dt0'] = T[['x_dt0', 'y_dt0']].mean(axis=1)
e = np.arange(-200, 201, 20)
for arm in 'ACD':
    a = T[(T.arm == arm) & (T.t_ms >= 25)]
    s = a[(a.n_gated_arm == 1) & a.wall]
    print(f'== {arm}: late singles (wall) {len(s)}')
    for p in 'xy':
        v = s[f'{p}_dt0']
        print(f'  {p} t0-pin: median {v.median():6.1f}  p16 {v.quantile(.16):6.1f} p84 {v.quantile(.84):6.1f}  '
              f'railed low {np.mean(v <= -175):.4f} high {np.mean(v >= 175):.4f}')
    h, _ = np.histogram(s.x_dt0, e); print('  x hist', h.tolist())
    for lo in (12, 25):
        pr = T[(T.arm == arm) & (T.t_ms >= lo) & (T.n_gated_arm == 2)].sort_values(['run', 'subrun', 'tag', 'event_id', 'track_id'])
        g = pr.groupby(['run', 'subrun', 'tag', 'event_id'])
        d = (g.dt0.first() - g.dt0.last()).abs()
        dx = (g.x_dt0.first() - g.x_dt0.last()).abs(); dy = (g.y_dt0.first() - g.y_dt0.last()).abs()
        rail = g.t0_railed.any()
        ns = len(T[(T.arm == arm) & (T.t_ms >= lo) & (T.n_gated_arm == 1)])
        print(f'  pairs >= {lo} ms: {len(d)} (per single {len(d)/ns:.4f}); |dt0| median {d.median():.1f}  '
              f'frac |dt0|<60 {np.mean(d<60):.3f}  60-120 {np.mean((d>=60)&(d<120)):.3f}  >=120 {np.mean(d>=120):.3f}  '
              f'x>=120 {np.mean(dx>=120):.3f} y>=120 {np.mean(dy>=120):.3f}  any railed {rail.mean():.3f}')
        h, _ = np.histogram(d, np.arange(0, 401, 20)); print('   |dt0| hist', h.tolist())
