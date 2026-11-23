# study script: wall-group time difference for same-chamber pairs whose two tracks hit different wall groups
import numpy as np, pandas as pd, glob, os
WE = np.array([-225.0, -125.0, -25.0, 75.0, 175.0]); LW = 97.4 * 0.85
fs = sorted(glob.glob(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/scint_stack/tracks/stack_run_*.parquet')))
cols = ['subrun', 'event_id', 'arm', 'track_id', 'n_trk', 't_since_flash_ns', 'u_mm', 'tan_raw_x', 'run'] + \
       [f'w{n}_{k}_on' for n in range(1, 9) for k in ('amp', 'dt', 'sat')]
S = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
S = S[S.arm.isin(['A', 'C', 'D'])]
S['t_ms'] = S.t_since_flash_ns / 1e6
S['grp'] = np.digitize(S.u_mm + LW * S.tan_raw_x, WE) - 1
def gtime(r, g):
    a, b = f'w{2*g+1}_dt_on', f'w{2*g+2}_dt_on'
    return np.nanmean([r[a], r[b]])
print('dt_on example values', S.w1_dt_on.dropna().describe().round(1).to_dict())
# singles: own-group time relative to trigger
for arm in 'ACD':
    s = S[(S.arm == arm) & (S.n_trk == 1) & (S.t_ms >= 25) & S.grp.between(0, 3)]
    t = np.array([np.nanmean([s[f'w{2*g+1}_dt_on'].to_numpy()[i], s[f'w{2*g+2}_dt_on'].to_numpy()[i]]) if False else 0 for i, g in []])
    tt = np.full(len(s), np.nan)
    for g in range(4):
        m = (s.grp == g).to_numpy()
        tt[m] = np.nanmean(np.c_[s[f'w{2*g+1}_dt_on'].to_numpy()[m], s[f'w{2*g+2}_dt_on'].to_numpy()[m]], 1)
    v = tt[np.isfinite(tt)]
    print(f'{arm} singles own group time: n {len(v)} median {np.median(v):.1f} p16 {np.percentile(v,16):.1f} p84 {np.percentile(v,84):.1f}  p1 {np.percentile(v,1):.1f} p99 {np.percentile(v,99):.1f}')
    p = S[(S.arm == arm) & (S.n_trk == 2) & (S.t_ms >= 25)].sort_values(['run', 'subrun', 'event_id', 'track_id'])
    g = p.groupby(['run', 'subrun', 'event_id'])
    f, l = g.first(), g.last()
    ok = (f.grp != l.grp) & f.grp.between(0, 3) & l.grp.between(0, 3)
    f, l = f[ok], l[ok]
    def tg(d):
        out = np.full(len(d), np.nan)
        for gg in range(4):
            m = (d.grp == gg).to_numpy()
            out[m] = np.nanmean(np.c_[d[f'w{2*gg+1}_dt_on'].to_numpy()[m], d[f'w{2*gg+2}_dt_on'].to_numpy()[m]], 1)
        return out
    t1, t2 = tg(f), tg(l); both = np.isfinite(t1) & np.isfinite(t2); d = np.abs(t1 - t2)[both]
    print(f'   pairs >= 25 ms in different groups: {len(f)}, both groups timed {both.sum()} ({both.mean():.2f}); '
          f'|dt_wall| median {np.median(d):.1f}  <5 {np.mean(d<5):.3f} 5-20 {np.mean((d>=5)&(d<20)):.3f} >=20 {np.mean(d>=20):.3f}')
    print('   hist 0-100 by 5:', np.histogram(d, np.arange(0, 101, 5))[0].tolist(), ' >100:', int((d > 100).sum()))
