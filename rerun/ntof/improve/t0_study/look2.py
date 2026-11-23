# study script: railed pairs by split-child status; in-time vs random t0 references; mixture fit of pair |dt0|
import numpy as np, pandas as pd, os
T = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/tracks_t0.parquet'))
T['t_ms'] = T.t_since_flash_ns / 1e6
T['child'] = T.x_split_child.fillna(False) | T.y_split_child.fillna(False)
T['dt0'] = T[['x_dt0', 'y_dt0']].mean(axis=1)
rng = np.random.default_rng(3)
for arm in 'AC':
    a = T[(T.arm == arm) & (T.t_ms >= 25)]
    pr = a[a.n_gated_arm == 2].copy()
    ev = pr.groupby(['run', 'subrun', 'tag', 'event_id'])
    pr['ev_child'] = ev.child.transform('any')
    pe = pr.groupby(['run', 'subrun', 'tag', 'event_id']).agg(child=('child', 'any'), rail=('t0_railed', 'any'),
                                                              d=('dt0', lambda v: abs(v.iloc[0] - v.iloc[-1])),
                                                              wall=('wall', 'first'))
    print(f'== {arm}: pairs >= 25 ms {len(pe)}; with a split child {pe.child.mean():.3f}')
    for c, g in pe.groupby('child'):
        print(f'   child={c}: n {len(g)} railed {g.rail.mean():.3f}  |dt0|>=120 {np.mean(g.d >= 120):.3f}  median {g.d.median():.1f}')
    rail_child = a[a.child].t0_railed.mean(); rail_s = a[(a.n_gated_arm == 1)].t0_railed.mean()
    print(f'   railed rate: split children {rail_child:.3f}, singles {rail_s:.4f}, non-child pair members '
          f'{a[(a.n_gated_arm == 2) & ~a.child].t0_railed.mean():.3f}')
    # references
    s_in = a[(a.n_gated_arm == 1) & a.wall & ~a.child].dt0.dropna().to_numpy()
    s_rand = a[(a.n_gated_arm == 1) & ~a.wall & ~a.child].dt0.dropna().to_numpy()
    print(f'   singles own wall fired: {len(s_in)}  t0-pin rsig {np.subtract(*np.percentile(s_in, [84, 16])) / 2:.1f} | '
          f'own wall NOT fired: {len(s_rand)} rsig {np.subtract(*np.percentile(s_rand, [84, 16])) / 2:.1f}; '
          f'|t0-pin|>=120: {np.mean(abs(s_in - np.median(s_in)) >= 120):.4f} vs {np.mean(abs(s_rand - np.median(s_in)) >= 120):.4f}')
    n = 200000
    m_in = abs(rng.choice(s_in, n) - rng.choice(s_in, n))
    m_acc = abs(rng.choice(s_in, n) - rng.choice(s_rand, n))
    e = np.arange(0, 401, 20)
    d = pe[~pe.child].d.to_numpy()
    h = np.histogram(d, e)[0].astype(float); hi = np.histogram(m_in, e)[0] / n; ha = np.histogram(m_acc, e)[0] / n
    best = min(((np.sum((h - len(d) * ((1 - f) * hi + f * ha)) ** 2 / np.maximum(len(d) * ((1 - f) * hi + f * ha), 1)), f)
                for f in np.linspace(0, 1, 201)))
    print(f'   non-child pairs: |dt0| mixture fit: accidental-like fraction {best[1]:.3f} (chi2 {best[0]:.1f}/{len(e) - 2})')
    print('     data   ', h.astype(int).tolist()[:12]); print('     in-time', np.round(len(d) * hi).astype(int).tolist()[:12])
