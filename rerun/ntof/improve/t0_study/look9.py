# study script: data-driven accidental template for |dtc| of same-chamber pairs, and a two-component fit
#   accidental partner: arrival offset D drawn from the off-time own-group wall-hit density (|D| in 60..1000 ns, the
#   in-time peak region filled by interpolation), its tc drawn from the off-time single tracks in the nearest D bin
#   (look6), the other track's tc from in-time singles.  in-time pairs: two in-time singles (mixed).
#   Fit |dtc| of non-child pairs = (1-f) in-time + f accidental, per ToF range.
import numpy as np, pandas as pd, os
W = pd.read_parquet(os.path.expandvars('$WORK_DATA/ntof_pass1/ana/t0_study/wall_hits.parquet'))
W['t_ms'] = W.t_since_flash_ns / 1e6
W['child'] = W.x_split_child.fillna(False).astype(bool) | W.y_split_child.fillna(False).astype(bool)
W['tc'] = 0.5 * ((W.x_dt0 + W.x_q_u50) + (W.y_dt0 + W.y_q_u50))
rng = np.random.default_rng(7)
E = np.arange(0, 1201, 50)
for arm in 'AC':
    a = W[(W.arm == arm) & (W.t_ms >= 12) & (W.t_ms < 75) & ~W.child]
    s = a[a.n_gated_arm == 1].copy()
    pk = s.t_wall[(s.t_wall > -60) & (s.t_wall < 40)].median(); s['dtw'] = s.t_wall - pk
    s['tid'] = s.groupby(['run', 'subrun', 'event_id', 'track_id'], sort=False).ngroup()
    s['in'] = np.abs(s.dtw) < 30; s['off'] = (np.abs(s.dtw) >= 60) & (np.abs(s.dtw) < 1000)
    agg = s.groupby('tid').agg(n_in=('in', 'sum'), n_off=('off', 'sum'))
    o = s[s.tid.isin(agg.index[(agg.n_in == 0) & (agg.n_off == 1)]) & s.off]
    tin = s[s['in']].drop_duplicates('tid').tc.dropna().to_numpy()
    # arrival-offset density from all off-time own-group hits of singles (per 50 ns), peak region interpolated
    eb = np.arange(-1000, 1001, 50); h = np.histogram(s.dtw[s.off], eb)[0].astype(float)
    c = 0.5 * (eb[1:] + eb[:-1]); hole = np.abs(c) < 60
    h[hole] = np.interp(c[hole], c[~hole], h[~hole])
    D = rng.choice(c, 300000, p=h / h.sum()) + rng.uniform(-25, 25, 300000)
    ob = np.array([-1000, -700, -500, -400, -300, -180, -120, -60, 60, 120, 180, 300, 400, 500, 700, 1000])
    o = o.assign(b=np.digitize(o.dtw, ob))
    pools = {b: g.tc.dropna().to_numpy() for b, g in o.groupby('b')}
    bD = np.digitize(D, ob)
    tc_acc = np.array([rng.choice(pools[b]) if b in pools and len(pools[b]) else np.nan for b in bD])
    acc = np.abs(tc_acc - rng.choice(tin, len(D))); acc = acc[np.isfinite(acc)]
    inn = np.abs(rng.choice(tin, 300000) - rng.choice(tin, 300000))
    hi_ = np.histogram(inn, E)[0] / len(inn); ha = np.histogram(acc, E)[0] / len(acc)
    p = a[a.n_gated_arm == 2].sort_values(['run', 'subrun', 'tag', 'event_id', 'track_id']).drop_duplicates(['run', 'subrun', 'tag', 'event_id', 'track_id'])
    g = p.groupby(['run', 'subrun', 'tag', 'event_id'])
    e = pd.DataFrame(dict(d=(g.tc.first() - g.tc.last()).abs(), t=g.t_ms.first(), n=g.tc.size()))
    e = e[e.n == 2]
    print(f'== {arm}: accidental template |dtc| > 300: {np.mean(acc > 300):.2f} (in-time {np.mean(inn > 300):.3f})')
    for lo, hi in ((12, 16), (16, 25), (25, 35), (35, 50), (50, 75), (25, 75)):
        d = e[(e.t >= lo) & (e.t < hi)].d.to_numpy(); hd = np.histogram(d, E)[0]; n = hd.sum()
        fs = np.linspace(0, 1, 401)
        nll = [-(hd * np.log(np.maximum((1 - f) * hi_ + f * ha, 1e-9))).sum() for f in fs]
        i = int(np.argmin(nll)); fb = fs[i]
        ok = np.array(nll) <= nll[i] + 0.5; err = (fs[ok].max() - fs[ok].min()) / 2
        mu = n * ((1 - fb) * hi_ + fb * ha); chi2 = np.sum((hd - mu) ** 2 / np.maximum(mu, 1))
        print(f'   {lo:2d}-{hi:2d} ms: non-child pairs {n:5d}  accidental fraction {fb:.3f} +- {err:.3f}  chi2 {chi2:.1f}/{len(E) - 2}')
