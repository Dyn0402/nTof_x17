"""d_lost.py -- the late (>= 25 ms) D tracks that only one set (all / stage3_r2_final) has gated, and why (T2)."""
import sys; sys.path.insert(0, '/media/ucla/work/claude-jobs/eb8200b3/T2_ref_compare')
import numpy as np, pandas as pd, compare as C
C.COLS = sorted(set(C.COLS + ['gated_precut', 'gated_reco', 'in_acceptance', 'x_cens_frac', 'y_cens_frac', 'amp_norm',
                              'x_n_strips', 'y_n_strips', 'x_n_seed', 'x_p0', 'y_p0']))
cfg = C.read_view(sys.argv[1]); cfg = cfg[cfg.arm == 'D']
ref = C.read_view(sys.argv[2], keep=C.acquisitions(cfg))
c, r = cfg[cfg.ms >= 25], ref[ref.ms >= 25]
p, oa, ob = C.pair(c, r)
print('late D: paired', len(p), 'only all', len(oa), 'only ref', len(ob))
ci = c.set_index(C.KEY)
for nm, t, other in (('only ref (lost by all)', ob, c), ('only all (added)', oa, r)):
    o = other.set_index(C.KEY)
    has = t.set_index(C.KEY).index.isin(o.index)
    print(f'-- {nm}: n {len(t)}, wall {t.wall.mean():.2f}; event has a track in the other set {has.mean():.2f}')
    # nearest track of the other set in the same event, and why it is not gated there
    rows = []
    for _, x in t.iterrows():
        k = tuple(x[C.KEY])
        if k not in o.index:
            rows.append('no track'); continue
        g = o.loc[[k]]
        g = g.iloc[[np.nanargmin(np.hypot(g.x_local - x.x_local, g.y_local - x.y_local).fillna(1e9).to_numpy())]].iloc[0]
        why = []
        if not g.gated_reco: why.append('not reco-gated')
        if g.gated_reco and not bool(g.in_acceptance): why.append('acceptance')
        if g.censored_window: why.append('censored')
        if g.d_faint: why.append('d_faint')
        if np.hypot(g.x_local - x.x_local, g.y_local - x.y_local) > 5: why.append('different track (>5 mm)')
        rows.append(' + '.join(why) or 'gated?')
    print(pd.Series(rows).value_counts().to_string())
    print(f'   own x_cens_frac median {t.x_cens_frac.median():.2f}, amp_norm median {t.amp_norm.median():.2f}, '
          f'x_p0 quartiles {np.nanquantile(t.x_p0, [.25, .5, .75]).round(0)}')
d = (p.x_t0 - p.x_t0_ref)
print('paired late D: x_t0 shift median %.1f ns, y_t0 %.1f ns' % (d.median(), (p.y_t0 - p.y_t0_ref).median()))
