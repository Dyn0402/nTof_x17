"""a2_part3.py -- robustness of the gain: time splits (a), where the gain comes from (b), index by reference (c), position residual (d), baselines (e)."""
import sys, json, time
from a2_lib import *
KS = 6; Ks = (9, 12, 16, 20); lams = 10.0 ** np.arange(-3, 2.1, 0.75)
D = {'run1': load('run1'), 'run2': load('run2')}


def select_fit(Dtr, variant='phi', Ks=Ks):
    kap = pilot_kappa(Dtr); best = None
    for K in Ks:
        spec = Spec(variant, K, KS, kap)
        for lam in lams:
            cx, cy = cv_score(spec, Dtr, lam)
            if best is None or cx + cy < best[0]: best = (cx + cy, K, lam)
    m = fit_map(Spec(variant, best[1], KS, kap), Dtr, best[2]); m['K'] = best[1]
    return m


def arms_for(Dtr, T, m):
    arms = baseline_arms(T, Dtr, grids=(8,)); arms['phi'] = apply_map(m, T)
    return arms


def split_report(name, Dtr, T, m=None):
    m = m or select_fit(Dtr)
    arms = arms_for(Dtr, T, m)
    dd = {}
    for a, pr in arms.items():
        dd[f'{a}|x'] = dth(pr['x'], T.trx); dd[f'{a}|y'] = dth(pr['y'], T.trY)
    rep = dict(name=name, ntrain=Dtr.n, ntest=T.n, K=m['K'], lam=m['lam'])
    for p in 'xy':
        ds = {a: dd[f'{a}|{p}'] for a in arms}
        S, Df = paired_boot(ds, 'pass1', nb=400)
        _, Dc = paired_boot({k: ds[k] for k in ('cells8', 'phi')}, 'cells8', nb=400)
        rep[p] = dict(s68={a: S[a] for a in S}, d_vs_pass1={a: Df[a] for a in Df}, phi_minus_cells8=Dc['phi'])
    print(name, {p: {a: round(rep[p]['s68'][a][0], 3) for a in rep[p]['s68']} for p in 'xy'}, 'K', m['K'], flush=True)
    return rep, dd, m


out = {}
# cross-run
cross = {}
for tr, te in (('run1', 'run2'), ('run2', 'run1')):
    rep, dd, m = split_report(f'{tr}->{te}', D[tr], D[te])
    out[f'{tr}->{te}'] = rep; cross[(tr, te)] = (dd, m)
# (a) time splits
for run in ('run1', 'run2'):
    d = D[run]; r = d.rank; first = r < d.n // 2
    for nm, mtr in (('first->second', first), ('second->first', ~first),
                    ('oddfiles->evenfiles', d.fid % 2 == 1), ('evenfiles->oddfiles', d.fid % 2 == 0),
                    ('random-half-A->B', (np.arange(d.n) % 2 == 0))):
        rep, dd, m = split_report(f'{run} {nm}', d.sub(mtr), d.sub(~mtr))
        out[f'{run} {nm}'] = rep
# (b) bins on cross-run predictions
binsr = {}
for (tr, te), (dd, m) in cross.items():
    T = D[te]; bb = {}
    for p in 'xy':
        trr = T.trx if p == 'x' else T.trY
        centre = (np.abs(T.U) < 0.8) & (np.abs(T.W) < 0.8)
        sels = {'|tan| 0-0.1': np.abs(trr) < 0.1, '0.1-0.2': (np.abs(trr) >= 0.1) & (np.abs(trr) < 0.2),
                '0.2-0.3': (np.abs(trr) >= 0.2) & (np.abs(trr) < 0.3), '0.3-0.6': np.abs(trr) >= 0.3,
                'centre': centre, 'outer': ~centre, 'all': np.ones(T.n, bool)}
        for sn, s in sels.items():
            row = {'n': int(s.sum())}
            for a in ('pass1', 'pass1c', 'cells8', 'phi'):
                v = dd[f'{a}|{p}']
                row[a] = s68(v[s])
                row[a + '_t2'] = float(np.mean(np.abs(v[s]) > 2)); row[a + '_t4'] = float(np.mean(np.abs(v[s]) > 4))
            bb[f'{p} {sn}'] = row
    binsr[f'{tr}->{te}'] = bb
out['bins'] = binsr
# (c) index by reference position, (d) mid-gap position residual
cd = {}
for tr, te in (('run1', 'run2'), ('run2', 'run1')):
    Dtr, T = D[tr], D[te]
    Dr = Data(**{**Dtr.__dict__}); Dr.U = (Dtr.xm - 200) / 100; Dr.W = (Dtr.ym - 200) / 100
    Tr = Data(**{**T.__dict__}); Tr.U = (T.xm - 200) / 100; Tr.W = (T.ym - 200) / 100
    kap = pilot_kappa(Dr); K = cross[(tr, te)][1]['K']; lam = cross[(tr, te)][1]['lam']
    mref = fit_map(Spec('phi', K, KS, kap), Dr, lam); pref = apply_map(mref, Tr)
    cref = baseline_arms(Tr, Dr, grids=(8,))['cells8']
    dd = cross[(tr, te)][0]
    row = {}
    for p in 'xy':
        trr = T.trx if p == 'x' else T.trY
        row[p] = dict(phi_recon=s68(dd[f'phi|{p}']), phi_ref=s68(dth(pref[p], trr)), cells8_recon=s68(dd[f'cells8|{p}']), cells8_ref=s68(dth(cref[p], trr)))
        # (d) mid-gap position residual (mm): p0 + 14 tan - reference at mid-gap
        p0 = T.p0x if p == 'x' else T.p0y; xm = T.xm if p == 'x' else T.ym
        pos = {}
        for a, pr in (('pass1', T.tpx if p == 'x' else T.tpy), ('phi', cross[(tr, te)][0] and None), ):
            pass
        arms = baseline_arms(T, Dtr, grids=(8,)); arms['phi'] = apply_map(cross[(tr, te)][1], T)
        for a in ('pass1', 'pass1c', 'cells8', 'phi'):
            pos[a] = s68(p0 + 14 * arms[a][p] - xm)
        row[p]['pos_mid_s68_mm'] = pos
    cd[f'{tr}->{te}'] = row
    print(tr, te, row, flush=True)
out['index_and_position'] = cd
json.dump(out, open(f'{OUT}/part3.json', 'w'), indent=1, default=float)
