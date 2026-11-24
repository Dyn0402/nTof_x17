"""r2_03_maps.py -- drift-time (ln T) fields of the four n_TOF chambers and their stability.  -> $OUT/fields.pkl, stability.json"""
import sys, pickle, json
from r2_lib import *
from r2_02_explore import design
K, LAM = 10, 0.1
uu, ww = grid(60)
INS = (np.abs(uu.ravel()) < 1.8) & (np.abs(ww.ravel()) < 1.8)
SP = Spl(K)
BG = SP.basis(uu.ravel(), ww.ravel())
NMAX = 250000


def fit(d, K_=K, lam=LAM, seed=1, boot=False, tanterms=True, nmax=NMAX):
    if len(d) > nmax:
        d = d.sample(nmax, random_state=seed)
    if boot:
        d = d.sample(len(d), replace=True, random_state=seed)
    y = np.log(d.Tm.values); y = y - y.mean()
    c, e, keep, rs = Spl(K_).fit(d.U.values, d.Wc.values, y, lam, extra=design(d, tanterms))
    return c, e, rs


def field(c, K_=K):
    return (Spl(K_).basis(uu.ravel(), ww.ravel()) @ c)


def r(a, b):
    return float(np.corrcoef(a[INS], b[INS])[0, 1])


def drop_dead(d, arm):
    if arm == 'A':
        d = d[~d.run.isin(['run_79', 'run_81'])]
    return d


FL, ST = {}, {}
for arm in 'ABCD':
    for w in ['cosmic', 'late']:
        d = drop_dead(load(w, arm), arm)
        if len(d) < 3000:
            print(arm, w, 'too few', len(d)); continue
        c, e, rs = fit(d)
        FL[(arm, w, 'all')] = dict(c=c, e=e, n=len(d), rs=rs)
        print(arm, w, len(d), 'sd', field(c)[INS].std().round(4), 'extras', np.round(e, 3), flush=True)
        ri = d.run_i % 2
        for par in (0, 1):
            dd = d[ri == par]
            if len(dd) > 2000:
                FL[(arm, w, f'par{par}')] = dict(c=fit(dd)[0], n=len(dd))
        med = d.run_i.median()
        for lab, dd in (('early', d[d.run_i <= med]), ('lateC', d[d.run_i > med])):
            if len(dd) > 2000:
                FL[(arm, w, lab)] = dict(c=fit(dd)[0], n=len(dd))
        # tan bands (both |tan| < band or >): bands on max(|tx|,|ty|)
        m = np.maximum(d.absx, d.absy)
        for lab, sel in (('tan<0.15', m < 0.15), ('tan0.15-0.3', (m >= 0.15) & (m < 0.3)), ('tan>0.3', m >= 0.3)):
            if sel.sum() > 3000:
                FL[(arm, w, lab)] = dict(c=fit(d[sel], tanterms=True)[0], n=int(sel.sum()))
        # no tan terms / K, lam sensitivity
        FL[(arm, w, 'notan')] = dict(c=fit(d, tanterms=False)[0], n=len(d))
        for k2, l2 in ((8, 0.1), (12, 0.1), (10, 0.03), (10, 0.3)):
            FL[(arm, w, f'K{k2}_l{l2}')] = dict(c=fit(d, k2, l2)[0], n=len(d), K=k2)
        # X-view / Y-view quantile only
        for q in ('x_q_u90', 'y_q_u90'):
            d2 = d.copy(); d2['Tm'] = d2[q]
            FL[(arm, w, q)] = dict(c=fit(d2)[0], n=len(d))
        # bootstrap replicates of the field
        FL[(arm, w, 'boot')] = dict(cb=[fit(d, seed=s, boot=True)[0] for s in range(1, 21)])
    # stability table
    for w in ['cosmic', 'late']:
        if (arm, w, 'all') not in FL: continue
        base = field(FL[(arm, w, 'all')]['c'])
        row = {}
        for lab in ['par0', 'early', 'tan<0.15', 'tan0.15-0.3', 'tan>0.3', 'notan', 'K8_l0.1', 'K12_l0.1', 'K10_l0.03', 'K10_l0.3', 'x_q_u90', 'y_q_u90']:
            if (arm, w, lab) in FL:
                other = {'par0': 'par1', 'early': 'lateC'}.get(lab)
                f1 = FL[(arm, w, lab)]
                K_ = f1.get('K', K)
                m1 = field(f1['c'], K_)
                if other and (arm, w, other) in FL:
                    m2 = field(FL[(arm, w, other)]['c'])
                    row[f'{lab} vs {other}'] = r(m1, m2)
                else:
                    row[f'{lab} vs all'] = r(m1, base)
        if w == 'cosmic' and (arm, 'late', 'all') in FL:
            row['cosmic all vs late all'] = r(base, field(FL[(arm, 'late', 'all')]['c']))
        row['sd_all'] = float(base[INS].std())
        cb = np.array([field(c) for c in FL[(arm, w, 'boot')]['cb']])
        row['boot_sd_mean'] = float(cb.std(0)[INS].mean())
        ST[f'{arm} {w}'] = row
        print(arm, w, {k: round(v, 3) for k, v in row.items()}, flush=True)
pickle.dump(FL, open(f'{OUT}/fields.pkl', 'wb'))
json.dump(ST, open(f'{OUT}/stability.json', 'w'), indent=1)
