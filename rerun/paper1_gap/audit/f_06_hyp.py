import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
from scipy import stats
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
H={'H1 flat cathode':dict(zero=('c_kuu','c_kww','c_kuw')),
   'H2 cathode parallel to mesh':dict(tie={'c_kuu':{'h_kuu':1},'c_kww':{'h_kww':1},'c_kuw':{'h_kuw':1}}),
   'H3 isotropic h and c':dict(tie={'h_kww':{'h_kuu':1},'c_kww':{'c_kuu':1}}),
   'H4 isotropic cathode':dict(tie={'c_kww':{'c_kuu':1}}),
   'H5 taut foil on board-following frame':dict(tie={'c_kuu':{'h_kuu':.5,'h_kww':-.5},'c_kww':{'h_kuu':-.5,'h_kww':.5}}),
   'H6 cathode x follows board, y flat (c_kuu=h_kuu, c_kww=0)':dict(tie={'c_kuu':{'h_kuu':1}},zero=('c_kww',)),
   'H7 c_kuu=h_kuu only':dict(tie={'c_kuu':{'h_kuu':1}}),
   'H8 c_kww=0 only':dict(zero=('c_kww',))}
rows=[];L=[]
for sc,lab in ((1.0,'committed floors'),(0.5,'T,v floors x0.5'),(2.0,'T,v floors x2')):
    fl=dict(FLOORS); fl['T']*=sc; fl['v']*=sc
    R0=Fitter(CLASSES,fl).fit()
    L.append(f'**{lab}** (free fit chi2 {R0.chi2:.1f}; per class '+', '.join(f'{c} {v[0]:.0f}/{v[1]}' for c,v in R0.per.items())+')\n')
    L.append('| hypothesis | n constraints | dchi2 | p (chi2) | sigma-equiv | per-class dchi2 (hp, w0, d, G, GY, T, v) |\n|---|---|---|---|---|---|')
    for n,kw in H.items():
        R=Fitter(CLASSES,fl,**kw).fit(); d=R.chi2-R0.chi2; k=R0.npar-R.npar
        p=stats.chi2.sf(d,k); sg=stats.norm.isf(p/2) if p>1e-300 else np.inf
        pc=' '.join(f'{R.per[c][0]-R0.per[c][0]:+.1f}' for c in CLASSES)
        L.append(f'| {n} | {k} | {d:+.1f} | {p:.2g} | {sg:.1f} | {pc} |')
    L.append('')
open(f'{OUT}/sec5.md','w').write('\n'.join(L)); print('\n'.join(L))
