import sys, pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
Q=list(QUANT); cells=[(i,j) for i in range(4) for j in range(4)]
def qv(R): return {k:v[0] for k,v in quants(R).items()}
kw=dict(zero=('g0',))
cl=ARMS['timing']
Rf=Fitter(cl,FLOORS,**kw).fit(); full=qv(Rf); fe=quants(Rf)
L=[qv(Fitter(cl,FLOORS,cells=set(cells)-{c},**kw).fit()) for c in cells]
rng=np.random.default_rng(7); B=[]
for b in range(400):
    pick=rng.integers(0,16,16); w={c:0 for c in cells}
    for p in pick: w[cells[p]]+=1
    try: B.append(qv(Fitter(cl,FLOORS,weights=w,**kw).fit()))
    except Exception: pass
pickle.dump(dict(full=full,fe=fe,L=L,B=B,chi2=Rf.chi2,n=Rf.n),open(f'{OUT}/timing_g0fixed.pkl','wb'))
out=['**timing, g0 fixed at 28.2 (T and v alone cannot fix g0: it is degenerate with the per-run offsets)**\n','| quantity | full | fit err | jackknife | bootstrap std | boot 16-84 hw |\n|---|---|---|---|---|---|']
for k in Q:
    v=np.array([x[k] for x in L]); n=len(v); jk=np.sqrt((n-1)/n*np.sum((v-v.mean())**2))
    b=np.array([x[k] for x in B])
    out.append(f'| {k} | {full[k]:+.2f} | {fe[k][1]:.2f} | {jk:.2f} | {np.std(b):.2f} | {(np.percentile(b,84)-np.percentile(b,16))/2:.2f} |')
open(f'{OUT}/sec2_timing.md','w').write('\n'.join(out)); print('\n'.join(out), Rf.chi2, Rf.n)
