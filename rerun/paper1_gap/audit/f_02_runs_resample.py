import sys, json, pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
ARMLIST=['all','tilt','gap','geo','timing','no_hp','geo_gap']
Q=list(QUANT)
def qv(R): return {k:v[0] for k,v in quants(R).items()}
res={}
# per-run
pr={}
for arm in ARMLIST:
    for runs in (['run1'],['run2'],['run1','run2']):
        try:
            R=Fitter(ARMS[arm],FLOORS,runs=runs).fit()
            pr[(arm,'+'.join(runs))]=dict(q=quants(R),chi2=R.chi2,n=R.n,npar=R.npar,per=R.per)
        except Exception as e:
            pr[(arm,'+'.join(runs))]=dict(err=str(e))
res['perrun']=pr
cells=[(i,j) for i in range(4) for j in range(4)]
rng=np.random.default_rng(12345)
jk={};bs={};rows={};cols={}
for arm in ARMLIST:
    cl=ARMS[arm]
    full=qv(Fitter(cl,FLOORS).fit())
    L=[]
    for c in cells:
        try: L.append(qv(Fitter(cl,FLOORS,cells=set(cells)-{c}).fit()))
        except Exception: L.append({k:np.nan for k in Q})
    jk[arm]=dict(full=full,leave=L)
    Lr=[];Lc=[]
    for a in range(4):
        try: Lr.append(qv(Fitter(cl,FLOORS,cells={c for c in cells if c[0]!=a}).fit()))
        except Exception: Lr.append({k:np.nan for k in Q})
        try: Lc.append(qv(Fitter(cl,FLOORS,cells={c for c in cells if c[1]!=a}).fit()))
        except Exception: Lc.append({k:np.nan for k in Q})
    rows[arm]=Lr;cols[arm]=Lc
    B=[]
    for b in range(400):
        pick=rng.integers(0,16,16)
        w={}
        for p in pick: w[cells[p]]=w.get(cells[p],0)+1
        w={c:w.get(c,0) for c in cells}
        try:
            F=Fitter(cl,FLOORS,weights=w)
            B.append(qv(F.fit()))
        except Exception: pass
    bs[arm]=B
    print(arm,'done',len(B),flush=True)
res.update(jk=jk,rows=rows,cols=cols,bs=bs)
pickle.dump(res,open(f'{OUT}/resample.pkl','wb'))
