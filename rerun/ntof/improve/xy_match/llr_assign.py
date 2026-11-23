# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, glob
from scipy.optimize import linear_sum_assignment
D='/media/ucla/work/ntof_pass1/ana/xy_match'
T=pd.concat([pd.read_parquet(f) for f in glob.glob(D+'/combos/*.parquet')],ignore_index=True)
ST=pd.concat([pd.read_parquet(f'/media/ucla/work/ntof_pass1/ana/scint_stack/tracks/stack_run_{r}.parquet') for r in (145,104,139)])
rng=np.random.default_rng(2)
edges={'tc':np.linspace(-800,800,81),'lq':np.linspace(-3,3,121),'du90':np.linspace(-800,800,41),'duend':np.linspace(-1200,1200,41)}
KEYS=['tc','lq','du90','duend']
UOFF={'A':182.94,'C':181.99,'D':183.79}; KX={'A':1.0423,'C':0.989,'D':0.9825}
WALL_EDGES=np.array([-225.0,-125.0,-25.0,75.0,175.0]); LW=97.4*0.85; LS=234.6
def tables(arm):
    S=T[(T.arm==arm)&(T.ncx==1)&(T.ncy==1)&(T.t_ms>=30)&T.wall&T.x_plausible&T.y_plausible]
    p=rng.permutation(len(S)); X=S.reset_index(drop=True); Y=S.iloc[p].reset_index(drop=True)
    ph=10*(X.x_ftst.values-Y.y_ftst.values)
    B=dict(tc=X.x_t0.values+np.nan_to_num(X.x_q_u50.values)-Y.y_t0.values-np.nan_to_num(Y.y_q_u50.values)+ph,
           lq=np.log(np.maximum(X.x_q_sum.values,1)/np.maximum(Y.y_q_sum.values,1)),
           du90=X.x_q_u90.values-Y.y_q_u90.values, duend=X.x_q_uend.values-Y.y_q_uend.values)
    L={}
    for k in KEYS:
        e=edges[k]; hs,_=np.histogram(S[k],e); hb,_=np.histogram(B[k],e)
        L[k]=np.log(((hs+0.5)/hs.sum())/((hb+0.5)/hb.sum()))
    return L
def llr(df,L):
    o=np.zeros(len(df))
    for k in KEYS:
        e=edges[k]; i=np.clip(np.digitize(np.nan_to_num(df[k].values,nan=1e9),e)-1,0,len(e)-2); o+=L[k][i]
    return o
out=[]
for arm in 'ACD':
    a=T[T.arm==arm].copy(); a['llr']=llr(a,tables(arm))
    a['ek']=a.groupby(['run','subrun','tag','event_id'],sort=False).ngroup(); a=a.sort_values('ek').reset_index(drop=True)
    ek=a.ek.values; xr=a.x_rank.values.astype(int); yr=a.y_rank.values.astype(int); ll=a.llr.values; g1=a.gated1.values
    multi=((a.ncx>1)|(a.ncy>1)).values
    bnd=np.flatnonzero(np.diff(ek))+1; starts=np.r_[0,bnd]; ends=np.r_[bnd,len(ek)]
    chosen=np.zeros(len(a),bool)
    for s0,e0 in zip(starts,ends):
        if not multi[s0]:
            chosen[s0:e0]=True; continue
        gx=xr[s0:e0]; gy=yr[s0:e0]; gl=ll[s0:e0]
        ux=np.unique(gx); uy=np.unique(gy)
        M=np.full((len(ux),len(uy)),-50.0); ix=np.searchsorted(ux,gx); iy=np.searchsorted(uy,gy); M[ix,iy]=gl
        r,c=linear_sum_assignment(-M); k=min(len(ux),len(uy),3); o=np.argsort(-M[r,c])[:k]
        bs={(r[i],c[i]) for i in o}
        chosen[s0:e0]=[(i,j) in bs for i,j in zip(ix,iy)]
    a['chosen']=chosen
    # for each pass-1 gated pair that is not chosen: the chosen pair sharing its x (y changed) or its y (x changed)
    p1=a[a.gated1]; ch=a[a.chosen]
    keyx=['ek','x_rank']; keyy=['ek','y_rank']
    m=p1.merge(ch[keyx+['y_rank','llr']],on=keyx,how='left',suffixes=('','_new'))
    m['kind']=np.where(m.y_rank_new==m.y_rank,'same',np.where(m.y_rank_new.notna(),'y_changed','x_moved'))
    out.append((arm,a,m))
    print(arm, m.kind.value_counts().to_dict())
import pickle; pickle.dump(out,open('/media/ucla/work/ntof_pass1/ana/xy_match/ana4.pkl','wb'))
