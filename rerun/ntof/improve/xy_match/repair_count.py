# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, glob, itertools
D='/media/ucla/work/ntof_pass1/ana/xy_match'
T=pd.concat([pd.read_parquet(f) for f in glob.glob(D+'/combos/*.parquet')],ignore_index=True)
rng=np.random.default_rng(2)
edges={'tc':np.linspace(-800,800,81),'lq':np.linspace(-3,3,121),'du90':np.linspace(-800,800,41),'duend':np.linspace(-1200,1200,41)}
KEYS=['tc','lq','du90','duend']
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
rows=[]
for arm in 'ACD':
    L=tables(arm); a=T[T.arm==arm].copy(); a['llr']=llr(a,L)
    a['tbin']=pd.cut(a.t_ms,[1,3,5,8,12,16,20,30,1000])
    multi=a[((a.ncx>1)|(a.ncy>1))].copy()
    multi['ek']=multi.groupby(['run','subrun','tag','event_id'],sort=False).ngroup()
    multi=multi.sort_values('ek')
    ek=multi.ek.values; xr=multi.x_rank.values.astype(int); yr=multi.y_rank.values.astype(int)
    ll=multi.llr.values; g1=multi.gated1.values; tb=multi.tbin.values
    bnd=np.flatnonzero(np.diff(ek))+1; starts=np.r_[0,bnd]; ends=np.r_[bnd,len(ek)]
    from scipy.optimize import linear_sum_assignment
    for s0,e0 in zip(starts,ends):
        gg=g1[s0:e0]
        if not gg.any():
            continue
        gx=xr[s0:e0]; gy=yr[s0:e0]; gl=ll[s0:e0]
        ux=np.unique(gx); uy=np.unique(gy)
        Mx=np.full((len(ux),len(uy)),-50.0)
        Mx[np.searchsorted(ux,gx),np.searchsorted(uy,gy)]=gl
        k=min(len(ux),len(uy),3)
        r,c=linear_sum_assignment(-Mx)
        o=np.argsort(-Mx[r,c])[:k]
        best={(ux[r[i]],uy[c[i]]) for i in o}
        g1s=[(gx[i],gy[i]) for i in np.flatnonzero(gg)]
        rows.append(dict(arm=arm,tbin=tb[s0],ncx=len(ux),ncy=len(uy),n_g1=len(g1s),
                         g1_changed=sum(q not in best for q in g1s)))
R=pd.DataFrame(rows)
R.to_parquet('/media/ucla/work/ntof_pass1/ana/xy_match/R3.parquet')
for arm in 'ACD':
    r=R[R.arm==arm]
    s=r.groupby('tbin',observed=True).agg(events=('n_g1','size'),gated=('n_g1','sum'),changed=('g1_changed','sum'))
    s['frac_changed']=(s.changed/s.gated).round(3)
    tot=T[(T.arm==arm)&T.gated1].assign(tbin=lambda d:pd.cut(d.t_ms,[1,3,5,8,12,16,20,30,1000])).groupby('tbin',observed=True).size()
    s['all_gated']=tot; s['changed_of_all']=(s.changed/s.all_gated).round(4)
    print(arm); print(s)
