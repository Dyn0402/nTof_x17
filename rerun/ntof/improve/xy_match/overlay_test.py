# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, glob, json
D='/media/ucla/work/ntof_pass1/ana/xy_match'
T=pd.concat([pd.read_parquet(f) for f in glob.glob(D+'/combos/*.parquet')])
BUN={'A':'A_E2_beam','C':'C_E2_beam','D':'D_E2_beam_L1'}
FTST=10.
def feats(xq,yq,xt0,yt0,xu50,yu50,xu90,yu90,xue,yue,xf,yf):
    ph=FTST*(xf-yf)
    return dict(tc=xt0+np.nan_to_num(xu50)-yt0-np.nan_to_num(yu50)+ph, lq=np.log(np.maximum(xq,1)/np.maximum(yq,1)),
                du50=xu50-yu50, du90=xu90-yu90, duend=xue-yue)
rng=np.random.default_rng(1)
def rsig(v): v=v[np.isfinite(v)]; q=np.percentile(v,[16,50,84]); return q[1],(q[2]-q[0])/2
for arm in 'ACD':
    p=json.load(open(f'/media/ucla/work/ntof_pass1/bundles/pf/{BUN[arm]}/bundle.json'))['xy_pairing']
    S=T[(T.arm==arm)&(T.ncx==1)&(T.ncy==1)&(T.t_ms>=30)&T.wall&T.x_plausible&T.y_plausible].reset_index(drop=True)
    n=len(S); half=n//2
    # calibrate n_TOF widths on first half (even rows), test on odd
    cal=S.iloc[::2]; tst=S.iloc[1::2].reset_index(drop=True)
    ntof={f:rsig(cal[f].to_numpy()) for f in ['tc','lq','du50','du90','duend']}
    # histogram LR: signal = cal; background = mixed calibration from shuffled cal
    perm=rng.permutation(len(cal))
    def g(df,pre,k): return df[f'{pre}_{k}'].to_numpy()
    def F(dx,dy): return feats(g(dx,'x','q_sum'),g(dy,'y','q_sum'),g(dx,'x','t0'),g(dy,'y','t0'),g(dx,'x','q_u50'),g(dy,'y','q_u50'),g(dx,'x','q_u90'),g(dy,'y','q_u90'),g(dx,'x','q_uend'),g(dy,'y','q_uend'),g(dx,'x','ftst'),g(dy,'y','ftst'))
    Bc=F(cal, cal.iloc[perm])
    edges={'tc':np.linspace(-800,800,81),'lq':np.linspace(-3,3,121),'du90':np.linspace(-800,800,41),'duend':np.linspace(-1200,1200,41)}
    def llr(f, keys):
        out=0
        for k in keys:
            e=edges[k]; hs,_=np.histogram(cal[k],e); hb,_=np.histogram(Bc[k],e)
            hs=(hs+0.5)/hs.sum(); hb=(hb+0.5)/hb.sum()
            i=np.clip(np.digitize(np.nan_to_num(f[k],nan=1e9),e)-1,0,len(e)-2)
            out=out+np.log(hs[i]/hb[i])
        return out
    def cost_bench(f):
        c=0
        for k in p['features']:
            kk={'u50':'du50','u90':'du90'}.get(k,k)
            m=p['median'][k]; s=p['rsig'][k]
            if k=='lq': m,s=p['lq_class']['sat']['median'],p['lq_class']['sat']['rsig']
            z=(f[kk]-m)/s; c=c+np.where(np.isfinite(z),np.minimum(z*z,25),25)
        return c
    def cost_ntof(f, keys):
        c=0
        for k in keys:
            m,s=ntof[k]; z=(f[k]-m)/s; c=c+np.where(np.isfinite(z),np.minimum(z*z,25),25)
        return c
    # overlay test: pair i with partner j = shuffled
    j=rng.permutation(len(tst)); A_=tst; B_=tst.iloc[j].reset_index(drop=True)
    keep=[F(A_,A_),F(B_,B_)]; swap=[F(A_,B_),F(B_,A_)]
    res={}
    for name,fn in [('bench cost',cost_bench),
                    ('ntof chi2 tc+lq+u50+u90',lambda f:cost_ntof(f,['tc','lq','du50','du90'])),
                    ('ntof chi2 lq',lambda f:cost_ntof(f,['lq'])),
                    ('ntof chi2 tc',lambda f:cost_ntof(f,['tc'])),
                    ('LLR tc+lq+u90+uend',lambda f:-llr(f,['tc','lq','du90','duend'])),
                    ('LLR lq',lambda f:-llr(f,['lq'])),
                    ('LLR tc+lq',lambda f:-llr(f,['tc','lq']))]:
        k=fn(keep[0])+fn(keep[1]); s=fn(swap[0])+fn(swap[1])
        res[name]=np.mean(k<s)+0.5*np.mean(k==s)
    print(arm, 'n test', len(tst), 'ntof widths', {k:round(v[1],3) for k,v in ntof.items()})
    for k,v in res.items(): print(f'   {k:28s} correct {v:.3f}')
