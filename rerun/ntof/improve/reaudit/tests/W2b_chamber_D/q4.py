import sys, numpy as np, pandas as pd
sys.path.insert(0,'/media/ucla/work/claude-jobs/eb8200b3/T2_ref_compare')
import compare as C
C.COLS=sorted(set(C.COLS+['gated_precut','gated_reco','in_acceptance','x_cens_frac','y_cens_frac','x_n_flagged','y_n_flagged','amp_norm','x_n_strips','y_n_strips','x_p0','y_p0','x_q_sum','y_q_sum','tanx']))
cfg=C.read_view('/media/ucla/work/claude-jobs/eb8200b3/T2_ref_compare/all/final'); cfg=cfg[cfg.arm=='D']
acq=C.acquisitions(cfg); print('D acquisitions',sorted(acq))
ref=C.read_view('/media/ucla/work/ntof_pass1/stage3_r2_final',keep=acq)
for t in (cfg,ref):
    for c in ('gated_precut','gated_reco','in_acceptance'): t[c]=t[c].astype('boolean').fillna(False).astype(bool)
K=C.KEY
def nearest(src,other):
    """for each src row the nearest other-set track (any state) in the same event, by (x_local,y_local)"""
    o=other.copy(); o['_j']=np.arange(len(o))
    g=o.groupby(K)['_j'].apply(list).to_dict()
    ox,oy=o.x_local.to_numpy(float),o.y_local.to_numpy(float)
    js=[];ds=[]
    for k,x,y in zip(map(tuple,src[K].to_numpy()),src.x_local.to_numpy(float),src.y_local.to_numpy(float)):
        l=g.get(k)
        if not l: js.append(-1); ds.append(np.nan); continue
        d=np.hypot(ox[l]-x,oy[l]-y); d=np.where(np.isfinite(d),d,1e9); i=int(np.argmin(d)); js.append(l[i]); ds.append(d[i])
    js=np.array(js); r=o.iloc[np.where(js>=0,js,0)].reset_index(drop=True).add_suffix('_o'); r.loc[js<0,:]=np.nan
    out=src.reset_index(drop=True).join(r); out['dist_o']=ds; return out
L_ref=ref[(ref.ms>=25)]; L_cfg=cfg[(cfg.ms>=25)]
lost=nearest(L_ref[L_ref.gated],L_cfg)
lost=lost[~(lost.gated_o.astype('boolean').fillna(False).astype(bool)&(lost.dist_o<=5))]   # ref gated with no gated cfg track within 5 mm
added=nearest(L_cfg[L_cfg.gated],L_ref)
added=added[~(added.gated_o.astype('boolean').fillna(False).astype(bool)&(added.dist_o<=5))]
def why(r):
    if not np.isfinite(r.dist_o): return 'no track in event'
    if r.dist_o>5: return 'nearest track >5 mm away'
    w=[]
    if not r.gated_reco_o: w.append('not reco-gated')
    elif not r.in_acceptance_o: w.append('acceptance')
    if r.gated_uncensored_o and r.censored_window_o: w.append('censored')
    if r.gated_precut_o and r.d_faint_o: w.append('d_faint')
    return '+'.join(w) or 'other'
for nm,t,sgn in (('LOST (ref gated, cfg not)',lost,1),('ADDED (cfg gated, ref not)',added,-1)):
    t['why']=[why(r) for r in t.itertuples()]
    # newly flagged channels in window: cfg n_flagged - ref n_flagged
    if sgn==1: t['dfx']=t.x_n_flagged_o-t.x_n_flagged; t['dfy']=t.y_n_flagged_o-t.y_n_flagged
    else: t['dfx']=t.x_n_flagged-t.x_n_flagged_o; t['dfy']=t.y_n_flagged-t.y_n_flagged_o
    print('\n==',nm,'n',len(t),'wall %.3f dca15 %.3f'%(t.wall.mean(),(t.dca_source_mm<15).mean()))
    print(t.why.value_counts().to_string())
    s=t[t.dist_o<=5]
    print(' same track (<=5 mm) n',len(s),': newly flagged in window x>0 %.2f y>0 %.2f; median dfx %.1f dfy %.1f'%((s.dfx>0).mean(),(s.dfy>0).mean(),s.dfx.median(),s.dfy.median()))
    for lab,c in (('censored',s.why.str.contains('censored')),('d_faint',s.why.str.contains('d_faint')),('not reco-gated',s.why.str.contains('not reco'))):
        u=s[c]
        if len(u): print('   %s n %d: y_cens ref %.2f -> cfg %.2f ; x_cens %.2f -> %.2f; dfy>0 %.2f; wall %.2f dca15 %.2f'%(lab,len(u),
            (u.y_cens_frac if sgn==1 else u.y_cens_frac_o).median(),(u.y_cens_frac_o if sgn==1 else u.y_cens_frac).median(),
            (u.x_cens_frac if sgn==1 else u.x_cens_frac_o).median(),(u.x_cens_frac_o if sgn==1 else u.x_cens_frac).median(),(u.dfy>0).mean(),u.wall.mean(),(u.dca_source_mm<15).mean()))
    t.to_csv(f'q4_{"lost" if sgn==1 else "added"}.csv',index=False)
# paired late gated tracks: t0 shift vs newly flagged
P,oa,ob=C.pair(L_cfg,L_ref)
P2=P.merge(L_ref[K+['x_local','y_local','x_n_flagged','y_n_flagged']].rename(columns={'x_n_flagged':'xf_ref','y_n_flagged':'yf_ref','x_local':'x_local_ref','y_local':'y_local_ref'}),on=K+['x_local_ref','y_local_ref'],how='left')
P2['dfx']=P2.x_n_flagged-P2.xf_ref; P2['dfy']=P2.y_n_flagged-P2.yf_ref
P2['dtx']=P2.x_t0-P2.x_t0_ref; P2['dty']=P2.y_t0-P2.y_t0_ref
print('\n== paired late gated D',len(P2))
for lab,s in (('no newly flagged (dfx=0&dfy=0)',(P2.dfx==0)&(P2.dfy==0)),('dfx>0',P2.dfx>0),('dfy>0',P2.dfy>0),('dfx=0',P2.dfx==0),('dfy=0',P2.dfy==0)):
    g=P2[s]; print(' %-32s n %4d (%.2f) x_t0 shift med %+.1f ns, y_t0 %+.1f ns; frac |dtx|>1ns %.2f; |dx|>0.5mm %.2f'%(lab,len(g),len(g)/len(P2),g.dtx.median(),g.dty.median(),(g.dtx.abs()>1).mean(),((g.x_local-g.x_local_ref).abs()>0.5).mean()))
P2.to_csv('q4_paired.csv',index=False)
