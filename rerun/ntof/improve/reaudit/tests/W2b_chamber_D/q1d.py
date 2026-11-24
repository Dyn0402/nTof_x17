import numpy as np, pandas as pd
reg=pd.read_csv('/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/ntof_registry.csv',usecols=['tag','rclass'])
cos=set(reg[reg.rclass=='cosmic'].tag)
cols=['run','subrun','tag','event_id','arm','gated','gated_precut','gated_uncensored','censored_window','d_faint','x_local','y_local','p0_x','p0_y','p0_z','d_x_nomap','d_y_nomap','d_z_nomap','rn']
t=pd.read_parquet('db_all.parquet',columns=cols)
t=t[t.tag.isin(cos)]
K=['run','subrun','tag','event_id']
B=t[(t.arm=='B')&t.gated]; B=B[B.groupby(K).arm.transform('size')==1]
D=t[(t.arm=='D')&t.gated_uncensored.notna()]
D=D[D.groupby(K).arm.transform('size')==1]          # one D track in the event
M=B.merge(D,on=K,suffixes=('_b','_d'))
tt=(M.p0_x_b-M.p0_x_d)/M.d_x_nomap_d
mz=M.p0_z_d+tt*M.d_z_nomap_d-M.p0_z_b; my=M.p0_y_d+tt*M.d_y_nomap_d-M.p0_y_b
ok=np.isfinite(mz)&np.isfinite(my)
M['sep']=np.hypot(mz-np.nanmedian(mz[ok]),my-np.nanmedian(my[ok]))
M['per']=np.where(M.rn_d<=132,'P1','P2'); M.loc[M.rn_d<=81,'per']='E1'
M['hole']=(M.x_local_d>0)&(M.x_local_d<60)
M['side']=np.where(M.hole,'hole 0..60',np.where(M.x_local_d<0,'x<0','x>60'))
for per in ('P1','P2'):
    for side in ('x<0','hole 0..60','x>60'):
        g=M[(M.per==per)&(M.side==side)&(M.y_local_d.abs()<150)]
        for lab,s in (('gated',g.gated_d),('cens',g.censored_window_d&g.gated_uncensored_d),('dfaint',g.d_faint_d&g.gated_precut_d),('notgated',~g.gated_uncensored_d)):
            gg=g[s]
            print(per,side,lab,'n',len(gg),'sep<30 %.3f'%((gg.sep<30).mean() if len(gg) else np.nan))
# count per B track: fraction of B tracks (whose line hits D in the hole) that have a matched D gated track
M.to_parquet('bd_match.parquet',index=False)
