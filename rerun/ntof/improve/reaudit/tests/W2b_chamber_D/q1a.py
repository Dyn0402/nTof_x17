import numpy as np, pandas as pd
S=pd.read_parquet('/media/ucla/work/claude-jobs/eb8200b3/T7_stability/stability.parquet',columns=['run','subrun','arm','rclass','n_trig_late','t'])
S=S[(S.arm=='D')]
d=pd.read_parquet('db_all.parquet',columns=['run','subrun','arm','t_since_flash_ns','gated','gated_precut','gated_uncensored','censored_window','d_faint','x_local','y_local','wall_D','dca_source_mm','rn'],filters=[('arm','==','D')])
d=d[d.t_since_flash_ns>=25e6]
d=d.merge(S[['run','subrun','n_trig_late','t']],on=['run','subrun'])
d['per']=np.select([d.rn<=81,d.rn<=132],['E1','P1'],'P2')
E=np.arange(-200,201,4.); 
def reg(x,lo,hi): return (x>=lo)&(x<hi)
ntr=S[S.rclass=='production'].assign(rn=lambda s:s.run.str[4:].astype(int))
ntr['per']=np.select([ntr.rn<=81,ntr.rn<=132],['E1','P1'],'P2')
NT=ntr.groupby('per').n_trig_late.sum()
print('late triggers',NT.to_dict())
rows=[]
for per,g in d.groupby('per'):
    for lab,(lo,hi) in {'x16-24':(16,24),'x8-40':(8,40),'x16-60':(16,60),'x-200..200':(-200,200),'x60-200':(60,200),'x-200..8':(-200,8)}.items():
        h=g[reg(g.x_local,lo,hi)&(g.y_local.abs()<200)]
        n=NT[per]
        rows.append(dict(per=per,region=lab,all=len(h)/n*1e3,precut=h.gated_uncensored.sum()/n*1e3,uncens=h.gated_uncensored.sum()/n*1e3,
                         gprecut=h.gated_precut.sum()/n*1e3,gated=h.gated.sum()/n*1e3,
                         cens=(h.gated_uncensored&h.censored_window).sum()/n*1e3,dfaint=(h.gated_precut&h.d_faint).sum()/n*1e3,
                         wall_g=h[h.gated].wall_D.mean(),dca15_g=(h[h.gated].dca_source_mm<15).mean(),
                         wall_cens=h[h.gated_uncensored&h.censored_window].wall_D.mean(),dca15_cens=(h[h.gated_uncensored&h.censored_window].dca_source_mm<15).mean(),
                         wall_df=h[h.gated_precut&h.d_faint].wall_D.mean(),dca15_df=(h[h.gated_precut&h.d_faint].dca_source_mm<15).mean()))
R=pd.DataFrame(rows); pd.set_option('display.width',250); print(R.round(3).to_string())
