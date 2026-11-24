import numpy as np, pandas as pd
S=pd.read_parquet('/media/ucla/work/claude-jobs/eb8200b3/T7_stability/stability.parquet',columns=['run','subrun','arm','rclass','n_trig_late'])
S=S[(S.arm=='D')&(S.rclass=='production')]
d=pd.read_parquet('db_all.parquet',columns=['run','subrun','arm','t_since_flash_ns','gated','gated_uncensored','censored_window','d_faint','gated_precut','x_local','y_local','wall_D','dca_source_mm','rn'],filters=[('arm','==','D')])
d=d[(d.t_since_flash_ns>=25e6)&(d.y_local.abs()<150)]
d['reg']=pd.cut(d.x_local,[-200,-60,0,8,40,60,120,200],labels=['<-60','-60..0','0..8','8..40','40..60','60..120','>120'])
g=d[d.gated].groupby(['run','reg'],observed=False).size().unstack()
gu=d[d.gated_uncensored].groupby(['run','reg'],observed=False).size().unstack()
nt=S.groupby('run').n_trig_late.sum()
g=g.div(nt,axis=0)*1e3; g['rn']=g.index.str[4:].astype(int); g=g.sort_values('rn')
pd.set_option('display.width',200); print(g.round(2).to_string())
# hole deficit
E=np.arange(-100,121,2.); c=0.5*(E[1:]+E[:-1])
for per,sel in (('P1',(d.rn>83)&(d.rn<=132)),('P2',d.rn>=135)):
    h=np.histogram(d[sel&d.gated].x_local,E)[0].astype(float)
    lev=np.interp(c,[ -6, 64],[h[(c>-10)&(c<-2)].mean(),h[(c>60)&(c<68)].mean()])
    hole=(c>0)&(c<60)
    print(per,'observed in 0..60',h[hole].sum(),'expected (linear interp of the edges)',lev[hole].sum().round(0),'missing',(lev[hole].sum()-h[hole].sum()).round(0),
      'fraction of late gated |y|<150 in -100..120:',((lev[hole].sum()-h[hole].sum())/(h.sum()+lev[hole].sum()-h[hole].sum())).round(4), 'eff in hole',(h[hole].sum()/lev[hole].sum()).round(4))
    for lo,hi in ((0,8),(8,16),(16,24),(24,32),(32,48),(48,60)):
        s=(c>lo)&(c<hi); print('   ',lo,hi,'eff',(h[s].sum()/lev[s].sum()).round(4))
print('--- 60..120 funnel per 1000 late trig')
s=d[d.reg=='60..120']
f=pd.DataFrame({'all':s.groupby('run').size(),'uncens':s[s.gated_uncensored].groupby('run').size(),'precut':s[s.gated_precut].groupby('run').size(),'gated':s[s.gated].groupby('run').size(),
 'cens':s[s.gated_uncensored&s.censored_window].groupby('run').size(),'dfaint':s[s.gated_precut&s.d_faint].groupby('run').size()})
f=f.div(nt,axis=0)*1e3; f['rn']=f.index.str[4:].astype(int); print(f.sort_values('rn').round(2).to_string())
