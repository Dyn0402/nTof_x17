import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=pd.read_parquet('db_all.parquet',columns=['run','subrun','arm','t_since_flash_ns','gated','gated_precut','gated_uncensored','x_local','y_local','wall_D','dca_source_mm','rn','x_n_strips','x_q_sum','chi2dof_x'],filters=[('arm','==','D')])
prod=np.isfinite(d.t_since_flash_ns)
L=d[(d.t_since_flash_ns>=25e6)]
L=L.assign(per=np.select([L.rn<=81,L.rn<=132],['E1','P1'],'P2'))
reg=pd.read_csv('/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/ntof_registry.csv',usecols=['run','subrun','rclass']).drop_duplicates(['run','subrun'])
reg['run']='run_'+reg.run.astype(str)
cosk=set(zip(reg[reg.rclass=='cosmic'].run,reg[reg.rclass=='cosmic'].subrun))
Cz=d[[ (r,s) in cosk for r,s in zip(d.run,d.subrun)]]
Cz=Cz.assign(per=np.select([Cz.rn<=81,Cz.rn<=132],['E1','P1'],'P2'))
E=np.arange(-100,121,2.); c=0.5*(E[1:]+E[:-1])
fig,ax=plt.subplots(2,1,figsize=(7,6.2),sharex=True)
out={}
for k,(S,lab,sel) in enumerate([(L,'late (>=25 ms) gated, |y|<150','gated'),(Cz,'beam-off cosmic runs, all tracks |y|<150',None)]):
    for per,col in (('P1','C0'),('P2','C3'),('E1','0.6')):
        g=S[(S.per==per)&(S.y_local.abs()<150)]
        if sel: g=g[g[sel]]
        h=np.histogram(g.x_local,E)[0].astype(float)
        norm=h[(c>60)&(c<120)].sum()
        out[(lab[:4],per)]=(h,norm)
        ax[k].step(c,h/norm*30,where='mid',color=col,label=f'{per} (n={len(g)})')
    ax[k].set_title(lab,fontsize=9); ax[k].legend(fontsize=8); ax[k].set_ylabel('per bin / mean bin of x 60-120')
    for x0 in (16,24): ax[k].axvline(x0,color='k',lw=0.5,ls=':')
ax[1].set_xlabel('D x_local (mm)')
fig.tight_layout(); fig.savefig('fig_q1_xprofile.png',dpi=130)
# table: ratio P2/P1 normalised to x60-120 in regions
for key in ('late','beam'):
    h1,n1=out[(key,'P1')]; h2,n2=out[(key,'P2')]
    for lo,hi in ((-40,-20),(-20,0),(0,8),(8,16),(16,24),(24,32),(32,40),(40,50),(50,60),(60,80),(80,120)):
        s=(c>lo)&(c<hi)
        a,b=h1[s].sum()/n1,h2[s].sum()/n2
        print(key,f'x {lo}..{hi}: P1 {a:.3f} P2 {b:.3f} ratio {b/a:.2f} +- {b/a*np.sqrt(1/h1[s].sum()+1/h2[s].sum()):.2f}')
# properties of late gated in 16-24 vs 40-60
for per in ('P1','P2'):
    for lo,hi in ((16,24),(24,40),(40,60),(60,100)):
        g=L[(L.per==per)&L.gated&(L.x_local>=lo)&(L.x_local<hi)&(L.y_local.abs()<150)]
        print(per,lo,hi,len(g),'wall %.3f dca15 %.3f nstr %.1f q %.0f chi2 %.2f'%(g.wall_D.mean(),(g.dca_source_mm<15).mean(),g.x_n_strips.median(),g.x_q_sum.median(),g.chi2dof_x.median()))
