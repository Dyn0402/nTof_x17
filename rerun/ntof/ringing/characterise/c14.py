import pandas as pd, numpy as np, glob, sys
sys.path.insert(0,'.')
from sept26_prelim_analysis.capsule_tof import crossing
d=pd.concat([pd.read_parquet(f) for f in glob.glob('/media/ucla/work/ntof_pass1/ana/ringing/wq_run_*.parquet')])
d=d[d.t_since_flash_ns.notna()].copy(); d['ms']=d.t_since_flash_ns/1e6
d['pos'],d['disp']=d.p0_x,-d.p0_z/d.d_z*d.d_x
d['qmin']=np.minimum(d.x_q_sum,d.y_q_sum); d['ncs']=np.minimum(d.x_n_sig_corr,d.y_n_sig_corr)
rng=np.random.default_rng(0); EDG=[1,3,5,8,12,16,20,30,75]; TB=[0,0.05,0.1,0.2,0.3,0.4,0.5]
out=[]
for run in ['all']+sorted(d.run.unique()):
  for a in 'AC':
    g=d[(d.arm==a)&((d.run==run)|(run=='all'))].copy()
    late=g[(g.ms>=20)&(g.qmin>=600)]; u0,_,_=crossing(late.pos.to_numpy(),late.disp.to_numpy())
    g['miss']=g.pos+g.disp-u0; g['wall']=g[f'wall_{a}'].astype('boolean').fillna(False).astype(bool)
    ref=((g.ms>=30)&g.wall&(g.miss.abs()<15)).to_numpy()
    def exc(s):
        if len(s)<40: return np.nan
        m=np.abs(s.miss); sh=np.abs(s.pos.to_numpy()+rng.permutation(s.disp.to_numpy())-u0)
        return (m<15).mean()-(sh<15).mean()
    exl=exc(g[g.ms>=30])
    rules={'gated (pass 1)':np.ones(len(g),bool),'q >= 600 ADC':(g.qmin>=600).to_numpy(),'ringing-free (>=1 ch > 5 sigma after removal)':(g.ncs>=1).to_numpy(),
           'q >= 600 and >= 1 ch after removal':((g.qmin>=600)&(g.ncs>=1)).to_numpy(),'q >= 600 and >= 2 ch after removal':((g.qmin>=600)&(g.ncs>=2)).to_numpy()}
    for nm,c in rules.items():
        row=dict(run=run,arm=a,rule=nm,ref_kept=round(c[ref].mean(),4))
        for lo,hi in zip(EDG[:-1],EDG[1:]):
            m=((g.ms>=lo)&(g.ms<hi)).to_numpy(); s0,s1=g[m],g[m&c]; e0,e1=exc(s0),exc(s1)
            row[f'{lo}-{hi}']=f'{len(s1)/len(s0):.2f}/{(len(s1)*e1/(len(s0)*e0) if e0>0.01 else np.nan):.2f}/{e1/exl:.2f}'
        L=((g.ms>=30)&g.wall).to_numpy(); i=np.digitize(g.tanx.abs()[L],TB)-1; cl=c[L]
        row['eff vs |tan x| (late wall)']=' '.join(f'{cl[i==k].mean():.3f}' for k in range(len(TB)-1))
        out.append(row)
o=pd.DataFrame(out); pd.set_option('display.width',400); pd.set_option('display.max_colwidth',60)
print(o.to_string(index=False)); o.to_csv('/media/ucla/work/ntof_pass1/ana/ringing/wq_rules.csv',index=False)
