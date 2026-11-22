import pandas as pd, numpy as np, glob, sys
sys.path.insert(0,'.')
from sept26_prelim_analysis.capsule_tof import crossing
d=pd.concat([pd.read_parquet(f) for f in glob.glob('/media/ucla/work/ntof_pass1/ana/ringing/wq_run_*.parquet')])
d=d[d.t_since_flash_ns.notna()].copy(); d['ms']=d.t_since_flash_ns/1e6
d['pos'],d['disp']=d.p0_x,-d.p0_z/d.d_z*d.d_x
d['qmin']=np.minimum(d.x_q_sum,d.y_q_sum); d['ncs']=np.minimum(d.x_n_sig_corr,d.y_n_sig_corr)
rng=np.random.default_rng(0)
for a in 'AC':
    g=d[d.arm==a].copy()
    late=g[(g.ms>=20)&(g.qmin>=600)]; u0,_,_=crossing(late.pos.to_numpy(),late.disp.to_numpy())
    def exc(s):
        if len(s)<40: return np.nan
        m=np.abs(s.pos+s.disp-u0); sh=np.abs(s.pos.to_numpy()+rng.permutation(s.disp.to_numpy())-u0)
        return (m<15).mean()-(sh<15).mean()
    exl=exc(g[g.ms>=30])
    g['cat']=np.select([(g.ncs<3)&(g.qmin<600),(g.ncs<3),(g.qmin<600)],['both','ring only','low-q only'],'neither')
    print(a,'late excess',round(exl,3),' purity (excess/late) and count per category:')
    for lo,hi in ((3,8),(8,12),(12,16),(16,20)):
        s=g[(g.ms>=lo)&(g.ms<hi)]
        print(f'  {lo}-{hi} ms:', '  '.join(f'{c}: {exc(s[s.cat==c])/exl:.2f} (n {int((s.cat==c).sum())})' for c in ('neither','ring only','low-q only','both')))
print('--- the "neither" tracks (ringing-free, normal charge): wall fraction and x/y charge balance')
for a in 'AC':
    g=d[d.arm==a].copy()
    late=g[(g.ms>=20)&(g.qmin>=600)]; u0,_,_=crossing(late.pos.to_numpy(),late.disp.to_numpy())
    g['miss']=g.pos+g.disp-u0; g['wall']=g[f'wall_{a}'].astype('boolean').fillna(False).astype(bool)
    g=g[(g.ncs>=3)&(g.qmin>=600)]
    g['lb']=np.log(g.x_q_sum/g.y_q_sum); mu=g[g.ms>=30].lb.median()
    for lo,hi in ((3,8),(8,12),(12,16),(30,75)):
        for nm,m in (('pointing |miss|<10',g.miss.abs()<10),('not pointing |miss|>40',g.miss.abs()>40)):
            s=g[(g.ms>=lo)&(g.ms<hi)&m]
            print(f'  {a} {lo}-{hi} ms {nm:24s} n {len(s):5d} wall {s.wall.mean():.2f}  balance MAD {1.4826*np.median(np.abs(s.lb-mu)):.3f}  frac |bal|>0.5 {np.mean(np.abs(s.lb-mu)>0.5):.2f}  qmin med {s.qmin.median():.0f}')
