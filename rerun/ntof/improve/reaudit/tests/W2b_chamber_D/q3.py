import numpy as np, pandas as pd
reg=pd.read_csv('/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/ntof_registry.csv',usecols=['tag','rclass'])
cos=set(reg[reg.rclass=='cosmic'].tag)
t=pd.read_parquet('db_all.parquet',columns=['run','tag','event_id','arm','gated','t_since_flash_ns','x_local','y_local','x_n_strips','y_n_strips','x_q_sum','y_q_sum','rn','wall_B'],filters=[('arm','==','B')])
t['cos']=t.tag.isin(cos)
t['per']=np.select([t.rn<=106,t.rn<=132],['early<=106','mid107-132'],'late>=133')
box=(t.x_local.between(-52,-44))&(t.y_local.between(-200,-160))
nb=(t.x_local.between(-68,-28))&(t.y_local.between(-200,-160))&~box    # same y band, neighbours in x
for cos_,lab in ((False,'beam late>=25ms'),(True,'cosmic')):
    s=t[(t.cos==cos_)&t.gated]
    if not cos_: s=s[s.t_since_flash_ns>=25e6]
    for per,g in s.groupby('per'):
        b=g[box.loc[g.index]]; n=g[nb.loc[g.index]]
        dens_b=len(b)/8; dens_n=len(n)/32
        print(lab,per,'n gated',len(g),'box',len(b),'neigh',len(n),'box/neigh density %.2f'%(dens_b/max(dens_n,1e-9)),
              'box frac of all %.4f'%(len(b)/len(g)), 'box q med x %.0f y %.0f nstr x %.1f y %.1f'%(b.x_q_sum.median(),b.y_q_sum.median(),b.x_n_strips.median(),b.y_n_strips.median()),
              'neigh q x %.0f y %.0f'%(n.x_q_sum.median(),n.y_q_sum.median()))
# B-D lines
L=pd.read_parquet('/media/ucla/work/ntof_pass1/ana_r2/track_quality/bd_lines.parquet')
L['rn']=L.run.str[4:].astype(int)
L['per']=np.select([L.rn<=106,L.rn<=132],['early<=106','mid107-132'],'late>=133')
bx=L.x_local_b.between(-52,-44)&L.y_local_b.between(-200,-160)
import sys; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17')
from rerun.ntof.improve import bd_map as BD
for per,g in L.groupby('per'):
    b=g[bx.loc[g.index]]
    cz=BD.clean(g,'z'); cy=BD.clean(g,'y')
    cbz=BD.clean(b,'z') if len(b) else []
    print('BD lines',per,'n',len(g),'with B in box',len(b),'clean z',len(cz),'clean-z with B in box',len(cbz),'res_z rms box %.3f all %.3f'%(b.res_z.std() if len(b)>1 else np.nan,g.res_z.std()))
