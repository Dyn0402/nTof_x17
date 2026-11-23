# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, pickle
out=pickle.load(open('/media/ucla/work/ntof_pass1/ana/xy_match/ana4.pkl','rb'))
def ang(tx1,ty1,tx2,ty2):
    v1=np.stack([tx1,ty1,np.ones_like(tx1)],1); v2=np.stack([tx2,ty2,np.ones_like(tx2)],1)
    c=(v1*v2).sum(1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1); return np.degrees(np.arccos(np.clip(c,-1,1)))
for arm,a,m in out:
    if arm=='D': continue
    late=a[(a.t_ms>=12)]
    g1=late[late.gated1]; ch=late[late.chosen]
    ev=g1.groupby('ek').size(); ek2=ev[ev==2].index
    rows=[]
    for e in ek2:
        p=g1[g1.ek==e]; c=ch[(ch.ek==e)&ch.x_rank.isin(p.x_rank)&ch.y_rank.isin(p.y_rank)]
        if len(c)<2: continue
        same=set(zip(p.x_rank,p.y_rank))==set(zip(c.x_rank,c.y_rank))
        o=ang(p.x_tan_theta.values[:1],p.y_tan_theta.values[:1],p.x_tan_theta.values[1:],p.y_tan_theta.values[1:])[0]
        n=ang(c.x_tan_theta.values[:1],c.y_tan_theta.values[:1],c.x_tan_theta.values[1:],c.y_tan_theta.values[1:])[0]
        rows.append(dict(same=same,old=o,new=n,dl=c.llr.sum()-p.llr.sum()))
    R=pd.DataFrame(rows); s=R[~R.same]
    print(arm,'late 2-track events with a 2x2 alternative:',len(R),'swapped by LLR:',len(s), f'({len(s)/len(R):.3f})')
    print('   |d opening angle| deg of swaps: median %.1f, p90 %.1f; llr gain median %.1f'%(np.median(abs(s.new-s.old)),np.percentile(abs(s.new-s.old),90),s.dl.median()))
    print('   opening angle of all: median %.1f'%R.old.median())
