import pandas as pd, numpy as np, json
R=pd.read_pickle('acq_occ.pkl')
P=json.load(open('posmap.json'))
def mean_occ(sel,fe):
    o=sum(r[f'occ_{fe}'].astype(float) for r in sel); n=sum(r['n'] for r in sel); return o/n
grp=lambda lo,hi,cls:[r for r in R if lo<=r['run']<=hi and r['rclass']==cls]
for arm,fes in (('D',(1,2)),('B',(5,6))):
  m=json.load(open(f'/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/channel_layer/masks/ntof_{arm}_E2_post_access.json'))
  for fe,v in zip(fes,'xy'):
    dead=set(m['planes'][v]['dead']); hot=set(m['planes'][v]['hot'])
    p=np.array(P[arm][v],float); loc= -(p-199.29) if v=='x' else (p-199.29)
    for lab,(a,b,c,d,cls) in {'cos':(117,131,133,134,'cosmic'),'beam':(124,132,139,139,'production')}.items():
        o1=mean_occ(grp(a,b,cls),fe); o2=mean_occ(grp(c,d,cls),fe)
        rat=o2/np.maximum(o1,1e-5)
        idx=np.where(((rat>1.6)|(rat<0.6))&(np.maximum(o1,o2)>0.003))[0]
        print(arm,v,lab,'channels changed:',[(int(i),round(float(loc[i]),1),round(100*o1[i],2),round(100*o2[i],2),'D' if i in dead else ('H' if i in hot else '')) for i in idx])
