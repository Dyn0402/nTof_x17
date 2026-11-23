# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, pickle
out=pickle.load(open('/media/ucla/work/ntof_pass1/ana/xy_match/ana4.pkl','rb'))
UOFF={'A':182.94,'C':181.99,'D':183.79}; KX={'A':1.0423,'C':0.989,'D':0.9825}; LS=234.6
def usrc(arm,p0,tan): return (UOFF[arm]-p0)-LS*KX[arm]*tan
for arm,a,m in out:
    late=a[a.gated1&(a.t_ms>=30)&(a.ncx==1)]; pk=np.median(usrc(arm,late.x_p0,late.x_tan_theta))
    def pfrac(uu): d=np.abs(uu-pk); return np.mean(d<15)-np.mean((d>60)&(d<120))*30/120
    ch=a[a.chosen][['ek','y_rank','x_p0','x_tan_theta','llr','x_dchi2','x_q_sum']].rename(columns={'x_p0':'nx_p0','x_tan_theta':'nx_tan','llr':'nllr','x_dchi2':'nx_dchi2','x_q_sum':'nxq'})
    xm=m[m.kind=='x_moved'].merge(ch,on=['ek','y_rank'])
    xm['dl']=xm.nllr-xm.llr
    xm['early']=np.where(xm.t_ms<12,'<12','>=12')
    print('==',arm,'x_moved: by llr gain  (old point, new point, n, new dchi2 > old)')
    for (e,b),g in xm.groupby(['early',pd.cut(xm.dl,[-99,0,2,5,10,99])],observed=True):
        print(f'  {e:4s} {str(b):10s} n {len(g):5d}  old {pfrac(usrc(arm,g.x_p0,g.x_tan_theta)):.3f} new {pfrac(usrc(arm,g.nx_p0,g.nx_tan)):.3f}  newdchi2>old {np.mean(g.nx_dchi2>g.x_dchi2):.2f} newq>old {np.mean(g.nxq>g.x_q_sum):.2f}')
    # two-track late events: swaps
    two=m[(m.t_ms>=12)]
    ev=two.groupby('ek').agg(n=('kind','size'),ny=('kind',lambda k:(k=='y_changed').sum()))
    print('  late events with >=2 pass-1 gated tracks:',int((ev.n>=2).sum()),' of which LLR re-pairs y:',int(((ev.n>=2)&(ev.ny>=2)).sum()))
