# study script (fixed paths), x/y matching study 2026-11-23; see README.md
import pandas as pd, numpy as np, pickle
out=pickle.load(open('/media/ucla/work/ntof_pass1/ana/xy_match/ana4.pkl','rb'))
UOFF={'A':182.94,'C':181.99,'D':183.79}; KX={'A':1.0423,'C':0.989,'D':0.9825}; LS=234.6
def usrc(arm,p0,tan): return (UOFF[arm]-p0)-LS*KX[arm]*tan
for arm,a,m in out:
    g=a[a.gated1].copy(); late=g[(g.t_ms>=30)&(g.ncx==1)]; pk=np.median(usrc(arm,late.x_p0,late.x_tan_theta))
    def ex(d): u=np.abs(usrc(arm,d.x_p0,d.x_tan_theta)-pk); return np.mean(u<15)-np.mean((u>60)&(u<120))*30/120
    ref=ex(g[g.t_ms>=30]); refw=g[(g.t_ms>=30)&g.wall]
    print('==',arm,'late excess',round(ref,3))
    for cut in [-99,-2,0,1,2]:
        s=[f'{cut:4d} keep-late-wall {np.mean(refw.llr>cut):.3f} |']
        for lo,hi in [(3,8),(8,12),(12,16),(16,20)]:
            d=g[(g.t_ms>=lo)&(g.t_ms<hi)]; k=d[d.llr>cut]
            s.append(f' {lo}-{hi}: pur {ex(k)/ref:.2f} kept {len(k)/len(d):.2f}')
        print(''.join(s))
