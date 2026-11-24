import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
L=[]; store={}
for arm in ('all','no_hp','tilt','timing'):
    cl=ARMS[arm]; R0=Fitter(cl,FLOORS,**({'zero':('g0',)} if arm=='timing' else {})).fit()
    L.append(f'**{arm}**\n\n| a | chi2 | dchi2 | h_kuu | h_kww | c_kuu | c_kww | gap_kuu | beta | mesh x@150 | cath x@150 | cath y@150 |\n|---|---|---|---|---|---|---|---|---|---|---|---|')
    rows=[]
    kwz={'zero':('g0',)} if arm=='timing' else {}
    for a in (0.3,0.4,0.5,0.6,0.7):
        R=Fitter(cl,FLOORS,a=a,**kwz).fit(); q=quants(R); f=five(R)
        rows.append((a,R.chi2,q,f))
        L.append(f'| {a} | {R.chi2:.1f} | {R.chi2-R0.chi2:+.1f} | '+' | '.join(f'{q[k][0]:+.2f}+-{q[k][1]:.2f}' for k in ['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','beta'])+f' | {f["mesh_x"][0]:+.2f}+-{f["mesh_x"][1]:.2f} | {f["cath_x"][0]:+.2f}+-{f["cath_x"][1]:.2f} | {f["cath_y"][0]:+.2f}+-{f["cath_y"][1]:.2f} |')
    for lab,mode in (('free a (one)','free'),('free a_x,a_y','free_xy')):
        try:
            R=Fitter(cl,FLOORS,a=mode,**kwz).fit(); q=quants(R); f=five(R)
            ax=(R['ax'],R.err('ax')); ay=(R['ay'],R.err('ay'))
            L.append(f'| {lab}: a_x={ax[0]:.2f}+-{ax[1]:.2f}, a_y={ay[0]:.2f}+-{ay[1]:.2f} | {R.chi2:.1f} | {R.chi2-R0.chi2:+.1f} | '+' | '.join(f'{q[k][0]:+.2f}+-{q[k][1]:.2f}' for k in ['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','beta'])+f' | {f["mesh_x"][0]:+.2f}+-{f["mesh_x"][1]:.2f} | {f["cath_x"][0]:+.2f}+-{f["cath_x"][1]:.2f} | {f["cath_y"][0]:+.2f}+-{f["cath_y"][1]:.2f} |')
            store[(arm,mode)]=(ax,ay,R.chi2)
        except Exception as e: L.append(f'| {lab} | failed {e} |')
    store[arm]=rows; L.append('')
# fine scan for the figure (all arm) + profile-likelihood interval of one a
aa=np.linspace(0.0,1.0,21); ch={}
for arm in ('all','no_hp','tilt'):
    ch[arm]=[Fitter(ARMS[arm],FLOORS,a=a).fit().chi2 for a in aa]
store['scan']=(aa,ch)
for arm in ch:
    c=np.array(ch[arm]); i=c.argmin(); ok=aa[c<c.min()+1]
    L.append(f'profile chi2 {arm}: min at a={aa[i]:.2f} (chi2 {c.min():.1f}); delta chi2<1 for a in [{ok.min():.2f},{ok.max():.2f}]; chi2(0.5)-min={c[10]-c.min():.1f}')
pickle.dump(store,open(f'{OUT}/mix.pkl','wb'))
open(f'{OUT}/sec4.md','w').write('\n'.join(L)); print('\n'.join(L))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(4.8,3.2))
for arm in ch:
    c=np.array(ch[arm]); ax.plot(aa,c-c.min(),'o-',ms=3,label=arm)
ax.axvline(0.5,color='k',lw=.6,ls=':'); ax.set_ylim(0,30); ax.set_xlabel('cathode weight a in w0 = v[(1-a) grad h + a grad c]'); ax.set_ylabel(r'$\chi^2 - \chi^2_{min}$'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/chi2_vs_a.png',dpi=140)
