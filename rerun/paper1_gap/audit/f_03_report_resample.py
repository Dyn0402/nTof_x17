import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
res=pickle.load(open(f'{OUT}/resample.pkl','rb'))
ARMLIST=['all','tilt','gap','geo','timing','no_hp','geo_gap']
Q=['h_kuu','h_kww','c_kuu','c_kww','gap_kuu','gap_kww','gtilt_U','gtilt_W','beta']
L=[]
L.append('### 1. Per-run fits (value +- fit error)\n')
for arm in ARMLIST:
    L.append(f'**{arm}**\n\n| fit | chi2/ndf | '+' | '.join(Q)+' |\n|'+'---|'*(len(Q)+2))
    for runs in ('run1','run2','run1+run2'):
        d=res['perrun'][(arm,runs)]
        if 'err' in d: L.append(f'| {runs} | failed |'); continue
        L.append(f'| {runs} | {d["chi2"]:.0f}/{d["n"]-d["npar"]} | '+' | '.join(f'{d["q"][k][0]:+.2f}+-{d["q"][k][1]:.2f}' for k in Q)+' |')
    # consistency
    d1,d2=res['perrun'][(arm,'run1')],res['perrun'][(arm,'run2')]
    if 'q' in d1 and 'q' in d2:
        z=[(d1['q'][k][0]-d2['q'][k][0])/np.hypot(d1['q'][k][1],d2['q'][k][1]) for k in Q]
        L.append('| (run1-run2)/err | | '+' | '.join(f'{x:+.1f}' for x in z)+' |')
    L.append('')
L.append('### 2. Resampling errors\n')
for arm in ARMLIST:
    J=res['jk'][arm]; full=J['full']
    B=res['bs'][arm]
    L.append(f'**{arm}** (bootstrap n={len(B)})\n\n| quantity | full fit | fit err | jackknife err | bootstrap err (std) | bootstrap 16-84 half-width | row-JK range | col-JK range |\n|---|---|---|---|---|---|---|---|')
    fe=quants(Fitter(ARMS[arm],FLOORS).fit())
    for k in Q:
        v=np.array([x[k] for x in J['leave']]); v=v[np.isfinite(v)]; n=len(v)
        jk=np.sqrt((n-1)/n*np.sum((v-v.mean())**2)) if n>2 else np.nan
        b=np.array([x[k] for x in B]); b=b[np.isfinite(b)]
        hw=(np.percentile(b,84)-np.percentile(b,16))/2
        rj=np.array([x[k] for x in res['rows'][arm]]); cj=np.array([x[k] for x in res['cols'][arm]])
        L.append(f'| {k} | {full[k]:+.2f} | {fe[k][1]:.2f} | {jk:.2f} | {np.std(b):.2f} | {hw:.2f} | {np.nanmin(rj):+.2f}..{np.nanmax(rj):+.2f} | {np.nanmin(cj):+.2f}..{np.nanmax(cj):+.2f} |')
    L.append('')
# influence of cells on c_kuu in arm all
cells=[(i,j) for i in range(4) for j in range(4)]
L.append('**Leave-one-cell-out influence on c_kuu (value when the cell is dropped; full fit in header)**\n')
for arm in ('all','tilt','geo','gap'):
    full=res['jk'][arm]['full']['c_kuu']
    L.append(f'{arm}: full {full:+.2f}\n\n| i\\j | j=0 | j=1 | j=2 | j=3 |\n|---|---|---|---|---|')
    v=np.array([x['c_kuu'] for x in res['jk'][arm]['leave']]).reshape(4,4)
    for i in range(4): L.append(f'| i={i} | '+' | '.join(f'{v[i,j]:+.2f} ({v[i,j]-full:+.2f})' for j in range(4))+' |')
    L.append('')
open(f'{OUT}/sec12.md','w').write('\n'.join(L)); print('\n'.join(L))
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(9,3.0))
for a,(arm,k) in zip(ax,[('all','c_kuu'),('tilt','gap_kuu'),('geo','c_kuu')]):
    full=res['jk'][arm]['full'][k]
    v=np.array([x[k] for x in res['jk'][arm]['leave']]).reshape(4,4)-full
    m=np.nanmax(abs(v))
    im=a.imshow(v.T,origin='lower',cmap='RdBu_r',vmin=-m,vmax=m,extent=[-.5,3.5,-.5,3.5])
    a.set_title(f'{arm}: {k} (full {full:+.2f})',fontsize=9); a.set_xlabel('cell i (x)'); a.set_ylabel('cell j (y)')
    plt.colorbar(im,ax=a,label='change when dropped (mm)')
plt.tight_layout(); plt.savefig(f'{OUT}/jackknife_influence.png',dpi=140)
