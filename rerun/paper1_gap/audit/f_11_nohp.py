import sys,pickle; sys.path.insert(0,'/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from f_lib import *
OUT='/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/f'
cells=[(i,j) for i in range(4) for j in range(4)]
Rh=Fitter(('hp',),FLOORS).fit(); Rn=Fitter(ARMS['no_hp'],FLOORS).fit()
print('hp alone: chi2',Rh.chi2,Rh.n,Rh.npar,'h_kuu %.2f+-%.2f h_kww %.2f+-%.2f'%(Rh['h_kuu'],Rh.err('h_kuu'),Rh['h_kww'],Rh.err('h_kww')))
print('no_hp (committed floors): h_kuu %.2f+-%.2f h_kww %.2f+-%.2f'%(Rn['h_kuu'],Rn.err('h_kuu'),Rn['h_kww'],Rn.err('h_kww')))
# floors retuned without hp: iterate floors in no_hp arm only
fl={c:0.0 for c in CLASSES}; scale={'hp': 0.1, 'w0': 0.02, 'd': 0.002, 'G': 0.2, 'GY': 0.2, 'T': 0.005, 'v': 0.005}
for it in range(12):
    R=Fitter(ARMS['no_hp'],fl).fit(); new={}
    for c,(chi,n) in R.per.items():
        red=chi/max(n,1); new[c]=fl[c] if red<=1.05 else fl[c]+scale[c]*np.sqrt(red-1)*(0.5 if it else 1)
    for c in CLASSES:
        new.setdefault(c,fl[c])
    if all(abs(new[c]-fl[c])<1e-6 for c in CLASSES): break
    fl=new
print('no_hp floors retuned without hp:',{k:round(v,4) for k,v in fl.items()},'(committed',{k:round(v,4) for k,v in FLOORS.items()},')')
fl['hp']=FLOORS['hp']
Rn2=Fitter(ARMS['no_hp'],fl).fit(); print('no_hp retuned: h_kuu %.2f+-%.2f h_kww %.2f+-%.2f chi2 %.1f'%(Rn2['h_kuu'],Rn2.err('h_kuu'),Rn2['h_kww'],Rn2.err('h_kww'),Rn2.chi2))
# hp-alone floor retune: reduced chi2 of hp alone
print('hp alone reduced chi2',Rh.chi2/(Rh.n-Rh.npar))
# bootstrap of difference
rng=np.random.default_rng(3); D=[]
for b in range(400):
    pick=rng.integers(0,16,16); w={c:0 for c in cells}
    for p in pick: w[cells[p]]+=1
    try:
        a=Fitter(('hp',),FLOORS,weights=w).fit(); n=Fitter(ARMS['no_hp'],FLOORS,weights=w).fit()
        D.append([a['h_kuu'],n['h_kuu'],a['h_kww'],n['h_kww']])
    except Exception: pass
D=np.array(D)
for k,(i,j) in {'h_kuu':(0,1),'h_kww':(2,3)}.items():
    d=D[:,j]-D[:,i]; hw=lambda x:(np.percentile(x,84)-np.percentile(x,16))/2; print('  robust: hp %.2f no_hp %.2f diff %.2f'%(hw(D[:,i]),hw(D[:,j]),hw(d)));print(k,'boot: hp-alone sd %.2f, no_hp sd %.2f, difference (no_hp - hp) = %.2f +- %.2f(boot)  ; corr %.2f'%(D[:,i].std(),D[:,j].std(),(Rn[k]-Rh[k]),d.std(),np.corrcoef(D[:,i],D[:,j])[0,1]))
