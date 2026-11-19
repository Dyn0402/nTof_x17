import uproot, numpy as np, glob, sys
s68 = lambda x: 0.5*np.diff(np.percentile(x,[16,84]))[0]
rows=[]
for r in sys.argv[1:]:
    for fn in sorted(glob.glob(f'/media/dylan/data/x17/beam_july/ntof_data/run{r}.parts/*.root')):
        f = uproot.open(fn); X={}
        for arm in 'AC':
            a = f['PSS'+arm].arrays(['BunchNumber','detn','tof','amp'], library='np')
            m = (a['tof']>5e4)&(a['amp']>300)
            X[arm] = np.c_[a['BunchNumber'][m], a['tof'][m], a['detn'][m], a['amp'][m]]
            # wall bars in same arm for the 'leg' requirement
        A, C = X['A'], X['C']
        kC = C[:,0]*1e9+C[:,1]; o=np.argsort(kC); C=C[o]; kC=kC[o]
        kA = A[:,0]*1e9+A[:,1]
        lo=np.searchsorted(kC,kA-300); hi=np.searchsorted(kC,kA+300)
        for i in np.flatnonzero(hi>lo):
            for j in range(lo[i],hi[i]): rows.append([A[i,1]-C[j,1], A[i,2], C[j,2], A[i,3], C[j,3]])
R=np.array(rows); dt,pA,pC,aA,aC=R.T
h,e=np.histogram(dt,bins=60,range=(-300,300)); print('plastic A-C dt 10ns bins:', h.tolist())
pk = e[h.argmax()]+5; core = np.abs(dt-pk)<20
print('peak ~%.0f ns, N core %d, bg/10ns ~%.1f' % (pk, core.sum(), np.median(np.r_[h[:15],h[-15:]])))
for i in (1,2):
    for j in (1,2):
        m=core&(pA==i)&(pC==j)
        if m.sum()>20: print(' PMT pair',i,j,m.sum(),'median %.2f s68 %.2f' % (np.median(dt[m]), s68(dt[m])))
res=dt.copy()
for i in (1,2):
    for j in (1,2):
        m=core&(pA==i)&(pC==j); res[m]-=np.median(dt[m])
print('after pair offsets s68 %.2f, core rms(|r|<6) %.2f' % (s68(res[core]), np.std(res[core][np.abs(res[core])<6])))
hh,ee=np.histogram(res[core],bins=24,range=(-6,6)); print('residual hist 0.5ns bins:', hh.tolist())
