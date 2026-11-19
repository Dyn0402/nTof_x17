import uproot, numpy as np, glob, sys
exec(open(sys.argv[1]).read().split('allA, allC')[0])   # reuse bars()
def bars_seg(a):
    out=[]
    k = a['BunchNumber'].astype(np.int64)
    for g in range(4):
        top = a['detn'] == 2*g+1; bot = a['detn'] == 2*g+2
        kt, tt, at = k[top], a['tof'][top], a['amp'][top]
        kb, tb, ab = k[bot], a['tof'][bot], a['amp'][bot]
        ob = np.lexsort((tb, kb)); kb, tb, ab = kb[ob], tb[ob], ab[ob]
        key_b = kb*1e9+tb; key_t = kt*1e9+tt
        j = np.clip(np.searchsorted(key_b, key_t), 1, len(key_b)-1)
        jj = np.where(np.abs(key_b[j]-key_t) < np.abs(key_b[j-1]-key_t), j, j-1)
        d = tt - tb[jj]; ok = (kb[jj]==kt) & (np.abs(d)<15)
        out.append(np.c_[kt[ok], 0.5*(tt[ok]+tb[jj][ok]), np.sqrt(at[ok]*ab[jj][ok]), d[ok], np.full(ok.sum(), g)])
    return np.concatenate(out)
rows=[]
for r in sys.argv[2:]:
    for fn in sorted(glob.glob(f'/media/dylan/data/x17/beam_july/ntof_data/run{r}.parts/*.root')):
        f = uproot.open(fn)
        X = {}
        for arm in 'AC':
            a = f['WAL'+arm].arrays(['BunchNumber','detn','tof','amp'], library='np'); a = {q:v[a['tof']>5e4] for q,v in a.items()}
            X[arm] = bars_seg(a)
        A, C = X['A'], X['C']
        kC = C[:,0]*1e9 + C[:,1]; o = np.argsort(kC); C = C[o]; kC = kC[o]
        kA = A[:,0]*1e9 + A[:,1]
        lo = np.searchsorted(kC, kA-25); hi = np.searchsorted(kC, kA+25)
        for i in np.flatnonzero(hi-lo == 1):
            c = C[lo[i]]; rows.append([A[i,1]-c[1], A[i,3], c[3], A[i,4], c[4], A[i,2], c[2]])
R = np.array(rows); dt, dA, dC, gA, gC, aA, aC = R.T
print('N single A-C bar coincidences:', len(R))
s68 = lambda x: 0.5*np.diff(np.percentile(x,[16,84]))[0]
print('dt all: median %.2f s68 %.2f' % (np.median(dt), s68(dt)))
print('corr(dt, dA_tb)=%.3f corr(dt, dC_tb)=%.3f corr(dt, dA-dC)=%.3f' % (np.corrcoef(dt,dA)[0,1], np.corrcoef(dt,dC)[0,1], np.corrcoef(dt,dA-dC)[0,1]))
print('segment pair (gA,gC): n, median dt, s68')
for i in range(4):
    for j in range(4):
        m = (gA==i)&(gC==j)
        if m.sum()>30: print('  ', i, j, m.sum(), '%.2f %.2f' % (np.median(dt[m]), s68(dt[m])))
# remove per-pair median and the linear dependence on tb differences
res = dt.copy()
for i in range(4):
    for j in range(4):
        m=(gA==i)&(gC==j)
        if m.sum(): res[m]-=np.median(dt[m])
print('after per-pair offsets: s68 %.2f' % s68(res))
X = np.c_[dA, dC, np.ones_like(dA)]
c, *_ = np.linalg.lstsq(X, res, rcond=None); r2 = res - X@c
print('fit res = a*dA_tb + b*dC_tb: a=%.3f b=%.3f -> s68 %.2f, rms(|r|<8) %.2f' % (c[0], c[1], s68(r2), np.std(r2[np.abs(r2)<8])))
for q in [(0,50),(50,80),(80,120),(120,200),(200,1e9)]:
    m = (np.minimum(aA,aC)>=q[0])&(np.minimum(aA,aC)<q[1])
    if m.sum()>30: print('  min geo-amp %s: n=%d s68=%.2f' % (q, m.sum(), s68(r2[m])))
print('top-bottom per arm s68: A %.2f C %.2f; medians per seg A' % (s68(dA), s68(dC)), [round(np.median(dA[gA==g]),1) for g in range(4)], 'C', [round(np.median(dC[gC==g]),1) for g in range(4)])
q = dA - dC
for lo_, hi_ in [(-30,-6),(-6,-3),(-3,-1),(-1,1),(1,3),(3,6),(6,30)]:
    m=(q>=lo_)&(q<hi_)
    if m.sum()>20: print('  (dA-dC) in [%d,%d): n=%d median res %.2f s68 %.2f' % (lo_,hi_,m.sum(),np.median(res[m]),s68(res[m])))
