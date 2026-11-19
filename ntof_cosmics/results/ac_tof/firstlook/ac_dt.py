import uproot, numpy as np, glob, sys
runs = sys.argv[1:]
def bars(a):
    # pair top (odd detn) and bottom (even) of the same segment within 15 ns -> mean time, geo-mean amp
    out_b, out_t, out_a, out_d = [], [], [], []
    k = a['BunchNumber'].astype(np.int64)
    for g in range(4):
        top = a['detn'] == 2*g+1; bot = a['detn'] == 2*g+2
        kt, tt, at = k[top], a['tof'][top], a['amp'][top]
        kb, tb, ab = k[bot], a['tof'][bot], a['amp'][bot]
        ob = np.lexsort((tb, kb)); kb, tb, ab = kb[ob], tb[ob], ab[ob]
        key_b = kb * 1e9 + tb
        key_t = kt * 1e9 + tt
        j = np.clip(np.searchsorted(key_b, key_t), 1, len(key_b)-1)
        jj = np.where(np.abs(key_b[j]-key_t) < np.abs(key_b[j-1]-key_t), j, j-1)
        d = tt - tb[jj]; ok = (kb[jj] == kt) & (np.abs(d) < 15)
        out_b.append(kt[ok]); out_t.append(0.5*(tt[ok]+tb[jj][ok])); out_a.append(np.sqrt(at[ok]*ab[jj][ok])); out_d.append(d[ok])
    return [np.concatenate(x) for x in (out_b, out_t, out_a, out_d)]
allA, allC, tbd = [], [], []
dts=[]; dtsP=[]
for r in runs:
    for fn in sorted(glob.glob(f'/media/dylan/data/x17/beam_july/ntof_data/run{r}.parts/*.root')):
        f = uproot.open(fn)
        A = f['WALA'].arrays(['BunchNumber','detn','tof','amp'], library='np'); A={q:v[A['tof']>5e4] for q,v in A.items()}
        C = f['WALC'].arrays(['BunchNumber','detn','tof','amp'], library='np'); C={q:v[C['tof']>5e4] for q,v in C.items()}
        bA, tA, aA, dA = bars(A); bC, tC, aC, dC = bars(C)
        tbd.append(dA); 
        kC = bC*1e9 + tC; o = np.argsort(kC); kC = kC[o]
        kA = bA*1e9 + tA
        lo = np.searchsorted(kC, kA-300); hi = np.searchsorted(kC, kA+300)
        for i in np.flatnonzero(hi > lo):
            dts.extend(kA[i] - kC[lo[i]:hi[i]])
dts = np.array(dts); tbd = np.concatenate(tbd)
print('wall A bars (top&bottom within 15 ns):', len(tbd), ' top-bottom dt: median %.2f, sigma68 %.2f ns' % (np.median(tbd), 0.5*np.diff(np.percentile(tbd,[16,84]))[0]))
h, e = np.histogram(dts, bins=120, range=(-300,300))
print('A-C wall-bar dt, 5 ns bins, |dt|<300:'); 
bg = np.median(np.r_[h[:30], h[-30:]])
for i in range(45, 75): print('%6.0f %5d %s' % (e[i], h[i], '#'*int(h[i]//max(1,h.max()//60))))
core = dts[np.abs(dts) < 25]
print('N(|dt|<25)=%d, bg per 5ns bin=%.1f -> expected bg in core %.0f' % (len(core), bg, bg*10))
print('core median %.2f, s68 %.2f' % (np.median(core), 0.5*np.diff(np.percentile(core,[16,84]))[0]))
h1, e1 = np.histogram(core, bins=50, range=(-25,25))
for i in range(10,40): print('%5.0f %4d %s' % (e1[i], h1[i], '#'*int(h1[i]//max(1,h1.max()//60))))
