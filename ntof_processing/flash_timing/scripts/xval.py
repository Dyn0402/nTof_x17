"""Cross-validate the divert-off calibration against the reprocessed tflash."""
import sys, glob, numpy as np, uproot
paths = sorted(glob.glob(sys.argv[1]))[:int(sys.argv[2])]
TREES = ['WALA','WALB','WALC','WALD','PSSA','PSSB','PSSC','PSSD','LIQA','LIQB','LIQC','LIQD']
acc = {t: [] for t in TREES}
nb = 0
for p in paths:
    f = uproot.open(p)
    keys = [k.split(';')[0] for k in f.keys(recursive=False)]
    a = f['PKUP'].arrays(['BunchNumber','tof','amp'], library='np')
    b,t,am = a['BunchNumber'],a['tof'],a['amp']
    o = np.lexsort((-am,b)); b,t = b[o],t[o]
    fi = np.ones(len(b),bool); fi[1:] = b[1:]!=b[:-1]; b,t = b[fi],t[fi]
    m = np.abs(t-np.median(t))<200
    pk = dict(zip(b[m].tolist(), t[m].tolist())); nb += m.sum()
    for tr in TREES:
        if tr not in keys or f[tr].num_entries == 0: continue
        x = f[tr].arrays(['BunchNumber','tflash'], entry_stop=2000000, library='np')
        bb, tf = x['BunchNumber'], x['tflash']
        o2 = np.argsort(bb, kind='stable'); bb, tf = bb[o2], tf[o2]
        fi2 = np.ones(len(bb),bool); fi2[1:] = bb[1:]!=bb[:-1]
        bb, tf = bb[fi2], tf[fi2]
        rel = np.array([tf[i]-pk[int(bb[i])] for i in range(len(bb)) if int(bb[i]) in pk])
        acc[tr].append(rel)
print(f"{len(paths)} partials, {nb} bunches")
print(f"\n{'tree':6} {'tflash-PKUP':>12} {'sigma':>7} {'nbunch':>7}")
for tr in TREES:
    if not acc[tr]: continue
    v = np.concatenate(acc[tr]); v = v[np.isfinite(v)]
    md = np.median(v); core = v[np.abs(v-md)<100]
    print(f"{tr:6} {core.mean():12.2f} {core.std():7.2f} {len(core):7d}   (in-core {len(core)/len(v):.1%})")
