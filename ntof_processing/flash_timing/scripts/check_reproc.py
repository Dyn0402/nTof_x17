"""Re-measure the flash-region pathology on a reprocessed run, and check the
PKUP+C calibration against the reprocessed stored tflash."""
import sys, glob, json, numpy as np, uproot

paths = sorted(glob.glob(sys.argv[1]))[:int(sys.argv[2]) if len(sys.argv) > 2 else 2]
CAL = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
TREES = ['PSSA','PSSB','PSSC','PSSD','LIQA','LIQB','LIQC','LIQD','WALA','WALB','WALC','WALD']

def load(paths):
    out = {}
    pk = {}
    for p in paths:
        f = uproot.open(p)
        keys = [k.split(';')[0] for k in f.keys(recursive=False)]
        if 'PKUP' not in keys: continue
        a = f['PKUP'].arrays(['BunchNumber','tof','amp','PulseIntensity','tflash'], library='np')
        b,t,am = a['BunchNumber'],a['tof'],a['amp']
        o = np.lexsort((-am,b)); b,t = b[o],t[o]
        fi = np.ones(len(b),bool); fi[1:] = b[1:]!=b[:-1]
        b,t = b[fi],t[fi]
        m = np.abs(t-np.median(t))<200
        pk.update(dict(zip(b[m].tolist(), t[m].tolist())))
        for tr in TREES:
            if tr not in keys or f[tr].num_entries==0: continue
            br = ['BunchNumber','detn','tof','amp','area','satuflag','fwhm','tflash']
            arr = f[tr].arrays(br, library='np')
            out.setdefault(tr, []).append(arr)
    merged = {}
    for tr, chunks in out.items():
        merged[tr] = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return merged, pk

d, pk = load(paths)
print(f"{len(paths)} partials, {len(pk)} bunches with a good PKUP pulse\n")
print(f"{'tree':5} {'found':>6} {'sigma':>7} {'amp':>8} {'area/amp':>9} {'satu':>6} {'nhit/b':>7} {'C_meas':>9} {'tflash-PKUP':>12}")
rows=[]
for tr in TREES:
    if tr not in d: continue
    r = d[tr]
    m1 = r['detn']==1
    ref = np.array([pk.get(int(b), np.nan) for b in r['BunchNumber'][m1]])
    dt = r['tof'][m1] - ref
    fin = np.isfinite(dt)
    nb = len(set(r['BunchNumber'][m1][fin].tolist()))
    big = fin & (r['amp'][m1]>20000) & (dt>-3000) & (dt<-500)
    if big.sum() < 20:
        print(f"{tr:5}   no flash population found"); continue
    h,e = np.histogram(dt[big], bins=np.arange(-3000,-500,5)); peak = e[h.argmax()]+2.5
    sel = fin & (np.abs(dt-peak)<60)
    b = r['BunchNumber'][m1][sel]; v = dt[sel]
    a = r['amp'][m1][sel]; ar = r['area'][m1][sel]; sa = r['satuflag'][m1][sel]
    o = np.lexsort((-a,b)); b,v,a,ar,sa = b[o],v[o],a[o],ar[o],sa[o]
    fi = np.ones(len(b),bool); fi[1:] = b[1:]!=b[:-1]
    v,a,ar,sa = v[fi],a[fi],ar[fi],sa[fi]
    win = fin & (np.abs(dt-peak)<1000)
    sig = 1.4826*np.median(np.abs(v-np.median(v)))
    # stored tflash of this tree, referenced to PKUP
    tfl = r['tflash'][m1][fin]
    tref = np.array([pk.get(int(x), np.nan) for x in r['BunchNumber'][m1][fin]])
    rel = tfl - tref
    md = np.median(rel[np.isfinite(rel)])
    print(f"{tr:5} {len(v)/nb:6.0%} {sig:7.2f} {np.median(a):8.0f} "
          f"{np.median(np.abs(ar)/np.abs(a)):9.1f} {np.mean(sa>0):6.0%} {win.sum()/nb:7.1f} "
          f"{np.mean(v):9.2f} {md:12.1f}")
    rows.append((tr, float(np.mean(v)), float(md)))
if CAL:
    print("\ncalibration check: measured C in this reprocessed run vs the divert-off calibration")
    for tr, cmeas, md in rows:
        if tr in CAL:
            print(f"  {tr}: reprocessed {cmeas:9.2f}   calibration {CAL[tr]:9.2f}   diff {cmeas-CAL[tr]:+7.2f} ns")
