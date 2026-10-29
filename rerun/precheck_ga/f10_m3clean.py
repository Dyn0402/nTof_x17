import uproot, glob, numpy as np, sys
B='/eos/experiment/ntof/data/x17/cosmic_bench/june_tests/'
runs={'6-23 det3/det4':'mx17_det3_det4_overnight_6-23-26','6-24 g_det4':'mx17_det4_day_6-24-26','6-27 sat det3':'mx17_det3_saturday_scan_6-27-26','6-22 det2/det3':'mx17_det2_det3_overnight_6-22-26'}
for lab,r in runs.items():
    for sd in sorted(glob.glob(B+r+'/*/m3_tracking_root_v2')):
        fs=sorted(glob.glob(sd+'/*rays.root'))[:3]
        n=c=c3=0
        for f in fs:
            import awkward as ak; t=uproot.open(f)['T'].arrays(['Chi2X','Chi2Y','NClusX','NClusY'],library='ak'); one=ak.num(t['Chi2X'])==1; a={k:ak.to_numpy(ak.firsts(t[k][one])) for k in t.fields}; ntot=len(one)
            n+=ntot; ok=(a['Chi2X']<1)&(a['Chi2Y']<1)
            c+=np.sum(ok&(a['NClusX']>=4)&(a['NClusY']>=4)); c3+=np.sum(ok&(a['NClusX']>=3)&(a['NClusY']>=3))
        if n: print(f"{lab:15s} {sd.split('/')[-2]:30s} rays={n:7d} clean4={c/n:.3f} clean3={c3/n:.3f}", flush=True)
        if lab!='6-23 det3/det4': break
