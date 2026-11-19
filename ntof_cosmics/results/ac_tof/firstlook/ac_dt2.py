import uproot, numpy as np, glob, sys
fn = sorted(glob.glob('/media/dylan/data/x17/beam_july/ntof_data/run224678.parts/*.root'))[0]
f = uproot.open(fn)
for tr in ['WALA','WALC']:
    a = f[tr].arrays(['BunchNumber','detn','tof','amp'], library='np')
    k = a['BunchNumber'].astype(np.int64)*1e9 + a['tof']
    o = np.argsort(k); k = k[o]; det = a['detn'][o]; amp = a['amp'][o]
    # cluster hits within 20 ns
    br = np.r_[0, np.flatnonzero(np.diff(k) > 20)+1, len(k)]
    mult = np.array([len(np.unique(det[br[i]:br[i+1]])) for i in range(len(br)-1)])
    print(tr, 'hits', len(k), 'clusters', len(mult), 'channel multiplicity counts:', np.bincount(mult)[:10])
    # tof mod structure: fraction of hits in clusters with all 8 channels
    big = np.repeat(mult, np.diff(br)) >= 7
    print('  amp median, mult>=7: %.0f, others %.0f' % (np.median(amp[big]), np.median(amp[~big])))
    tt = (k % 1e9)
    print('  tof of mult>=7 clusters, percentiles:', np.percentile(tt[big], [0,25,50,75,100]).round(0) if big.any() else None)
