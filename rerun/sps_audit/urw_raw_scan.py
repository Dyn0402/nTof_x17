#!/usr/bin/env python3
"""urw_raw_scan.py --out O.npz FEU1.root

The uRWELL telescope of RAW run_71, built like the ZS one (extract2.py / tel_job.telescope_zs) instead of the 2026-10-30
followup's scan (rerun/sps_followup/urwell_scan.py).  Why: that scan needed an in-time 3-strip sum > 100 ADC with a
single peak in every plane-view, but the uRWELL lead strip in RAW is mostly 40-100 ADC (noise 3.9 ADC), so it kept
0.054 % of triggers as correlated four-view tracks against 3.7 % on ZS run66 two hours earlier, same HV and mount.  With
a 40 ADC (10 sigma) single-strip threshold the correlated four-view rate is 0.7 % and the correlated x pairs 3.6 %
(rerun/sps_audit/README.md).

Per event: RAW baseline of urwell_run71 (pre-sample median, 64-channel block median per sample, pre median again); a
strip's amplitude is its maximum in samples T_LO..T_HI (the uRWELL pulse peaks at 12-13); hit = amplitude > THR_ADC and
the strip is not oscillating (> 30 % of the file's first 400 events above threshold); hits clustered in position
(gap 3 mm), the leading cluster by charge, the number of clusters and the lead's strip count, per plane-view.
Writes eid and {fx,fy,bx,by}_{p,q,n,size} for every event (the selection is offline).
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'sps_followup')]
from urwell_run71 import urwell_map, grid_of, baseline, RAW_PRE   # noqa: E402
from tel_job import clusters                                       # noqa: E402

T_LO, T_HI = 9, 17
THR_ADC = 40.0


def main():
    import uproot
    ap = argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=THR_ADC)
    a = ap.parse_args()
    view, pos, det = urwell_map()
    t = uproot.open(a.file)['nt']
    eids, A = [], []
    for lo in range(0, t.num_entries, 2000):
        arr = t.arrays(['eventId', 'sample', 'channel', 'amplitude'], entry_start=lo, entry_stop=min(lo + 2000, t.num_entries), library='np')
        for j in range(len(arr['eventId'])):
            g = baseline(grid_of(arr['channel'][j].astype(int), arr['sample'][j].astype(int), arr['amplitude'][j].astype(np.float32)))
            w = np.nan_to_num(g[:, T_LO:T_HI + 1], nan=-1e9)
            A.append(w.max(1).astype(np.float32)); eids.append(int(arr['eventId'][j]))
    A = np.array(A); eids = np.array(eids, np.int64)
    bad = np.mean(A[:400] > a.thr, axis=0) > 0.3
    hit = (A > a.thr) & ~bad[None, :]
    ev, ch = np.nonzero(hit)
    out = dict(eid=eids, n_bad=np.array([int(bad.sum())]))
    for d in 'fb':
        for v in 'xy':
            k = (det[ch] == d) & (view[ch] == v)
            p, q, n, size = clusters(ev[k], pos[ch[k]], A[ev[k], ch[k]].astype(float), len(eids))
            out[f'{d}{v}_p'], out[f'{d}{v}_q'], out[f'{d}{v}_n'], out[f'{d}{v}_size'] = p, q, n, size
    np.savez_compressed(a.out, **out)
    four = np.all([out[f'{p}_n'] >= 1 for p in ('fx', 'fy', 'bx', 'by')], axis=0)
    print(f'{os.path.basename(a.file)}: {len(eids)} events, {int(bad.sum())} oscillating strips, four-view {four.mean():.4f}')


if __name__ == '__main__':
    main()
