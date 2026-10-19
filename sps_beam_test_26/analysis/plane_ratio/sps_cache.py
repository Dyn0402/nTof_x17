#!/usr/bin/env python3
"""sps_cache.py -- det4 H4 head-on events in the bench calibration-cache format.

run_71 RAW (Ar/CF4/iso 88/10/2, flat mount, no zero suppression): both views
at normal incidence, so the drift ladder is absent (w = 0) and the neighbour
pattern is the kernel plus the lateral spread, independent of v.  Each event
gets both views' waveform windows (+-HALF_WIN strips around the uRWELL
prediction), tan = 0, ref_mesh = the uRWELL prediction, and a constant ftst
class (the beam trigger is asynchronous to the DREAM clock in the same way
for every event; the 5 ns t0 prior is not used here).

    sps_cache.py [--per-plateau 1500] [--out DIR]
Output: <out>/sps_run71_<plateau>.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ANA, '..', '..'))
sys.path[:0] = [ANA, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import datasets                                              # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

HALF_WIN = 8           # head-on: +-8 strips is +-6 mm, far past any sharing
NOISE_ADC = 10.0       # post-CNS noise, RAW_RUN71_PHYSICS §1
Q_LO = 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-plateau', type=int, default=1500)
    ap.add_argument('--out', default='/media/dylan/data/x17/sps_run53_det4_check/plane_ratio')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = datasets.get('run71_raw')
    Z = np.load(D['stage'] + 'wf_run71_raw_det4only.npz', allow_pickle=True)
    ev, ch, samp, amp = Z['ev'], Z['ch'].astype(int), Z['samp'], Z['amp']
    plat, pX, pY = Z['ev_plateau'], Z['ev_pX'], Z['ev_pY']
    nsmp = int(D['n_samples'])
    order = np.lexsort((ch, ev))
    ev, ch, samp, amp = ev[order], ch[order], samp[order], amp[order]
    starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1, len(ev)]
    out = {}
    for s, e in zip(starts[:-1], starts[1:]):
        eid = int(ev[s])
        lab = str(plat[eid])
        if not lab or len(out.get(lab, {})) >= a.per_plateau:
            continue
        if not (np.isfinite(pX[eid]) and np.isfinite(pY[eid])):
            continue
        rec = dict(eid=eid, tan_x=0.0, tan_y=0.0,
                   ref_mesh_x=float(pX[eid]), ref_mesh_y=float(pY[eid]))
        c, sm, am = ch[s:e], samp[s:e], amp[s:e]
        ok = True
        for view, p0 in (('x', pX[eid]), ('y', pY[eid])):
            k = (VIEW[c] == view) & (np.abs(POSITION_MM[c] - p0) <= HALF_WIN * PITCH_MM)
            if k.sum() == 0:
                ok = False
                break
            cc, ss, aa = c[k], sm[k], am[k]
            chs = np.unique(cc)
            W = np.zeros((len(chs), nsmp), np.float32)
            ci = np.searchsorted(chs, cc)
            m = (ss >= 0) & (ss < nsmp)
            W[ci[m], ss[m].astype(int)] = aa[m]
            if W.max() < Q_LO or len(chs) < 4:
                ok = False
                break
            o = np.argsort(POSITION_MM[chs])
            rec[view] = dict(ch=chs[o].astype(np.int16),
                             pos=POSITION_MM[chs[o]].astype(np.float32),
                             W=W[o], noise=np.full(len(chs), NOISE_ADC, np.float32))
            rec[f'ftst_{view}'] = 0
        if ok:
            out.setdefault(lab, {})[eid] = rec
    for lab, evs in out.items():
        p = os.path.join(a.out, f'sps_run71_{lab}.pkl')
        with open(p, 'wb') as f:
            pickle.dump(evs, f, protocol=4)
        print(f'{lab}: {len(evs)} events -> {p}')


if __name__ == '__main__':
    main()
