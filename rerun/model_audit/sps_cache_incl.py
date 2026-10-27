#!/usr/bin/env python3
"""sps_cache_incl.py [--wf wf_run63_operating.npz] [--dataset run63_rot25] [--half-win 16] [--per-plateau 800] [--out DIR]

det4 H4 INCLINED events (25.64 deg mount) in the bench calibration-cache format, one pickle per drift plateau, for the
known-angle test of the waveform fit (RERUN_PLAN S4 R7): the fitted w must equal tan(25.64 deg) x v(E) in the
inclined view and ~0 in the other.  Events are re-labelled by wall clock against datasets.py's plateau windows (the
local wf npz predates the current labels).  tan_x = tan_y = tan(mount) is stored as the magnitude; the sign and the
inclined view are read off the data.  ref_mesh = ev_pX/ev_pY of the wf npz, which is det4's OWN leading strip (extract_det4_only.py), not the uRWELL
prediction (corrected 2026-10-27, ws_sps; the uRWELL clusters are in the pair npz: ws_sps/resolution_cache.py).  ZS data:
exactly-zero samples are not measurements (run with WFT_ZS_CENSOR=1).
"""
import argparse, os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'analysis'), os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import datasets                                              # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

NOISE_ADC = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wf', default='/media/dylan/data/x17/sps_run53_det4_check/staging/run_63/wf_run63_operating.npz')
    ap.add_argument('--dataset', default='run63_rot25'); ap.add_argument('--half-win', type=int, default=16)
    ap.add_argument('--per-plateau', type=int, default=800); ap.add_argument('--mount-deg', type=float, default=25.64)
    ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/sps'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = datasets.get(a.dataset); nsmp = int(D['n_samples']); tanv = float(np.tan(np.radians(a.mount_deg)))
    Z = np.load(a.wf, allow_pickle=True)
    ev, ch, samp, amp = Z['ev'], Z['ch'].astype(int), Z['samp'], Z['amp']
    pX, pY, tw = Z['ev_pX'], Z['ev_pY'], Z['ev_t_wall']
    lab = np.full(len(pX), '', dtype='<U8')
    for name, lo, hi, dv, rv in D['plateaus']:
        lab[(tw >= lo) & (tw < hi)] = name
    order = np.lexsort((ch, ev)); ev, ch, samp, amp = ev[order], ch[order], samp[order], amp[order]
    starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1, len(ev)]
    out = {}
    for s, e in zip(starts[:-1], starts[1:]):
        eid = int(ev[s]); L = str(lab[eid])
        if not L or len(out.get(L, {})) >= a.per_plateau or not (np.isfinite(pX[eid]) and np.isfinite(pY[eid])):
            continue
        rec = dict(eid=eid, tan_x=tanv, tan_y=tanv, ref_mesh_x=float(pX[eid]), ref_mesh_y=float(pY[eid]), t_wall=float(tw[eid]))
        c, sm, am = ch[s:e], samp[s:e], amp[s:e]
        ok = True
        for view, p0 in (('x', pX[eid]), ('y', pY[eid])):
            k = (VIEW[c] == view) & (np.abs(POSITION_MM[c] - p0) <= a.half_win * PITCH_MM)
            if k.sum() == 0:
                ok = False; break
            cc, ss, aa = c[k], sm[k], am[k]
            chs = np.unique(cc); W = np.zeros((len(chs), nsmp), np.float32)
            ci = np.searchsorted(chs, cc); m = (ss >= 0) & (ss < nsmp)
            W[ci[m], ss[m].astype(int)] = aa[m]
            if W.max() < 300 or len(chs) < 4:
                ok = False; break
            o = np.argsort(POSITION_MM[chs])
            rec[view] = dict(ch=chs[o].astype(np.int16), pos=POSITION_MM[chs[o]].astype(np.float32), W=W[o],
                             noise=np.full(len(chs), NOISE_ADC, np.float32))
            rec[f'ftst_{view}'] = 0
        if ok:
            out.setdefault(L, {})[eid] = rec
    for L, evs in out.items():
        p = os.path.join(a.out, f'{a.dataset}_{L}.pkl')
        pickle.dump(evs, open(p, 'wb'), protocol=4)
        dv = {x[0]: x[3] for x in D['plateaus']}[L]
        print(f'{L}: {len(evs)} events, drift {dv} V -> {p}')


if __name__ == '__main__':
    main()
