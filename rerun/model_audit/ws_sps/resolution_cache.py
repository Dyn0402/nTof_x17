#!/usr/bin/env python3
"""resolution_cache.py --set flat|rot25 [--per-plateau 3000] [--out DIR]

det4 H4 events (run_63, Ar/CF4/iso, ZS 4 sigma) in the bench calibration-cache format WITH the raw uRWELL cluster
positions (front/back, both views) of each event, for the position resolution of the waveform fit at 0 deg and at
25.64 deg against the same reference (F14 / F20, ws_sps 2026-10-27).  Same selection as sps_cache_incl.py (uRWELL
prediction finite, +-16 strips around it, window max >= 300 ADC); rec['urw'] = (fx, fy, bx, by) from the pair npz.

  flat : wf_run63_flat.npz      / pair_run63_flat.npz      plateau flat700 (243 V/cm)
  rot25: wf_run63_operating.npz / pair_run63_operating.npz plateaus d425 (148 V/cm), d325 (113 V/cm)
Output <out>/res_<set>_<plateau>.pkl
"""
import argparse, os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'analysis'), os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import datasets                                              # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

ST = '/media/dylan/data/x17/sps_run53_det4_check/staging/run_63/'
SETS = {'flat': ('wf_run63_flat.npz', 'pair_run63_flat.npz', 'run63_flat', 0.0),
        'rot25': ('wf_run63_operating.npz', 'pair_run63_operating.npz', 'run63_rot25', 25.64)}
NOISE_ADC = 10.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--set', required=True, choices=list(SETS))
    ap.add_argument('--per-plateau', type=int, default=3000); ap.add_argument('--half-win', type=int, default=16)
    ap.add_argument('--out', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    wf, pair, ds, mount = SETS[a.set]
    D = datasets.get(ds); nsmp = int(D['n_samples']); tanv = float(np.tan(np.radians(mount)))
    Z = np.load(ST + wf, allow_pickle=True); Pz = np.load(ST + pair, allow_pickle=True)
    urw = np.stack([Pz['fx_p'], Pz['fy_p'], Pz['bx_p'], Pz['by_p']], 1)
    ev, ch, samp, amp = Z['ev'], Z['ch'].astype(int), Z['samp'], Z['amp']
    pX, pY, tw = Z['ev_pX'], Z['ev_pY'], Z['ev_t_wall']
    lab = np.full(len(pX), '', dtype='<U12')
    for name, lo, hi, dv, rv in D['plateaus']:
        lab[(tw >= lo) & (tw < hi)] = name
    order = np.lexsort((ch, ev)); ev, ch, samp, amp = ev[order], ch[order], samp[order], amp[order]
    starts = np.r_[0, np.flatnonzero(ev[1:] != ev[:-1]) + 1, len(ev)]
    out = {}
    for s, e in zip(starts[:-1], starts[1:]):
        eid = int(ev[s]); L = str(lab[eid])
        if not L or len(out.get(L, {})) >= a.per_plateau or not (np.isfinite(pX[eid]) and np.isfinite(pY[eid])):
            continue
        if not np.isfinite(urw[eid]).all():
            continue
        rec = dict(eid=eid, tan_x=tanv, tan_y=tanv, ref_mesh_x=float(pX[eid]), ref_mesh_y=float(pY[eid]), t_wall=float(tw[eid]),
                   urw=urw[eid].astype(float))
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
        p = os.path.join(a.out, f'res_{a.set}_{L}.pkl')
        pickle.dump(evs, open(p, 'wb'), protocol=4)
        print(f'{L}: {len(evs)} events -> {p}')


if __name__ == '__main__':
    main()
