#!/usr/bin/env python3
"""hits_baseline.py --run run_53 --subrun cfg_gain4.5_peaktime200_deflt [--gate 600 1850] --out O.npz

The run-53 hits estimator on EVERY telescope event of a ZS run (not only the ones the pass's det4 selection keeps):
uRWELL clusters as tel_job.telescope_zs (= extract2.py), det4 lead cluster per view from combined_hits feu 3
(amp > 60, hit-time gate), one cluster per uRWELL plane-view and per det4 view.  Writes per event the four telescope
positions, det4 hits X/Y and the det4 per-view maximum hit amplitude (to emulate the pass's peak >= 300 ADC selection),
for analyze_baseline in analyze.py.  Reads only combined_hits: a few minutes per run, interactive.
"""
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tel_job as tj                                                  # noqa: E402
from tel_job import POSITION_MM, VIEW                                 # noqa: E402,F401
sys.path.insert(0, os.path.join(tj.REPO, 'sps_beam_test_26', 'analysis', 'spatial_resolution'))
from resolution import det4_clusters                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--subrun', required=True)
    ap.add_argument('--gate', type=float, nargs=2, default=None); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(tj.sbc.RUNS, a.run, a.subrun, 'combined_hits_root', '*_feu-combined_hits.root')))
    acc = {}
    for f in fs:
        g = f.split('_feu-combined')[0][-3:]
        tel, h = tj.telescope_zs(f)
        n = len(tel['eid']); idx = {int(e): i for i, e in enumerate(tel['eid'])}
        m = np.ones(len(h['ev']), bool)
        if a.gate:
            m &= (h['time'] > a.gate[0]) & (h['time'] < a.gate[1])
        ev = np.array([idx[int(e)] for e in h['ev'][m]], int); ch = h['ch'][m].astype(int); amp = h['amp'][m].astype(float)
        isx = VIEW[ch] == 'x'
        rec = dict(group=np.full(n, int(g)), **tel)
        for v, k in (('x', isx), ('y', ~isx)):
            lead, nc, size = det4_clusters(ev[k], POSITION_MM[ch[k]], amp[k], n) if k.any() else \
                (np.full(n, np.nan), np.zeros(n, int), np.zeros(n, int))
            amax = np.zeros(n); np.maximum.at(amax, ev[k], amp[k])
            rec[f'd{v}'], rec[f'd{v}_n'], rec[f'd{v}_size'], rec[f'd{v}_amax'] = lead, nc, size, amax
        for k_, v_ in rec.items():
            acc.setdefault(k_, []).append(np.asarray(v_))
        print(f'{os.path.basename(f)}: {n} four-view telescope events, det4 both views {np.mean(np.isfinite(rec["dx"]) & np.isfinite(rec["dy"])):.3f}', flush=True)
    np.savez_compressed(a.out, **{k: np.concatenate(v) for k, v in acc.items()})
    print('wrote', a.out)


if __name__ == '__main__':
    main()
