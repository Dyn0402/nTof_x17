#!/usr/bin/env python3
"""
10d_hv_scan_wft_reco.py — run the frozen waveform-first reconstruction over the
det3 mesh-voltage ladder, one events.parquet per sub-run.

Why this and not the hits chain: everything this pass exists to measure —
total deposited charge, how much of the drift column is filled, the track
angle — is geometry or charge, and `combined_hits` times/amplitudes are not a
basis for either (RECONSTRUCTION_BASIS.md). The forward fit also *censors*
saturated samples (one-sided penalty, wft.model.chi2_plane), which is the whole
point here: the peak strip rails above ~500 V, so the peak-amplitude gain
ladder (10c) must stop there, while the deconvolved charge need not.

Calibration: the frozen det3 r06 bundle (c2 = 0.6 c1). It was fitted on the
490 V long run of THIS run at THIS drift field, and every hyper in it is a
property of the drift gap, the resistive layer or the electronics — none of
them is a function of the mesh voltage — so it transfers along the ladder.
The one thing that *is* mesh-dependent, the amplitude, is what we measure.
v_drift is not refit per sub-run: the drift field is 1000 V throughout.

Selection: exactly the population 10b/10c use — M3 golden rays (chi2 < 1,
NClus = 4) landing inside the same fiducial box. Voltage-independent by
construction, so detector effects show up as changed reconstruction, never as
a changed sample. Spark-vetoed events are dropped by the seeder; the loss is
recorded per sub-run (it is the discharge fraction, and it is large at 525 V).

    ../.venv/bin/python 10d_hv_scan_wft_reco.py            # all 18 sub-runs
    ../.venv/bin/python 10d_hv_scan_wft_reco.py --only hv_scan_resist_425V_drift_1000V
    ../.venv/bin/python 10d_hv_scan_wft_reco.py --jobs 8 --force
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE, os.path.join(REPO, 'cosmic_bench_analysis')]

RUN = 'mx17_det3_saturday_scan_6-27-26'
BASE = '/home/dylan/x17/cosmic_bench/det3/'
BUNDLE = ('/media/dylan/data/x17/cosmic_bench/condor_campaign_r06/'
          'local_bundles/mx17_3/calib_bundle_r06')
SUBRUN_RE = re.compile(r'^hv_scan(2?)_resist_(\d+)V_drift_1000V$')
DET_Z = 702.0
FID_X = (-190.0, 115.0)          # identical to mx17_sim_wft/hv_slope/analyse.py
FID_Y = (-190.0, 165.0)
TAG = 'events_hvscan.parquet'


def subruns(base, run):
    out = []
    for d in sorted(os.listdir(os.path.join(base, run))):
        m = SUBRUN_RE.match(d)
        if m:
            out.append((d, int(m.group(2)), 'scan2' if m.group(1) else 'scan1'))
    return sorted(out, key=lambda t: (t[2], t[1]))


def fiducial_ids(cfg):
    """M3-golden event ids whose ray crosses the fiducial box at the det plane."""
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    x, y, evn = rays.get_xy_positions(DET_Z)
    x, y, evn = np.asarray(x), np.asarray(y), np.asarray(evn)
    ok = (np.isfinite(x) & np.isfinite(y)
          & (x >= FID_X[0]) & (x <= FID_X[1])
          & (y >= FID_Y[0]) & (y <= FID_Y[1]))
    return set(int(e) for e in evn[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--bundle', default=BUNDLE)
    ap.add_argument('--jobs', type=int, default=7)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force', action='store_true',
                    help='re-reconstruct sub-runs that already have a table')
    a = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.calib import CalibrationBundle
    from wft.reco import reconstruct_run

    cal = CalibrationBundle.load(a.bundle)
    print('[10d]', cal.summary())

    todo = [s for s in subruns(a.base, a.run)
            if not a.only or s[0] in a.only]
    print(f'[10d] {len(todo)} sub-runs')
    for sub, volt, scan in todo:
        cfg = get_config('sat_det3')
        cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN = a.base, a.run, sub
        # OUT_BASE is frozen at construction from (run, sub_run) -- retarget it,
        # or every sub-run writes over the golden long-run table
        cfg.OUT_BASE = os.path.join(
            os.path.dirname(a.base.rstrip('/')), 'Analysis', a.run, sub,
            cfg.DET_NAME)
        out = os.path.join(cfg.out_dir('wft'), TAG)
        if os.path.exists(out) and not a.force:
            print(f'[10d] {sub}: have {out}, skip')
            continue
        if not os.path.isdir(os.path.join(a.base, a.run, sub, 'decoded_root')):
            print(f'[10d] {sub}: no decoded_root, skip')
            continue
        ids = fiducial_ids(cfg)
        t0 = time.time()
        print(f'[10d] === {sub}  {volt} V ({scan})  '
              f'{len(ids):,} fiducial M3 rays', flush=True)
        df = reconstruct_run(cfg, cal, out, event_filter=ids, jobs=a.jobs,
                             limit=a.limit, bundle_path=a.bundle)
        dt = time.time() - t0
        side = {'n_fiducial_rays': len(ids), 'volt': volt, 'scan': scan,
                'subrun': sub, 'n_reco': int(len(df)),
                'seconds': round(dt, 1), 'bundle': a.bundle,
                'fid_x': list(FID_X), 'fid_y': list(FID_Y), 'det_z': DET_Z}
        with open(out.replace('.parquet', '.hvscan.json'), 'w') as f:
            json.dump(side, f, indent=1)
        print(f'[10d] {sub}: {len(df):,}/{len(ids):,} reconstructed '
              f'in {dt:.0f} s', flush=True)


if __name__ == '__main__':
    main()
