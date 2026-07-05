#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_recofar.py

Characterise the reco_far events for det3 (the headline g_det3_wknd run).

reco_far = a clean M3 muon crossing the active area for which the detector DID
form a valid X+Y reconstructed point, but that point lands > R mm from the M3
ray projection, AND the event is NOT a spark (spark = > SPARK_THRESH strips —
those are pulled out separately, matching 09_efficiency_breakdown.py).

We replicate 09's categorisation EXACTLY (same alignment, same active box, same
spark cut) so the reco_far set here == the reco_far bar on the June overview
PDF, then join every reco_far/reco_near event back to its EventResult to pull
per-plane residuals and cluster-quality fields.

Heavy step (load M3 rays + attach) runs once; results cached to recofar_data.npz.
Run:  ../../.venv/bin/python extract_recofar.py [KEY]   (default g_det3_wknd)
"""
import os, sys, json, pickle
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions

KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
R = 5.0
SPARK_THRESH = 50
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, 'recofar_data.npz')
META = os.path.join(HERE, 'recofar_meta.json')

import uproot


def main():
    CFG = get_config(KEY)
    print(f'KEY={KEY}  DET={CFG.DET_NAME}  RUN={CFG.RUN}/{CFG.SUB_RUN}')

    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20.0)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    result_by_id = {r.event_id: r for r in res}
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # raw multiplicity per event (spark tagging) -- same as 09
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    det_raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    det_hit = set(int(e) for e in det_raw['eventId'].unique())
    mult_by_ev = det_raw.groupby('eventId').size().to_dict()

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr); evn = [int(v) for v in evn]
    nray_by_ev = Counter(evn)                      # rays per event (multi-track flag)

    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    box = [float(ax0), float(ax1), float(ay0), float(ay1)]

    # per-event record for reco_near / reco_far (non-spark)
    cols = ['ray_x', 'ray_y', 'det_x', 'det_y', 'r', 'dx', 'dy',
            'nsx', 'nsy', 'chi2x', 'chi2y', 'durx', 'dury',
            'tthx', 'tthy', 'mult', 'nray', 'is_far']
    rec = {c: [] for c in cols}
    n_no_hit = n_hit_no_reco = n_spark = 0

    for e, x, y in zip(evn, px, py):
        if not (np.isfinite(x) and np.isfinite(y) and ax0 <= x <= ax1 and ay0 <= y <= ay1):
            continue
        if mult_by_ev.get(e, 0) > SPARK_THRESH:
            n_spark += 1; continue
        if e in reco:
            dxp, dyp = reco[e]
            r = float(np.hypot(x - dxp, y - dyp))
            R_ = result_by_id.get(e)
            xf, yf = R_.x_fit, R_.y_fit
            rec['ray_x'].append(x); rec['ray_y'].append(y)
            rec['det_x'].append(float(dxp)); rec['det_y'].append(float(dyp))
            rec['r'].append(r)
            rec['dx'].append(float(dxp - x)); rec['dy'].append(float(dyp - y))
            rec['nsx'].append(int(xf.n_strips)); rec['nsy'].append(int(yf.n_strips))
            rec['chi2x'].append(float(xf.red_chi2)); rec['chi2y'].append(float(yf.red_chi2))
            rec['durx'].append(float(xf.cluster_duration_ns)); rec['dury'].append(float(yf.cluster_duration_ns))
            rec['tthx'].append(float(xf.tan_theta_estimate)); rec['tthy'].append(float(yf.tan_theta_estimate))
            rec['mult'].append(int(mult_by_ev.get(e, 0)))
            rec['nray'].append(int(nray_by_ev.get(e, 1)))
            rec['is_far'].append(bool(r > R))
        elif e in det_hit:
            n_hit_no_reco += 1
        else:
            n_no_hit += 1

    arrs = {c: np.array(rec[c]) for c in cols}
    n_total = (len(arrs['r']) + n_spark + n_hit_no_reco + n_no_hit)
    np.savez(NPZ, box=np.array(box), R=R, SPARK_THRESH=SPARK_THRESH, **arrs)

    is_far = arrs['is_far'].astype(bool)
    meta = dict(
        key=KEY, det=CFG.DET_NAME, run=CFG.RUN, subrun=CFG.SUB_RUN,
        R=R, spark_thresh=SPARK_THRESH, box=box,
        n_active_crossings=int(n_total),
        n_reco_near=int((~is_far).sum()), n_reco_far=int(is_far.sum()),
        n_spark=int(n_spark), n_hit_no_reco=int(n_hit_no_reco), n_no_hit=int(n_no_hit),
        pct_reco_near=100 * float((~is_far).sum()) / n_total,
        pct_reco_far=100 * float(is_far.sum()) / n_total,
        pct_spark=100 * float(n_spark) / n_total,
    )
    json.dump(meta, open(META, 'w'), indent=2)
    print(json.dumps(meta, indent=2))
    print(f'\nWrote {NPZ}\nWrote {META}')


if __name__ == '__main__':
    main()
