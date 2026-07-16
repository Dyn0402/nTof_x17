#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_chi2_extract.py

Follow-up extraction for the det3 position-miss investigation (2026-07-13 owner ask).

Beyond extract_recofar.py this records, for EVERY active-area M3 crossing:
  * the M3 REFERENCE-TRACK quality (Chi2X, Chi2Y, NClusX, NClusY, incidence angle)
    -- so we can test whether the > ~2 mm misses are the reference tracker's fault
    (a slightly-bad M3 track that still passes chi2<5) rather than the chamber's.
  * the crossing category (reco_near / reco_far / spark / hit_no_reco / no_hit) and
    ray position -- so we can build 1-D efficiency-vs-position profiles and locate the
    physical active-area edges / the ~2 cm passivated Y strips for the efficiency
    denominator.

Runs on the M3 v2 recipe (chi2<5, NClus>=3) to match 09_efficiency_breakdown.py.
Heavy step cached to edge_chi2_data.npz.

Run:  ../../.venv/bin/python edge_chi2_extract.py [KEY]     (default g_det3_wknd)
"""
import os, sys, json, pickle
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions
import awkward as ak
import uproot

KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
R = 5.0
SPARK_THRESH = 50
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, f'edge_chi2_data_{KEY}.npz')

# category codes for the all-crossing arrays
CAT = {'reco_near': 0, 'reco_far': 1, 'spark': 2, 'hit_no_reco': 3, 'no_hit': 4}


def main():
    CFG = get_config(KEY)
    print(f'KEY={KEY}  DET={CFG.DET_NAME}  RUN={CFG.RUN}/{CFG.SUB_RUN}')
    print(f'm3 dir: {CFG.m3_tracking_dir}')

    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    # intentionally decoupled from qa_config.M3_CHI2_CUT: this is the scan CEILING --
    # records Chi2X/Chi2Y/NClus per event so m3_cut_scan.py can scan tighter cuts
    # post-hoc; must stay >= the loosest cut of interest (old chi2<5 recipe). NClus
    # stays at the M3RefTracking default (3) unchanged, matching the cached
    # edge_chi2_data_*.npz / m3_cut_scan.json this script already produced.
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=5.0)          # v2 recipe (+NClus>=3 default)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)

    result_by_id = {r.event_id: r for r in res}
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}

    # ----- M3 per-event reference-track quality (ray_data is flat: 1 track/event) -----
    m3_evn = ak.to_numpy(rays.ray_data['evn']).astype(int)
    m3_chi2x = ak.to_numpy(rays.ray_data['Chi2X']).astype(float)
    m3_chi2y = ak.to_numpy(rays.ray_data['Chi2Y']).astype(float)
    m3_nclusx = ak.to_numpy(rays.ray_data['NClusX']).astype(float)
    m3_nclusy = ak.to_numpy(rays.ray_data['NClusY']).astype(float)
    # incidence angle from vertical [deg] (combined x & y slopes)
    xang, yang, ang_evn = get_xy_angles(rays.ray_data)
    tanth = np.hypot(np.tan(np.asarray(xang)), np.tan(np.asarray(yang)))
    m3_theta_deg = np.degrees(np.arctan(tanth))
    q_by_id = {int(e): (m3_chi2x[i], m3_chi2y[i], m3_nclusx[i], m3_nclusy[i], m3_theta_deg[i])
               for i, e in enumerate(m3_evn)}

    # ----- raw multiplicity per event (spark tagging) + which events fired at all -----
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                             expressions=['eventId', 'feu'], library='pd')
    det_raw = raw[raw['feu'].isin(CFG.MX17_FEUS)]
    det_hit = set(int(e) for e in det_raw['eventId'].unique())
    mult_by_ev = det_raw.groupby('eventId').size().to_dict()
    det_lo = int(det_raw['eventId'].min()); det_hi = int(det_raw['eventId'].max())

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr); evn = [int(v) for v in evn]
    nray_by_ev = Counter(evn)

    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    box = [float(ax0), float(ax1), float(ay0), float(ay1)]

    # ----- per reco crossing (near+far) with full M3 + detector quality -----
    cols = ['ray_x', 'ray_y', 'det_x', 'det_y', 'r', 'dx', 'dy',
            'mesh_x', 'mesh_y',
            'nsx', 'nsy', 'fchi2x', 'fchi2y', 'durx', 'dury', 'tthx', 'tthy',
            'mult', 'nray',
            'm3_chi2x', 'm3_chi2y', 'm3_nclusx', 'm3_nclusy', 'm3_theta']
    rec = {c: [] for c in cols}

    # ----- all-crossing (any category) ray position + category + M3 chi2 -----
    # Recorded over a WIDE window (box +/- MARGIN) rather than clipped to the percentile
    # box, so the efficiency turn-off at the physical active-area edges / passivated Y
    # strips is visible for the edge study. `in_box` flags the 09 denominator subset.
    EDGE_MARGIN = 40.0
    wx0, wx1 = ax0 - EDGE_MARGIN, ax1 + EDGE_MARGIN
    wy0, wy1 = ay0 - EDGE_MARGIN, ay1 + EDGE_MARGIN
    allc = {c: [] for c in ('cx', 'cy', 'cat', 'cm3chi2x', 'cm3chi2y', 'ctheta', 'in_box')}

    n_outside = 0
    for e, x, y in zip(evn, px, py):
        if e < det_lo or e > det_hi:
            n_outside += 1; continue
        if not (np.isfinite(x) and np.isfinite(y) and wx0 <= x <= wx1 and wy0 <= y <= wy1):
            continue
        in_box = bool(ax0 <= x <= ax1 and ay0 <= y <= ay1)
        q = q_by_id.get(e, (np.nan,) * 5)
        # classify
        if mult_by_ev.get(e, 0) > SPARK_THRESH:
            cat = 'spark'
        elif e in reco:
            dxp, dyp = reco[e]
            r = float(np.hypot(x - dxp, y - dyp))
            cat = 'reco_far' if r > R else 'reco_near'
        elif e in det_hit:
            cat = 'hit_no_reco'
        else:
            cat = 'no_hit'

        allc['cx'].append(x); allc['cy'].append(y); allc['cat'].append(CAT[cat])
        allc['cm3chi2x'].append(q[0]); allc['cm3chi2y'].append(q[1]); allc['ctheta'].append(q[4])
        allc['in_box'].append(in_box)

        if in_box and cat in ('reco_near', 'reco_far'):
            R_ = result_by_id[e]
            xf, yf = R_.x_fit, R_.y_fit
            dxp, dyp = reco[e]
            rec['ray_x'].append(x); rec['ray_y'].append(y)
            rec['det_x'].append(float(dxp)); rec['det_y'].append(float(dyp))
            rec['r'].append(float(np.hypot(x - dxp, y - dyp)))
            rec['dx'].append(float(dxp - x)); rec['dy'].append(float(dyp - y))
            rec['mesh_x'].append(float(xf.mesh_position_mm)); rec['mesh_y'].append(float(yf.mesh_position_mm))
            rec['nsx'].append(int(xf.n_strips)); rec['nsy'].append(int(yf.n_strips))
            rec['fchi2x'].append(float(xf.red_chi2)); rec['fchi2y'].append(float(yf.red_chi2))
            rec['durx'].append(float(xf.cluster_duration_ns)); rec['dury'].append(float(yf.cluster_duration_ns))
            rec['tthx'].append(float(xf.tan_theta_estimate)); rec['tthy'].append(float(yf.tan_theta_estimate))
            rec['mult'].append(int(mult_by_ev.get(e, 0))); rec['nray'].append(int(nray_by_ev.get(e, 1)))
            rec['m3_chi2x'].append(float(q[0])); rec['m3_chi2y'].append(float(q[1]))
            rec['m3_nclusx'].append(float(q[2])); rec['m3_nclusy'].append(float(q[3]))
            rec['m3_theta'].append(float(q[4]))

    arrs = {c: np.array(rec[c]) for c in cols}
    allarr = {c: np.array(allc[c]) for c in allc}
    np.savez(NPZ, box=np.array(box), R=R, SPARK_THRESH=SPARK_THRESH,
             det_lo=det_lo, det_hi=det_hi, n_outside=n_outside,
             cat_codes=json.dumps(CAT), **arrs, **allarr)

    catarr = allarr['cat']; inbox = allarr['in_box'].astype(bool)
    cb = catarr[inbox]; n = len(cb)
    print(f'\nin-box (09-denominator) crossings: {n}  |  wide-window total: {len(catarr)}  '
          f'(excluded {n_outside} outside det data range)')
    for k, v in CAT.items():
        print(f'  {k:12s}: {int((cb == v).sum()):6d}  ({100*(cb==v).mean():5.2f}%)')
    print(f'\nWrote {NPZ}')


if __name__ == '__main__':
    main()
