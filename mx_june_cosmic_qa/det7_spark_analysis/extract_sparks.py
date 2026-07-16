#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_sparks.py

Characterise the SPARKS (full-detector discharges, > SPARK_THRESH strips firing)
for det3 (headline g_det3_wknd run). Answers four questions, dumping arrays for
make_plots.py:

  Q1  random in time?          per-event timestamp + spark flag  -> events.npz
  Q2  muon-induced or random?  M3 ray projected onto det plane, in-box vs miss,
                               vs spark flag                     -> events.npz
  Q3  particular places?       per-spark-hit strip position (X=FEU7, Y=FEU8)
                               and per-spark centroid            -> spark_hits.npz
  Q4  all-at-once or "walk"?   per-spark-hit time (ns) vs strip position

Heavy step (load hits + M3 rays + attach) runs once; cached to the .npz files.
Run:  ../../.venv/bin/python extract_sparks.py [KEY]   (default g_det3_wknd)
"""
import os, sys, json, pickle
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_config import get_config, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from common.Mx17StripMap import RunConfig
from M3RefTracking import M3RefTracking, get_xy_angles, get_xy_positions
import uproot

KEY = next((a for a in sys.argv[1:] if not a.startswith('-')), 'g_det3_wknd')
SPARK_THRESH = 50
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    CFG = get_config(KEY)
    fx, fy = CFG.MX17_FEUS            # 7 = X plane, 8 = Y plane
    print(f'KEY={KEY}  DET={CFG.DET_NAME}  FEU X/Y={fx}/{fy}  RUN={CFG.RUN}/{CFG.SUB_RUN}')

    # ---- channel -> strip position lookup (raw detector frame, 0..~399 mm) ----
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    posmap = {}                      # (feu, channel) -> position along that plane's axis
    for ch in range(512):
        px = det.map_hit(fx, ch); py = det.map_hit(fy, ch)
        if px and px[0] is not None:
            posmap[(fx, ch)] = float(px[0])
        if py and py[1] is not None:
            posmap[(fy, ch)] = float(py[1])

    # ---- all detector hits ----
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    raw = uproot.concatenate(
        [f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
        expressions=['eventId', 'trigger_timestamp_ns', 'channel', 'feu',
                     'time', 'amplitude', 'saturated'], library='pd')
    raw = raw[raw['feu'].isin([fx, fy])].reset_index(drop=True)
    raw['eventId'] = raw['eventId'].astype(np.int64)
    print(f'det hits: {len(raw)}  events firing: {raw["eventId"].nunique()}')

    # per-event multiplicity + spark flag + per-plane counts
    g = raw.groupby('eventId')
    mult = g.size()
    nx = raw[raw['feu'] == fx].groupby('eventId').size()
    ny = raw[raw['feu'] == fy].groupby('eventId').size()
    ts = g['trigger_timestamp_ns'].first()
    ev = mult.index.values.astype(np.int64)
    mult_by_ev = mult.to_dict()
    spark_ev = set(int(e) for e in mult.index[mult > SPARK_THRESH])
    print(f'spark events (>{SPARK_THRESH} strips): {len(spark_ev)}')

    # ---- M3 rays: aligned projection onto the detector plane (same frame as 09) ----
    params = cm.load_alignment(os.path.join(CFG.OUT_BASE, 'alignment_tpc_veto50', 'alignment.json'))
    res = pickle.load(open(os.path.join(CFG.OUT_BASE, 'cache', 'event_results.pkl'), 'rb'))
    # intentionally loose (not qa_config.M3_CHI2_CUT): needs the full M3 crossing
    # population to classify spark events, not a precision-position subsample.
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20.0)
    xa, ya, an = get_xy_angles(rays.ray_data); xa = params.ref_x_sign * np.array(xa)
    cm.attach_reference_positions(res, rays, params, xa, an)
    reco = {r.event_id: (r.det_x_aligned_mm, r.det_y_aligned_mm) for r in res if r.has_both
            and np.isfinite(r.det_x_aligned_mm) and np.isfinite(r.det_y_aligned_mm)}
    recpos = np.array(list(reco.values()))
    ax0, ax1 = np.percentile(recpos[:, 0], [0.5, 99.5])
    ay0, ay1 = np.percentile(recpos[:, 1], [0.5, 99.5])
    box = [float(ax0), float(ax1), float(ay0), float(ay1)]

    xr, yr, evn = get_xy_positions(rays.ray_data, params.z_mean)
    px = params.ref_x_sign * np.array(xr); py = np.array(yr)
    evn = np.array([int(v) for v in evn], dtype=np.int64)
    nray = Counter(evn)
    # keep single-ray events only for a clean geometry test (multi-ray ~0% anyway)
    ray_x, ray_y = {}, {}
    for e, x, y in zip(evn, px, py):
        ray_x[int(e)] = float(x); ray_y[int(e)] = float(y)

    # ---- per-event table (union of firing events and ray events) ----
    all_ev = sorted(set(int(e) for e in ev) | set(int(e) for e in evn))
    E = len(all_ev)
    ev_arr = np.array(all_ev, dtype=np.int64)
    e_ts = np.array([int(ts.get(e, 0)) for e in all_ev], dtype=np.int64)
    e_mult = np.array([int(mult_by_ev.get(e, 0)) for e in all_ev])
    e_spark = np.array([e in spark_ev for e in all_ev])
    e_rx = np.array([ray_x.get(e, np.nan) for e in all_ev])
    e_ry = np.array([ray_y.get(e, np.nan) for e in all_ev])
    e_hasray = np.isfinite(e_rx) & np.isfinite(e_ry)
    e_nray = np.array([nray.get(e, 0) for e in all_ev])
    np.savez(os.path.join(HERE, 'events.npz'),
             box=np.array(box), spark_thresh=SPARK_THRESH,
             eventId=ev_arr, ts=e_ts, mult=e_mult, spark=e_spark,
             ray_x=e_rx, ray_y=e_ry, has_ray=e_hasray, nray=e_nray)

    # ---- per-spark-hit arrays (for Q3 places, Q4 walk) ----
    sp = raw[raw['eventId'].isin(spark_ev)].copy()
    sp['pos'] = [posmap.get((f, c), np.nan) for f, c in zip(sp['feu'].values, sp['channel'].values)]
    isx = (sp['feu'] == fx).values
    np.savez(os.path.join(HERE, 'spark_hits.npz'),
             eventId=sp['eventId'].values.astype(np.int64),
             is_x=isx, pos=sp['pos'].values.astype(float),
             time=sp['time'].values.astype(float),
             amp=sp['amplitude'].values.astype(float),
             chan=sp['channel'].values.astype(np.int64),
             sat=sp['saturated'].values.astype(bool))

    # ---- a normal-cluster time-spread reference (non-spark firing events) ----
    norm_ev = [int(e) for e in mult.index[(mult >= 4) & (mult <= 12)]]
    norm_sample = norm_ev[::max(1, len(norm_ev) // 4000)][:4000]   # deterministic subsample
    nm = raw[raw['eventId'].isin(set(norm_sample))]
    tspread_norm = nm.groupby('eventId')['time'].agg(lambda v: float(np.std(v))).values

    meta = dict(
        key=KEY, det=CFG.DET_NAME, feu_x=fx, feu_y=fy,
        run=CFG.RUN, subrun=CFG.SUB_RUN, spark_thresh=SPARK_THRESH, box=box,
        n_events_union=E, n_firing=int(len(mult)), n_spark=len(spark_ev),
        n_ray_events=int(e_hasray.sum()),
        tspread_norm_median_ns=float(np.median(tspread_norm)),
        run_duration_s=float((e_ts[e_ts > 0].max() - e_ts[e_ts > 0].min()) / 1e9),
        # detector-local -> aligned/ref frame map (see common/mx17_active_area.alignment_transform),
        # so make_plots.py can draw the active-area outline without re-loading qa_config/CFG.
        alignment=dict(theta_deg=params.theta_deg, centre_x=params.centre_x, centre_y=params.centre_y,
                      x_offset=params.x_offset, y_offset=params.y_offset),
    )
    json.dump(meta, open(os.path.join(HERE, 'spark_meta.json'), 'w'), indent=2)
    print(json.dumps(meta, indent=2))
    print('\nwrote events.npz, spark_hits.npz, spark_meta.json')


if __name__ == '__main__':
    main()
