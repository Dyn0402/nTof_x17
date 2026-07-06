#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_core_geometry_vdrift.py

Core-only geometry drift velocity: the event displays (22_*.py) show the
strip-time ladder is corrupted at BOTH cluster ends — low-amplitude mesh-end
skirt strips start late, and deep-end strips fire early (RC-fed charge from
neighbours rather than direct deep charge). Full-cluster estimators are
therefore biased in opposite directions:
    ridge fit (times, incl. skirts)        -> v low  (28)
    full extent / full T_sat (positions)   -> v high (34)
Restricting BOTH axes to the amplitude CORE (>= CORE_FRAC of cluster max)
removes the corrupted ends coherently:
    v_core = (core-extent slope vs |tan th_ref|) / (core time span plateau)

Usage: ../.venv/bin/python 23_core_geometry_vdrift.py sat_det3 [--veto=50]
Prints per-plane results for the long run + drift-scan points with caches.
"""
import os
import re
import sys
import pickle

import numpy as np
import pandas as pd

from qa_config import config_from_argv, setup_paths, _Config
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
MIN_STRIPS = 4
RES_CUT_MM = 10.0
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking); was 20 pre-v2
GAP_MM_CLUSTER = getattr(cm, 'GAP_THRESHOLD_MM', 2.0)
CORE_FRACS = (0.20, 0.30, 0.40)
SAT_DEG = 10.0
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def largest_cluster(pos):
    o = np.argsort(pos)
    breaks = np.where(np.diff(pos[o]) > GAP_MM_CLUSTER)[0]
    return max(np.split(o, breaks + 1), key=len)


def measure(subrun, seed, det):
    cfg = cfg_for(subrun)
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not os.path.exists(cache) or not os.path.isdir(cfg.combined_hits_dir):
        return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)
    th = np.deg2rad(seed.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)

    fs = sorted(f for f in os.listdir(cfg.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    df = uproot.concatenate(
        [f'{cfg.combined_hits_dir}{f}:hits' for f in fs],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'time'], library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)]
    hpe = df.groupby('eventId')['eventId'].transform('size')
    df = df[(hpe <= VETO) & df['eventId'].isin(ref)].copy()
    df = cm._map_strip_positions(df, det)

    data = {cf: {'x': ([], [], []), 'y': ([], [], [])} for cf in CORE_FRACS}
    for eid, g in df.groupby('eventId'):
        for pi, (p, pcol) in enumerate([('x', 'x_position_mm'), ('y', 'y_position_mm')]):
            gp = g[g[pcol].notna()]
            if len(gp) < MIN_STRIPS:
                continue
            pos_all = gp[pcol].to_numpy()
            idx = largest_cluster(pos_all)
            if len(idx) < MIN_STRIPS:
                continue
            pos = gp[pcol].to_numpy()[idx]
            t = gp['time'].to_numpy()[idx]
            amp = gp['amplitude'].to_numpy()[idx]
            for cf in CORE_FRACS:
                m = amp >= cf * amp.max()
                if m.sum() < 3:
                    continue
                tans, exts, durs = data[cf][p]
                tans.append(ref[eid][pi])
                exts.append(np.ptp(pos[m]))
                durs.append(np.ptp(t[m]))

    rows = []
    for cf in CORE_FRACS:
        for p in ('x', 'y'):
            tans, exts, durs = map(np.asarray, data[cf][p])
            if len(tans) < 300:
                continue
            at = np.abs(tans)
            bins = np.arange(TAN_LO, TAN_HI + TAN_STEP, TAN_STEP)
            ctr, med = [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                m = (at >= b0) & (at < b1)
                if m.sum() >= 40:
                    ctr.append(0.5 * (b0 + b1))
                    med.append(np.median(exts[m]))
            if len(ctr) < 4:
                continue
            slope, icpt = np.polyfit(ctr, med, 1)
            m_sat = np.abs(np.degrees(np.arctan(tans))) > SAT_DEG
            tsat = float(np.median(durs[m_sat]))
            rows.append(dict(subrun=subrun, core_frac=cf, plane=p,
                             z_slope_mm=float(slope), w0_mm=float(icpt),
                             t_sat_ns=tsat,
                             v_core=float(slope * 1000.0 / tsat)))
    return rows


def main():
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    seed = cm.load_alignment(os.path.join(CFG.OUT_BASE,
                                          f'alignment_tpc_veto{VETO}', 'alignment.json'))
    subruns = [CFG.SUB_RUN]
    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_\d+V_drift_(\d+)V$', name)
        if m and int(m.group(1)) >= 700:
            subruns.append(name)

    all_rows = []
    for sub in subruns:
        print(f'=== {sub} ===')
        rows = measure(sub, seed, det)
        if rows:
            all_rows.extend(rows)
            for r in rows:
                print(f"  core>{r['core_frac']:.0%} {r['plane']}: "
                      f"z_slope = {r['z_slope_mm']:5.1f} mm  w0 = {r['w0_mm']:4.1f} mm  "
                      f"T_sat = {r['t_sat_ns']:4.0f} ns  →  v = {r['v_core']:5.2f} µm/ns")
    pd.DataFrame(all_rows).to_csv(
        os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                     CFG.RUN, 'drift_velocity', CFG.DET_NAME, 'core_geometry_vdrift.csv'),
        index=False)


if __name__ == '__main__':
    main()
