#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_slope_reference_vdrift_scan.py

Drift velocity vs field from the SLOPE-vs-REFERENCE method, waveform-level,
across the whole det3 drift scan.

Idea (user, 2026-07-15): a track's recorded (strip position, arrival time)
points obey  x = x0 + tanθ · v · (t − t0),  so the LOCAL slope
    dx/dt = v · tanθ.
With tanθ known from the M3 reference track,
    v = (dx/dt) / tanθ.
dx/dt is a LOCAL quantity — it is the same whether the window records the
full drift column or only the near-mesh part — so this estimator is
ROBUST TO WINDOW TRUNCATION and reaches the low fields where the geometry
estimator (which needs spatial extent to beat the ~3.9 mm charge-spreading
floor) fails.

The only bias is the ~20 % low pull of charge sharing / resistive-strip RC on
the raw strip times.  We remove it with the measured UNSHARING kernel (a
field-independent design property, CSHARE below), then fit S vs tanθ_ref and
take the SLOPE (removes the additive charge-spreading offset).  This is
`ridge_v` applied to the unshared strip-time slopes.

  * NOT independent of the reference (uses M3 tanθ) — a cross-check / low-field
    tool, not a standalone measurement.
  * At high field it must agree with v_geom (21); at low field it should stay
    on the Magboltz curve where v_geom collapses.

Reuses the machinery of 27_unsharing_refinement.py, parameterised per subrun.

Usage: ../.venv/bin/python 45_slope_reference_vdrift_scan.py sat_det3 [--veto=50] [--alpha=0.5]
Output: <Analysis>/<run>/drift_velocity/<det>/slope_reference_vdrift_scan.csv/.png
"""
import os
import re
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths, _Config
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
ALPHA = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--alpha=')), 0.5)
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS

SAMPLE_NS = 60.0
MIN_STRIPS_AFTER = 3
MIN_STRIPS_BEFORE = 4
# tightened track-to-reference match (was 10.0) + degrader-edge fiducial
RES_CUT_MM = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--rescut=')), 2.0)
EDGE_FID_MM = next((float(a.split('=')[1]) for a in sys.argv if a.startswith('--edge=')), 25.0)
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
GAP_CM = 3.0

# det3's own measured sharing constants (FEU 7 = X, FEU 8 = Y); design property,
# field-independent — see MICROTPC_RUNBOOK.md §5.  {feu: (c1, c2)}
CSHARE = {7: (0.449, 0.052), 8: (0.516, 0.152)}

ALIGN_SEED = os.path.join(CFG.OUT_BASE, f'alignment_tpc_veto{VETO}', 'alignment.json')
OUT = os.path.join(os.path.dirname(CFG.BASE_PATH.rstrip('/')), 'Analysis',
                   CFG.RUN, 'drift_velocity', CFG.DET_NAME)


def robust_line(x, y, n_iter=4, clip=3.0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    keep = np.ones(len(x), bool)
    p = (np.nan, np.nan)
    for _ in range(n_iter):
        if keep.sum() < 10:
            return np.nan, np.nan, np.nan, 0
        p = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(p, x)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) < clip * s
    n = int(keep.sum())
    xk = x[keep]
    resid = y[keep] - np.polyval(p, xk)
    se = np.sqrt(np.sum(resid**2) / max(n - 2, 1) / np.sum((xk - xk.mean())**2))
    return float(p[0]), float(p[1]), float(se), n


def cfd_time(w):
    ipk = int(np.argmax(w))
    a = w[ipk]
    if a < THR_WF or ipk == 0:
        return np.nan
    lvl = 0.5 * a
    for i in range(1, ipk + 1):
        if w[i] >= lvl > w[i - 1]:
            return SAMPLE_NS * (i - 1 + (lvl - w[i - 1]) / (w[i] - w[i - 1]))
    return np.nan


def nsum(x, k):
    out = np.zeros_like(x)
    out[k:] += x[:-k]
    out[:-k] += x[k:]
    return out


def unshare(wb, c1, c2, alpha):
    n, ns = wb.shape
    if n < 3:
        return wb
    ab = np.zeros((5, n))
    ab[0, 2:] = alpha * c2
    ab[1, 1:] = alpha * c1
    ab[2, :] = 1.0
    ab[3, :-1] = alpha * c1
    ab[4, :-2] = alpha * c2
    X = np.zeros_like(wb)
    for s in range(ns):
        rhs = wb[:, s].copy()
        if alpha < 1.0:
            if s >= 1:
                rhs -= (1 - alpha) * c1 * nsum(X[:, s - 1], 1)
            if s >= 2:
                rhs -= (1 - alpha) * c2 * nsum(X[:, s - 2], 2)
        X[:, s] = solve_banded((2, 2), ab, rhs)
    return X


def load_events(ref, det, feus, feu_x, feu_y, dec_dir):
    plane_of_feu = {feu_x: 'x', feu_y: 'y'}
    blocks, pos_of = {}, {}
    for feu in feus:
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                        [0 if plane_of_feu[feu] == 'x' else 1]
                        for ch in range(512)], dtype=float)
        pos_of[feu] = pos
        ok = np.where(np.isfinite(pos))[0]
        o = ok[np.argsort(pos[ok])]
        brk = np.where(np.diff(pos[o]) > 1.5 * PITCH_MM)[0]
        blocks[feu] = np.split(o, brk + 1)

    rows = []
    for feu in feus:
        pi = 0 if plane_of_feu[feu] == 'x' else 1
        fs = sorted(glob.glob(os.path.join(dec_dir, f'*_{feu:02d}.root')))
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(32, 512) for a in a0
                                      if a.size == 32 * 512]), axis=(0, 1))
            for lo in range(0, t.num_entries, CHUNK):
                hi = min(lo + CHUNK, t.num_entries)
                want = [i for i in range(lo, hi) if int(eids_all[i]) in ref]
                if not want:
                    continue
                arr = t.arrays(['eventId', 'amplitude'], entry_start=lo,
                               entry_stop=hi, library='np')
                for i in want:
                    j = i - lo
                    eid = int(arr['eventId'][j])
                    if arr['amplitude'][j].size != 32 * 512:
                        continue
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    rows.append((feu, eid, ref[eid][pi], wfm.T.astype(np.float16)))
    return blocks, pos_of, rows


def process(rows, blocks, pos_of, feus, mode, alpha=1.0):
    out = {f: {'S': [], 'tan': []} for f in feus}
    min_strips = MIN_STRIPS_BEFORE if mode == 'before' else MIN_STRIPS_AFTER
    for feu, eid, tanv, wfm16 in rows:
        wfm = wfm16.astype(np.float32)
        c1, c2 = CSHARE[feu]
        best = None
        for blk in blocks[feu]:
            wb = wfm[blk]
            if mode == 'after':
                wb = unshare(wb, c1, c2, alpha)
            amax = wb.max(axis=1)
            hit = np.where(amax >= THR_HIT)[0]
            if len(hit) < min_strips:
                continue
            brk = np.where(np.diff(hit) > 2)[0]
            for grp in np.split(hit, brk + 1):
                if len(grp) < min_strips:
                    continue
                if best is None or len(grp) > len(best[0]):
                    best = (grp, blk, wb)
        if best is None:
            continue
        grp, blk, wb = best
        pos = pos_of[feu][blk[grp]]
        amp = wb[grp].max(axis=1)
        tt = np.array([cfd_time(wb[g]) for g in grp])
        ok = np.isfinite(tt)
        mcore = (amp >= CORE_FRAC * amp.max()) & ok
        if mcore.sum() < 3 or np.ptp(pos[mcore]) == 0:
            continue
        sl = np.polyfit(pos[mcore], tt[mcore], 1)[0]
        if sl == 0:
            continue
        out[feu]['S'].append(1000.0 / sl)
        out[feu]['tan'].append(tanv)
    return out


def ridge_v(d):
    """Fit strip-slope S [µm/ns] vs tan_ref per sign; slope of the line = v."""
    S, T = np.array(d['S']), np.array(d['tan'])
    vs, ses = [], []
    for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
        m = (T > lo) & (T < hi) & np.isfinite(S)
        v, b, se, n = robust_line(T[m], S[m])
        if np.isfinite(v) and n > 60:
            vs.append(v); ses.append(se)
    if not vs:
        return np.nan, np.nan, 0
    vs, ses = np.array(vs), np.array(ses)
    w = 1.0 / ses**2
    return float(np.sum(vs * w) / np.sum(w)), float(np.std(vs) if len(vs) > 1 else ses[0]), len(vs)


def cfg_for(subrun):
    return _Config(f'{CFG.KEY}_{subrun}', CFG.RUN, subrun, feus=CFG.MX17_FEUS,
                   det_z=CFG.DET_PLANE_Z, det_name=CFG.DET_NAME,
                   base_path=CFG.BASE_PATH, zero_suppressed=CFG.ZERO_SUPPRESSED)


def active_bounds(det):
    lims = {}
    for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))[axis]
                        for ch in range(512)], dtype=float)
        pos = pos[np.isfinite(pos)]
        lims[axis] = (float(pos.min()), float(pos.max()))
    return lims[0], lims[1]


def edge_dist(r, bounds):
    (xmn, xmx), (ymn, ymx) = bounds
    if not (np.isfinite(r.ref_mesh_x_mm) and np.isfinite(r.ref_mesh_y_mm)):
        return np.nan
    return min(r.ref_mesh_x_mm - xmn, xmx - r.ref_mesh_x_mm,
               r.ref_mesh_y_mm - ymn, ymx - r.ref_mesh_y_mm)


def measure_point(subrun, seed, det, bounds):
    cfg = cfg_for(subrun)
    dec_dir = os.path.join(cfg.BASE_PATH, cfg.RUN, subrun, 'decoded_root')
    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{VETO}.pkl')
    if not (os.path.isdir(dec_dir) and glob.glob(os.path.join(dec_dir, '*.root'))):
        print(f'  [SKIP] {subrun}: no decoded_root'); return None
    if not os.path.exists(cache):
        print(f'  [SKIP] {subrun}: no cache'); return None
    results = pickle.load(open(cache, 'rb'))
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = seed.ref_x_sign * np.array(xang)
    params = cm.translation_alignment(results, rays, seed)
    cm.attach_reference_positions(results, rays, params, xang, anum)
    th = np.deg2rad(seed.theta_deg)
    ref = {}
    n_match = 0
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or np.isnan(r.ref_tan_theta_x):
            continue
        if r.radial_residual_mm < 10.0:
            n_match += 1
        if r.radial_residual_mm > RES_CUT_MM:
            continue
        if EDGE_FID_MM > 0 and not (edge_dist(r, bounds) > EDGE_FID_MM):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    print(f'    cuts: {n_match} (res<10) -> {len(ref)} (res<{RES_CUT_MM:g} & '
          f'edge>{EDGE_FID_MM:g}mm)  [{100.0*len(ref)/max(n_match,1):.0f}% kept]')
    if len(ref) < 100:
        print(f'  [SKIP] {subrun}: only {len(ref)} matched'); return None

    blocks, pos_of, rows = load_events(ref, det, CFG.MX17_FEUS,
                                       CFG.MX17_FEU_X, CFG.MX17_FEU_Y, dec_dir)
    row = dict(subrun=subrun, n_matched=len(ref), n_planes=len(rows))
    for mode, key in (('before', 'raw'), ('after', 'unshared')):
        out = process(rows, blocks, pos_of, CFG.MX17_FEUS, mode, alpha=ALPHA)
        vx, ex, _ = ridge_v(out[CFG.MX17_FEU_X])
        vy, ey, _ = ridge_v(out[CFG.MX17_FEU_Y])
        row[f'v_{key}_x'] = vx; row[f'v_{key}_y'] = vy
        vs = [v for v in (vx, vy) if np.isfinite(v)]
        es = [e for v, e in ((vx, ex), (vy, ey)) if np.isfinite(v)]
        row[f'v_{key}'] = float(np.mean(vs)) if vs else np.nan
        row[f'v_{key}_err'] = float(np.hypot(np.mean(es) if es else np.nan,
                                             0.5 * abs(vx - vy) if len(vs) == 2 else 0.0))
    print(f'  {subrun}: v_raw = {row["v_raw"]:.2f}  v_unshared = {row["v_unshared"]:.2f}  '
          f'µm/ns   (x/y unshared {row["v_unshared_x"]:.1f}/{row["v_unshared_y"]:.1f}, '
          f'n_planes={len(rows):,})')
    return row


def main():
    seed = cm.load_alignment(ALIGN_SEED)
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
    bounds = active_bounds(det)
    (xmn, xmx), (ymn, ymx) = bounds
    print(f'active area: x [{xmn:.0f}, {xmx:.0f}]  y [{ymn:.0f}, {ymx:.0f}] mm   '
          f'cuts: res<{RES_CUT_MM:g}mm, edge>{EDGE_FID_MM:g}mm')

    run_dir = os.path.join(CFG.BASE_PATH, CFG.RUN)
    points = []
    for name in sorted(os.listdir(run_dir)):
        m = re.match(r'drift_scan_resist_(\d+)V_drift_(\d+)V$', name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            points.append((int(m.group(2)), name))
    m = re.search(r'drift_(\d+)V', CFG.SUB_RUN)
    if m:
        points.append((int(m.group(1)), CFG.SUB_RUN))
    points.sort()

    rows = []
    for hv, subrun in points:
        print(f'=== drift {hv} V — {subrun} ===')
        r = measure_point(subrun, seed, det, bounds)
        if r is not None:
            r['drift_hv'] = hv
            r['E_Vcm'] = hv / GAP_CM
            r['is_long_run'] = (subrun == CFG.SUB_RUN)
            rows.append(r)

    df = pd.DataFrame(rows).sort_values('drift_hv')
    df.to_csv(os.path.join(OUT, 'slope_reference_vdrift_scan.csv'), index=False)
    print('\n' + df[['drift_hv', 'E_Vcm', 'n_planes', 'v_raw', 'v_unshared',
                     'v_unshared_err']].to_string(index=False))
    print(f'\nWritten: {OUT}/slope_reference_vdrift_scan.csv')


if __name__ == '__main__':
    main()
