#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_microtpc_metrics.py

Micro-TPC MODE performance metrics — the hit-mode-style scoreboard for the
drift/segment measurement, built on the final analysis chain (waveform
unsharing + additive tan calibration, scripts 26-28).

Hit mode answers "did the detector see the muon and where"; micro-TPC mode
answers "did it also measure the track DIRECTION through the drift gap".
The metrics here are the direct analogues of efficiency / resolution /
within-5-mm:

  1. SEGMENT EFFICIENCY LADDER (denominator = good single M3 rays traversing
     the fiducial active area, spark-vetoed events removed):
        tier 0  hit-mode: valid X+Y cluster within 10 mm of the ray
        tier 1  micro-TPC segment: unshared cluster fit in both planes
        tier 2  direction agreement: both planes within |dtheta| < 5 deg
  2. ANGULAR RESPONSE: theta_det vs theta_ref correlation, bias and 68 %
     resolution vs theta_ref (per plane and combined), after unsharing +
     calibration (constants remeasured here, not hardcoded).
  3. ANGULAR AGREEMENT: fraction of segments within 3 / 5 deg of the ray,
     vs theta_ref (the angular "within-5-mm" analogue) + Pearson r.
  4. 3D DIRECTION: opening angle between the (x,y)-combined micro-TPC
     direction and the M3 ray direction; median / 68 % quantiles.

The per-event unshared segment table is cached to CSV so downstream studies
(e.g. 32_edge_fringe_field.py) reuse it without re-streaming waveforms.

Usage: ../.venv/bin/python 31_microtpc_metrics.py sat_det3 [--veto=50]
       [--rebuild]   force waveform re-pass (ignore segment CSV cache)
Output: <alignment_tpc_vetoN>/microtpc_metrics/
        microtpc_segments.csv (cache), microtpc_metrics.png,
        microtpc_metrics_summary.csv + stdout table
"""
import os
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
REBUILD = '--rebuild' in sys.argv
SAMPLE_NS = 60.0
MIN_STRIPS_AFTER = 3
RES_CUT_MM = 10.0
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking)
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
ALPHA = 0.5
PLATEAU_TAN = (0.12, 0.55)     # |tan| window for the calibration constant
FID_MARGIN_MM = 5.0            # fiducial inset from the active-area edges
AGREE_DEG = (3.0, 5.0)         # angular agreement windows
MAX_REF_TAN = 0.7              # analysis angle range

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'microtpc_metrics')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
SEG_CSV = os.path.join(OUT, 'microtpc_segments.csv')
# measured in 26_unsharing_analysis.py (design property; det2 validated within a few %)
CSHARE = {7: (0.449, 0.052), 8: (0.516, 0.151), 6: (0.449, 0.052),
          3: (0.449, 0.052), 4: (0.516, 0.151)}


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


def unshare(wb, c1, c2, alpha=ALPHA):
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
        if s >= 1:
            rhs -= (1 - alpha) * c1 * nsum(X[:, s - 1], 1)
        if s >= 2:
            rhs -= (1 - alpha) * c2 * nsum(X[:, s - 2], 2)
        X[:, s] = solve_banded((2, 2), ab, rhs)
    return X


def build_segment_table(ref, det):
    """Waveform pass: unshared CFD-core segment fit per matched event & plane.
    Returns a DataFrame(eid, plane, S_um_ns, n_strips, amp_sum, pos_lead_mm)."""
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    blocks, pos_of = {}, {}
    for feu in CFG.MX17_FEUS:
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))
                        [0 if plane_of_feu[feu] == 'x' else 1]
                        for ch in range(512)], dtype=float)
        pos_of[feu] = pos
        ok = np.where(np.isfinite(pos))[0]
        o = ok[np.argsort(pos[ok])]
        brk = np.where(np.diff(pos[o]) > 1.5 * PITCH_MM)[0]
        blocks[feu] = np.split(o, brk + 1)

    rows = []
    for feu in CFG.MX17_FEUS:
        c1, c2 = CSHARE[feu]
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
        if not fs:
            print(f'  [WARN] no decoded_root for FEU {feu} -- '
                  f'segment table will be hits-only for this plane')
            continue
        for fn in fs:
            t = uproot.open(fn)['nt']
            eids_all = t.arrays(['eventId'], library='np')['eventId']
            a0 = t.arrays(['amplitude'], entry_stop=N_PED_EVENTS, library='np')['amplitude']
            ped = np.median(np.stack([a.reshape(32, 512) for a in a0]), axis=(0, 1))
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
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    wfm = wfm.T
                    bst = None
                    for blk in blocks[feu]:
                        wb = unshare(wfm[blk], c1, c2)
                        amax = wb.max(axis=1)
                        hit = np.where(amax >= THR_HIT)[0]
                        if len(hit) < MIN_STRIPS_AFTER:
                            continue
                        brk = np.where(np.diff(hit) > 2)[0]
                        for grp in np.split(hit, brk + 1):
                            if len(grp) < MIN_STRIPS_AFTER:
                                continue
                            if bst is None or len(grp) > len(bst[0]):
                                bst = (grp, blk, wb)
                    if bst is None:
                        continue
                    grp, blk, wb = bst
                    pos = pos_of[feu][blk[grp]]
                    amp = wb[grp].max(axis=1)
                    tt = np.array([cfd_time(wb[g]) for g in grp])
                    ok2 = np.isfinite(tt)
                    mcore = (amp >= CORE_FRAC * amp.max()) & ok2
                    if mcore.sum() < 3 or np.ptp(pos[mcore]) == 0:
                        continue
                    sl = np.polyfit(pos[mcore], tt[mcore], 1)[0]
                    if sl == 0:
                        continue
                    ilead = int(np.nanargmin(np.where(ok2, tt, np.inf)))
                    rows.append(dict(eid=eid, plane=plane_of_feu[feu],
                                     S_um_ns=1000.0 / sl,
                                     n_strips=int(len(grp)),
                                     amp_sum=float(amp.sum()),
                                     pos_lead_mm=float(pos[ilead]),
                                     t_span_ns=float(np.nanmax(tt[ok2]) - np.nanmin(tt[ok2]))
                                     if ok2.sum() >= 2 else np.nan))
            print(f'  {os.path.basename(fn)} done ({len(rows):,} segments)')
    return pd.DataFrame(rows)


def vetoed_event_ids():
    """Re-derive the spark-vetoed eids (events with > VETO hit-rows in the
    detector FEUs) from combined_hits -- the veto'd cache silently drops them
    and they must leave the efficiency denominator too. Returns (vetoed set,
    (eid_min, eid_max)): zero-hit events are absent from the hits tree but are
    genuine inefficiency, so the denominator uses the covered eid RANGE."""
    files = [f for f in os.listdir(CFG.combined_hits_dir)
             if f.endswith('.root') and '_datrun_' in f]
    counts = {}
    for f in files:
        arr = uproot.open(os.path.join(CFG.combined_hits_dir, f))['hits'].arrays(
            ['eventId', 'feu'], library='np')
        m = np.isin(arr['feu'], CFG.MX17_FEUS)
        eid, cnt = np.unique(arr['eventId'][m], return_counts=True)
        for e, c in zip(eid.tolist(), cnt.tolist()):
            counts[e] = counts.get(e, 0) + c
    vetoed = {e for e, c in counts.items() if c > VETO}
    return vetoed, (min(counts), max(counts))


def decoded_coverage_ranges():
    """eid ranges covered by decoded waveform files for BOTH planes (the
    Saturday run's file 003 exists only for FEU 7 -- events outside dual-FEU
    coverage cannot have segments and must leave the ladder denominator)."""
    per_feu = {}
    for feu in CFG.MX17_FEUS:
        rngs = []
        for fn in sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root'))):
            e = uproot.open(fn)['nt'].arrays(['eventId'], library='np')['eventId']
            if len(e):
                rngs.append((int(e.min()), int(e.max())))
        per_feu[feu] = rngs
    def covered(e):
        return all(any(lo <= e <= hi for lo, hi in per_feu[f]) for f in CFG.MX17_FEUS)
    n_files = {f: len(per_feu[f]) for f in per_feu}
    print(f'decoded coverage: files per FEU = {n_files}')
    return covered


def active_bounds(det):
    """Active-area bounding box from the strip map itself (det.map_hit);
    cm.get_active_det_bounds returns (0,0) for this run's config."""
    lims = {}
    for feu, axis in ((CFG.MX17_FEU_X, 0), (CFG.MX17_FEU_Y, 1)):
        pos = np.array([(det.map_hit(feu, ch) or (np.nan, np.nan))[axis]
                        for ch in range(512)], dtype=float)
        pos = pos[np.isfinite(pos)]
        lims[axis] = (float(pos.min()), float(pos.max()))
    return lims[0], lims[1]


def profile(x, y, edges, min_n=60, stat='eff'):
    ctr, val, err = [], [], []
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (x >= b0) & (x < b1)
        n = m.sum()
        if n < min_n:
            continue
        ctr.append(0.5 * (b0 + b1))
        if stat == 'eff':
            k = float(np.sum(y[m]))
            val.append(k / n)
            err.append(np.sqrt(max(k, 0.25) * (1 - k / n)) / n)
        elif stat == 'med':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(q[1]); err.append(0.5 * (q[2] - q[0]) / np.sqrt(n))
        elif stat == 's68':
            q = np.percentile(y[m], [16, 50, 84])
            val.append(0.5 * (q[2] - q[0])); err.append(0.5 * (q[2] - q[0]) / np.sqrt(2 * n))
    return np.array(ctr), np.array(val), np.array(err)


def main():
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)

    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)

    # ---- matched events (for the angular metrics) ----
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    print(f'{len(ref):,} matched events (hit-mode, r<{RES_CUT_MM:g} mm)')

    # ---- segment table (cached waveform pass) ----
    if os.path.exists(SEG_CSV) and not REBUILD:
        seg = pd.read_csv(SEG_CSV)
        print(f'Loaded cached segment table: {len(seg):,} rows ({SEG_CSV})')
    else:
        seg = build_segment_table(ref, det)
        seg.to_csv(SEG_CSV, index=False)
        print(f'Segment table cached: {len(seg):,} rows -> {SEG_CSV}')

    tan_ref = {p: {} for p in 'xy'}
    for eid, (tx, ty) in ref.items():
        tan_ref['x'][eid] = tx
        tan_ref['y'][eid] = ty
    seg['tan_ref'] = [tan_ref[p].get(e, np.nan) for p, e in zip(seg['plane'], seg['eid'])]
    seg = seg[np.isfinite(seg['tan_ref']) & (np.abs(seg['tan_ref']) < MAX_REF_TAN)].copy()

    # ---- per-plane velocity + additive tan calibration (remeasured on v2) ----
    consts = {}
    for p in 'xy':
        d = seg[seg['plane'] == p]
        S, T = d['S_um_ns'].to_numpy(), d['tan_ref'].to_numpy()
        vs = []
        for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
            m = (T > lo) & (T < hi) & np.isfinite(S)
            v, b, se, n = robust_line(T[m], S[m])
            if np.isfinite(v) and n > 100:
                vs.append(v)
        v = float(np.mean(vs))
        tan_det = S / v
        m_pl = (np.abs(T) > PLATEAU_TAN[0]) & (np.abs(T) < PLATEAU_TAN[1]) & np.isfinite(tan_det)
        b_res = float(np.median(np.abs(tan_det[m_pl]) - np.abs(T[m_pl])))
        consts[p] = (v, b_res)
        print(f'plane {p}: ridge v = {v:.2f} um/ns, calibration b = {b_res:+.4f} '
              f'({np.degrees(np.arctan(b_res)):+.2f} deg at 0)')
    seg['tan_det'] = [s / consts[p][0] for s, p in zip(seg['S_um_ns'], seg['plane'])]
    seg['tan_cal'] = seg['tan_det'] - np.sign(seg['tan_det']) * \
        np.array([consts[p][1] for p in seg['plane']])
    seg['th_ref'] = np.degrees(np.arctan(seg['tan_ref']))
    seg['th_det'] = np.degrees(np.arctan(seg['tan_cal']))
    seg['dth'] = seg['th_det'] - seg['th_ref']

    # ================= 1. EFFICIENCY LADDER =================
    (xmn, xmx), (ymn, ymx) = active_bounds(det)
    print(f'active area: x [{xmn:.0f}, {xmx:.0f}]  y [{ymn:.0f}, {ymx:.0f}] mm')
    vetoed, (eid_lo, eid_hi) = vetoed_event_ids()
    covered = decoded_coverage_ranges()
    print(f'{len(vetoed):,} spark-vetoed events removed from the denominator; '
          f'detector eid range [{eid_lo}, {eid_hi}]')

    # fiducial ray sample in RAW detector coordinates (inverse alignment)
    x_ref, _, evx = rays.get_xy_positions(best.z_x)
    _, y_ref, evy = rays.get_xy_positions(best.z_y)
    x_by, y_by = dict(zip(evx, best.ref_x_sign * np.array(x_ref))), dict(zip(evy, y_ref))
    cos_t, sin_t = np.cos(th), np.sin(th)
    cxc, cyc = best.centre_x, best.centre_y
    denom_eids = []
    for e in set(evx) & set(evy):
        u = x_by[e] - cxc - best.x_offset
        v = y_by[e] - cyc - best.y_offset
        mx = cxc + cos_t * u + sin_t * v
        my = cyc - sin_t * u + cos_t * v
        if (xmn + FID_MARGIN_MM < mx < xmx - FID_MARGIN_MM
                and ymn + FID_MARGIN_MM < my < ymx - FID_MARGIN_MM
                and eid_lo <= e <= eid_hi and e not in vetoed and covered(e)):
            denom_eids.append(e)
    denom_eids = np.array(sorted(denom_eids))
    print(f'denominator: {len(denom_eids):,} good single rays in fiducial '
          f'(margin {FID_MARGIN_MM:g} mm, spark-vetoed removed, '
          f'dual-FEU decoded coverage required)')

    hit_ok = {r.event_id for r in results
              if r.has_both and np.isfinite(r.radial_residual_mm)
              and r.radial_residual_mm < RES_CUT_MM}
    seg_x = set(seg[seg['plane'] == 'x']['eid'])
    seg_y = set(seg[seg['plane'] == 'y']['eid'])
    agree = seg[np.abs(seg['dth']) < AGREE_DEG[1]]
    agr_x, agr_y = set(agree[agree['plane'] == 'x']['eid']), set(agree[agree['plane'] == 'y']['eid'])

    in_hit = np.array([e in hit_ok for e in denom_eids])
    in_seg = np.array([e in seg_x and e in seg_y for e in denom_eids])
    in_agr = np.array([e in agr_x and e in agr_y for e in denom_eids])
    tiers = [('hit-mode (X+Y cluster, r<10mm)', in_hit),
             ('micro-TPC segment (X+Y unshared fit)', in_seg),
             (f'direction agrees (both |dth|<{AGREE_DEG[1]:g} deg)', in_agr)]
    print('\n== efficiency ladder ==')
    ladder = {}
    for name, m in tiers:
        eff = m.mean()
        err = np.sqrt(eff * (1 - eff) / len(m))
        ladder[name] = (eff, err)
        print(f'  {name:45s} {100*eff:5.1f} +/- {100*err:.1f} %')

    # conditional: segment given hit
    m_cond = in_seg[in_hit]
    print(f'  {"segment | hit-mode":45s} {100*m_cond.mean():5.1f} %')

    # efficiency vs reference angle (space angle of the ray)
    tanx_all, tany_all, ev_ang = get_xy_angles(rays.ray_data)
    tan_by_e = {e: (tx, ty) for tx, ty, e in
                zip(np.tan(best.ref_x_sign * np.array(tanx_all)),
                    np.tan(tany_all), ev_ang)}
    th_space = np.array([np.degrees(np.arctan(np.hypot(*tan_by_e[e])))
                         if e in tan_by_e else np.nan for e in denom_eids])
    mfin = np.isfinite(th_space)
    edges_th = np.arange(0, 40, 2.5)
    eff_curves = {}
    for name, m in tiers:
        eff_curves[name] = profile(th_space[mfin], m[mfin].astype(float), edges_th, min_n=100)

    # ================= 2-3. ANGULAR METRICS =================
    edges = np.arange(-35, 35.1, 2.5)
    prof_bias, prof_s68, prof_agree = {}, {}, {}
    for p in 'xy':
        d = seg[seg['plane'] == p]
        prof_bias[p] = profile(d['th_ref'].to_numpy(), d['dth'].to_numpy(), edges, 80, 'med')
        prof_s68[p] = profile(d['th_ref'].to_numpy(), d['dth'].to_numpy(), edges, 80, 's68')
        prof_agree[p] = profile(d['th_ref'].to_numpy(),
                                (np.abs(d['dth']) < AGREE_DEG[1]).to_numpy().astype(float),
                                edges, 80)
    dth_all = seg['dth'].to_numpy()
    thr_all = seg['th_ref'].to_numpy()
    pl_mask = np.abs(thr_all) > 8
    plateau_bias = float(np.median(dth_all[pl_mask]))
    q = np.percentile(dth_all[pl_mask], [16, 84])
    plateau_s68 = float(0.5 * (q[1] - q[0]))
    frac3 = float(np.mean(np.abs(dth_all) < AGREE_DEG[0]))
    frac5 = float(np.mean(np.abs(dth_all) < AGREE_DEG[1]))
    # Pearson on tan values; clip runaway slopes (near-vertical fits give
    # |S| -> huge; a handful of those destroy the moment-based coefficient)
    mpl = (np.abs(seg['tan_ref']) < MAX_REF_TAN) & (np.abs(seg['tan_cal']) < 1.5) \
        & np.isfinite(seg['tan_cal'])
    pear = float(np.corrcoef(seg['tan_ref'][mpl], seg['tan_cal'][mpl])[0, 1])
    mplat = mpl & (np.abs(seg['th_ref']) > 8)
    pear_pl = float(np.corrcoef(seg['tan_ref'][mplat], seg['tan_cal'][mplat])[0, 1])
    print(f'\n== angular metrics (both planes) ==')
    print(f'  plateau (|th|>8deg) bias = {plateau_bias:+.2f} deg, s68 = {plateau_s68:.2f} deg')
    print(f'  fraction |dth| < 3 / 5 deg = {100*frac3:.1f} / {100*frac5:.1f} %')
    print(f'  Pearson r(tan_det, tan_ref) = {pear:.4f} (all), {pear_pl:.4f} (plateau)')

    # ================= 4. 3D OPENING ANGLE =================
    sx = seg[seg['plane'] == 'x'].set_index('eid')
    sy = seg[seg['plane'] == 'y'].set_index('eid')
    common = sx.index.intersection(sy.index)
    txd, tyd = sx.loc[common, 'tan_cal'].to_numpy(), sy.loc[common, 'tan_cal'].to_numpy()
    txr, tyr = sx.loc[common, 'tan_ref'].to_numpy(), sy.loc[common, 'tan_ref'].to_numpy()

    def unit(tx, ty):
        n = np.sqrt(tx**2 + ty**2 + 1.0)
        return np.stack([tx / n, ty / n, 1.0 / n])
    ud, ur = unit(txd, tyd), unit(txr, tyr)
    psi = np.degrees(np.arccos(np.clip(np.sum(ud * ur, axis=0), -1, 1)))
    q_psi = np.percentile(psi, [50, 68])
    psi5 = float(np.mean(psi < 5.0))
    psi10 = float(np.mean(psi < 10.0))
    print(f'\n== 3D direction (n={len(psi):,} dual-plane segments) ==')
    print(f'  opening angle psi: median = {q_psi[0]:.2f} deg, 68% = {q_psi[1]:.2f} deg')
    print(f'  fraction psi < 5 / 10 deg = {100*psi5:.1f} / {100*psi10:.1f} %')

    # ================= summary CSV =================
    summ = dict(run=CFG.RUN, subrun=CFG.SUB_RUN, det=CFG.DET_NAME,
                n_denom=len(denom_eids), n_matched=len(ref),
                n_segments_x=len(seg_x), n_segments_y=len(seg_y),
                v_x=consts['x'][0], v_y=consts['y'][0],
                b_x=consts['x'][1], b_y=consts['y'][1],
                eff_hit=ladder[tiers[0][0]][0], eff_seg=ladder[tiers[1][0]][0],
                eff_agree=ladder[tiers[2][0]][0],
                eff_seg_given_hit=float(m_cond.mean()),
                plateau_bias_deg=plateau_bias, plateau_s68_deg=plateau_s68,
                frac_dth_lt3=frac3, frac_dth_lt5=frac5, pearson_tan=pear,
                pearson_tan_plateau=pear_pl,
                psi_median_deg=q_psi[0], psi_68_deg=q_psi[1],
                frac_psi_lt5=psi5, frac_psi_lt10=psi10)
    pd.DataFrame([summ]).to_csv(os.path.join(OUT, 'microtpc_metrics_summary.csv'),
                                index=False)

    # ================= FIGURE =================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))

    ax = axes[0, 0]
    ax.hist2d(seg['th_ref'], seg['th_det'], bins=[70, 70],
              range=[[-35, 35], [-35, 35]], norm=LogNorm(), cmap='viridis')
    ax.plot([-35, 35], [-35, 35], 'r--', lw=1)
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('θ_TPC (calibrated) [deg]')
    ax.set_title(f'angle correlation (both planes, r={pear:.3f})')

    ax = axes[0, 1]
    for p, c in [('x', 'tab:blue'), ('y', 'tab:orange')]:
        ctr, med, err = prof_bias[p]
        ax.errorbar(ctr, med, yerr=err, fmt='o-', ms=4, color=c, label=f'{p} plane')
    ax.axhline(0, color='k', lw=1)
    ax.axhspan(-plateau_s68, plateau_s68, color='gray', alpha=0.12,
               label=f'±σ68 = {plateau_s68:.1f}°')
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('median θ_TPC − θ_ref [deg]')
    ax.set_title(f'angular bias (plateau {plateau_bias:+.2f}°)')
    ax.set_ylim(-5, 5); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 2]
    for p, c in [('x', 'tab:blue'), ('y', 'tab:orange')]:
        ctr, s68, err = prof_s68[p]
        ax.errorbar(ctr, s68, yerr=err, fmt='s-', ms=4, color=c, label=f'{p} plane')
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('σ68(Δθ) [deg]')
    ax.set_title(f'angular resolution (plateau {plateau_s68:.2f}°)')
    ax.set_ylim(0, 8); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 0]
    for (name, _), c in zip(tiers, ('tab:green', 'tab:blue', 'crimson')):
        ctr, eff, err = eff_curves[name]
        ax.errorbar(ctr, 100 * eff, yerr=100 * err, fmt='o-', ms=4,
                    color=c, label=name.split(' (')[0])
    ax.set_xlabel('ray space angle θ [deg]'); ax.set_ylabel('efficiency [%]')
    ax.set_title('efficiency ladder vs track angle'); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')

    ax = axes[1, 1]
    ax.hist(psi, bins=np.arange(0, 15, 0.25), histtype='stepfilled',
            color='tab:blue', alpha=0.65)
    for qq, lab, ls in [(q_psi[0], f'median {q_psi[0]:.1f}°', '-'),
                        (q_psi[1], f'68% {q_psi[1]:.1f}°', '--')]:
        ax.axvline(qq, color='crimson', ls=ls, label=lab)
    ax.set_xlabel('3D opening angle ψ(TPC, ray) [deg]'); ax.set_ylabel('events')
    ax.set_title('3D direction agreement'); ax.legend(fontsize=9)

    ax = axes[1, 2]
    ax.axis('off')
    txt = [
        f"MICRO-TPC SCOREBOARD — {CFG.DET_NAME}",
        f"{CFG.RUN}",
        "",
        f"good rays in fiducial       {len(denom_eids):,}",
        f"hit-mode efficiency         {100*ladder[tiers[0][0]][0]:.1f} %",
        f"micro-TPC segment eff       {100*ladder[tiers[1][0]][0]:.1f} %",
        f"  (given hit-mode)          {100*m_cond.mean():.1f} %",
        f"direction-agreement eff     {100*ladder[tiers[2][0]][0]:.1f} %",
        "",
        f"plateau angular bias        {plateau_bias:+.2f} deg",
        f"plateau angular s68         {plateau_s68:.2f} deg",
        f"|dth|<3deg / <5deg          {100*frac3:.0f} % / {100*frac5:.0f} %",
        f"Pearson r all / plateau     {pear:.3f} / {pear_pl:.3f}",
        f"psi 3D median / 68%         {q_psi[0]:.1f} / {q_psi[1]:.1f} deg",
        f"psi<5deg / <10deg           {100*psi5:.0f} % / {100*psi10:.0f} %",
        "",
        f"v_drift x/y  {consts['x'][0]:.1f} / {consts['y'][0]:.1f} um/ns",
        f"calib b x/y  {consts['x'][1]:+.3f} / {consts['y'][1]:+.3f}",
    ]
    ax.text(0.02, 0.98, '\n'.join(txt), transform=ax.transAxes, va='top',
            fontsize=11.5, family='monospace')

    fig.suptitle(f'{CFG.RUN} — micro-TPC mode performance metrics '
                 f'(unshared + calibrated, M3 v2 rays)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'microtpc_metrics.png'), dpi=160)

    # ================= EXPLAINER FIGURE =================
    # What "direction agrees" and psi mean, and WHY they look low: both are
    # dominated by near-vertical tracks, where (a) the unshared cluster
    # shrinks below the 3-strip segment minimum (segment-efficiency loss)
    # and (b) a fixed ~2 deg per-plane angular error is a large RELATIVE
    # direction error, so psi and |dth| cuts bite hardest at small theta.
    th_sp_seg = np.degrees(np.arctan(np.hypot(txr, tyr)))   # ray space angle, dual-plane segs
    fig2, ax2 = plt.subplots(2, 3, figsize=(18, 10))

    ax = ax2[0, 0]   # schematic of the two metrics
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # drift gap sketch: ray and TPC segment through the gap
    ax.plot([2, 4.0], [2, 8], '-', color='tab:green', lw=3)
    ax.plot([2, 4.8], [2, 8], '--', color='tab:blue', lw=3)
    ax.annotate('M3 ray\n(reference direction)', xy=(3.1, 5.5), xytext=(0.7, 6.8),
                color='tab:green', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='tab:green'))
    ax.annotate('micro-TPC segment\n(measured direction)', xy=(4.0, 6.6),
                xytext=(5.6, 7.6), color='tab:blue', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='tab:blue'))
    ax.annotate('', xy=(4.55, 7.5), xytext=(3.85, 7.5),
                arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.6))
    ax.text(4.2, 7.75, 'ψ', color='crimson', fontsize=14, ha='center')
    ax.plot([1, 9], [2, 2], '-', color='k', lw=1)
    ax.text(9, 1.6, 'mesh', fontsize=9, ha='right')
    txt = ('METRIC DEFINITIONS\n\n'
           '|Δθ| < 5°  (per plane):\n'
           '  the 2D projected angle measured by ONE\n'
           '  strip plane is within 5° of the ray\'s\n'
           '  projection.  "Direction agrees" tier =\n'
           '  BOTH planes pass simultaneously.\n\n'
           'ψ (3D opening angle):\n'
           '  combine tanθx + tanθy into one 3D unit\n'
           '  vector; ψ = angle between it and the\n'
           '  ray direction.  One number for "how well\n'
           '  does the TPC point", both planes at once.')
    ax.text(0.3, 0.02, txt, transform=ax.transAxes, fontsize=9.5,
            family='monospace', va='bottom')

    ax = ax2[0, 1]   # psi vs space angle
    hb = ax.hist2d(th_sp_seg, np.clip(psi, 0, 20), bins=[36, 40],
                   range=[[0, 30], [0, 20]], norm=LogNorm(), cmap='viridis')
    ctr_p, med_p, err_p = profile(th_sp_seg, psi, np.arange(0, 30, 2.5), 60, 'med')
    ax.errorbar(ctr_p, med_p, yerr=err_p, fmt='o-', color='crimson', ms=5,
                label='median ψ')
    ax.axhline(5, color='w', ls='--', lw=1)
    ax.set_xlabel('ray space angle θ [deg]'); ax.set_ylabel('ψ [deg]')
    ax.set_title('ψ vs track angle — small-θ tracks dominate the tail')
    ax.legend(fontsize=9)

    ax = ax2[0, 2]   # agreement fraction vs angle, conditional on segment
    both_ok = (np.abs(np.degrees(np.arctan(txd)) - np.degrees(np.arctan(txr))) < AGREE_DEG[1]) \
        & (np.abs(np.degrees(np.arctan(tyd)) - np.degrees(np.arctan(tyr))) < AGREE_DEG[1])
    for arr, lab, c in [(both_ok.astype(float), f'both planes |Δθ|<{AGREE_DEG[1]:g}°', 'crimson'),
                        ((psi < 5).astype(float), 'ψ < 5°', 'tab:blue'),
                        ((psi < 10).astype(float), 'ψ < 10°', 'tab:cyan')]:
        c1_, v1, e1 = profile(th_sp_seg, arr, np.arange(0, 30, 2.5), 60)
        ax.errorbar(c1_, 100 * v1, yerr=100 * e1, fmt='o-', ms=4, color=c, label=lab)
    ax.set_xlabel('ray space angle θ [deg]')
    ax.set_ylabel('fraction of dual-plane segments [%]')
    ax.set_title('agreement GIVEN a segment, vs angle')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')

    ax = ax2[1, 0]   # ladder waterfall
    vals = [100.0, 100 * ladder[tiers[0][0]][0], 100 * ladder[tiers[1][0]][0],
            100 * ladder[tiers[2][0]][0]]
    labs = ['good rays\nin fiducial', 'hit-mode\n(X+Y, r<10mm)',
            'micro-TPC\nsegment', f'direction\nagrees (<{AGREE_DEG[1]:g}°)']
    cols = ['gray', 'tab:green', 'tab:blue', 'crimson']
    bars = ax.bar(labs, vals, color=cols, alpha=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center',
                fontsize=11)
    ax.annotate('mostly near-vertical tracks:\nunshared cluster < 3 strips\n(no lever arm in time)',
                xy=(2, vals[2] + 3), xytext=(1.15, 72), fontsize=9,
                arrowprops=dict(arrowstyle='->'))
    ax.annotate('mostly the |θ|<5°\ntransition region',
                xy=(3, vals[3] + 3), xytext=(3.0, 62), fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->'))
    ax.set_ylabel('efficiency [%]'); ax.set_ylim(0, 112)
    ax.set_title('the ladder, factorised')

    ax = ax2[1, 1]   # same ladder, restricted to inclined tracks
    m_i = mfin & (th_space > 10)
    rows_tbl = []
    for name, m in tiers:
        rows_tbl.append((name.split(' (')[0], 100 * m[m_i].mean()))
    seg_i = np.array([e in seg_x and e in seg_y for e in denom_eids])[m_i]
    ax.axis('off')
    t2 = ['SAME LADDER, INCLINED TRACKS ONLY (θ > 10°):', '']
    for nm, v in rows_tbl:
        t2.append(f'  {nm:28s} {v:5.1f} %')
    n_seg_i = seg_i.sum()
    t2 += ['',
           f'  (n = {m_i.sum():,} rays, {n_seg_i:,} with segments)',
           '',
           'Read: for tracks with real inclination the',
           'micro-TPC works most of the time; the global',
           f'{100*ladder[tiers[1][0]][0]:.0f}% / {100*ladder[tiers[2][0]][0]:.0f}% '
           'numbers are dominated by the',
           'cosmic angular distribution peaking near 0°,',
           'where a drift TPC fundamentally has no',
           'time-position lever arm (2-3 direct strips).',
           '',
           'ψ median 2.4° BUT at θ_ref=2° even a perfect',
           '2°-resolution detector gives ψ ~ 2-3°: the',
           'RELATIVE direction error at small θ is O(1).']
    ax.text(0.02, 0.98, '\n'.join(t2), transform=ax.transAxes, va='top',
            fontsize=10.5, family='monospace')

    ax = ax2[1, 2]   # per-plane dth distribution, inclined vs vertical
    dth_x = np.degrees(np.arctan(txd)) - np.degrees(np.arctan(txr))
    for m_band, lab, c in [(th_sp_seg < 5, 'θ_ray < 5°', 'tab:orange'),
                           (th_sp_seg > 10, 'θ_ray > 10°', 'tab:blue')]:
        ax.hist(np.clip(dth_x[m_band], -14.8, 14.8), bins=np.arange(-15, 15.2, 0.5),
                histtype='step', lw=2, density=True, color=c, label=lab)
    ax.axvline(-AGREE_DEG[1], color='k', ls=':'); ax.axvline(AGREE_DEG[1], color='k', ls=':')
    ax.set_xlabel('Δθ_x (TPC − ray) [deg]'); ax.set_ylabel('normalised')
    ax.set_title('x-plane Δθ: vertical vs inclined tracks')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig2.suptitle(f'{CFG.RUN} — direction metrics, explained', fontsize=13.5)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(os.path.join(OUT, 'microtpc_direction_explainer.png'), dpi=155)
    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
