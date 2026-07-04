#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_det2_validation.py

OUT-OF-SAMPLE validation of the det3 chain on det2 (mx17_2, FEU 6/8, top
slot, 6-22 overnight run, resist ~525 V / drift 1000 V).  Same gas line and
same detector design as det3, so the frozen-data predictions are:

  v_geom(1000 V)   ~ 34 um/ns      (Ar/iso 95/5 + 1% H2O (+N2), E = 333 V/cm)
  lambda_att       ~ 16-18 mm      (same O2 contamination)
  sharing c1 / c2  ~ 0.45-0.52 / 0.05-0.15, ~ +60 ns neighbour delay
  v after unsharing = v_geom       (time-fit/geometry convergence)

Chain (single run, everything from scratch on this detector):
  1. geometry estimator: cluster-extent slope [mm/unit-tan] / T_sat (hits);
  2. amplitude vs drift depth -> lambda_att (hits);
  3. sharing measurement from vertical tracks (decoded waveforms);
  4. unshare (alpha=0.5, measured c's) + CFD re-time + core-OLS ridge.

Usage: ../.venv/bin/python 29_det2_validation.py o22_long_det2 [--veto=50]
Output: <out>/alignment_tpc_veto50/bias_study/det2_validation.png + stdout
"""
import os
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
MIN_STRIPS = 4
MIN_STRIPS_AFTER = 3
RES_CUT_MM = 10.0
CHI2_CUT = 20.0
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
ALPHA = 0.5
SAT_DEG = 10.0
TAN_LO, TAN_HI, TAN_STEP = 0.06, 0.44, 0.04
INCL_DEG = 12.0
LAM_FIT_MM = (8.0, 22.0)

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
TS = np.arange(32) * SAMPLE_NS

DET3_REF = dict(v_geom=33.9, lam=(16, 18), c1=(0.449, 0.516), c2=(0.052, 0.151),
                v_unshared=33.5)


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


def main():
    cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    print(f'{len(results):,} cached events; alignment {best}')
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)
    ref, dur, ns_ = {}, {}, {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        if r.x_fit.n_strips < MIN_STRIPS or r.y_fit.n_strips < MIN_STRIPS:
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
        dur[r.event_id] = (r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns,
                           r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns)
        ns_[r.event_id] = (r.x_fit.n_strips, r.y_fit.n_strips)
    print(f'{len(ref):,} matched quality events')

    # ---------- 1. geometry estimator ----------
    print('\n== 1. geometry estimator ==')
    v_geoms = {}
    for pi, p in enumerate(('x', 'y')):
        at = np.abs(np.array([v[pi] for v in ref.values()]))
        ext = np.array([(ns_[e][pi] - 1) * PITCH_MM for e in ref])
        dd = np.array([dur[e][pi] for e in ref])
        bins = np.arange(TAN_LO, TAN_HI + TAN_STEP, TAN_STEP)
        ctr, med = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (at >= b0) & (at < b1)
            if m.sum() >= 40:
                ctr.append(0.5 * (b0 + b1))
                med.append(np.median(ext[m]))
        slope = np.polyfit(ctr, med, 1)[0]
        msat = np.degrees(np.arctan(at)) > SAT_DEG
        tsat = float(np.median(dd[msat]))
        v_geoms[p] = slope * 1000.0 / tsat
        print(f'  {p}: extent slope = {slope:5.1f} mm/unit-tan   '
              f'T_sat = {tsat:4.0f} ns   v_geom = {v_geoms[p]:5.2f} µm/ns   '
              f'(det3: 23.2 mm, 690 ns, 33.9)')

    # ---------- 2. amplitude vs depth -> lambda ----------
    print('\n== 2. amplitude vs drift depth ==')
    v_use = float(np.mean(list(v_geoms.values())))
    fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
                if f.endswith('.root') and '_datrun_' in f)
    hf = uproot.concatenate([f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
                            expressions=['eventId', 'feu', 'amplitude', 'sample'],
                            library='pd')
    hf = hf[hf['feu'].isin(CFG.MX17_FEUS)]
    hpe = hf.groupby('eventId')['eventId'].transform('size')
    incl = {e for e, v in ref.items()
            if max(abs(np.degrees(np.arctan(v[0]))),
                   abs(np.degrees(np.arctan(v[1])))) > INCL_DEG}
    hf = hf[(hpe <= VETO) & hf['eventId'].isin(incl)].copy()
    hf['t'] = hf['sample'] * SAMPLE_NS
    hf['t_rel'] = hf['t'] - hf.groupby(['eventId', 'feu'])['t'].transform('min')
    t_rel = hf['t_rel'].to_numpy()
    amp = hf['amplitude'].to_numpy()
    bins = np.arange(0.0, 1920.0 + SAMPLE_NS, SAMPLE_NS)
    tc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(len(tc), np.nan)
    for i in range(len(tc)):
        m = (t_rel >= bins[i]) & (t_rel < bins[i + 1])
        if m.sum() >= 30:
            med[i] = np.median(amp[m])
    z = tc * v_use / 1000.0
    mfit = (z > LAM_FIT_MM[0]) & (z < LAM_FIT_MM[1]) & np.isfinite(med) & (med > 0)
    pfit = np.polyfit(z[mfit], np.log(med[mfit]), 1)
    lam = -1.0 / pfit[0]
    print(f'  lambda_att = {lam:.1f} mm  (fit {LAM_FIT_MM[0]:g}-{LAM_FIT_MM[1]:g} mm, '
          f'v = {v_use:.1f})   (det3: 16-18 mm)')

    # ---------- 3+4. sharing + unsharing from decoded waveforms ----------
    print('\n== 3. sharing measurement + 4. unshare & re-time ==')
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(CFG.run_config_path, CFG.MAP_CSV_PATH)
    det = rc.get_detector(CFG.DET_NAME)
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

    share = {f: {'r1': [], 'r2': [], 'dt1': []} for f in CFG.MX17_FEUS}
    buffered = []
    for feu in CFG.MX17_FEUS:
        pi = 0 if plane_of_feu[feu] == 'x' else 1
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
        print(f'  FEU {feu}: {len(fs)} decoded files')
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
                    tanv = ref[eid][pi]
                    wfm = arr['amplitude'][j].reshape(32, 512).astype(np.float32) - ped
                    cms = np.median(wfm.reshape(32, 8, 64), axis=2)
                    wfm -= np.repeat(cms, 64, axis=1)
                    wfm = wfm.T
                    if abs(tanv) < 0.03:
                        for blk in blocks[feu]:
                            wb = wfm[blk]
                            amax = wb.max(axis=1)
                            k = int(np.argmax(amax))
                            if amax[k] < 500 or k < 2 or k > len(blk) - 3:
                                continue
                            a_0 = amax[k]
                            a_m, a_p = amax[k - 1], amax[k + 1]
                            if min(a_m, a_p) <= 0 or max(a_m, a_p) > 0.8 * a_0:
                                continue
                            share[feu]['r1'].extend([a_m / a_0, a_p / a_0])
                            share[feu]['r2'].extend([amax[k - 2] / a_0,
                                                     amax[k + 2] / a_0])
                            t0c = cfd_time(wb[k])
                            for kk in (k - 1, k + 1):
                                tcn = cfd_time(wb[kk])
                                if np.isfinite(tcn) and np.isfinite(t0c):
                                    share[feu]['dt1'].append(tcn - t0c)
                    buffered.append((feu, eid, tanv, wfm.astype(np.float16)))

    cshare = {}
    for feu in CFG.MX17_FEUS:
        r1 = np.array(share[feu]['r1'])
        r2 = np.array(share[feu]['r2'])
        dt1 = np.array(share[feu]['dt1'])
        c1, c2 = float(np.median(r1)), float(np.median(np.clip(r2, 0, None)))
        cshare[feu] = (c1, c2)
        p = plane_of_feu[feu]
        print(f'  {p} (FEU {feu}): c1 = {c1:.3f}  c2 = {c2:.3f}  '
              f'Δt = {np.median(dt1):+.0f} ns  (n={len(r1)//2:,})   '
              f'(det3: c1 0.45/0.52, c2 0.05/0.15, +69 ns)')

    print('\n  ridge v (CFD core OLS):')
    summary = {}
    for mode in ('before', 'after'):
        Sd = {f: ([], []) for f in CFG.MX17_FEUS}
        for feu, eid, tanv, wfm16 in buffered:
            wfm = wfm16.astype(np.float32)
            c1, c2 = cshare[feu]
            min_strips = MIN_STRIPS if mode == 'before' else MIN_STRIPS_AFTER
            bst = None
            for blk in blocks[feu]:
                wb = wfm[blk]
                if mode == 'after':
                    wb = unshare(wb, c1, c2)
                amax = wb.max(axis=1)
                hit = np.where(amax >= THR_HIT)[0]
                if len(hit) < min_strips:
                    continue
                brk = np.where(np.diff(hit) > 2)[0]
                for grp in np.split(hit, brk + 1):
                    if len(grp) < min_strips:
                        continue
                    if bst is None or len(grp) > len(bst[0]):
                        bst = (grp, blk, wb)
            if bst is None:
                continue
            grp, blk, wb = bst
            pos = pos_of[feu][blk[grp]]
            ampv = wb[grp].max(axis=1)
            tt = np.array([cfd_time(wb[g]) for g in grp])
            ok = np.isfinite(tt)
            mcore = (ampv >= CORE_FRAC * ampv.max()) & ok
            if mcore.sum() < 3 or np.ptp(pos[mcore]) == 0:
                continue
            sl = np.polyfit(pos[mcore], tt[mcore], 1)[0]
            if sl == 0:
                continue
            Sd[feu][0].append(1000.0 / sl)
            Sd[feu][1].append(tanv)
        for feu in CFG.MX17_FEUS:
            S, T = map(np.array, Sd[feu])
            vs = []
            for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
                m = (T > lo) & (T < hi) & np.isfinite(S)
                v, b, se, n = robust_line(T[m], S[m])
                if np.isfinite(v) and n > 100:
                    vs.append(v)
            p = plane_of_feu[feu]
            summary[(mode, p)] = np.mean(vs) if vs else np.nan
            print(f'    {mode:6s} {p}: v = {summary[(mode, p)]:6.2f} µm/ns')

    # ---------- summary figure ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    items = [
        ('v_geom x', v_geoms['x'], 33.9), ('v_geom y', v_geoms['y'], 33.9),
        ('v before x', summary[('before', 'x')], None),
        ('v before y', summary[('before', 'y')], None),
        ('v unshared x', summary[('after', 'x')], 33.5),
        ('v unshared y', summary[('after', 'y')], 33.5),
    ]
    yp = np.arange(len(items))
    ax.barh(yp, [i[1] for i in items],
            color=['crimson', 'crimson', 'gray', 'gray', 'steelblue', 'steelblue'],
            alpha=0.85)
    for y, (lab, v, ref3) in zip(yp, items):
        if ref3:
            ax.plot([ref3], [y], '*', color='k', ms=14)
    ax.axvline(33.9, color='crimson', lw=1.5, ls='--',
               label='det3 v_geom = 33.9 (prediction)')
    ax.set_yticks(yp); ax.set_yticklabels([i[0] for i in items])
    ax.set_xlabel('v [µm/ns]'); ax.set_xlim(20, 40)
    ax.grid(alpha=0.3, axis='x'); ax.legend()
    ax.set_title(f'{CFG.DET_NAME} ({CFG.RUN}): out-of-sample validation — '
                 f'λ_att = {lam:.1f} mm (det3: 16–18)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'det2_validation.png'), dpi=160)
    print(f'\nPlot: {OUT}/det2_validation.png')


if __name__ == '__main__':
    main()
