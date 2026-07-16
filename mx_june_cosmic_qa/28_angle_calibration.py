#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_angle_calibration.py

Final step of the micro-TPC angle chain: the ADDITIVE TAN-SPACE CALIBRATION
on top of waveform unsharing.

After unsharing (26/27_*.py) a residual sign-following offset of about
+-2 deg remains -- the incompletely-nulled spreading floor (genuine
transverse-diffusion charge, which must NOT be deconvolved away as if it
were electronic sharing).  Because it is additive and angle-independent in
tan space (the same structure as the original off-diagonal bias, just 2-3x
smaller), one constant per plane removes it:

    tan(th_corr) = tan(th_det) - sign(tan th_det) * b_plane

with b_plane measured as the plateau median of
|tan th_det| - |tan th_ref| on the unshared angles themselves.

Pipeline: identical to 27 (alpha = 0.5 mixed kernel), then measure b per
plane, apply, and produce the final bias/resolution figures and numbers.

Usage: ../.venv/bin/python 28_angle_calibration.py sat_det3 [--veto=50]
Output: <bias_study>/angle_calibration.png + stdout numbers
"""
import os
import sys
import glob
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
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
SAMPLE_NS = 60.0
MIN_STRIPS_AFTER = 3
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
ALPHA = 0.5
PLATEAU_TAN = (0.12, 0.55)   # |tan| window for measuring the residual offset

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
# NB: FEU numbers are REUSED across detectors/runs (e.g. FEU 8 = det3's Y in some runs,
# det7's Y, AND det4's Y) -- this dict must hold ONLY the CURRENT detector's own FEUs,
# freshly measured by 26_unsharing_analysis.py for THAT run, edited immediately before
# each detector's 27/28 pass. Do not carry stale entries from a previous detector.
CSHARE = {6: (0.247, 0.057), 8: (0.514, 0.232)}   # det7 (g_det7_long), measured 2026-07-14 (chi2<1+NClus4)


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
    cache_res = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache_res, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)
    th = np.deg2rad(best.theta_deg)
    ref = {}
    for r in results:
        if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
                or r.radial_residual_mm > RES_CUT_MM or np.isnan(r.ref_tan_theta_x):
            continue
        tx = np.cos(th) * r.ref_tan_theta_x + np.sin(th) * r.ref_tan_theta_y
        ty = -np.sin(th) * r.ref_tan_theta_x + np.cos(th) * r.ref_tan_theta_y
        ref[r.event_id] = (tx, ty)
    print(f'{len(ref):,} matched events')

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

    S_all = {f: [] for f in CFG.MX17_FEUS}
    T_all = {f: [] for f in CFG.MX17_FEUS}
    for feu in CFG.MX17_FEUS:
        pi = 0 if plane_of_feu[feu] == 'x' else 1
        c1, c2 = CSHARE[feu]
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
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
                        continue                         # malformed multi-frame event
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
                    S_all[feu].append(1000.0 / sl)
                    T_all[feu].append(ref[eid][pi])
            print(f'  {os.path.basename(fn)} done ({len(S_all[feu]):,})')

    # per-plane velocity, residual offset b, and calibration
    print('\n== calibration constants ==')
    dth_raw, dth_cal, thr_comb = [], [], []
    for feu in CFG.MX17_FEUS:
        p = plane_of_feu[feu]
        S = np.array(S_all[feu])
        T = np.array(T_all[feu])
        vs = []
        for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
            m = (T > lo) & (T < hi) & np.isfinite(S)
            v, b, se, n = robust_line(T[m], S[m])
            if np.isfinite(v) and n > 100:
                vs.append(v)
        v = np.mean(vs)
        tan_det = S / v
        m_pl = (np.abs(T) > PLATEAU_TAN[0]) & (np.abs(T) < PLATEAU_TAN[1]) \
            & np.isfinite(tan_det)
        b_res = float(np.median(np.abs(tan_det[m_pl]) - np.abs(T[m_pl])))
        print(f'  {p}: v = {v:.2f} µm/ns   residual offset b = {b_res:+.4f} '
              f'(= {np.degrees(np.arctan(b_res)):+.2f}° at 0)')
        tan_cor = tan_det - np.sign(tan_det) * b_res
        m = np.isfinite(tan_det) & (np.abs(T) < 0.7)
        dth_raw.extend(np.degrees(np.arctan(tan_det[m])) - np.degrees(np.arctan(T[m])))
        dth_cal.extend(np.degrees(np.arctan(tan_cor[m])) - np.degrees(np.arctan(T[m])))
        thr_comb.extend(np.degrees(np.arctan(T[m])))
    dth_raw, dth_cal = np.array(dth_raw), np.array(dth_cal)
    thr_comb = np.array(thr_comb)

    # final bias/resolution profiles
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    prof = {}
    for name, dth, c in [('unshared', dth_raw, 'tab:blue'),
                         ('unshared + calibrated', dth_cal, 'crimson')]:
        ctr, med, sig = [], [], []
        for b0 in np.arange(-32, 32, 2.5):
            m = (thr_comb >= b0) & (thr_comb < b0 + 2.5)
            if m.sum() < 80:
                continue
            q = np.percentile(dth[m], [16, 50, 84])
            ctr.append(b0 + 1.25); med.append(q[1]); sig.append(0.5 * (q[2] - q[0]))
        prof[name] = (np.array(ctr), np.array(med), np.array(sig))
        axes[1].plot(ctr, med, 'o-', color=c, label=name)
        axes[2].plot(ctr, sig, 's-', color=c, label=name)

    ax = axes[0]
    h = ax.hist2d(thr_comb, thr_comb + dth_cal, bins=[70, 70],
                  range=[[-35, 35], [-35, 35]], norm=LogNorm(), cmap='viridis')
    ax.plot([-35, 35], [-35, 35], 'r--', lw=1)
    ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('θ_det (corrected) [deg]')
    ax.set_title('final: unshared + calibrated correlation')

    axes[1].axhline(0, color='k', lw=1)
    axes[1].set_xlabel('θ_ref [deg]'); axes[1].set_ylabel('median Δθ [deg]')
    axes[1].set_title('angular bias'); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-5, 5); axes[1].legend(fontsize=9)
    axes[2].set_xlabel('θ_ref [deg]'); axes[2].set_ylabel('σ(Δθ) [deg] (68%)')
    axes[2].set_title('angular resolution'); axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, 10); axes[2].legend(fontsize=9)
    fig.suptitle(f'{CFG.RUN} — final micro-TPC angles: unsharing (α={ALPHA}) '
                 '+ additive tan-space calibration', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'angle_calibration.png'), dpi=160)

    ctr, med, sig = prof['unshared + calibrated']
    pl = np.abs(ctr) > 8
    print(f'\nfinal calibrated: plateau |bias| median = '
          f'{np.median(np.abs(med[pl])):.2f}°, plateau σ = '
          f'{np.median(sig[pl]):.2f}°')
    print(f'Plot: {OUT}/angle_calibration.png')


if __name__ == '__main__':
    main()
