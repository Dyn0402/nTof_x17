#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_unsharing_analysis.py

Full from-scratch waveform analysis of the long run with charge UNSHARING —
the prototype of the tracking fix, to establish feasibility before touching
the production hit converter.

Pipeline (per FEU, decoded_root waveforms):
  1. pedestal (per channel) + common-noise (per chip, per sample) subtraction;
  2. MEASURE the sharing from near-vertical tracks: amplitude ratios
     A(+-1)/A(0), A(+-2)/A(0) and the CFD time offset of neighbours relative
     to the lead strip (prompt vs delayed), per plane;
  3. UNSHARE: solve the banded system (I + c1 E1 + c2 E2) x = w per time
     sample, over position-ordered contiguous strip blocks;
  4. re-extract strip times (CFD) and amplitudes from the unshared
     waveforms, recluster, core-OLS ladder fits;
  5. detector-frame ridge -> v BEFORE vs AFTER unsharing, against
     v_geom = 33.9 (acceptance criterion), plus geometry observables after.

Toy closure (25_signal_formation_toy.py --unshare): with known sharing the
same procedure recovers v_true to ~3% (v_cfd 27.4 -> 33.1 at v_true=34).

Usage: ../.venv/bin/python 26_unsharing_analysis.py sat_det3 [--veto=50] [--refit]
Output: <bias_study>/unsharing_{summary.csv, ladders.png, sharing.png, v.png}
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
from scipy.linalg import solve_banded

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
REFIT = '--refit' in sys.argv
SAMPLE_NS = 60.0
MIN_STRIPS_BEFORE = 4
MIN_STRIPS_AFTER = 3
RES_CUT_MM = 10.0
CHI2_CUT = 5.0   # M3 v2 recipe (chi2<5; NClus>=3 automatic in M3RefTracking); was 20 pre-v2
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
V_GEOM = 33.9
SAT_DEG = 10.0

tag = f'_veto{VETO}'
OUT = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
DEC_DIR = os.path.join(CFG.BASE_PATH, CFG.RUN, CFG.SUB_RUN, 'decoded_root')
CACHE = os.path.join(OUT, 'unsharing_striptimes.pkl')
TS = np.arange(32) * SAMPLE_NS


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


def unshare_block(wf_block, c1, c2):
    """wf_block: (n_strips, 32). Solve (I + c1 E1 + c2 E2) x = w per sample."""
    n = wf_block.shape[0]
    if n < 3:
        return wf_block
    ab = np.zeros((5, n))
    ab[0, 2:] = c2
    ab[1, 1:] = c1
    ab[2, :] = 1.0
    ab[3, :-1] = c1
    ab[4, :-2] = c2
    return solve_banded((2, 2), ab, wf_block)


def build(ref, det):
    """Loop decoded files: measure sharing, then unshare + retime."""
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    # position-ordered channel blocks per feu
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
        print(f'FEU {feu}: {len(ok)} mapped channels in {len(blocks[feu])} '
              f'contiguous blocks (sizes {[len(b) for b in blocks[feu]][:4]}...)')

    # pass 1: sharing measurement on near-vertical tracks
    share = {f: {'r1': [], 'r2': [], 'dt1': []} for f in CFG.MX17_FEUS}
    # pass 2 storage
    rows = []

    for feu in CFG.MX17_FEUS:
        p = plane_of_feu[feu]
        pi = 0 if p == 'x' else 1
        fs = sorted(glob.glob(os.path.join(DEC_DIR, f'*_{feu:02d}.root')))
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
                    wfm = wfm.T   # (512, 32)

                    # --- sharing measurement (vertical tracks only) ---
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
                                tc = cfd_time(wb[kk])
                                if np.isfinite(tc) and np.isfinite(t0c):
                                    share[feu]['dt1'].append(tc - t0c)
                    rows.append((feu, eid, tanv, wfm.astype(np.float16)))
            print(f'  {os.path.basename(fn)}: {len(rows):,} event-planes buffered')
    return blocks, pos_of, share, rows


def retime(rows, blocks, pos_of, cshare, unshared=True):
    """Per event-plane: (un)share, threshold, cluster, CFD, core OLS slope."""
    plane_of_feu = {CFG.MX17_FEU_X: 'x', CFG.MX17_FEU_Y: 'y'}
    out = {f: {'S': [], 'tan': [], 'ext': [], 'span': []} for f in CFG.MX17_FEUS}
    min_strips = MIN_STRIPS_AFTER if unshared else MIN_STRIPS_BEFORE
    for feu, eid, tanv, wfm16 in rows:
        wfm = wfm16.astype(np.float32)
        best = None
        for blk in blocks[feu]:
            wb = wfm[blk]
            if unshared:
                c1, c2 = cshare[feu]
                wb = unshare_block(wb, c1, c2)
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
        d = out[feu]
        d['S'].append(1000.0 / sl)
        d['tan'].append(tanv)
        d['ext'].append(np.ptp(pos))
        d['span'].append(np.nanmax(tt[ok]) - np.nanmin(tt[ok]))
    return out


def ridge_v(d):
    S, T = np.array(d['S']), np.array(d['tan'])
    vs = []
    for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
        m = (T > lo) & (T < hi) & np.isfinite(S)
        v, b, se, n = robust_line(T[m], S[m])
        if np.isfinite(v) and n > 100:
            vs.append(v)
    return (np.mean(vs), np.std(vs, ddof=1) / np.sqrt(len(vs))) if vs else (np.nan, np.nan)


def geom_v(d):
    T = np.abs(np.array(d['tan']))
    ext = np.array(d['ext'])
    span = np.array(d['span'])
    ctr, med = [], []
    for b0 in np.arange(0.06, 0.44, 0.04):
        m = (T >= b0) & (T < b0 + 0.04)
        if m.sum() >= 40:
            ctr.append(b0 + 0.02)
            med.append(np.median(ext[m]))
    if len(ctr) < 4:
        return np.nan, np.nan, np.nan
    slope = np.polyfit(ctr, med, 1)[0]
    msat = T > np.tan(np.radians(SAT_DEG))
    tsat = np.median(span[msat])
    return slope * 1000.0 / tsat, slope, tsat


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

    blocks, pos_of, share, rows = build(ref, det)

    # ---- sharing summary ----
    cshare = {}
    print('\n== measured sharing (near-vertical tracks) ==')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, feu in enumerate(CFG.MX17_FEUS):
        r1 = np.array(share[feu]['r1']); r2 = np.array(share[feu]['r2'])
        dt1 = np.array(share[feu]['dt1'])
        c1, c2 = float(np.median(r1)), float(np.median(np.clip(r2, 0, None)))
        cshare[feu] = (c1, c2)
        print(f'  FEU {feu}: c1 = {c1:.3f}  c2 = {c2:.3f}  '
              f'neighbour dt = {np.median(dt1):+.0f} ns '
              f'(n={len(r1)//2:,} leads)')
        ax = axes[j]
        ax.hist(r1, bins=np.linspace(-0.1, 1.0, 56), histtype='step', lw=2,
                color='tab:blue', label=f'A(±1)/A(0), med={c1:.3f}')
        ax.hist(r2, bins=np.linspace(-0.1, 1.0, 56), histtype='step', lw=2,
                color='tab:orange', label=f'A(±2)/A(0), med={c2:.3f}')
        ax.set_xlabel('neighbour / lead amplitude')
        ax.set_title(f'FEU {feu} ({"x" if feu==CFG.MX17_FEU_X else "y"} plane), '
                     f'Δt(±1) med = {np.median(dt1):+.0f} ns')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle(f'{CFG.RUN} — charge sharing measured from vertical tracks')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'unsharing_sharing.png'), dpi=150)
    plt.close(fig)

    # ---- before/after ----
    print('\n== ridge & geometry, BEFORE vs AFTER unsharing ==')
    res = {}
    for name, do_un in [('before', False), ('after', True)]:
        out = retime(rows, blocks, pos_of, cshare, unshared=do_un)
        res[name] = out
        for feu in CFG.MX17_FEUS:
            v, ve = ridge_v(out[feu])
            vg, zs, ts = geom_v(out[feu])
            p = 'x' if feu == CFG.MX17_FEU_X else 'y'
            print(f'  {name:6s} {p}: v_cfd_core = {v:6.2f} ± {ve:4.2f}   '
                  f'v_geom = {vg:6.2f} (z_slope {zs:5.1f} mm, T_sat {ts:4.0f} ns)')

    # ---- summary figure ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, vals, errs, cols = [], [], [], []
    for name, c in [('before', 'gray'), ('after', 'steelblue')]:
        for feu in CFG.MX17_FEUS:
            p = 'x' if feu == CFG.MX17_FEU_X else 'y'
            v, ve = ridge_v(res[name][feu])
            labels.append(f'{name} {p}')
            vals.append(v); errs.append(ve); cols.append(c)
    yp = np.arange(len(labels))
    ax.barh(yp, vals, xerr=errs, color=cols, alpha=0.85)
    ax.axvline(V_GEOM, color='crimson', lw=2, label=f'v_geom = {V_GEOM}')
    ax.set_yticks(yp); ax.set_yticklabels(labels)
    ax.set_xlabel('CFD core-OLS ridge v [µm/ns]')
    ax.set_xlim(20, 40); ax.grid(alpha=0.3, axis='x'); ax.legend()
    ax.set_title('time-fit velocity before/after waveform unsharing')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'unsharing_v.png'), dpi=160)
    print(f'\nPlots in {OUT}')


if __name__ == '__main__':
    main()
