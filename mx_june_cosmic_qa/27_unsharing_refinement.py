#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_unsharing_refinement.py

(a) Refine the unsharing kernel with the measured one-sample neighbour
    delay: shared charge = alpha*prompt + (1-alpha)*delayed(1 sample) for
    +-1 neighbours (2 samples for +-2). The delayed part makes the system
    causal in time, so it is solved sample-by-sample: a banded solve for
    the prompt part with an explicit RHS from already-unshared earlier
    samples. alpha is tuned so that AFTER unsharing the vertical-track
    neighbour/lead ratio nulls (the physical target: neighbours of an
    isolated strip should keep only real transverse-diffusion charge).

(b) Angular resolution with unshared times: per-plane angle correlation
    theta_det vs theta_ref and the resolution sigma(theta_det - theta_ref)
    vs theta_ref, BEFORE vs AFTER unsharing. Each frame uses its own ridge
    velocity for tan conversion, so the comparison isolates shape/resolution
    gains from the velocity-scale fix.

Also: per-event ladder displays (same events, before vs after).

Usage: ../.venv/bin/python 27_unsharing_refinement.py sat_det3 [--veto=50]
Output: <bias_study>/unsharing_kernel_scan (stdout table),
        unsharing_ladders.png, unsharing_angles.png, unsharing_resolution.png
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
MIN_STRIPS_BEFORE = 4
RES_CUT_MM = 10.0
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
PITCH_MM = 0.78
THR_HIT = 100.0
THR_WF = 150.0
CORE_FRAC = 0.30
N_PED_EVENTS = 300
CHUNK = 400
V_GEOM = 33.9
ALPHAS = (1.0, 0.5, 0.0)

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
    """Sum of k-shifted copies along strips: x[i-k] + x[i+k]."""
    out = np.zeros_like(x)
    out[k:] += x[:-k]
    out[:-k] += x[k:]
    return out


def unshare(wb, c1, c2, alpha):
    """Mixed prompt/delayed kernel, causal in samples.
    W[:,s] = X[:,s] + a*c1*E1 X[:,s] + (1-a)*c1*E1 X[:,s-1]
                    + a*c2*E2 X[:,s] + (1-a)*c2*E2 X[:,s-2]."""
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


def load_events(ref, det):
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
        pi = 0 if plane_of_feu[feu] == 'x' else 1
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
                    rows.append((feu, eid, ref[eid][pi], wfm.T.astype(np.float16)))
            print(f'  loaded {os.path.basename(fn)} ({len(rows):,} event-planes)')
    return blocks, pos_of, rows


def process(rows, blocks, pos_of, mode, alpha=1.0):
    """mode: 'before' (no unsharing) or 'after'. Returns per-feu dicts and
    vertical-track post-processing neighbour ratios."""
    out = {f: {'S': [], 'tan': [], 'eid': []} for f in CFG.MX17_FEUS}
    vratio = {f: [] for f in CFG.MX17_FEUS}
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
            # vertical-track nulling metric
            if abs(tanv) < 0.03:
                k = int(np.argmax(amax))
                if amax[k] > 500 and 2 <= k <= len(blk) - 3:
                    vratio[feu].extend([amax[k - 1] / amax[k], amax[k + 1] / amax[k]])
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
        d['eid'].append(eid)
    return out, vratio


def ridge_v(d):
    S, T = np.array(d['S']), np.array(d['tan'])
    vs = []
    for lo, hi in [(0.06, 0.55), (-0.55, -0.06)]:
        m = (T > lo) & (T < hi) & np.isfinite(S)
        v, b, se, n = robust_line(T[m], S[m])
        if np.isfinite(v) and n > 100:
            vs.append(v)
    return np.mean(vs) if vs else np.nan


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
    blocks, pos_of, rows = load_events(ref, det)

    # ---- (a) kernel scan ----
    print('\n== kernel scan (alpha = prompt fraction) ==')
    print(f'{"mode":>12} {"resid r1 x":>10} {"resid r1 y":>10} '
          f'{"v_x":>7} {"v_y":>7}')
    results_by_mode = {}
    outB, vrB = process(rows, blocks, pos_of, 'before')
    results_by_mode['before'] = outB
    fx, fy = CFG.MX17_FEU_X, CFG.MX17_FEU_Y
    print(f'{"before":>12} {np.median(vrB[fx]):10.3f} {np.median(vrB[fy]):10.3f} '
          f'{ridge_v(outB[fx]):7.2f} {ridge_v(outB[fy]):7.2f}')
    best_alpha, best_score = None, 1e9
    for a in ALPHAS:
        outA, vrA = process(rows, blocks, pos_of, 'after', alpha=a)
        r1x, r1y = np.median(vrA[fx]), np.median(vrA[fy])
        vx, vy = ridge_v(outA[fx]), ridge_v(outA[fy])
        results_by_mode[f'after a={a:.1f}'] = outA
        score = abs(r1x) + abs(r1y)
        if score < best_score:
            best_alpha, best_score = a, score
        print(f'{"a="+format(a,".1f"):>12} {r1x:10.3f} {r1y:10.3f} '
              f'{vx:7.2f} {vy:7.2f}')
    print(f'best alpha by neighbour nulling: {best_alpha}')
    outA = results_by_mode[f'after a={best_alpha:.1f}']

    # ---- (a) ladder displays: same events before vs after ----
    ev_pick = []
    for feu, eid, tanv, wfm16 in rows:
        if feu == CFG.MX17_FEU_X and 0.25 < abs(tanv) < 0.55:
            ev_pick.append((feu, eid, tanv, wfm16))
        if len(ev_pick) >= 4:
            break
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (feu, eid, tanv, wfm16) in enumerate(ev_pick):
        wfm = wfm16.astype(np.float32)
        c1, c2 = CSHARE[feu]
        blk = max(blocks[feu], key=len)
        for rowi, (name, wb) in enumerate([
                ('before', wfm[blk]),
                (f'after (α={best_alpha:g})', unshare(wfm[blk], c1, c2, best_alpha))]):
            ax = axes[rowi, col]
            amax = wb.max(axis=1)
            hit = np.where(amax >= THR_HIT)[0]
            if len(hit) < 3:
                continue
            brk = np.where(np.diff(hit) > 2)[0]
            grp = max(np.split(hit, brk + 1), key=len)
            pos = pos_of[feu][blk[grp]]
            amp = wb[grp].max(axis=1)
            tt = np.array([cfd_time(wb[g]) for g in grp])
            sc = ax.scatter(pos, tt, c=np.clip(amp, 80, 4000), s=45,
                            cmap='viridis', norm=LogNorm(vmin=80, vmax=4000),
                            zorder=5, edgecolors='k', linewidths=0.4)
            ok = np.isfinite(tt)
            mcore = (amp >= CORE_FRAC * amp.max()) & ok
            if mcore.sum() >= 3:
                pfit = np.polyfit(pos[mcore], tt[mcore], 1)
                xx = np.linspace(pos.min() - 1, pos.max() + 1, 10)
                ax.plot(xx, np.polyval(pfit, xx), '-', color='crimson', lw=1.4,
                        label=f'fit: {1000.0/pfit[0]/tanv:+.1f} µm/ns / tan')
            i0 = np.nanargmin(np.where(ok, tt, np.inf))
            xx = np.linspace(pos.min() - 1, pos.max() + 1, 10)
            ax.plot(xx, tt[i0] + (xx - pos[i0]) * 1000.0 / (V_GEOM * tanv),
                    '--', color='gray', lw=1.2, label='geometric (v=33.9)')
            ax.set_title(f'evt {eid} tan={tanv:+.2f} — {name}', fontsize=9)
            ax.set_xlabel('x [mm]'); ax.set_ylabel('t_CFD [ns]')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f'{CFG.RUN} — ladders before/after unsharing (X plane)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'unsharing_ladders.png'), dpi=150)
    plt.close(fig)

    # ---- (b) angle correlation + resolution, before vs after ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    res_curves = {}
    for j, (name, out) in enumerate([('before', outB),
                                     (f'after (α={best_alpha:g})', outA)]):
        # combine planes with per-plane velocity scale
        dth_all, thr_all = [], []
        for feu in CFG.MX17_FEUS:
            d = out[feu]
            v = ridge_v(d)
            S, T = np.array(d['S']), np.array(d['tan'])
            m = np.isfinite(S) & (np.abs(T) < 0.7)
            th_det = np.degrees(np.arctan(S[m] / v))
            th_ref = np.degrees(np.arctan(T[m]))
            dth_all.extend(th_det - th_ref)
            thr_all.extend(th_ref)
        dth_all, thr_all = np.array(dth_all), np.array(thr_all)

        ax = axes[j, 0]
        h = ax.hist2d(thr_all, thr_all + dth_all, bins=[70, 70],
                      range=[[-35, 35], [-35, 35]], norm=LogNorm(), cmap='viridis')
        ax.plot([-35, 35], [-35, 35], 'r--', lw=1)
        ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('θ_det [deg]')
        ax.set_title(f'{name}: angle correlation (own-plane v scale)')

        ctr, med, sig = [], [], []
        for b0 in np.arange(-32, 32, 2.5):
            m = (thr_all >= b0) & (thr_all < b0 + 2.5)
            if m.sum() < 80:
                continue
            q = np.percentile(dth_all[m], [16, 50, 84])
            ctr.append(b0 + 1.25); med.append(q[1]); sig.append(0.5 * (q[2] - q[0]))
        res_curves[name] = (np.array(ctr), np.array(med), np.array(sig))

        ax = axes[j, 1]
        ax.plot(ctr, med, 'o-', color='tab:blue')
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('median θ_det − θ_ref [deg]')
        ax.set_title(f'{name}: angular bias'); ax.grid(alpha=0.3)
        ax.set_ylim(-8, 8)

        ax = axes[j, 2]
        ax.plot(ctr, sig, 's-', color='tab:red')
        ax.set_xlabel('θ_ref [deg]'); ax.set_ylabel('σ(θ_det − θ_ref) [deg]  (68%)')
        ax.set_title(f'{name}: angular resolution'); ax.grid(alpha=0.3)
        ax.set_ylim(0, 10)

    fig.suptitle(f'{CFG.RUN} — micro-TPC angles before vs after waveform '
                 'unsharing (CFD core fits)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, 'unsharing_angles.png'), dpi=155)
    plt.close(fig)

    # resolution overlay
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, c in [('before', 'gray'), (f'after (α={best_alpha:g})', 'crimson')]:
        ctr, med, sig = res_curves[name]
        ax.plot(ctr, sig, 'o-', color=c, label=name)
    ax.set_xlabel('θ_ref [deg]')
    ax.set_ylabel('σ(θ_det − θ_ref) [deg]  (68% half-width)')
    ax.set_title(f'{CFG.DET_NAME} micro-TPC angular resolution — effect of unsharing')
    ax.grid(alpha=0.3); ax.legend()
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'unsharing_resolution.png'), dpi=160)
    print(f'\nPlots in {OUT}')


if __name__ == '__main__':
    main()
