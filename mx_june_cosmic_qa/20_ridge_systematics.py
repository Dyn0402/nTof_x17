#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_ridge_systematics.py

Stress-test the ridge-fit drift velocity against an ANGLE-DEPENDENT spreading
bias, and measure the resistive-strip anisotropy per physical plane.

Motivation: the ridge model S = v*(tan(th) +/- w/z) assumes w is
angle-independent. If w = w(th), the fitted slope is v*(1 + dw/dz/dtan) and
v trades 1:1 against w'. The hard time cutoff T_sat is solid, so
z_rec = v*T_sat inherits any v bias: v could in principle be as high as
gap/T_sat = 30mm/690ns = 43.5 um/ns (then the cutoff IS the full gap).

Tests (all in the DETECTOR frame, per physical plane FEU-X / FEU-Y):
 1. Per-plane, per-sign ridge fits -> v_plane, intercept_plane. Anisotropic
    spreading (resistive strips!) => very different intercepts; if the
    low-spreading plane gives the same slope, the w(th) loophole shrinks.
 2. Cluster spatial extent [mm] vs |tan th_ref| (NO time information):
    slope = z_rec + w'(th). Extent floor at th~0 = w(0).
 3. Ridge straightness: independent fits in split windows
    (0.06-0.28 / 0.28-0.55) -> local slopes; curvature bounds w'.
 4. Extent vs cluster max amplitude at fixed angle (threshold-skirt check).

Usage: ../.venv/bin/python 20_ridge_systematics.py [sat_det3] [--veto=50]
Output: <out>/alignment_tpc_veto50/bias_study/ridge_systematics.png + stdout
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
MIN_STRIPS = 4
RES_CUT_MM = 10.0
PITCH_MM = 0.78
from qa_config import M3_CHI2_CUT as CHI2_CUT, M3_MIN_NCLUS  # centralized M3 recipe (see qa_config.py)
GAP_MM = 30.0


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


def main():
    tag = f'_veto{VETO}'
    out_dir = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
    cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
    align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align_json)
    rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    xang, _, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)

    sel = [r for r in results if r.has_x and r.has_y
           and np.isfinite(r.ref_tan_theta_x) and np.isfinite(r.ref_tan_theta_y)
           and np.isfinite(r.x_fit.slope_mm_per_ns) and np.isfinite(r.y_fit.slope_mm_per_ns)
           and r.x_fit.n_strips >= MIN_STRIPS and r.y_fit.n_strips >= MIN_STRIPS
           and np.isfinite(r.radial_residual_mm) and r.radial_residual_mm < RES_CUT_MM]
    print(f'{len(sel):,} quality events')

    s_x = np.array([r.x_fit.slope_mm_per_ns for r in sel]) * 1000.0
    s_y = np.array([r.y_fit.slope_mm_per_ns for r in sel]) * 1000.0
    ns_x = np.array([r.x_fit.n_strips for r in sel])
    ns_y = np.array([r.y_fit.n_strips for r in sel])
    dur_x = np.array([r.x_fit.latest_time_ns - r.x_fit.earliest_time_ns for r in sel])
    dur_y = np.array([r.y_fit.latest_time_ns - r.y_fit.earliest_time_ns for r in sel])
    tan_rx = np.array([r.ref_tan_theta_x for r in sel])
    tan_ry = np.array([r.ref_tan_theta_y for r in sel])

    # reference angles rotated INTO the detector frame (inverse of the slope
    # rotation used in production: S_M3 = R(theta) s_det  =>  t_det = R(-theta)^T ... )
    th = np.deg2rad(best.theta_deg)
    t_x_det = np.cos(th) * tan_rx + np.sin(th) * tan_ry
    t_y_det = -np.sin(th) * tan_rx + np.cos(th) * tan_ry

    # sanity: sign/pairing via correlation
    for name, s, t in [('X(FEU%d)' % CFG.MX17_FEU_X, s_x, t_x_det),
                       ('Y(FEU%d)' % CFG.MX17_FEU_Y, s_y, t_y_det)]:
        c = np.corrcoef(s, t)[0, 1]
        print(f'  pairing check {name}: corr(s, t_det) = {c:+.3f}')

    planes = {
        f'X strips (FEU {CFG.MX17_FEU_X})': dict(s=s_x, t=t_x_det, ns=ns_x, dur=dur_x),
        f'Y strips (FEU {CFG.MX17_FEU_Y})': dict(s=s_y, t=t_y_det, ns=ns_y, dur=dur_y),
    }

    # ---- 1+3. per-plane per-sign ridge fits, full and split windows ----
    print('\n== Ridge fits per physical plane (detector frame) ==')
    WINDOWS = [('full 0.06-0.55', 0.06, 0.55),
               ('inner 0.06-0.28', 0.06, 0.28),
               ('outer 0.28-0.55', 0.28, 0.55)]
    summary = {}
    for pname, d in planes.items():
        s, t = d['s'], d['t']
        for wname, lo, hi in WINDOWS:
            vs, bs = [], []
            for sgn in (+1, -1):
                m = (sgn * t > lo) & (sgn * t < hi)
                v, b, ve, n = robust_line(t[m], s[m])
                if np.isfinite(v) and n > 50:
                    vs.append(v)
                    bs.append(abs(b))
            if vs:
                v_m, b_m = np.mean(vs), np.mean(bs)
                summary[(pname, wname)] = (v_m, b_m)
                print(f'  {pname:22s} {wname:17s} v = {v_m:6.2f} µm/ns   '
                      f'|intercept| = {b_m:5.2f} µm/ns   (w=b*T_sat={b_m*0.69:.2f} mm @1000V)')

    # ---- per-plane time-span saturation ----
    print('\n== Time-span plateau per plane ==')
    tsat = {}
    for pname, d in planes.items():
        m = np.abs(np.degrees(np.arctan(d['t']))) > 10
        tsat[pname] = np.median(d['dur'][m])
        print(f'  {pname}: T_sat = {tsat[pname]:.0f} ns  '
              f'(v_max if full {GAP_MM:.0f} mm gap: {GAP_MM*1000/tsat[pname]:.1f} µm/ns)')

    # ---- 2. spatial extent vs |tan| in mm (geometry only, NO times) ----
    print('\n== Cluster extent [mm] vs |tanθ_ref| (slope = z_rec + w\') ==')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    zslopes = {}
    for j, (pname, d) in enumerate(planes.items()):
        ext = (d['ns'] - 1) * PITCH_MM
        at = np.abs(d['t'])
        # median extent per |tan| bin, then line through the medians (robust to tails)
        bins = np.arange(0.06, 0.60, 0.04)
        ctr, med = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (at >= b0) & (at < b1)
            if m.sum() >= 50:
                ctr.append(0.5 * (b0 + b1))
                med.append(np.median(ext[m]))
        ctr, med = np.array(ctr), np.array(med)
        p = np.polyfit(ctr, med, 1)
        zslopes[pname] = p[0]
        floor = np.median(ext[at < 0.05])
        print(f'  {pname}: slope = {p[0]:5.1f} mm/unit-tan   '
              f'intercept = {p[1]:4.1f} mm   floor(|tan|<0.05) = {floor:.1f} mm '
              f'(min-strips cut floor = {(MIN_STRIPS-1)*PITCH_MM:.1f} mm)')

        ax = axes[0, j]
        ax.plot(ctr, med, 'o', color='k', ms=6, label='median extent')
        tt = np.linspace(0, 0.6, 10)
        ax.plot(tt, np.polyval(p, tt), '-', color='crimson', lw=1.6,
                label=f'fit: slope = {p[0]:.1f} mm')
        for z, ls in [(19.4, '--'), (30.0, ':')]:
            ax.plot(tt, z * tt + p[1], ls, color='gray', lw=1.2,
                    label=f'{z:g} mm · tanθ + const')
        ax.set_xlabel('|tanθ_ref| (detector frame)')
        ax.set_ylabel('cluster extent [mm]')
        ax.set_title(f'{pname}: extent vs angle (no time info)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ---- 4. extent vs n-strip amplitude proxy at fixed angle band ----
        ax = axes[1, j]
        for lo, hi, c in [(0.0, 0.05, 'tab:blue'), (0.15, 0.30, 'tab:green'),
                          (0.35, 0.55, 'tab:red')]:
            m = (at >= lo) & (at < hi)
            if m.sum() < 200:
                continue
            ax.hist(ext[m], bins=np.arange(0, 25, PITCH_MM), histtype='step',
                    density=True, lw=1.8, color=c,
                    label=f'|tan| {lo:.2f}-{hi:.2f} (med {np.median(ext[m]):.1f} mm)')
        ax.set_xlabel('cluster extent [mm]'); ax.set_ylabel('norm.')
        ax.set_title(f'{pname}: extent distributions by angle band')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{CFG.RUN} — ridge systematics: per-plane anisotropy & '
                 f'geometry-only depth measurement', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'ridge_systematics.png'), dpi=160)

    # ---- verdict table ----
    print('\n== Verdict inputs ==')
    for pname in planes:
        v_full = summary[(pname, 'full 0.06-0.55')][0]
        z_t = v_full * tsat[pname] / 1000.0
        print(f'  {pname}: v_ridge = {v_full:.1f} → z_rec(time) = {z_t:.1f} mm ; '
              f'z_slope(geom) = {zslopes[pname]:.1f} mm ; '
              f'v(geom) = z_slope/T_sat = {zslopes[pname]*1000/tsat[pname]:.1f} µm/ns')
    print(f'\nPlot: {out_dir}/ridge_systematics.png')


if __name__ == '__main__':
    main()
